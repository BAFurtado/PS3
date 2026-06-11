import argparse
import sys
from pathlib import Path
from glob import glob

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from analysis.output import OUTPUT_DATA_SPEC


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "output"
OUT_DIR = BASE_DIR / "analysis" / "validation"

# Approximate "Brasil" benchmark ranges. Deliberately broad sanity bands, not
# precise calibration targets -- a model run inside these ranges is plausible,
# outside is worth a second look.
TARGETS = {
    "housing_stock_gdp": (1.5, 2.5),
    "price_income": (6, 12),
    "housing_production_per_1000": (2, 8),
    "consumption_gdp": (0.55, 0.65),
    "vacancy": (0.08, 0.12),
    "housing_stock_permanent_income": (5, 8),
    # Pooled-median target across cities/years, not a per-city or per-period
    # requirement. Cross-city dispersion is itself realistic: smaller/poorer
    # cities trend lower (~0.35-0.45), large unequal metros (SAO PAULO,
    # PORTO ALEGRE) trend higher, up to ~0.60. Early-period (pre-2015) model
    # gini (~0.32-0.34) sits below this floor -- a known, currently
    # unaddressed structural/initialization issue, not something to chase
    # by re-tuning housing-supply parameters.
    "gini": (0.40, 0.60),
    "zero_cons": (0.0, 0.05),
    "unemployment": (0.02, 0.15),
}

# Calibration window used across waves: 2011-2024 is the historical/calibration
# period, 2025-2039 is the long-run projection.
EVAL_START, EVAL_END = "2011-01-01", "2024-12-31"
LONG_START, LONG_END = "2025-01-01", "2039-12-01"


def find_stats_files(stats_file=None, base_dir=None):
    """Return the file(s) to load.

    If `stats_file` is given, use it as-is (any path, e.g. an externally
    aggregated final_statsNN.csv). Otherwise, glob for final_stats*.csv under
    `base_dir` (default OUTPUT_DIR) and pick the most recently modified one --
    `output/` accumulates one of these per wave, so a fixed filename goes
    stale every wave.
    """
    if stats_file:
        return [stats_file]

    base = Path(base_dir) if base_dir else OUTPUT_DIR
    files = glob(str(base / "**" / "final_stats*.csv"), recursive=True)
    files = [f for f in files if "/avg/" not in f and "sensitivity" not in Path(f).name]
    if not files:
        return []
    latest = max(files, key=lambda f: Path(f).stat().st_mtime)
    return [latest]


def load_stats(path):
    with open(path) as fh:
        first = fh.readline()
    if first.startswith("month"):
        return pd.read_csv(path, sep=",", low_memory=False)
    cols = OUTPUT_DATA_SPEC["stats"]["columns"]
    df = pd.read_csv(path, header=None, sep=";")
    df.columns = cols
    return df


def path_to_run_id(path):
    p = Path(path)
    try:
        rel = p.relative_to(OUTPUT_DIR)
        return str(rel.parent)
    except ValueError:
        return str(p.parent)


def safe_ratio(num, den):
    den = den.replace(0, np.nan)
    r = num / den
    return r.replace([np.inf, -np.inf], np.nan)


def compute_derived_monthly_indicators(df):
    df = df.copy()

    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    # -----------------------------
    # Core monthly level variables
    # -----------------------------
    df["housing_stock_value"] = df["number_domiciles"] * df["house_price"]

    # -----------------------------
    # Housing stock / GDP
    # stock over annualized GDP flow
    # -----------------------------
    df["housing_stock_gdp"] = safe_ratio(
        df["housing_stock_value"],
        df["gdp_level"] * 12
    )

    # -----------------------------
    # Housing stock / family wealth
    # proxy: median family wealth * number of domiciles
    # -----------------------------
    df["families_permanent_income"] = df["families_median_permanent_income"] * 12 * df["number_domiciles"]

    df["housing_stock_permanent_income"] = safe_ratio(
        df["housing_stock_value"],
        df["families_permanent_income"]
    )

    # -----------------------------
    # Price / monthly family income (months of income to buy a house)
    # Uses firms_wage_per_worker when available (correct per-worker median computed
    # only over active firms); falls back to families_wages_received (family level).
    # The old firms_median_wage_paid / firms_median_employment ratio is incorrect:
    # dividing two medians across all firms (including 0-wage firms) gives near-zero
    # per-worker wages and price/wage ratios in the thousands.
    # -----------------------------
    if "firms_wage_per_worker" in df.columns:
        df["price_wage"] = safe_ratio(df["house_price"], df["firms_wage_per_worker"])
    else:
        df["price_wage"] = safe_ratio(df["house_price"], df["families_wages_received"])

    # -----------------------------
    # Price / annual household wage income
    # -----------------------------
    df["price_income"] = safe_ratio(
        df["house_price"],
        df["families_wages_received"] * 12
    )

    # -----------------------------
    # Housing production
    # calculated within each run
    # -----------------------------
    if "run_id" in df.columns:
        df = df.sort_values(["run_id", "month"]).copy()
        df["new_houses"] = (
            df.groupby("run_id")["number_domiciles"]
            .diff()
            .fillna(0)
            .clip(lower=0)
        )
    else:
        df = df.sort_values("month").copy()
        df["new_houses"] = df["number_domiciles"].diff().fillna(0).clip(lower=0)

    df["houses_year"] = df["new_houses"] * 12

    df["housing_production_per_1000"] = safe_ratio(
        df["houses_year"] * 1000,
        df["pop"]
    )

    # -----------------------------
    # Consumption / GDP
    # assumes average_utility is monthly spending
    # and is already aligned with number_domiciles
    # -----------------------------
    df["total_consumption"] = df["average_utility"] * df["number_domiciles"]

    df["consumption_gdp"] = safe_ratio(
        df["total_consumption"],
        df["gdp_level"]
    )

    # -----------------------------
    # Vacancy
    # -----------------------------
    df["vacancy"] = df["house_vacancy"]

    # -----------------------------
    # Inequality / poverty / labor indicators
    # -----------------------------
    df["gini"] = df["gini_index"]
    df["zero_cons"] = df["pct_zero_consumption"]
    # "unemployment" already exists as a raw column under that name.

    return df


def summarize_columns(df, cols):
    rows = []

    for col in cols:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        valid = s.dropna()

        if valid.empty:
            rows.append({
                "variable": col,
                "count": 0,
                "missing": int(s.isna().sum()),
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "p10": np.nan,
                "p25": np.nan,
                "p75": np.nan,
                "p90": np.nan,
                "min": np.nan,
                "max": np.nan
            })
            continue

        rows.append({
            "variable": col,
            "count": int(valid.shape[0]),
            "missing": int(s.isna().sum()),
            "mean": valid.mean(),
            "median": valid.median(),
            "std": valid.std(),
            "p10": valid.quantile(0.10),
            "p25": valid.quantile(0.25),
            "p75": valid.quantile(0.75),
            "p90": valid.quantile(0.90),
            "min": valid.min(),
            "max": valid.max()
        })

    return pd.DataFrame(rows)


def compute_sanity_checks(df):
    checks = {}

    def share(condition):
        if len(condition) == 0:
            return np.nan
        return float(np.mean(condition))

    checks["rows"] = len(df)
    checks["month_min"] = df["month"].min() if "month" in df.columns else pd.NaT
    checks["month_max"] = df["month"].max() if "month" in df.columns else pd.NaT
    checks["unique_runs"] = df["run_id"].nunique() if "run_id" in df.columns else np.nan

    checks["share_gdp_le_zero"] = share(pd.to_numeric(df["gdp_level"], errors="coerce") <= 0)
    checks["share_wages_le_zero"] = share(pd.to_numeric(df["families_wages_received"], errors="coerce") <= 0)
    checks["share_firm_employment_le_zero"] = share(pd.to_numeric(df["firms_median_employment"], errors="coerce") <= 0)
    checks["share_house_price_le_zero"] = share(pd.to_numeric(df["house_price"], errors="coerce") <= 0)
    checks["share_pop_le_zero"] = share(pd.to_numeric(df["pop"], errors="coerce") <= 0)

    if "run_id" in df.columns:
        tmp = df.sort_values(["run_id", "month"]).copy()
        delta = tmp.groupby("run_id")["number_domiciles"].diff()
        checks["share_months_domiciles_fall"] = share(delta < 0)
    else:
        delta = df.sort_values("month")["number_domiciles"].diff()
        checks["share_months_domiciles_fall"] = share(delta < 0)

    derived_cols = [
        "housing_stock_gdp",
        "price_wage",
        "price_income",
        "housing_production_per_1000",
        "consumption_gdp",
        "vacancy",
        "gini",
        "zero_cons",
        "unemployment",
    ]

    for col in derived_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            checks[f"share_{col}_missing"] = float(s.isna().mean())

    return pd.Series(checks, name="value")


def aggregate_by_run(df, cols):
    if "run_id" not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby("run_id")[cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    return grouped


def write_csv(df, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    df.to_csv(path, index=False)
    return path


def compute_pass_fail(summary):
    """Compare pooled-mean indicators against TARGETS. Single-value indicators
    only -- price_wage is reported as a p10-p90 range and is excluded here."""
    mapping = {
        "housing_stock_gdp": "housing_stock_gdp_mean",
        "housing_stock_permanent_income": "housing_stock_permanent_income_mean",
        "price_income": "price_income_mean",
        "housing_production_per_1000": "housing_production_per_1000_mean",
        "consumption_gdp": "consumption_gdp_mean",
        "vacancy": "vacancy_mean",
        "gini": "gini_mean",
        "zero_cons": "zero_cons_mean",
        "unemployment": "unemployment_mean",
    }
    rows = []
    for indicator, summary_key in mapping.items():
        value = summary.get(summary_key, np.nan)
        low, high = TARGETS[indicator]
        passed = bool(pd.notna(value) and low <= value <= high)
        rows.append({
            "indicator": indicator,
            "value": value,
            "target_low": low,
            "target_high": high,
            "pass": passed,
        })
    return pd.DataFrame(rows)


def print_main_results(raw_summary, derived_summary, sanity, by_run_summary, pass_fail):
    print("\n==============================")
    print("POOLED STATS VALIDATION")
    print("==============================\n")

    print("General:")
    print(f"Rows: {sanity.get('rows')}")
    print(f"Unique runs: {sanity.get('unique_runs')}")
    print(f"Month range: {sanity.get('month_min')} -> {sanity.get('month_max')}")

    print("\nSanity checks:")
    for k, v in sanity.items():
        if k in {"rows", "unique_runs", "month_min", "month_max"}:
            continue
        print(f"{k}: {v}")

    print("\nDerived indicators summary:")
    if not derived_summary.empty:
        cols = ["variable", "mean", "median", "p10", "p90", "min", "max"]
        print(derived_summary[cols].to_string(index=False))

    if not by_run_summary.empty:
        print("\nBy-run means (summary across runs):")
        print(by_run_summary.to_string(index=False))

    print("\nRaw variables summary (selected):")
    if not raw_summary.empty:
        cols = ["variable", "mean", "median", "p10", "p90", "min", "max"]
        print(raw_summary[cols].to_string(index=False))

    print("\nPass/fail vs Brasil (aprox.) targets:")
    print(pass_fail.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main(stats_file=None, base_dir=None, city=None, run_type=None,
         month_start=None, month_cut_off=None, suffix=""):
    files = find_stats_files(stats_file=stats_file, base_dir=base_dir)
    if not files:
        raise RuntimeError("No stats.csv files found.")

    pooled = []

    for f in files:
        df = load_stats(f)
        df["source_file"] = str(f)
        if "simulation_id" in df.columns:
            df["run_id"] = df["simulation_id"]
        else:
            df["run_id"] = path_to_run_id(f)
        df["month"] = pd.to_datetime(df["month"], errors="coerce")

        if city and "processing_acps" in df.columns:
            df = df[df["processing_acps"] == city]
        if run_type and "run_type" in df.columns:
            df = df[df["run_type"] == run_type]
        if month_start:
            df = df[df["month"] >= pd.Timestamp(month_start)]
        if month_cut_off:
            df = df[df["month"] < pd.Timestamp(month_cut_off)]

        pooled.append(df)

    pooled_df = pd.concat(pooled, ignore_index=True)
    if pooled_df.empty:
        raise RuntimeError("No rows left after filtering (check --city/--run-type/--month-* filters).")
    pooled_df = pooled_df.sort_values(["run_id", "month"]).reset_index(drop=True)

    pooled_df = compute_derived_monthly_indicators(pooled_df)

    raw_cols = [
        "gdp_level",
        "house_price",
        "number_domiciles",
        "house_vacancy",
        "families_permanent_income",
        "families_wages_received",
        "firms_median_wage_paid",
        "firms_median_employment",
        "average_utility",
        "pop",
    ]

    derived_cols = [
        "housing_stock_gdp",
        "housing_stock_permanent_income",
        "price_wage",
        "price_income",
        "housing_production_per_1000",
        "consumption_gdp",
        "vacancy",
        "gini",
        "zero_cons",
        "unemployment",
    ]

    raw_summary = summarize_columns(pooled_df, raw_cols)
    derived_summary = summarize_columns(pooled_df, derived_cols)
    sanity = compute_sanity_checks(pooled_df)

    by_run = aggregate_by_run(pooled_df, derived_cols)
    by_run_summary = summarize_columns(by_run, derived_cols) if not by_run.empty else pd.DataFrame()

    summary = {
        "housing_stock_gdp_mean": pooled_df["housing_stock_gdp"].mean(),
        "housing_stock_permanent_income_mean": pooled_df["housing_stock_permanent_income"].mean(),
        "price_wage_p10": pooled_df["price_wage"].quantile(0.10),
        "price_wage_p90": pooled_df["price_wage"].quantile(0.90),
        "price_income_mean": pooled_df["price_income"].mean(),
        "housing_production_per_1000_mean": pooled_df["housing_production_per_1000"].mean(),
        "consumption_gdp_mean": pooled_df["consumption_gdp"].mean(),
        "vacancy_mean": pooled_df["vacancy"].mean(),
        "gini_mean": pooled_df["gini"].mean(),
        "zero_cons_mean": pooled_df["zero_cons"].mean(),
        "unemployment_mean": pooled_df["unemployment"].mean(),
    }
    summary = pd.Series(summary)

    pass_fail = compute_pass_fail(summary)

    pooled_path = write_csv(pooled_df, f"all_stats_pooled{suffix}.csv")
    raw_summary_path = write_csv(raw_summary, f"raw_variables_summary{suffix}.csv")
    derived_summary_path = write_csv(derived_summary, f"derived_indicators_summary{suffix}.csv")
    sanity_path = write_csv(sanity.reset_index().rename(columns={"index": "check", "value": "value"}),
                            f"sanity_checks{suffix}.csv")
    by_run_path = write_csv(by_run, f"derived_indicators_by_run{suffix}.csv") if not by_run.empty else None
    by_run_summary_path = write_csv(by_run_summary,
                                    f"derived_indicators_by_run_summary{suffix}.csv") if not by_run_summary.empty else None
    pass_fail_path = write_csv(pass_fail, f"pass_fail{suffix}.csv")
    latex_path = write_latex(summary, pass_fail, suffix)

    print_main_results(raw_summary, derived_summary, sanity, by_run_summary, pass_fail)

    print("\nSource file(s):")
    for f in files:
        print(f"  {f}")

    print("\nOutputs:")
    print(pooled_path)
    print(raw_summary_path)
    print(derived_summary_path)
    print(sanity_path)
    if by_run_path:
        print(by_run_path)
    if by_run_summary_path:
        print(by_run_summary_path)
    print(pass_fail_path)

    print(latex_path)


def write_latex(summary, pass_fail, suffix=""):
    pf = pass_fail.set_index("indicator")["pass"]
    mark = lambda ind: "OK" if pf.get(ind, False) else "X"
    rng = lambda ind: "{:g}--{:g}".format(*TARGETS[ind])

    table = f"""
\\begin{{tabular}}{{lccc}}
\\hline
Indicador & Modelo & Brasil (aprox.) & OK? \\\\
\\hline
Estoque imobiliário / PIB & {summary['housing_stock_gdp_mean']:.2f} & {rng('housing_stock_gdp')} & {mark('housing_stock_gdp')} \\\\
Preço / salário mensal por trabalhador & {summary['price_wage_p10']:.0f}--{summary['price_wage_p90']:.0f} & 60--150 & -- \\\\
Preço / renda anual familiar & {summary['price_income_mean']:.2f} & {rng('price_income')} & {mark('price_income')} \\\\
Produção habitacional (por 1000 hab) & {summary['housing_production_per_1000_mean']:.2f} & {rng('housing_production_per_1000')} & {mark('housing_production_per_1000')} \\\\
Consumo / PIB & {summary['consumption_gdp_mean']:.2f} & {rng('consumption_gdp')} & {mark('consumption_gdp')} \\\\
Vacância habitacional & {summary['vacancy_mean']:.2f} & {rng('vacancy')} & {mark('vacancy')} \\\\
Estoque imobiliário / renda permanente das famílias & {summary['housing_stock_permanent_income_mean']:.2f} & {rng('housing_stock_permanent_income')} & {mark('housing_stock_permanent_income')} \\\\
Índice de Gini & {summary['gini_mean']:.2f} & {rng('gini')} & {mark('gini')} \\\\
Famílias com consumo zero & {summary['zero_cons_mean']:.2f} & {rng('zero_cons')} & {mark('zero_cons')} \\\\
Taxa de desemprego & {summary['unemployment_mean']:.2f} & {rng('unemployment')} & {mark('unemployment')} \\\\
\\hline
\\end{{tabular}}
"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"housing_validation_table{suffix}.tex"
    path = OUT_DIR / filename

    with open(path, "w") as f:
        f.write(table)

    print("\nLaTeX table:\n")
    print(table)

    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate PS3 housing/macro indicators against Brasil benchmarks.")
    parser.add_argument("--stats-file", help="Path to a specific final_statsNN.csv. Default: most recently "
                                               "modified output/**/final_stats*.csv.")
    parser.add_argument("--base-dir", help="Base directory to glob final_stats*.csv from (default: output/).")
    parser.add_argument("--city", help="Filter to a single processing_acps (e.g. GOIANIA).")
    parser.add_argument("--run-type", help="Filter to a single run_type (e.g. sensitivity, planhab, other).")
    parser.add_argument("--month-start", help="Keep rows with month >= this date (e.g. 2011-01-01).")
    parser.add_argument("--month-cutoff", help="Keep rows with month < this date (e.g. 2025-01-01).")
    parser.add_argument("--suffix", default="", help="Suffix for output filenames.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if any([args.stats_file, args.base_dir, args.city, args.run_type, args.month_start, args.month_cutoff]):
        main(stats_file=args.stats_file, base_dir=args.base_dir, city=args.city, run_type=args.run_type,
             month_start=args.month_start, month_cut_off=args.month_cutoff, suffix=args.suffix)
    else:
        cut_off = "2025-01-01"

        print("\nRunning FULL pooled descriptive validation\n")
        main(month_cut_off=None, suffix="_full")

        print("\nRunning PRE-2025 pooled descriptive validation\n")
        main(month_cut_off=cut_off, suffix="_pre_2025")
