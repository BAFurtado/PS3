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


def find_stats_files():
    files = glob(str(OUTPUT_DIR / "**" / "stats.csv"), recursive=True)
    return [f for f in files if "/avg/" not in f]


def load_stats(path):
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
    df["families_permanent_income"] = df["families_median_wealth"] * 12 * df["number_domiciles"]

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
        "housing_stock_wealth",
        "price_wage",
        "price_income",
        "housing_production_per_1000",
        "consumption_gdp",
        "vacancy",
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


def print_main_results(raw_summary, derived_summary, sanity, by_run_summary):
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


def main(month_cut_off=None, suffix=""):
    files = find_stats_files()
    if not files:
        raise RuntimeError("No stats.csv files found.")

    pooled = []

    for f in files:
        df = load_stats(f)
        df["source_file"] = str(f)
        df["run_id"] = path_to_run_id(f)
        df["month"] = pd.to_datetime(df["month"], errors="coerce")

        if month_cut_off:
            cutoff = pd.Timestamp(month_cut_off)
            df = df[df["month"] < cutoff]

        pooled.append(df)

    pooled_df = pd.concat(pooled, ignore_index=True)
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
    }
    summary = pd.Series(summary)

    pooled_path = write_csv(pooled_df, f"all_stats_pooled{suffix}.csv")
    raw_summary_path = write_csv(raw_summary, f"raw_variables_summary{suffix}.csv")
    derived_summary_path = write_csv(derived_summary, f"derived_indicators_summary{suffix}.csv")
    sanity_path = write_csv(sanity.reset_index().rename(columns={"index": "check", "value": "value"}),
                            f"sanity_checks{suffix}.csv")
    by_run_path = write_csv(by_run, f"derived_indicators_by_run{suffix}.csv") if not by_run.empty else None
    by_run_summary_path = write_csv(by_run_summary,
                                    f"derived_indicators_by_run_summary{suffix}.csv") if not by_run_summary.empty else None
    latex_path = write_latex(summary, suffix)

    print_main_results(raw_summary, derived_summary, sanity, by_run_summary)

    print("\nOutputs:")
    print(pooled_path)
    print(raw_summary_path)
    print(derived_summary_path)
    print(sanity_path)
    if by_run_path:
        print(by_run_path)
    if by_run_summary_path:
        print(by_run_summary_path)

    print(latex_path)


def write_latex(summary, suffix=""):
    table = f"""
\\begin{{tabular}}{{lcc}}
\\hline
Indicador & Modelo & Brasil (aprox.) \\\\
\\hline
Estoque imobiliário / PIB & {summary['housing_stock_gdp_mean']:.2f} & 1.5--2.5 \\\\
Preço / salário mensal por trabalhador & {summary['price_wage_p10']:.0f}--{summary['price_wage_p90']:.0f} & 60--150 \\\\
Preço / renda anual familiar & {summary['price_income_mean']:.2f} & 6--12 \\\\
Produção habitacional (por 1000 hab) & {summary['housing_production_per_1000_mean']:.2f} & 2--4 \\\\
Consumo / PIB & {summary['consumption_gdp_mean']:.2f} & 0.55--0.65 \\\\
Vacância habitacional & {summary['vacancy_mean']:.2f} & 0.08--0.12 \\\\
Estoque imobiliário / renda permanente das famílias & {summary['housing_stock_permanent_income_mean']:.2f} & 5--8 \\\\
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


if __name__ == "__main__":
    cut_off = "2025-01-01"

    print("\nRunning FULL pooled descriptive validation\n")
    main(month_cut_off=None, suffix="_full")

    print("\nRunning PRE-2025 pooled descriptive validation\n")
    main(month_cut_off=cut_off, suffix="_pre_2025")