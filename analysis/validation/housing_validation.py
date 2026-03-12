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


def safe_ratio(num, den):
    """
    Compute num/den safely by replacing zero denominators
    and removing infinite results.
    """
    den = den.replace(0, np.nan)
    ratio = num / den
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    return ratio


def compute_indicators(df):

    results = {}

    # ---------- housing stock / GDP ----------
    housing_stock_value = df["number_domiciles"] * df["house_price"]

    housing_to_gdp = safe_ratio(
        housing_stock_value,
        df["gdp_month"] * 12
    )

    # household wealth
    family_wealth = df["families_median_wealth"] * df["number_domiciles"]
    housing_wealth_ratio = safe_ratio(housing_stock_value, family_wealth)

    results["housing_stock_wealth_mean"] = housing_wealth_ratio.mean()
    results["housing_stock_gdp_mean"] = housing_to_gdp.mean()
    results["housing_stock_gdp_median"] = housing_to_gdp.median()

    # ---------- price / wage ----------
    price_wage = safe_ratio(
        df["house_price"],
        df["firms_median_wage_paid"]
    )

    results["price_wage_min"] = price_wage.min()
    results["price_wage_max"] = price_wage.max()
    results["price_wage_median"] = price_wage.median()

    # ---------- price / household income ----------
    price_income = safe_ratio(
        df["house_price"],
        df["families_wages_received"] * 12
    )

    results["price_income_mean"] = price_income.mean()
    results["price_income_median"] = price_income.median()

    # ---------- housing production ----------
    new_houses = df["number_domiciles"].diff().fillna(0).clip(lower=0)

    houses_year = new_houses * 12

    houses_per_1000 = safe_ratio(
        houses_year * 1000,
        df["pop"]
    )

    results["housing_production_per_1000_mean"] = houses_per_1000.mean()

    # ---------- consumption / GDP ----------
    consumption_gdp = safe_ratio(
        df["average_utility"],
        df["gdp_month"]
    )

    results["consumption_gdp_mean"] = consumption_gdp.mean()

    # ---------- vacancy ----------
    results["vacancy_mean"] = df["house_vacancy"].replace([np.inf, -np.inf], np.nan).mean()

    return results


def aggregate_runs(results_list):

    df = pd.DataFrame(results_list)
    df = df.replace([np.inf, -np.inf], np.nan)

    summary = df.mean()

    return summary


def write_csv(summary):

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = summary.reset_index()
    out.columns = ["indicator", "value"]

    path = OUT_DIR / "housing_validation_results.csv"
    out.to_csv(path, index=False)

    return path


def write_latex(summary):

    table = f"""
\\begin{{tabular}}{{lcc}}
\\hline
Indicador & Modelo & Brasil (aprox.) \\\\
\\hline
Estoque imobiliário / PIB & {summary['housing_stock_gdp_mean']:.2f} & 1.5--2.5 \\\\
Preço / salário mensal (trabalhador) & {summary['price_wage_min']:.0f}--{summary['price_wage_max']:.0f} & 40--90 \\\\
Preço / renda anual familiar & {summary['price_income_mean']:.2f} & 6--12 \\\\
Produção habitacional (por 1000 hab) & {summary['housing_production_per_1000_mean']:.2f} & 2--6 \\\\
Consumo / PIB & {summary['consumption_gdp_mean']:.2f} & 0.55--0.65 \\\\
Vacância habitacional & {summary['vacancy_mean']:.2f} & 0.08--0.12 \\\\
Estoque imobiliário / riqueza das famílias & {summary['housing_stock_wealth_mean']:.2f} & 0.45--0.65 \\\\
\\hline
\\end{{tabular}}
"""

    path = OUT_DIR / "housing_validation_table.tex"

    with open(path, "w") as f:
        f.write(table)

    return path


def main():

    files = find_stats_files()
    if not files:
        raise RuntimeError("No stats.csv files found.")

    results = []

    for f in files:
        df = load_stats(f)
        indicators = compute_indicators(df)
        results.append(indicators)

    summary = aggregate_runs(results)

    csv_path = write_csv(summary)
    tex_path = write_latex(summary)

    print("\nHousing validation summary\n")
    print(summary)

    print("\nOutputs:")
    print(csv_path)
    print(tex_path)


if __name__ == "__main__":
    main()