"""B-plan closure dashboard over PS3 run directories (B_PLAN.md §1, §5 B0).

usage:
  python dashboard.py <repo> <run_dir> [<run_dir> ...]     checks + flow table, one column per run
  python dashboard.py <repo> --multipliers                  export multiplier per ACP (for B1's smoke city)

<repo> is the PS3 checkout whose analysis/output.py resolves the stats.csv layout (legacy layouts included).
Checks use the last 36 months; "trend" is the slope per year of log(ratio) over years 2 to the end.
Ratios to GDP are in months of monthly GDP, except m2_gdp and housing credit / GDP, which are over annual GDP to match the
references in references.csv (BCB SGS 27810 / 4382 and 20612 / 4382, yearly means 2010-2019).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CHANNELS = ["public_transfers", "public_taxes_out", "bank_interest", "bank_profit_out", "ogu", "fgts_sbpe", "fgts_sbpe_repaid", "immigrants", "exports",
            "imports", "import_tax", "firm_entry", "firm_writeoff", "eco_investment"]
HOLDERS = ["households", "deposits", "firms", "public", "bank"]


def load(repo, run_dir):
    run_dir = os.path.abspath(run_dir)
    sys.path.insert(0, repo)
    cwd = os.getcwd()
    os.chdir(repo)
    from analysis.output import columns_for
    os.chdir(cwd)
    f = glob.glob(os.path.join(run_dir, "**", "stats.csv"), recursive=True)[0]
    n = len(open(f).readline().split(";"))
    return pd.read_csv(f, sep=";", header=None, names=columns_for("stats", n))


def trend(ratio):
    """Slope per year of log(ratio), from month 12 to the end."""
    r = ratio.iloc[12:]
    r = r[r > 0]
    if len(r) < 24:
        return np.nan
    return np.polyfit(np.arange(len(r)) / 12, np.log(r.values), 1)[0]


def checks(df):
    t = df.tail(36)
    m = t.mean(numeric_only=True)
    gdp = df.gdp_level.replace(0, np.nan)
    wealth = df.money_households + df.money_deposits
    r = {}
    r["unemployment"] = m.unemployment
    r["max unemployment"] = df.unemployment.max()
    r["gini"] = m.gini_index
    r["zero consumption"] = m.pct_zero_consumption
    r["vacancy"] = m.house_vacancy
    r["price / income (years)"] = m.house_price / (12 * m.families_median_permanent_income)
    r["production / 1,000 / yr"] = ((t.number_domiciles.iloc[-1] - t.number_domiciles.iloc[0])
                                    / (len(t) / 12) / m["pop"] * 1000)
    pr = df.price_level.iloc[-1] / df.price_level.iloc[12]
    r["price growth %/yr"] = 100 * (pr ** (12 / (len(df) - 12)) - 1)
    g = df.gdp_level.tail(12).sum() / df.gdp_level.iloc[12:24].sum()
    r["real GDP growth %/yr"] = 100 * ((g / (df.price_level.tail(12).mean() / df.price_level.iloc[12:24].mean()))
                                       ** (12 / (len(df) - 24)) - 1)
    r["real GDP growth y6-10 %/yr"] = 100 * (np.exp(log_growth(real_gdp(df), 60)) - 1)
    for k, v in decomposition(df).items():
        r[k] = v
    r["unmet share"] = m.firms_unmet_share
    if "household_unserved" in df:
        r["household unserved share"] = m.household_unserved / (m.demand_household - m.unmet_household
                                                                + m.household_unserved)
    r["money / GDP (months)"] = (t.money_total / t.gdp_level).mean()
    r["money / GDP trend (log/yr)"] = trend(df.money_total / gdp)
    r["hh wealth / GDP (months)"] = (wealth.tail(36) / t.gdp_level).mean()
    r["hh wealth / GDP trend (log/yr)"] = trend(wealth / gdp)
    if "families_total_permanent_income" in df:
        pi = df.families_total_permanent_income.replace(0, np.nan)
        r["hh wealth / hh PI (months)"] = (wealth.tail(36) / pi.tail(36)).mean()
        r["hh wealth / hh PI trend (log/yr)"] = trend(wealth / pi)
        r["hh PI / GDP"] = (t.families_total_permanent_income / t.gdp_level).mean()
    r["firm cash / GDP (months)"] = (t.money_firms / t.gdp_level).mean()
    r["firm cash / GDP trend (log/yr)"] = trend(df.money_firms / gdp)
    r["m2_gdp (ref 0.34-0.40)"] = ((t.money_households + t.money_deposits + t.money_firms) / (12 * t.gdp_level)).mean()
    r["housing credit / GDP (ref 0.086)"] = (t.credit_stock / (12 * t.gdp_level)).mean()
    sgdp = t.gdp_level.sum()
    r["trade balance / GDP"] = (t.ext_exports.sum() + t.ext_recycled.sum() - t.ext_imports.sum()) / sgdp
    last = df.tail(37)
    d = lambda c: last[c].iloc[-1] - last[c].iloc[0]
    r["net public transfers / GDP"] = (d("money_public_transfers") - d("money_public_taxes_out")) / sgdp
    r["max |unexplained| / money"] = (df.money_unexplained.abs() / df.money_total).max()
    return r


def real_gdp(df):
    return df.gdp_level / df.price_level


def log_growth(series, start):
    """Log growth per year of a level between the 12-month means starting at month `start` and ending the run."""
    years = (len(series) - 12 - start) / 12
    return np.log(series.tail(12).mean() / series.iloc[start:start + 12].mean()) / years


def decomposition(df):
    """Real GDP growth split into population, employment / population and real value added per worker, %/yr
    (log points, so the three parts add up to the total), over years 2-10 and 6-10."""
    emp = df.firms_total_employment.replace(0, np.nan)
    parts = {"pop": df["pop"], "emp/pop": emp / df["pop"], "real VA/worker": real_gdp(df) / emp,
             "nominal VA/worker": df.gdp_level / emp, "price level": df.price_level}
    return {f"growth {k} {label} %/yr": 100 * log_growth(v, start)
            for start, label in ((12, "y2-10"), (60, "y6-10")) for k, v in parts.items()}


# Density paper Table 2 (submitted), ranges as published; definitions from analysis/validation/housing_validation.py.
HOUSING_TABLE2 = {"housing_stock_gdp": (1.5, 2.5), "price_wage": (60, 150), "price_income": (6, 12),
                  "vacancy": (0.08, 0.13), "consumption_gdp": (0.55, 0.65), "housing_production_per_1000": (2, 4),
                  "housing_stock_permanent_income": (5, 8)}
# Validation targets: Table 2, except price / income, price / wage and stock / permanent income, which are the range over
# the 25 capitals of the Census 2010 values on the model's definitions, prices = Census mean rent / FipeZAP 2010 yield
# (r2/census_housing.py -> r2/census_housing_2010.csv)
HOUSING_TARGETS = {**HOUSING_TABLE2, "price_wage": (81, 119), "price_income": (3.7, 6.3),
                   "housing_stock_permanent_income": (3.1, 4.7)}


def housing_table(repo, df):
    """Table 2 moments, mean of the last 36 months, with the target range (HOUSING_TARGETS)."""
    sys.path.insert(0, repo)
    from analysis.validation.housing_validation import compute_derived_monthly_indicators
    h = compute_derived_monthly_indicators(df).tail(36)
    return {f"T2 {k} ({lo}-{hi})": h[k].mean() for k, (lo, hi) in HOUSING_TARGETS.items()}


def flows(df):
    """Ledger channels over the whole run (money) and over the last 36 months (share of GDP); holders start/end."""
    last = df.tail(37)
    sgdp = df.gdp_level.tail(36).sum()
    r = {}
    for c in CHANNELS:
        col = f"money_{c}"
        if col in df:
            r[f"in {c} (run)"] = df[col].iloc[-1]
            r[f"in {c} / GDP (36 m)"] = (last[col].iloc[-1] - last[col].iloc[0]) / sgdp
    for h in HOLDERS:
        r[f"held {h} start"] = df[f"money_{h}"].iloc[0]
        r[f"held {h} end"] = df[f"money_{h}"].iloc[-1]
    r["money start"], r["money end"] = df.money_total.iloc[0], df.money_total.iloc[-1]
    return r


def multipliers(repo):
    """Aggregate export multiplier per ACP: sum over sectors of the sector's multiplier times its household
    final-demand share, from input/final_demand/<ACP>_final_demand.json (as markets/goods.read_final_demand_matrix)."""
    sys.path.insert(0, repo)
    os.chdir(repo)
    from markets.goods import read_final_demand_matrix, final_demand
    hh = final_demand.set_index("sector")["HouseholdConsumption"]
    rows = {}
    for f in sorted(glob.glob("input/final_demand/*_final_demand.json")):
        acp = os.path.basename(f)[:-len("_final_demand.json")]
        m = read_final_demand_matrix(acp)
        rows[acp] = sum(m[s] * float(hh.get(s, 0)) for s in m)
    return pd.Series(rows).sort_values(ascending=False)


if __name__ == "__main__":
    repo = os.path.abspath(sys.argv[1])
    pd.set_option("display.width", 250)
    fmt = lambda x: f"{x:.4g}"
    if sys.argv[2:] == ["--multipliers"]:
        print(multipliers(repo).to_string(float_format=fmt))
        sys.exit()
    runs = sys.argv[2:]
    dfs = {os.path.basename(os.path.normpath(r)): load(repo, r) for r in runs}
    print(pd.DataFrame({k: {**checks(v), **housing_table(repo, v)} for k, v in dfs.items()}).to_string(float_format=fmt))
    print()
    print(pd.DataFrame({k: flows(v) for k, v in dfs.items()}).to_string(float_format=fmt))
