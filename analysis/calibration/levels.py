"""Level targets of the calibration wave: each run's levels over its last months, the hard constraints every run must
meet, and the implausibility of a level against its observed band (data/level_targets.csv)."""
import os

import numpy as np
import pandas as pd

from analysis.validation.housing_validation import compute_derived_monthly_indicators
from conf.default.params import REAIS_PER_MONEY_UNIT

TARGETS_PATH = os.path.join(os.path.dirname(__file__), "data", "level_targets.csv")
HOUSING = ["vacancy", "consumption_gdp", "housing_production_per_1000", "price_income", "price_wage",
           "housing_stock_permanent_income"]


def load_targets(regions):
    """{region: {moment: (low, high)}}; a band for region 'ALL' applies to every region without its own"""
    t = pd.read_csv(TARGETS_PATH)
    out = {}
    for region in regions:
        bands = {r.moment: (r.low, r.high) for r in t[t.region == "ALL"].itertuples()}
        bands.update({r.moment: (r.low, r.high) for r in t[t.region == region].itertuples()})
        out[region] = bands
    return out


def growth(series, start=12):
    """Growth %/yr of a level from month `start` to the end"""
    return 100 * ((series.iloc[-1] / series.iloc[start]) ** (12 / (len(series) - 1 - start)) - 1)


def level_moments(df, window=36):
    """Levels of one run, means over its last `window` months, and the measures of the hard constraints. Income per
    resident is in money of the first month: deflated by the model price level"""
    t = df.tail(window)
    deflator = (df.price_level / df.price_level.iloc[0]).tail(window)
    h = compute_derived_monthly_indicators(df.copy()).tail(window)
    m = {
        "unemployment": t.unemployment.mean(),
        "household_income_gdp": (t.families_total_income / t.gdp_level).mean(),
        "household_income_pc": (t.families_total_income / t["pop"] / deflator).mean() * REAIS_PER_MONEY_UNIT,
        "nontradable_inflation": growth(df.price_nontradable),
        **{k: h[k].mean() for k in HOUSING},
    }
    g = df.gdp_level.tail(12).sum() / df.gdp_level.iloc[12:24].sum()
    p = df.price_level.tail(12).mean() / df.price_level.iloc[12:24].mean()
    m["real_gdp_growth"] = 100 * ((g / p) ** (12 / (len(df) - 24)) - 1)
    m["price_growth"] = growth(df.price_level)
    m["max_unemployment_after_m24"] = df.unemployment.iloc[24:].max()
    m["max_unexplained_money"] = (df.money_unexplained.abs() / df.money_total).max()
    return m


def explodes(m):
    """A run that fails any hard constraint: unemployment >= 30 % (mean of the window or any month after month 24),
    price growth >= 5 %/yr, real GDP growth outside (-5, 15) %/yr, or money created or lost outside the ledger"""
    return not (m["unemployment"] < 0.30 and m["max_unemployment_after_m24"] < 0.30 and m["price_growth"] < 5
                and -5 < m["real_gdp_growth"] < 15 and m["max_unexplained_money"] < 1e-9)


def implausibility(value, band, seed_sd, discrepancy):
    """Distance of a level outside its band, over sqrt(seed_sd^2 + (discrepancy x band midpoint)^2); 0 inside"""
    low, high = band
    distance = max(low - value, value - high, 0.0)
    scale = np.sqrt(seed_sd ** 2 + (discrepancy * (low + high) / 2) ** 2)
    return distance / scale if scale > 0 else (np.inf if distance > 0 else 0.0)
