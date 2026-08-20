"""Numbers main.tex needs that compare_density_tier2.py does not print.

  * Table 4  -- Eq. 3 re-estimated on alternative averaging windows (new batch)
  * Table 5  -- ADF / KPSS on the pooled monthly mean dY series (new batch)
  * Figure 2 caption -- pre-2020 mean/std, peak month and value, tail-window mean
  * misc text numbers -- aggregate dY, share negative, N per level
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.validation import compare_density_tier2 as C  # noqa: E402

OUT = Path(__file__).parent


def build_panels_window(stats, reg, t0, t1):
    """C.build_panels, with the averaging window given explicitly."""
    sw = stats[(stats["month"] >= t0) & (stats["month"] <= t1)]
    rw = reg[(reg["month"] >= t0) & (reg["month"] <= t1)]

    gacp = ["simulation_id", "processing_acps", "interest_housing",
            "policy_melhorias", "seed"]
    gmun = ["simulation_id", "processing_acps", "mun_id", "interest_housing",
            "policy_melhorias", "seed"]
    sm = sw.groupby(gacp)[C.ACP_COLS].mean().reset_index()
    rm = rw.groupby(gmun)[C.MUN_COLS].mean().reset_index()

    base_a = (sm["policy_melhorias"] == C.BASELINE_PM) & (sm["interest_housing"] == C.BASELINE_IH)
    base_r = (rm["policy_melhorias"] == C.BASELINE_PM) & (rm["interest_housing"] == C.BASELINE_IH)

    ba = (sm[base_a][["processing_acps", "seed"] + C.ACP_COLS]
          .rename(columns={c: f"{c}_base" for c in C.ACP_COLS}))
    br = (rm[base_r][["processing_acps", "mun_id", "seed"] + C.MUN_COLS]
          .rename(columns={c: f"{c}_base" for c in C.MUN_COLS}))

    acp = sm[~base_a].merge(ba, on=["processing_acps", "seed"], how="inner")
    mun = rm[~base_r].merge(br, on=["processing_acps", "mun_id", "seed"], how="inner")

    for c in ["gini_index", "house_price", "affordability_median",
              "house_quality", "house_vacancy", "unemployment"]:
        acp[f"delta_{c}"] = acp[c] - acp[f"{c}_base"]
    acp = acp.rename(columns={"delta_gini_index": "delta_gini",
                              "delta_house_vacancy": "delta_vacancy"})
    for c in ["regional_gini", "regional_house_values", "median_affordability"]:
        mun[f"delta_{c}"] = mun[c] - mun[f"{c}_base"]

    cap_pop = br.groupby(["processing_acps", "mun_id"])["pop_base"].mean().reset_index()
    caps = (cap_pop.loc[cap_pop.groupby("processing_acps")["pop_base"].idxmax(),
                        ["processing_acps", "mun_id"]]
            .rename(columns={"mun_id": "capital_mun_id"}))
    mun = mun.merge(caps, on="processing_acps", how="left")
    mun["is_capital"] = mun["mun_id"] == mun["capital_mun_id"]
    cap = mun[mun["is_capital"]].copy()

    per = mun[~mun["is_capital"]].copy()
    for src, dst in [("delta_regional_gini", "w_g"),
                     ("delta_regional_house_values", "w_h"),
                     ("delta_median_affordability", "w_a")]:
        per[dst] = per[src] * per["pop_base"]
    gp = ["processing_acps", "interest_housing", "policy_melhorias", "seed"]
    pa = per.groupby(gp).agg(sg=("w_g", "sum"), sh=("w_h", "sum"), sa=("w_a", "sum"),
                             total_pop=("pop_base", "sum")).reset_index()
    pa["delta_gini_periphery"] = pa["sg"] / pa["total_pop"]
    pa["delta_house_values_periphery"] = pa["sh"] / pa["total_pop"]
    pa["delta_median_affordability_periphery"] = pa["sa"] / pa["total_pop"]
    return acp, cap, pa


def main():
    print("loading new batch ...")
    stats = C.load_stats(C.NEW_STATS, "new")
    reg = C.load_regional(C.NEW_REG, "new")

    # ---------------------------------------------------------------- Table 4
    windows = {
        "Tail (2035-2039, headline)": ("2035-01-01", "2039-12-01"),
        "Peak (2027-2031)": ("2027-01-01", "2031-12-01"),
        "Peak (2028-2030, narrow)": ("2028-01-01", "2030-12-01"),
    }
    print("\n=== Table 4: Eq.3 (ACP, secondary spec) across averaging windows ===")
    rows = []
    for name, (a, b) in windows.items():
        acp, cap, per = build_panels_window(stats, reg,
                                            pd.Timestamp(a), pd.Timestamp(b))
        r = C.eq3(acp, "delta_gini", "delta_affordability_median")
        r.update({"window": name, "mean_dY": acp["delta_gini"].mean()})
        rows.append(r)
        print(f"{name:<28} mean dY={r['mean_dY']:+.4f}  beta={r['beta']:+.4f}"
              f" ({r['se']:.4f}){r['stars']:<3}  R2={r['r2']:.3f}  N={r['n']}")
    pd.DataFrame(rows).to_csv(OUT / "table4_windows_new26.csv", index=False)

    # ------------------------------------------------- monthly pooled dY series
    print("\n=== monthly pooled mean dY (new batch) ===")
    keys = ["processing_acps", "seed", "month"]
    base = stats[(stats["policy_melhorias"] == C.BASELINE_PM)
                 & (stats["interest_housing"] == C.BASELINE_IH)][keys + ["gini_index"]]
    base = base.rename(columns={"gini_index": "gini_base"})
    treated = stats[~((stats["policy_melhorias"] == C.BASELINE_PM)
                      & (stats["interest_housing"] == C.BASELINE_IH))]
    t = treated[keys + ["gini_index", "interest_housing", "policy_melhorias"]].merge(
        base, on=keys, how="inner")
    t["delta_gini"] = t["gini_index"] - t["gini_base"]
    # Unbalanced: 10 seeds in 26 cities, 6 in Sao Paulo.
    expect = 5 * 360 * stats.groupby("processing_acps")["seed"].nunique().sum()
    print("merged monthly rows:", len(t), " expect", expect)

    series = t.groupby("month")["delta_gini"].mean().sort_index()
    series.to_csv(OUT / "timeseries_delta_gini_new26.csv")

    pre = series[series.index < "2020-01-01"]
    tail = series[(series.index >= "2035-01-01")]
    print(f"pre-2020 : mean={pre.mean():+.6f} std={pre.std():.6f} (n={len(pre)})")
    print(f"tail 35-39: mean={tail.mean():+.6f}")
    post = series[series.index >= "2020-01-01"]
    print(f"most negative month: {post.idxmin():%Y-%m} at {post.min():+.6f}")
    print(f"most positive month: {post.idxmax():%Y-%m} at {post.max():+.6f}")
    ann = post.groupby(post.index.year).mean()
    print("annual means post-2020:")
    print(ann.round(5).to_string())

    # ---------------------------------------------------------------- Table 5
    print("\n=== Table 5: stationarity of pooled monthly mean dY ===")
    segs = {"Full horizon (2010-2039)": series,
            "Post-divergence (2020-2039)": post,
            "Tail window (2035-2039)": tail}
    srows = []
    for name, s in segs.items():
        adf_p = adfuller(s.values, autolag="AIC")[1]
        kp_p = kpss(s.values, regression="c", nlags="auto")[1]
        srows.append({"series": name, "n": len(s), "adf_p": adf_p, "kpss_p": kp_p})
        print(f"{name:<30} N={len(s):>3}  ADF p={adf_p:.3f}  KPSS p={kp_p:.3f}")
    pd.DataFrame(srows).to_csv(OUT / "table5_stationarity_new26.csv", index=False)

    # ------------------------------------------------------- misc text numbers
    print("\n=== misc ===")
    acp, cap, per = build_panels_window(stats, reg,
                                        pd.Timestamp("2035-01-01"),
                                        pd.Timestamp("2039-12-01"))
    g = acp["delta_gini"]
    print(f"ACP N={len(acp)}  Capital N={len(cap)}  Periphery N={len(per)}"
          f"  periphery cities={per['processing_acps'].nunique()}")
    print(f"mean dY={g.mean():+.6f}  std={g.std():.6f}  "
          f"share negative={100 * (g < 0).mean():.1f}%  "
          f"mean in pp={100 * g.mean():+.4f}  std in pp={100 * g.std():.3f}")
    single = sorted(set(acp["processing_acps"]) - set(per["processing_acps"]))
    print("single-municipality ACPs (excluded from periphery):", single)


if __name__ == "__main__":
    main()
