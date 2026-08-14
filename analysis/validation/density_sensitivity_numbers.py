"""
Numbers for the density paper's sensitivity subsection (main.tex, sec:appendix-sensitivity),
computed on the corrected model's OAT batch in `output/sensitivity/`.

    python analysis/validation/density_sensitivity_numbers.py [output/sensitivity]

Writes `analysis/validation/density_sensitivity_cells.csv` (one row per parameter value)
and `analysis/validation/density_sensitivity_seedrows.csv` (one row per run), and prints
the report the prose is written from.

Design of the batch it reads: five parameters swept one-at-a-time over three levels
straddling the default, ten replications each, INTEREST_HOUSING="media", MCMV and
melhorias both active, everything else at `conf/default/params.py`. Three sweeps ran on
Goiania. UPGRADE_COST and MELHORIAS_INCOME_QUANTILE act only on the improvement
programme, which is a structural zero in Goiania, so both were re-run on Belem, where the
programme fires; the Goiania originals are kept in `output/sensitivity_goiania_melhorias/`.
The city of each run is read from its own conf.json rather than assumed, so moving a
sweep between cities re-panels the table on its own. Levels are comparable *within* a
city and not across two, and the script keeps the cities apart everywhere: seed-noise
floor, bands and the table all panel by city.

Seeds are matched across the three levels *within* a sweep and differ *between* sweeps,
so each city's sweeps share their middle cell as a configuration under independent seed
sets -- which is what gives the seed-noise floor the bands are judged against.
"""
import sys
import json
import glob
import os
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from analysis.output import OUTPUT_DATA_SPEC

COLS = OUTPUT_DATA_SPEC["stats"]["columns"]
REGIONAL_COLS = OUTPUT_DATA_SPEC["regional"]["columns"]
TAIL_START, TAIL_END = "2035-01-01", "2039-12-31"

# Outcome -> how the tail window is collapsed. `families_helped` is a monthly count of
# families assisted, so it sums; everything else is a level and averages.
OUTCOMES = {
    "gini_index": "mean",
    "unemployment": "mean",
    "house_vacancy": "mean",
    "pct_zero_consumption": "mean",
    "affordability_median": "mean",
    "rent_default": "mean",
    "families_helped": "sum",
    "amount_subsidised": "mean",
    "perc_policy_money_spent": "mean",
}

# Improvement-programme response, read from regional.csv. Only meaningful for the sweep
# that runs in a city where the programme fires (UPGRADE_COST, Belem). Upgrades and the
# pot are cumulated over the WHOLE horizon, not the tail: the programme's binding
# constraint builds up over thirty years. The melhorias_stop_* fields are mutually
# exclusive 0/1 indicators per municipality-month, so their mean reads as a probability.
PROGRAMME = {
    "melhorias_upgrades": "sum_all",
    "melhorias_eligible": "mean_all",
    "melhorias_money_residual": "sum_all",
    "melhorias_stop_no_eligible": "mean_all",
    "melhorias_stop_budget": "mean_all",
    "melhorias_stop_families_exhausted": "mean_all",
    "melhorias_stop_no_builder": "mean_all",
    "melhorias_stop_no_capacity": "mean_all",
}

# The level of each swept parameter that equals the model default, i.e. the cell the
# sweeps within a city share.
DEFAULTS = {
    "UPGRADE_COST": 1.0,
    "MAX_RENT_TO_INCOME_RATIO": 0.3,
    "MELHORIAS_INCOME_QUANTILE": 0.38,
    "NEIGHBORHOOD_EFFECT": 0.2,
    "HOUSING_FINANCIAL_WEIGHT": 60.0,
}


# Per-run summaries of the improvement programme's transition, which is what
# UPGRADE_COST actually moves: the run opens with a stock of substandard dwellings and
# a pot at zero, so money binds while that backlog is cleared and stops binding once it
# is. Reported over the whole horizon, not the tail window.
BACKLOG_END = 2014
TRANSITION = ["upgrades_early", "upgrades_late", "stop_budget_early", "stop_budget_late",
              "allocation_disbursed", "last_year_budget_bound"]


def programme_transition(reg):
    """Backlog-clearance measures for one run's regional.csv."""
    year = pd.to_datetime(reg.month).dt.year
    early, late = year <= BACKLOG_END, year > BACKLOG_END
    allocated = reg["melhorias_money_topup"].sum()
    # The pot is per municipality and persists, so what is left at the end is the last
    # month's residual in each municipality, not a sum over months.
    left = reg.loc[year == year.max()].groupby("mun_id")["melhorias_money_residual"].last().sum()
    budget_bound = reg.loc[reg["melhorias_stop_budget"] == 1]
    return {
        "upgrades_early": reg.loc[early, "melhorias_upgrades"].sum(),
        "upgrades_late": reg.loc[late, "melhorias_upgrades"].sum(),
        "stop_budget_early": reg.loc[early, "melhorias_stop_budget"].mean(),
        "stop_budget_late": reg.loc[late, "melhorias_stop_budget"].mean(),
        "allocation_disbursed": (allocated - left) / allocated if allocated else np.nan,
        # Last calendar year in which money bound anywhere in the ACP.
        "last_year_budget_bound": year[budget_bound.index].max() if len(budget_bound) else np.nan,
    }


def city_of(seed_dir):
    """The ACP a run was executed on, from the run's own saved configuration."""
    with open(os.path.join(seed_dir, "conf.json")) as f:
        return ", ".join(json.load(f)["PARAMS"]["PROCESSING_ACPS"])


def load(run_dir):
    """One row per completed run, with the run's stats.csv digest for identity checks."""
    rows = []
    for param_dir in sorted(glob.glob(os.path.join(run_dir, "*=*"))):
        param_name, param_value = os.path.basename(param_dir).split("=", 1)
        for seed_dir in sorted(glob.glob(os.path.join(param_dir, "*"))):
            stats_f = os.path.join(seed_dir, "stats.csv")
            if not (os.path.isdir(seed_dir) and os.path.exists(os.path.join(seed_dir, "DONE"))):
                continue
            df = pd.read_csv(stats_f, sep=";", header=None, names=COLS)
            df["month_dt"] = pd.to_datetime(df["month"])
            tail = df[(df.month_dt >= TAIL_START) & (df.month_dt <= TAIL_END)]
            row = {"param_name": param_name, "param_value": float(param_value),
                   "city": city_of(seed_dir),
                   "rep": os.path.basename(seed_dir),
                   "digest": hashlib.md5(open(stats_f, "rb").read()).hexdigest()}
            for col, how in OUTCOMES.items():
                row[col] = tail[col].sum() if how == "sum" else tail[col].mean()
            reg = pd.read_csv(os.path.join(seed_dir, "regional.csv"), sep=";",
                              header=None, names=REGIONAL_COLS)
            for col, how in PROGRAMME.items():
                row[col] = reg[col].sum() if how == "sum_all" else reg[col].mean()
            row.update(programme_transition(reg))
            rows.append(row)
    return pd.DataFrame(rows)


def spearman_sign(values, levels):
    """+1 / -1 for a strictly monotone cell-mean sequence, 0 otherwise."""
    order = np.argsort(levels)
    v = np.asarray(values)[order]
    d = np.diff(v)
    if np.all(d > 0):
        return +1
    if np.all(d < 0):
        return -1
    return 0


PRETTY = {
    "UPGRADE_COST": r"\texttt{UPGRADE\_COST}",
    "MAX_RENT_TO_INCOME_RATIO": r"\texttt{MAX\_RENT\_TO\_INCOME\_RATIO}",
    "MELHORIAS_INCOME_QUANTILE": r"\texttt{MELHORIAS\_INCOME\_QUANTILE}",
    "NEIGHBORHOOD_EFFECT": r"\texttt{NEIGHBORHOOD\_EFFECT}",
    "HOUSING_FINANCIAL_WEIGHT": r"\texttt{HOUSING\_FINANCIAL\_WEIGHT}",
}
# Which city each sweep ran on is read from the runs themselves, not assumed here, so
# that moving a sweep to another city re-panels the table without further edits. Within
# a panel the parameters keep this order, which is the order the appendix discusses them.
PARAM_ORDER = ["MAX_RENT_TO_INCOME_RATIO", "NEIGHBORHOOD_EFFECT",
               "HOUSING_FINANCIAL_WEIGHT", "UPGRADE_COST", "MELHORIAS_INCOME_QUANTILE"]
PANEL_TITLE = {"GOIANIA": r"Goi\^ania", "BELEM": r"Bel\'em"}


def panels(raw):
    """[(city, [param, ...]), ...], the city with the most sweeps first."""
    by_city = (raw.groupby("city")["param_name"].unique()
                  .apply(lambda ps: sorted(ps, key=PARAM_ORDER.index)))
    return sorted(by_city.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def panel_title(i, city):
    return rf"\emph{{Panel {chr(ord('A') + i)}: {PANEL_TITLE.get(city, city.title())}}}"


TABLE_COLS = [("gini_index", "{:.4f}"), ("unemployment", "{:.4f}"),
              ("house_vacancy", "{:.4f}"), ("pct_zero_consumption", "{:.4f}"),
              ("affordability_median", "{:.4f}")]


def write_table(cells, layout, path):
    """Table 8: the OAT cell means, with the within-cell seed sd underneath each mean."""
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{One-at-a-time sensitivity of tail-window (2035--2039) outcome "
         r"levels}",
         r"\label{tab:sensitivity}",
         r"\begin{threeparttable}",
         r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"Level & Gini & Unemploy- & House & Zero & Median \\",
         r" &  & ment & vacancy & consumption & affordability \\", r"\midrule"]
    for i, (panel, params) in enumerate(layout):
        L.append(r"\multicolumn{6}{@{}l}{" + panel_title(i, panel) + r"} \\")
        L.append(r"\addlinespace[2pt]")
        for param in params:
            g = cells[cells.param_name == param].sort_values("param_value")
            L.append(r"\multicolumn{6}{@{}l}{" + PRETTY[param] + r"} \\")
            for _, row in g.iterrows():
                is_default = row.param_value == DEFAULTS[param]
                lvl = f"{row.param_value:g}"
                lvl = rf"\textbf{{{lvl}}}" if is_default else lvl
                vals = " & ".join(fmt.format(row[f"{c}_mean"]) for c, fmt in TABLE_COLS)
                sds = " & ".join("({})".format(fmt.format(row[f"{c}_sd"]))
                                 for c, fmt in TABLE_COLS)
                L.append(rf"\quad {lvl} & {vals} \\")
                L.append(f" & {sds} \\\\")
            if param != params[-1]:
                L.append(r"\addlinespace")
        if i < len(layout) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{tablenotes}", r"\footnotesize",
          # Kept deliberately short: the design, the seed-matching scheme and the
          # HOUSING_FINANCIAL_WEIGHT explanation all live in the appendix prose, and
          # the full-length version of these notes ran past the bottom of the page.
          r"\item Notes: improvement policy active, baseline credit-rate scenario; ten "
          r"replications per cell. Cell means over 2035--2039, with the within-cell "
          r"seed-to-seed standard deviation in parentheses. Defaults in bold. Levels "
          r"are comparable within a panel and not across panels, which are different "
          r"cities: the two parameters that act only on the improvement programme were "
          r"swept on Bel\'em, because that programme is a structural zero in "
          r"Goi\^ania; see the text.",
          r"\item The \texttt{HOUSING\_FINANCIAL\_WEIGHT} cells at 30 and 60 are "
          r"identical by construction; see the text.",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}", ""]
    path.write_text("\n".join(L))
    print(f"\nTable written to {path}")


def main(run_dir):
    raw = load(run_dir)
    layout = panels(raw)
    raw.to_csv(ROOT / "analysis/validation/density_sensitivity_seedrows.csv", index=False)
    n_params = raw.param_name.nunique()
    print(f"{len(raw)} runs, {n_params} parameters, "
          f"{raw.groupby(['param_name', 'param_value']).size().unique()} reps per cell")
    print(raw.groupby(["city", "param_name"]).size().to_string(), "\n")

    cells = (raw.groupby(["param_name", "param_value"])
                .agg(n=("rep", "count"),
                     city=("city", "first"),
                     **{f"{c}_mean": (c, "mean") for c in list(OUTCOMES) + list(PROGRAMME) + TRANSITION},
                     **{f"{c}_sd": (c, "std") for c in list(OUTCOMES) + list(PROGRAMME) + TRANSITION})
                .reset_index())
    cells.to_csv(ROOT / "analysis/validation/density_sensitivity_cells.csv", index=False)

    # ── 1. Which levels actually changed the simulation ──────────────────────────────
    print("=" * 78)
    print("1. LEVELS THAT PRODUCE AN IDENTICAL RUN (matched seed, byte-identical stats.csv)")
    print("=" * 78)
    for param, g in raw.groupby("param_name"):
        wide = g.pivot(index="rep", columns="param_value", values="digest")
        levels = list(wide.columns)
        msgs = []
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                same = int((wide[levels[i]] == wide[levels[j]]).sum())
                if same:
                    msgs.append(f"{levels[i]} == {levels[j]} in {same}/{len(wide)} reps")
        print(f"  {param:<28} {'; '.join(msgs) if msgs else 'all three levels differ in every rep'}")

    # ── 2. Seed-noise floor, per city ────────────────────────────────────────────────
    # In Goiania four sweeps share the default cell, so the floor has both a within-cell
    # and a between-seed-set component. In Belem there is a single sweep, so only the
    # within-cell sd of its own default cell is available.
    for city, params in layout:
        print()
        print("=" * 78)
        print(f"2. SEED-NOISE FLOOR IN {city}: the middle cell of each of that city's "
              f"{len(params)} sweep(s)")
        print("   is the SAME configuration under independent seed sets")
        print("=" * 78)
        at_default = np.array([r.param_value == DEFAULTS[r.param_name]
                               for r in raw.itertuples()])
        dflt = raw[(raw.city == city) & at_default]
        print(f"  {len(dflt)} runs at the default configuration\n")
        print(f"  {'outcome':<24}{'grand mean':>12}{'within-cell sd':>16}"
              f"{'spread of cell means':>24}")
        for c in OUTCOMES:
            cm = dflt.groupby("param_name")[c].mean()
            within = dflt.groupby("param_name")[c].std().mean()
            print(f"  {c:<24}{dflt[c].mean():>12.5g}{within:>16.4g}"
                  f"{cm.max() - cm.min():>24.4g}")

    # ── 3. Bands across the cells of each city, against that floor ───────────────────
    for city, params in layout:
        sub = cells[cells.param_name.isin(params)]
        print()
        print("=" * 78)
        print(f"3. BAND ACROSS THE {len(sub)} {city} PARAMETER-VALUE CELLS vs THE "
              f"SEED-NOISE FLOOR")
        print("=" * 78)
        print(f"  {'outcome':<24}{'min':>11}{'max':>11}{'band':>11}"
              f"{'within-cell sd':>16}{'band/sd':>10}")
        for c in OUTCOMES:
            m = sub[f"{c}_mean"]
            sd = sub[f"{c}_sd"]
            band = m.max() - m.min()
            print(f"  {c:<24}{m.min():>11.4g}{m.max():>11.4g}{band:>11.4g}"
                  f"{sd.min():>8.3g}-{sd.max():<7.3g}{band / sd.mean():>10.2f}")

    # ── 4. Monotonicity, parameter by parameter ──────────────────────────────────────
    print()
    print("=" * 78)
    print("4. MONOTONICITY OF THE CELL MEANS, AND WHETHER THE END-TO-END MOVE CLEARS")
    print("   THE SEED-NOISE FLOOR  (* = |end-to-end change| > within-cell sd)")
    print("=" * 78)
    for param, g in cells.groupby("param_name"):
        g = g.sort_values("param_value")
        print(f"\n  {param}  levels {list(g.param_value)}  [{g.city.iloc[0]}]")
        for c in OUTCOMES:
            means = g[f"{c}_mean"].values
            sd = g[f"{c}_sd"].mean()
            sign = spearman_sign(means, g.param_value.values)
            delta = means[-1] - means[0]
            flag = "*" if abs(delta) > sd else " "
            arrow = {1: "increasing", -1: "decreasing", 0: "non-monotone"}[sign]
            print(f"    {c:<24}" + " ".join(f"{v:>10.4g}" for v in means)
                  + f"   {arrow:<13} d={delta:>+10.4g} {flag}")

    # ── 5. The improvement programme: where it fires, and how it responds ────────────
    print()
    print("=" * 78)
    print("5. THE IMPROVEMENT PROGRAMME: WHY UPGRADE_COST MOVED CITY, AND WHAT IT DOES")
    print("=" * 78)
    tk = ROOT / "analysis/validation/density_tier2_melhorias_takeup.csv"
    if tk.exists():
        t = pd.read_csv(tk).set_index("processing_acps")
        print(t.loc[["GOIANIA", "BELEM", "PORTO ALEGRE", "RECIFE"]]
              [["eligible0", "upgrades_total", "eligible0_per_1000"]].to_string())
        print("\n  `families_helped` is funds.families_subsided, which counts MCMV")
        print("  acquisitions AND melhorias upgrades in one running total, so it is")
        print("  dominated by MCMV in either city.\n")
    for param in ("UPGRADE_COST", "MELHORIAS_INCOME_QUANTILE"):
        g = cells[cells.param_name == param].sort_values("param_value")
        print(f"\n  {param}  [{g.city.iloc[0]}]  levels {list(g.param_value)}")
        for c in list(PROGRAMME) + TRANSITION + ["perc_policy_money_spent",
                                                 "amount_subsidised", "families_helped"]:
            means = g[f"{c}_mean"].values
            sd = g[f"{c}_sd"].mean()
            delta = means[-1] - means[0]
            flag = "*" if abs(delta) > sd else " "
            arrow = {1: "increasing", -1: "decreasing",
                     0: "non-monotone"}[spearman_sign(means, g.param_value.values)]
            print(f"    {c:<34}" + " ".join(f"{v:>12.5g}" for v in means)
                  + f"  sd={sd:>10.4g}  {arrow:<13} d={delta:>+12.5g} {flag}")

    write_table(cells, layout, ROOT / "text/text_density/tables/Table8_sensitivity.tex")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "output/sensitivity"))