"""
Okun's Law and Phillips Curve regressions by scenario.

Reproduces colleague's R analysis and generates publication-quality charts.

Usage:
    python analysis/validation/stylized_facts.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SIM_PATH = ROOT / "output" / "regional_stats_final_statss" / "final_stats.csv"
OUT_DIR  = ROOT / "analysis" / "validation" / "results"

BURN_IN_MONTHS = 24
VALIDATION_END = "2024-12-01"

# ── STYLE ─────────────────────────────────────────────────────────────────────

INTEREST_COLORS = {
    "alta":  "#d62728",
    "media": "#2ca02c",
    "baixa": "#1f77b4",
}

POLICY_STYLES = {
    "False": "-",
    "True":  "--",
}

INTEREST_LABELS = {
    "alta":  "High interest",
    "media": "Medium interest",
    "baixa": "Low interest",
}

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "figure.dpi":     150,
})


# ── DATA ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(SIM_PATH)
    df["month"] = pd.to_datetime(df["month"])

    # burn-in
    rank = df.groupby("simulation_id").cumcount()
    df = df[rank >= BURN_IN_MONTHS].copy()
    df = df[df["month"] <= VALIDATION_END].copy()

    df["policy_melhorias"] = df["policy_melhorias"].astype(str)
    df["scenario"] = df["interest_housing"] + "_" + df["policy_melhorias"]

    # Δunemployment and lagged unemployment per run
    df = df.sort_values(["simulation_id", "month"]).reset_index(drop=True)
    df["d_unemployment"]    = df.groupby("simulation_id")["unemployment"].diff()
    df["unemployment_lag1"] = df.groupby("simulation_id")["unemployment"].shift(1)

    return df


def aggregate_across_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average across simulation runs, keeping one row per (scenario, city, month).
    Mirrors what the colleague's R analysis likely did before running regressions.
    Δunemployment and lagged unemployment are recomputed on the averaged series.
    """
    group_cols = ["scenario", "interest_housing", "policy_melhorias", "processing_acps", "month"]
    numeric_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64]
                    and c not in ["d_unemployment", "unemployment_lag1"]]

    agg = df.groupby(group_cols)[numeric_cols].mean().reset_index()

    # Recompute Δunemployment on the averaged series
    agg = agg.sort_values(["scenario", "processing_acps", "month"]).reset_index(drop=True)
    group_key = ["scenario", "processing_acps"]
    agg["d_unemployment"]    = agg.groupby(group_key)["unemployment"].diff()
    agg["unemployment_lag1"] = agg.groupby(group_key)["unemployment"].shift(1)

    return agg


# ── REGRESSIONS ───────────────────────────────────────────────────────────────

def _ols(formula, data):
    """Fit OLS and return (beta, se, pval) for the first non-intercept coefficient."""
    m = smf.ols(formula, data=data.dropna()).fit()
    key = [k for k in m.params.index if k != "Intercept"][0]
    return m.params[key], m.bse[key], m.pvalues[key], m


def run_summary_regressions(df) -> pd.DataFrame:
    rows = []

    # Phillips ─────────────────────────────────────────────────────────────────
    beta, se, pv, _ = _ols(
        "inflation ~ unemployment",
        df[["inflation", "unemployment"]],
    )
    rows.append(("Phillips", "Full panel, contemp.", beta, se, pv))

    beta, se, pv, _ = _ols(
        "inflation ~ unemployment + C(processing_acps) + C(scenario)",
        df[["inflation", "unemployment", "processing_acps", "scenario"]],
    )
    rows.append(("Phillips", "Full panel, city + scenario FE", beta, se, pv))

    sub = df[(df["interest_housing"] == "media") & (df["policy_melhorias"] == "False")]
    beta, se, pv, _ = _ols("inflation ~ unemployment", sub[["inflation", "unemployment"]])
    rows.append(("Phillips", "media/False, contemp.", beta, se, pv))

    beta, se, pv, _ = _ols(
        "inflation ~ unemployment_lag1",
        sub[["inflation", "unemployment_lag1"]],
    )
    rows.append(("Phillips", "media/False, lagged", beta, se, pv))

    # Okun ─────────────────────────────────────────────────────────────────────
    beta, se, pv, _ = _ols(
        "gdp_growth_rate ~ d_unemployment",
        df[["gdp_growth_rate", "d_unemployment"]],
    )
    rows.append(("Okun", "Full panel", beta, se, pv))

    beta, se, pv, _ = _ols(
        "gdp_growth_rate ~ d_unemployment + C(processing_acps) + C(scenario)",
        df[["gdp_growth_rate", "d_unemployment", "processing_acps", "scenario"]],
    )
    rows.append(("Okun", "Full panel, city + scenario FE", beta, se, pv))

    sub = df[(df["interest_housing"] == "media") & (df["policy_melhorias"] == "False")]
    beta, se, pv, _ = _ols(
        "gdp_growth_rate ~ d_unemployment",
        sub[["gdp_growth_rate", "d_unemployment"]],
    )
    rows.append(("Okun", "media/False", beta, se, pv))

    return pd.DataFrame(rows, columns=["test", "spec", "beta", "std_error", "p_value"])


def run_by_scenario(df) -> pd.DataFrame:
    """OLS within each interest-rate group, pooling across policy values."""
    rows = []
    for interest in ["alta", "media", "baixa"]:
        sub = df[df["interest_housing"] == interest]

        okun_sub = sub.dropna(subset=["gdp_growth_rate", "d_unemployment"])
        if len(okun_sub) > 30:
            m = smf.ols("gdp_growth_rate ~ d_unemployment", data=okun_sub).fit()
            rows.append({
                "test": "Okun", "scenario": interest,
                "beta": m.params["d_unemployment"],
                "std_error": m.bse["d_unemployment"],
                "p_value": m.pvalues["d_unemployment"],
            })

        phil_sub = sub.dropna(subset=["inflation", "unemployment"])
        if len(phil_sub) > 30:
            m = smf.ols("inflation ~ unemployment", data=phil_sub).fit()
            rows.append({
                "test": "Phillips", "scenario": interest,
                "beta": m.params["unemployment"],
                "std_error": m.bse["unemployment"],
                "p_value": m.pvalues["unemployment"],
            })

    return pd.DataFrame(rows)


# ── CHARTS ────────────────────────────────────────────────────────────────────

def _add_ols_line(ax, x_vals, fit_model, color, ls, label):
    """Draw OLS regression line over the data range."""
    x_range = np.linspace(x_vals.min(), x_vals.max(), 200)
    # Reconstruct y from intercept + slope (single-regressor model)
    intercept = fit_model.params["Intercept"]
    key = [k for k in fit_model.params.index if k != "Intercept"][0]
    slope = fit_model.params[key]
    y_range = intercept + slope * x_range
    ax.plot(x_range, y_range, color=color, ls=ls, lw=1.6, label=label)


def plot_scatter_with_ols(df, out_dir):
    """
    Figure 1 — scatter plots with per-scenario OLS lines.
    Left: Phillips Curve (inflation ~ unemployment)
    Right: Okun's Law (gdp_growth_rate ~ Δunemployment)
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

    specs = [
        # (ax,  y_col,           x_col,           xlabel,                ylabel,          title)
        (axes[0], "inflation",    "unemployment",  "Unemployment rate",  "Monthly inflation", "Phillips Curve"),
        (axes[1], "gdp_growth_rate", "d_unemployment", r"$\Delta$ Unemployment rate", "Monthly GDP growth", "Okun's Law"),
    ]

    for ax, y_col, x_col, xlabel, ylabel, title in specs:
        all_data = df.dropna(subset=[y_col, x_col])

        # Scatter — all observations, light gray background
        ax.scatter(
            all_data[x_col], all_data[y_col],
            s=0.4, alpha=0.06, color="gray", rasterized=True,
        )

        for scenario in sorted(df["scenario"].unique()):
            sub = all_data[all_data["scenario"] == scenario]
            if len(sub) < 30:
                continue
            interest, policy = scenario.split("_", 1)
            color = INTEREST_COLORS[interest]
            ls    = POLICY_STYLES[policy]
            x_key = x_col if x_col != "unemployment_lag1" else "unemployment_lag1"
            formula = f"{y_col} ~ {x_col}"
            m = smf.ols(formula, data=sub).fit()
            _add_ols_line(
                ax, sub[x_col], m,
                color=color, ls=ls,
                label=f"{INTEREST_LABELS[interest]}, policy={'on' if policy == 'True' else 'off'}",
            )

        ax.axhline(0, color="black", lw=0.5, ls=":")
        ax.axvline(0, color="black", lw=0.5, ls=":")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    # Shared legend
    legend_handles = []
    for interest, color in INTEREST_COLORS.items():
        for policy, ls in POLICY_STYLES.items():
            label = f"{INTEREST_LABELS[interest]}, policy={'on' if policy == 'True' else 'off'}"
            legend_handles.append(
                mlines.Line2D([], [], color=color, ls=ls, lw=1.6, label=label)
            )
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3,
        frameon=False, bbox_to_anchor=(0.5, -0.08),
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    path = os.path.join(out_dir, "fig_stylized_facts_scatter.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_coefficient_forest(by_scenario_df, summary_df, out_dir):
    """
    Figure 2 — forest plot of beta coefficients ± 1.96 SE, by interest-rate group.
    """
    interest_order = ["alta", "media", "baixa"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    for ax, test in zip(axes, ["Okun", "Phillips"]):
        sub = (
            by_scenario_df[by_scenario_df["test"] == test]
            .set_index("scenario")
            .reindex(interest_order)
            .reset_index()
        )

        y_pos  = np.arange(len(sub))
        colors = [INTEREST_COLORS[r] for r in sub["scenario"]]

        for idx, row in sub.iterrows():
            ci = 1.96 * row["std_error"]
            ax.errorbar(
                row["beta"], idx,
                xerr=ci,
                fmt="none", color=colors[idx], capsize=4, lw=1.4,
            )
            ax.scatter(row["beta"], idx, color=colors[idx], s=55, zorder=5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([INTEREST_LABELS[s] for s in sub["scenario"]], fontsize=8)
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.set_xlabel(r"OLS $\hat{\beta}$ (95% CI)", fontsize=8)
        ax.set_title("Okun's Law" if test == "Okun" else "Phillips Curve", fontsize=9)
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#cccccc")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_stylized_facts_forest.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


COLLEAGUE = pd.DataFrame([
    ("Okun",     "alta_False",  -0.522894782, 0.1451211161, 1.303974e-03),
    ("Okun",     "alta_True",   -0.508850990, 0.1111837039, 1.027602e-04),
    ("Okun",     "baixa_False", -0.553634308, 0.1362449629, 3.960668e-04),
    ("Okun",     "baixa_True",  -0.376125634, 0.1205162237, 4.379266e-03),
    ("Okun",     "media_False", -0.284279197, 0.1297349808, 3.759128e-02),
    ("Okun",     "media_True",  -0.407982011, 0.1515659845, 1.226377e-02),
    ("Phillips", "alta_False",  -0.001736603, 0.0002702598, 8.286670e-07),
    ("Phillips", "alta_True",   -0.001799561, 0.0003795091, 6.644198e-05),
    ("Phillips", "baixa_False", -0.001680179, 0.0003068651, 9.629784e-06),
    ("Phillips", "baixa_True",  -0.001768381, 0.0003060680, 4.376016e-06),
    ("Phillips", "media_False", -0.001609456, 0.0003510166, 1.004914e-04),
    ("Phillips", "media_True",  -0.001819075, 0.0003339818, 1.038070e-05),
], columns=["test", "scenario", "beta", "std_error", "p_value"])


def _compare(ours: pd.DataFrame, label: str) -> pd.DataFrame:
    # Colleague results are by full scenario (interest_policy); aggregate to interest level
    colleague_agg = (
        COLLEAGUE.assign(interest=COLLEAGUE["scenario"].str.split("_").str[0])
        .groupby(["test", "interest"])[["beta", "std_error"]]
        .mean()
        .reset_index()
        .rename(columns={"interest": "scenario"})
    )
    merged = colleague_agg.merge(
        ours[["test", "scenario", "beta", "std_error", "p_value"]],
        on=["test", "scenario"], suffixes=("_R", "_py"),
    )
    merged["beta_ratio"] = merged["beta_py"] / merged["beta_R"]
    merged["source"] = label
    return merged


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df     = load_data()
    df_avg = aggregate_across_runs(df)

    print(f"  Full panel rows  : {len(df):,}")
    print(f"  Averaged rows    : {len(df_avg):,}")
    print(f"  Scenarios        : {sorted(df['scenario'].unique())}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Full-panel regressions ─────────────────────────────────────────────────
    print("\n── Full-panel summary regressions ────────────────────────────────")
    summary_full = run_summary_regressions(df)
    print(summary_full.to_string(index=False))

    print("\n── Full-panel by-scenario ────────────────────────────────────────")
    by_scen_full = run_by_scenario(df)
    print(by_scen_full[["test", "scenario", "beta", "std_error", "p_value"]].to_string(index=False))

    # ── Averaged regressions ───────────────────────────────────────────────────
    print("\n── Averaged (city×month mean) summary regressions ────────────────")
    summary_avg = run_summary_regressions(df_avg)
    print(summary_avg.to_string(index=False))

    print("\n── Averaged by-scenario ──────────────────────────────────────────")
    by_scen_avg = run_by_scenario(df_avg)
    print(by_scen_avg[["test", "scenario", "beta", "std_error", "p_value"]].to_string(index=False))

    # ── Comparison with colleague's R results ──────────────────────────────────
    print("\n── Comparison: Python (averaged) vs colleague R ──────────────────")
    comp_avg  = _compare(by_scen_avg,  "averaged")
    comp_full = _compare(by_scen_full, "full_panel")
    comp_all  = pd.concat([comp_avg, comp_full], ignore_index=True)

    pd.set_option("display.float_format", "{:.6f}".format)
    print(comp_avg[["test", "scenario", "beta_R", "beta_py", "beta_ratio"]].to_string(index=False))

    # ── Save outputs ──────────────────────────────────────────────────────────
    summary_full.to_csv(os.path.join(OUT_DIR, "stylized_facts_summary_full.csv"), index=False)
    summary_avg.to_csv( os.path.join(OUT_DIR, "stylized_facts_summary_avg.csv"),  index=False)
    by_scen_full.to_csv(os.path.join(OUT_DIR, "stylized_facts_by_scenario_full.csv"), index=False)
    by_scen_avg.to_csv( os.path.join(OUT_DIR, "stylized_facts_by_scenario_avg.csv"),  index=False)
    comp_all.to_csv(    os.path.join(OUT_DIR, "stylized_facts_comparison.csv"),    index=False)

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\nGenerating charts (averaged data)...")
    plot_scatter_with_ols(df_avg, OUT_DIR)
    plot_coefficient_forest(by_scen_avg, summary_avg, OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
