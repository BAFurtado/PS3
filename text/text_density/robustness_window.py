"""
robustness_window.py
=====================
Two robustness checks for the "Validation and robustness check" subsection:

1. Window sensitivity: does beta (Eq. 3, ACP level, affordability mechanism)
   hold up if the averaging window is moved from the long-run tail
   (2035-01 -- 2039-12, the paper's headline window) to the peak-effect
   window identified in the time series (Figure 1 / fig_timeseries,
   ~2028-2030)? Mirrors the ACP-level portion of prepare_data.py's pipeline
   (Sections 1-7), parameterised by an explicit window instead of "final N
   months of T_max", then re-runs the same OLS-with-city-FE spec used in
   regression.py.

2. Stationarity: ADF and KPSS tests on the pooled monthly mean-Delta-Gini
   series, for the full horizon, the post-divergence period, and the tail
   window — a light-weight check on whether the tail window is a defensibly
   "calmer" period than earlier ones (as opposed to the full per-series,
   per-seed structural-break exercise, which is out of scope here; see the
   Future work stub in main.tex).

Run from the project root:
    python text/text_density/robustness_window.py

Outputs (written to text/text_density/data/):
    robustness_window_table.tex        window-sensitivity LaTeX table
    robustness_stationarity_table.tex  ADF/KPSS LaTeX table
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller, kpss
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
STATS_FILE   = PROJECT_ROOT / 'output' / 'final_stats.csv'
TS_FILE      = SCRIPT_DIR / 'data' / 'timeseries_delta_gini.csv'
OUT_DIR      = SCRIPT_DIR / 'data'

BASELINE_IH = 'medium'
BASELINE_PM = False
IH_MAP = {'baixa': 'low', 'media': 'medium', 'alta': 'high'}
PM_MAP = {True: 'active', False: 'inactive'}

ACP_OUTCOME_COLS = ['gini_index', 'house_price', 'affordability_median']

WINDOWS = {
    'tail (headline, 2035-2039)': ('2035-01-01', '2039-12-01'),
    'peak (2027-2031)':           ('2027-01-01', '2031-12-01'),
    'peak (2028-2030, narrow)':   ('2028-01-01', '2030-12-01'),
}


def build_acp_deltas(stats, start, end):
    w = stats[(stats['month'] >= start) & (stats['month'] <= end)].copy()

    group = ['simulation_id', 'processing_acps', 'interest_housing', 'policy_melhorias', 'seed']
    means = w.groupby(group)[ACP_OUTCOME_COLS].mean().reset_index()

    is_base = (means['policy_melhorias'] == BASELINE_PM) & (means['interest_housing'] == BASELINE_IH)
    base = means[is_base][['processing_acps', 'seed'] + ACP_OUTCOME_COLS].rename(
        columns={c: f'{c}_base' for c in ACP_OUTCOME_COLS}
    )
    treated = means[~is_base].copy()

    merged = treated.merge(base, on=['processing_acps', 'seed'], how='inner')
    merged['delta_gini'] = merged['gini_index'] - merged['gini_index_base']
    merged['delta_affordability_median'] = (
        merged['affordability_median'] - merged['affordability_median_base']
    )
    return merged


def run_regression(df):
    res = smf.ols(
        'delta_gini ~ delta_affordability_median + C(processing_acps)', data=df
    ).fit(cov_type='HC3')
    beta = res.params['delta_affordability_median']
    se = res.bse['delta_affordability_median']
    p = res.pvalues['delta_affordability_median']
    stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
    return beta, se, p, stars, res.rsquared, int(res.nobs)


print('Loading output/final_stats.csv ...')
stats = pd.read_csv(STATS_FILE, low_memory=False)
stats['month'] = pd.to_datetime(stats['month'])
stats['interest_housing'] = stats['interest_housing'].map(IH_MAP)
stats['seed'] = stats['simulation_id'].str.split('__').str[-1].astype(int)
print(f'  {len(stats):,} rows, {stats["simulation_id"].nunique()} simulations\n')

print(f"{'Window':32s}  {'mean ΔY':>9s}  {'β':>10s}  {'SE':>8s}  {'p':>10s}  {'R²':>6s}  {'N':>5s}")
print('-' * 95)

results = {}
for label, (start, end) in WINDOWS.items():
    acp = build_acp_deltas(stats, start, end)
    beta, se, p, stars, r2, n = run_regression(acp)
    mean_dy = acp['delta_gini'].mean()
    results[label] = dict(beta=beta, se=se, p=p, r2=r2, n=n, mean_dy=mean_dy)
    print(f"{label:32s}  {mean_dy:+9.4f}  {beta:+7.4f}{stars:<3s}  {se:8.4f}  {p:10.2e}  {r2:6.3f}  {n:5d}")

print()
tail = results['tail (headline, 2035-2039)']
peak = results['peak (2027-2031)']
ratio = peak['beta'] / tail['beta']
print(f"beta(peak) / beta(tail) = {ratio:.2f}  "
      f"({'stable' if 0.7 < ratio < 1.4 else 'meaningfully different'} across windows)")

# ── Window-sensitivity LaTeX table ──────────────────────────────────────────
ROW_LABELS = {
    'tail (headline, 2035-2039)': 'Tail (2035--2039, headline)',
    'peak (2027-2031)':           'Peak (2027--2031)',
    'peak (2028-2030, narrow)':   'Peak (2028--2030, narrow)',
}

window_tex = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Affordability-channel coefficient across averaging windows}',
    r'\label{tab:window_robustness}',
    r'\begin{threeparttable}',
    r'\begin{tabular}{lcccc}',
    r'\toprule',
    r'Window & mean $\Delta Y$ & $\hat{\beta}$ & $R^2$ & $N$ \\',
    r'\midrule',
]
for key, label in ROW_LABELS.items():
    r = results[key]
    stars = '***' if r['p'] < 0.01 else '**' if r['p'] < 0.05 else '*' if r['p'] < 0.10 else ''
    window_tex.append(
        f"{label} & {r['mean_dy']:+.4f} & {r['beta']:+.4f}{stars} ({r['se']:.4f}) "
        f"& {r['r2']:.3f} & {r['n']} \\\\"
    )
window_tex += [
    r'\bottomrule',
    r'\end{tabular}',
    r'\begin{tablenotes}',
    r'\footnotesize',
    r'\item Notes: Equation~(3) (ACP level, secondary spec), re-estimated with the averaging window '
    r'moved to the peak-effect period identified in Figure~\ref{fig:timeseries}. HC3 SE in parentheses.',
    r'\item *** $p<0.01$, ** $p<0.05$, * $p<0.10$.',
    r'\end{tablenotes}',
    r'\end{threeparttable}',
    r'\end{table}',
]
(OUT_DIR / 'robustness_window_table.tex').write_text('\n'.join(window_tex))
print(f"\n  Saved: {OUT_DIR / 'robustness_window_table.tex'}")


# ══════════════════════════════════════════════════════════════════════════
# STATIONARITY CHECK
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Stationarity check (ADF / KPSS on pooled monthly mean Delta-Gini)")
print("=" * 70)

warnings.filterwarnings('ignore')  # KPSS emits an interpolation-range warning

ts = pd.read_csv(TS_FILE)
ts['month'] = pd.to_datetime(ts['month'])
agg = ts.groupby('month')['delta_gini'].mean().sort_index()

STATIONARITY_SERIES = {
    'Full horizon (2010--2039)':      agg,
    'Post-divergence (2020--2039)':   agg.loc['2020-01-01':],
    'Tail window (2035--2039)':       agg.loc['2035-01-01':],
}

stat_rows = []
print(f"{'Series':30s}  {'n':>4s}  {'ADF p':>8s}  {'KPSS p':>8s}")
for label, series in STATIONARITY_SERIES.items():
    adf_stat, adf_p, *_ = adfuller(series, autolag='AIC')
    kpss_stat, kpss_p, *_ = kpss(series, regression='c', nlags='auto')
    print(f"{label:30s}  {len(series):4d}  {adf_p:8.4f}  {kpss_p:8.4f}")
    stat_rows.append((label, len(series), adf_p, kpss_p))

stat_tex = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Stationarity of the pooled monthly mean $\Delta Y$ series}',
    r'\label{tab:stationarity}',
    r'\begin{threeparttable}',
    r'\begin{tabular}{lccc}',
    r'\toprule',
    r'Series & $N$ (months) & ADF $p$ & KPSS $p$ \\',
    r'\midrule',
]
for label, n, adf_p, kpss_p in stat_rows:
    stat_tex.append(f"{label} & {n} & {adf_p:.3f} & {kpss_p:.3f} \\\\")
stat_tex += [
    r'\bottomrule',
    r'\end{tabular}',
    r'\begin{tablenotes}',
    r'\footnotesize',
    r'\item Notes: ADF null is a unit root; KPSS null is stationarity. Series is the monthly mean '
    r'$\Delta Y$ pooled across all 27 metropolitan areas. KPSS $p$-values are capped at 0.10 '
    r'(the largest tabulated value) by the implementation.',
    r'\end{tablenotes}',
    r'\end{threeparttable}',
    r'\end{table}',
]
(OUT_DIR / 'robustness_stationarity_table.tex').write_text('\n'.join(stat_tex))
print(f"\n  Saved: {OUT_DIR / 'robustness_stationarity_table.tex'}")
