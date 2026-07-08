"""
make_figures.py
===============
Generates all figures for the density paper from the prepared delta panels.

Prerequisites:
    1. Run prepare_data.py   (produces text/text_density/data/*.csv)
    2. Run regression.py     (produces text/text_density/data/*.csv + regression_table.tex)

Usage (from project root):
    python text/text_density/make_figures.py

Outputs (written to text/text_density/):
    MAIN PAPER:
        fig_mechanism_scatter.pdf   Figure 1 — affordability channel scatter
        fig_timeseries.pdf          Figure 2 — trajectory of mean ΔGini over time

    APPENDIX:
        fig_appendix_ranked_bars.pdf          A1 — ranked ΔGini per city (95% CI)
        fig_appendix_violin.pdf               A2 — pooled ΔGini distribution
        fig_appendix_pop_scatter.pdf          A3 — population vs ΔGini

    All figures also saved as .png (150 dpi) for quick preview.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / 'data'
OUT_DIR    = SCRIPT_DIR              # figures sit next to main.tex

DPI_PDF = 300
DPI_PNG = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_city(name: str) -> str:
    """ARACAJU → Aracaju, RIO DE JANEIRO → Rio de Janeiro."""
    return name.title().replace('De ', 'de ').replace('Do ', 'do ')


def save(fig, stem: str) -> None:
    """Save a figure as both PDF and PNG, then close."""
    fig.savefig(OUT_DIR / f'{stem}.pdf', dpi=DPI_PDF, bbox_inches='tight')
    fig.savefig(OUT_DIR / f'{stem}.png', dpi=DPI_PNG, bbox_inches='tight')
    plt.close(fig)
    print(f'  {stem}.pdf / .png')


# ── Load data ─────────────────────────────────────────────────────────────────

print('Loading prepared data...')
acp = pd.read_csv(DATA_DIR / 'acp_deltas.csv')
pop = pd.read_csv(DATA_DIR / 'acp_population.csv')
ts  = pd.read_csv(DATA_DIR / 'timeseries_delta_gini.csv')
ts['month'] = pd.to_datetime(ts['month'])
ts['year']  = ts['month'].dt.year

acp_mean_by_city = acp.groupby('processing_acps')['delta_gini'].mean()

print(f'  acp_deltas:  {len(acp)} rows, {acp["processing_acps"].nunique()} cities')
print(f'  timeseries:  {len(ts)} rows')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE 1 — Affordability channel scatter (ΔM vs ΔY)
# ═══════════════════════════════════════════════════════════════════════════════

print('\nFigure 1: mechanism scatter')

fig, ax = plt.subplots(figsize=(8, 6))

# Policy configuration does not separate cleanly in (ΔM, ΔY) space, so points
# are left uncolored; the config-level decomposition is reported in §4.2.
ax.scatter(
    acp['delta_affordability_median'],
    acp['delta_gini'],
    c='#2166ac', alpha=0.45, s=25,
    edgecolors='white', linewidths=0.3,
    zorder=3,
)

# Simple OLS trend line (visual; the reported β uses city FE)
x = acp['delta_affordability_median'].values
y = acp['delta_gini'].values
mask = np.isfinite(x) & np.isfinite(y)
X_ols = sm.add_constant(x[mask])
res = sm.OLS(y[mask], X_ols).fit()
x_range = np.linspace(x[mask].min(), x[mask].max(), 100)
ax.plot(x_range, res.params[0] + res.params[1] * x_range,
        'k-', linewidth=1.5, alpha=0.7, zorder=4)

ax.axhline(0, color='grey', linewidth=0.5, zorder=1)
ax.axvline(0, color='grey', linewidth=0.5, zorder=1)
ax.set_xlabel(r'$\Delta M$ (change in median rent/income ratio)', fontsize=11)
ax.set_ylabel(r'$\Delta Y$ (change in Gini coefficient)', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.grid(True, linestyle=':', alpha=0.3)
ax.yaxis.grid(True, linestyle=':', alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
save(fig, 'fig_mechanism_scatter')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE 2 — Time series of mean ΔGini
# ═══════════════════════════════════════════════════════════════════════════════

print('Figure 2: time series')

annual = ts.groupby('year')['delta_gini'].agg(['mean', 'std', 'count'])
annual['se'] = annual['std'] / np.sqrt(annual['count'])

years = annual.index.values
means = annual['mean'].values * 100   # percentage points
sds   = annual['std'].values * 100

fig, ax = plt.subplots(figsize=(9, 4.5))

ax.fill_between(years, means - sds, means + sds, alpha=0.15, color='#2166ac')
ax.plot(years, means, 'o-', color='#2166ac', linewidth=1.8, markersize=4, zorder=4)

# Policy divergence vertical line
ax.axvline(2020, color='#b2182b', linewidth=1, linestyle='--', alpha=0.7, zorder=3)
ax.text(2020.3, max(means) + 0.15, 'Policy\ndivergence',
        fontsize=8, color='#b2182b', va='top')

# Averaging window shading
ax.axvspan(2035, 2039, alpha=0.10, color='#f4a582', zorder=1)
ax.text(2037, min(means) - 0.05, 'Averaging\nwindow',
        fontsize=8, color='#b2182b', ha='center', va='top', alpha=0.8)

ax.axhline(0, color='grey', linewidth=0.5, zorder=1)
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel(r'Mean $\Delta Y$ (Gini change, pp)', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle=':', alpha=0.3)
ax.set_axisbelow(True)
ax.set_xlim(2010, 2040)
plt.tight_layout()
save(fig, 'fig_timeseries')


# ═══════════════════════════════════════════════════════════════════════════════
# APPENDIX FIGURE A1 — Ranked bar chart of ΔGini per city (95% CI)
# ═══════════════════════════════════════════════════════════════════════════════

print('Figure A1: ranked bars')

city_mean = acp.groupby('processing_acps')['delta_gini'].mean().sort_values()
city_se = acp.groupby('processing_acps')['delta_gini'].agg(
    lambda x: x.std() / np.sqrt(len(x))
).reindex(city_mean.index)

cities  = [fmt_city(c) for c in city_mean.index]
values  = city_mean.values * 100
errors  = (1.96 * city_se.values) * 100   # 95% CI

colors = ['#2166ac' if v < 0 else '#b2182b' for v in values]

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(range(len(cities)), values, xerr=errors,
        color=colors, edgecolor='white', linewidth=0.5,
        capsize=2, error_kw={'linewidth': 0.7, 'color': '0.45'})
ax.set_yticks(range(len(cities)))
ax.set_yticklabels(cities, fontsize=8.5)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel(r'Mean $\Delta$ Gini (percentage points, 95% CI)', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.grid(True, linestyle=':', alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
save(fig, 'fig_appendix_ranked_bars')


# ═══════════════════════════════════════════════════════════════════════════════
# APPENDIX FIGURE A2 — Violin plot of pooled ΔGini
# ═══════════════════════════════════════════════════════════════════════════════

print('Figure A2: violin')

fig, ax = plt.subplots(figsize=(6, 4))
parts = ax.violinplot(
    acp['delta_gini'].values * 100,
    positions=[0],
    showmeans=True, showmedians=True, showextrema=False,
)
for pc in parts['bodies']:
    pc.set_facecolor('#2166ac')
    pc.set_alpha(0.4)
parts['cmeans'].set_color('#b2182b')
parts['cmedians'].set_color('black')

ax.axhline(0, color='grey', linewidth=0.5)
ax.set_xticks([0])
ax.set_xticklabels([r'Pooled $\Delta Y$ ($N = 405$)'])
ax.set_ylabel(r'$\Delta$ Gini (percentage points)', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle=':', alpha=0.3)
ax.set_axisbelow(True)

mean_pp = acp['delta_gini'].mean() * 100
ax.text(0.25, mean_pp + 0.15, f'mean = {mean_pp:+.03f} pp',
        fontsize=9, color='#b2182b')

plt.tight_layout()
save(fig, 'fig_appendix_violin')


# ═══════════════════════════════════════════════════════════════════════════════
# APPENDIX FIGURE A3 — Population vs ΔGini scatter
# ═══════════════════════════════════════════════════════════════════════════════

print('Figure A3: population scatter')

merged = pd.merge(
    acp_mean_by_city.reset_index(),
    pop, on='processing_acps',
)
merged.columns = ['city', 'delta_gini', 'mean_pop']

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(
    merged['mean_pop'] / 1000, merged['delta_gini'] * 100,
    color='#2166ac', s=50, edgecolors='white', linewidths=0.5, zorder=3,
)

# Label cities with large population or large |ΔGini|
for _, r in merged.iterrows():
    if r['mean_pop'] > 40000 or abs(r['delta_gini']) > 0.005:
        ax.annotate(
            fmt_city(r['city']),
            (r['mean_pop'] / 1000, r['delta_gini'] * 100),
            fontsize=7, xytext=(5, 3), textcoords='offset points',
        )

ax.axhline(0, color='grey', linewidth=0.5)
ax.set_xlabel('Metropolitan area population (thousands)', fontsize=10)
ax.set_ylabel(r'Mean $\Delta$ Gini (percentage points)', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.grid(True, linestyle=':', alpha=0.3)
ax.yaxis.grid(True, linestyle=':', alpha=0.3)
ax.set_axisbelow(True)

corr = np.corrcoef(np.log(merged['mean_pop']), merged['delta_gini'])[0, 1]
ax.text(0.95, 0.95, f'$r = {corr:.2f}$', transform=ax.transAxes,
        fontsize=10, ha='right', va='top')

plt.tight_layout()
save(fig, 'fig_appendix_pop_scatter')


# ═══════════════════════════════════════════════════════════════════════════════
print('\nDone. All figures saved to', OUT_DIR)
