"""
regression.py
=============
Estimates the mechanism regression (Eq. 3) from the density paper at three
geographic levels and saves a publication-ready LaTeX table.

Equation 3:
    ΔY_ips = α + β * ΔM_ips + μ_i + ε_ips

where:
    ΔY  = policy-induced change in Gini coefficient
    ΔM  = policy-induced change in average house prices  (primary M)
    μ_i = metropolitan-area fixed effects
    i   = city, p = policy config, s = seed

Three geographic levels:
    ACP       — full metropolitan area       (acp_deltas.csv)
    Capital   — capital municipality only    (capital_deltas.csv)
    Periphery — non-capital munis, pop-wtd   (periphery_deltas.csv)

Secondary check (INTEREST dimension):
    Re-runs Eq. 3 replacing ΔM with delta_affordability_median /
    delta_median_affordability (median rent/income among renters).

Run from the project root:
    /home/b/miniconda3/envs/ps3/bin/python text/text_density/regression.py

Outputs (written to text/text_density/data/):
    regression_primary.csv      β, SE, t, p, R², N for all levels (primary M)
    regression_secondary.csv    same with affordability median as M
    regression_table.tex        LaTeX table ready for inclusion in main.tex
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / 'data'
OUT_DIR    = DATA_DIR


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: Load prepared delta panels
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: Loading delta panels")
print("=" * 70)

acp = pd.read_csv(DATA_DIR / 'acp_deltas.csv').convert_dtypes(convert_string=False)
cap = pd.read_csv(DATA_DIR / 'capital_deltas.csv').convert_dtypes(convert_string=False)
per = pd.read_csv(DATA_DIR / 'periphery_deltas.csv').convert_dtypes(convert_string=False)

# patsy requires plain object dtype for categorical columns, not pandas StringDtype
for frame in (acp, cap, per):
    for col in frame.select_dtypes(include='string').columns:
        frame[col] = frame[col].astype(object)

print(f"  acp_deltas:       {len(acp):4d} rows, {acp['processing_acps'].nunique()} cities")
print(f"  capital_deltas:   {len(cap):4d} rows, {cap['processing_acps'].nunique()} cities")
print(f"  periphery_deltas: {len(per):4d} rows, {per['processing_acps'].nunique()} cities")
print(f"  (periphery excludes {acp['processing_acps'].nunique() - per['processing_acps'].nunique()} "
      f"single-municipality ACPs)")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: Define regression specs
# ──────────────────────────────────────────────────────────────────────────────
# Each spec is (label, dataframe, outcome_col, mechanism_col)
PRIMARY_SPECS = [
    ('ACP',       acp, 'delta_gini',            'delta_house_price'),
    ('Capital',   cap, 'delta_regional_gini',   'delta_regional_house_values'),
    ('Periphery', per, 'delta_gini_periphery',  'delta_house_values_periphery'),
]

SECONDARY_SPECS = [
    ('ACP',       acp, 'delta_gini',            'delta_affordability_median'),
    ('Capital',   cap, 'delta_regional_gini',   'delta_median_affordability'),
    ('Periphery', per, 'delta_gini_periphery',  'delta_median_affordability_periphery'),
]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: Fit regressions
# ──────────────────────────────────────────────────────────────────────────────

def run_regression(label, df, outcome, mechanism, city_col='processing_acps'):
    """OLS with city fixed effects. Returns a result dict."""
    formula = f'{outcome} ~ {mechanism} + C({city_col})'
    res = smf.ols(formula, data=df).fit(cov_type='HC3')

    beta    = res.params[mechanism]
    se      = res.bse[mechanism]
    tstat   = res.tvalues[mechanism]
    pval    = res.pvalues[mechanism]
    r2      = res.rsquared
    r2_adj  = res.rsquared_adj
    n       = int(res.nobs)
    n_cities = df[city_col].nunique()

    stars = ''
    if pval < 0.01:  stars = '***'
    elif pval < 0.05: stars = '**'
    elif pval < 0.10: stars = '*'

    return {
        'level':    label,
        'outcome':  outcome,
        'mechanism': mechanism,
        'beta':     beta,
        'se':       se,
        't':        tstat,
        'p':        pval,
        'stars':    stars,
        'r2':       r2,
        'r2_adj':   r2_adj,
        'n':        n,
        'n_cities': n_cities,
    }


print("\n" + "=" * 70)
print("SECTION 3: Running regressions")
print("=" * 70)

primary_results   = []
secondary_results = []

print("\n  PRIMARY (ΔM = house price change):")
for label, df, outcome, mechanism in PRIMARY_SPECS:
    row = run_regression(label, df, outcome, mechanism)
    primary_results.append(row)
    sig = row['stars'] if row['stars'] else 'n.s.'
    print(f"    {label:12s}  β={row['beta']:+.6f}  SE={row['se']:.6f}  "
          f"t={row['t']:+.3f}  p={row['p']:.4f} {sig:4s}  "
          f"R²={row['r2']:.4f}  N={row['n']}")

print("\n  SECONDARY (ΔM = median rent/income ratio, renters only):")
for label, df, outcome, mechanism in SECONDARY_SPECS:
    row = run_regression(label, df, outcome, mechanism)
    secondary_results.append(row)
    sig = row['stars'] if row['stars'] else 'n.s.'
    print(f"    {label:12s}  β={row['beta']:+.6f}  SE={row['se']:.6f}  "
          f"t={row['t']:+.3f}  p={row['p']:.4f} {sig:4s}  "
          f"R²={row['r2']:.4f}  N={row['n']}")

primary_df   = pd.DataFrame(primary_results)
secondary_df = pd.DataFrame(secondary_results)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: Descriptive stats on ΔM and ΔY (helps interpret magnitudes)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: Descriptive statistics on ΔM and ΔY")
print("=" * 70)

for label, df, outcome, mechanism in PRIMARY_SPECS:
    print(f"\n  {label}:")
    print(f"    ΔY ({outcome}):   "
          f"mean={df[outcome].mean():+.6f}  std={df[outcome].std():.6f}  "
          f"min={df[outcome].min():+.6f}  max={df[outcome].max():+.6f}")
    print(f"    ΔM ({mechanism}): "
          f"mean={df[mechanism].mean():+.6f}  std={df[mechanism].std():.6f}  "
          f"min={df[mechanism].min():+.6f}  max={df[mechanism].max():+.6f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: Policy-config breakdown of ΔY (ACP level)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5: Mean ΔY by policy configuration (ACP level)")
print("=" * 70)

config_means = (
    acp.groupby('policy_config')[['delta_gini', 'delta_house_price']]
    .agg(['mean', 'std'])
    .round(6)
)
print(config_means.to_string())


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: Build LaTeX table
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6: Building LaTeX table")
print("=" * 70)


def fmt_coef(beta, stars):
    return f"{beta:+.4f}{stars}"

def fmt_se(se):
    return f"({se:.4f})"

def fmt_r2(r2):
    return f"{r2:.3f}"


levels = [r['level'] for r in primary_results]

tex_lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Housing wealth channel: regression results (Equation~3)}',
    r'\label{tab:regression}',
    r'\begin{tabular}{lccc}',
    r'\toprule',
    r' & ACP & Capital & Periphery \\',
    r'\midrule',
    r'\multicolumn{4}{l}{\textit{Panel A: Primary mechanism} ($\Delta M$ = house price change)} \\[2pt]',
]

# β row
beta_cells = ' & '.join(fmt_coef(r['beta'], r['stars']) for r in primary_results)
tex_lines.append(r'$\hat{\beta}$ & ' + beta_cells + r' \\')

# SE row
se_cells = ' & '.join(fmt_se(r['se']) for r in primary_results)
tex_lines.append(r' & ' + se_cells + r' \\[4pt]')

# R² row
r2_cells = ' & '.join(fmt_r2(r['r2']) for r in primary_results)
tex_lines.append(r'$R^2$ & ' + r2_cells + r' \\')

# N row
n_cells = ' & '.join(str(r['n']) for r in primary_results)
tex_lines.append(r'$N$ & ' + n_cells + r' \\')

# Cities row
nc_cells = ' & '.join(str(r['n_cities']) for r in primary_results)
tex_lines.append(r'Cities & ' + nc_cells + r' \\')

tex_lines += [
    r'\midrule',
    r'\multicolumn{4}{l}{\textit{Panel B: Secondary check} ($\Delta M$ = median rent/income ratio, renters)} \\[2pt]',
]

beta_cells_s = ' & '.join(fmt_coef(r['beta'], r['stars']) for r in secondary_results)
tex_lines.append(r'$\hat{\beta}$ & ' + beta_cells_s + r' \\')

se_cells_s = ' & '.join(fmt_se(r['se']) for r in secondary_results)
tex_lines.append(r' & ' + se_cells_s + r' \\[4pt]')

r2_cells_s = ' & '.join(fmt_r2(r['r2']) for r in secondary_results)
tex_lines.append(r'$R^2$ & ' + r2_cells_s + r' \\')

n_cells_s = ' & '.join(str(r['n']) for r in secondary_results)
tex_lines.append(r'$N$ & ' + n_cells_s + r' \\')

tex_lines += [
    r'\midrule',
    r'\multicolumn{4}{l}{\footnotesize Notes: OLS with metropolitan-area fixed effects (HC3 standard errors in parentheses).} \\',
    r'\multicolumn{4}{l}{\footnotesize $\Delta M$ and $\Delta Y$ are 60-month window averages (2035--2039) of policy-induced changes (Eq.~1--2).} \\',
    r'\multicolumn{4}{l}{\footnotesize Periphery excludes five single-municipality ACPs (Boa Vista, Campo Grande, Manaus, Palmas, Rio Branco).} \\',
    r'\multicolumn{4}{l}{\footnotesize *** $p<0.01$, ** $p<0.05$, * $p<0.10$.} \\',
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{table}',
]

tex_str = '\n'.join(tex_lines)
print(tex_str)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8: Regressions separated by policy dimension
# ──────────────────────────────────────────────────────────────────────────────
# Split on policy_melhorias to isolate each lever:
#   active   (True)  — 3 configs: low/medium/high × improvement ON
#   inactive (False) — 2 configs: low/high         × improvement OFF
# Running both primary M (house price) and secondary M (rent/income) on each
# subset reveals which policy dimension drives each channel.
print("\n" + "=" * 70)
print("SECTION 8: Regressions by policy dimension")
print("=" * 70)

acp_active   = acp[acp['policy_melhorias'] == True].copy()
acp_inactive = acp[acp['policy_melhorias'] == False].copy()
cap_active   = cap[cap['policy_melhorias'] == True].copy()
cap_inactive = cap[cap['policy_melhorias'] == False].copy()
per_active   = per[per['policy_melhorias'] == True].copy()
per_inactive = per[per['policy_melhorias'] == False].copy()

SPLIT_GROUPS = [
    ('Improvement ON  (MELHORIAS=True,  interest varies)',
     acp_active,   cap_active,   per_active),
    ('Improvement OFF (MELHORIAS=False, interest varies)',
     acp_inactive, cap_inactive, per_inactive),
]

split_results = []

for group_label, acp_sub, cap_sub, per_sub in SPLIT_GROUPS:
    print(f"\n  {group_label}")
    print(f"  N: ACP={len(acp_sub)}, Capital={len(cap_sub)}, Periphery={len(per_sub)}")

    for mech_label, level_specs in [
        ('Primary M (house price)',
         [('ACP',       acp_sub, 'delta_gini',          'delta_house_price'),
          ('Capital',   cap_sub, 'delta_regional_gini', 'delta_regional_house_values'),
          ('Periphery', per_sub, 'delta_gini_periphery','delta_house_values_periphery')]),
        ('Secondary M (rent/income)',
         [('ACP',       acp_sub, 'delta_gini',          'delta_affordability_median'),
          ('Capital',   cap_sub, 'delta_regional_gini', 'delta_median_affordability'),
          ('Periphery', per_sub, 'delta_gini_periphery','delta_median_affordability_periphery')]),
    ]:
        print(f"    {mech_label}:")
        for level, df, outcome, mechanism in level_specs:
            if len(df) < 10 or df[mechanism].std() < 1e-10:
                print(f"      {level:12s}  insufficient variation — skipped")
                continue
            row = run_regression(level, df, outcome, mechanism)
            row['group'] = group_label
            split_results.append(row)
            sig = row['stars'] if row['stars'] else 'n.s.'
            print(f"      {level:12s}  β={row['beta']:+.6f}  SE={row['se']:.6f}  "
                  f"t={row['t']:+.3f}  p={row['p']:.4f} {sig:4s}  "
                  f"R²={row['r2']:.4f}  N={row['n']}")

split_df = pd.DataFrame(split_results)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9: House quality as alternative mechanism variable
# ──────────────────────────────────────────────────────────────────────────────
# delta_house_quality is the direct output of MELHORIAS (upgrade 0.5 → 1.0).
# Testing it as M asks: does a larger quality improvement translate into a
# larger Gini change? Only available at ACP level.
print("\n" + "=" * 70)
print("SECTION 9: House quality (delta_house_quality) as mechanism variable")
print("=" * 70)
print("  Direct target of MELHORIAS upgrade 0.5 → 1.0. ACP level only.")

quality_results = []
for subset_label, subset_df in [
    ('All configs',            acp),
    ('Improvement ON only',    acp_active),
    ('Improvement OFF only',   acp_inactive),
]:
    if subset_df['delta_house_quality'].std() < 1e-10:
        print(f"  {subset_label}: no variation in delta_house_quality — skipped")
        continue
    row = run_regression(subset_label, subset_df, 'delta_gini', 'delta_house_quality')
    row['group'] = subset_label
    quality_results.append(row)
    sig = row['stars'] if row['stars'] else 'n.s.'
    dq = subset_df['delta_house_quality']
    print(f"\n  {subset_label}  (N={len(subset_df)})")
    print(f"    delta_house_quality: mean={dq.mean():+.6f}  std={dq.std():.6f}  "
          f"pct > 0: {(dq > 0).mean()*100:.1f}%")
    print(f"    β={row['beta']:+.6f}  SE={row['se']:.6f}  "
          f"t={row['t']:+.3f}  p={row['p']:.4f} {sig}  R²={row['r2']:.4f}")

quality_df = pd.DataFrame(quality_results)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10: Cross-city heterogeneity in the rental channel
# ──────────────────────────────────────────────────────────────────────────────
# Per-city correlation between delta_affordability_median and delta_gini
# across all 15 obs (5 configs × 3 seeds). Split by improvement ON/OFF to see
# which lever drives the rental-burden → inequality link in each city.
print("\n" + "=" * 70)
print("SECTION 10: Cross-city heterogeneity in the rental channel")
print("=" * 70)

pop_df = pd.read_csv(DATA_DIR / 'acp_population.csv')

city_rows = []
for city in sorted(acp['processing_acps'].unique()):
    cdf      = acp[acp['processing_acps'] == city]
    cactive  = cdf[cdf['policy_melhorias'] == True]
    cinactive= cdf[cdf['policy_melhorias'] == False]

    corr_all     = cdf['delta_affordability_median'].corr(cdf['delta_gini'])
    corr_active  = (cactive['delta_affordability_median'].corr(cactive['delta_gini'])
                    if len(cactive) > 2 else np.nan)
    corr_inactive= (cinactive['delta_affordability_median'].corr(cinactive['delta_gini'])
                    if len(cinactive) > 2 else np.nan)
    corr_price   = cdf['delta_house_price'].corr(cdf['delta_gini'])
    mean_dy      = cdf['delta_gini'].mean()
    avg_pop      = pop_df.loc[pop_df['processing_acps'] == city, 'avg_population'].values[0]

    city_rows.append({
        'city':              city,
        'avg_population':    avg_pop,
        'mean_delta_gini':   mean_dy,
        'corr_rent_all':     corr_all,
        'corr_rent_active':  corr_active,
        'corr_rent_inactive':corr_inactive,
        'corr_price_gini':   corr_price,
        'n_obs':             len(cdf),
    })

city_df = pd.DataFrame(city_rows).sort_values('mean_delta_gini')

print(f"\n  {'City':20s}  {'mean ΔGini':>10s}  {'log(pop)':>8s}  "
      f"{'corr(rent,Δgini)':>16s}  {'ON':>8s}  {'OFF':>8s}  {'corr(price,Δgini)':>17s}")
print("  " + "-" * 98)
for _, r in city_df.iterrows():
    print(f"  {r['city']:20s}  {r['mean_delta_gini']:+10.6f}  "
          f"{np.log(r['avg_population']):8.2f}  "
          f"{r['corr_rent_all']:+16.4f}  "
          f"{r['corr_rent_active']:+8.4f}  "
          f"{r['corr_rent_inactive']:+8.4f}  "
          f"{r['corr_price_gini']:+17.4f}")

corr_pop_gini = city_df['mean_delta_gini'].corr(np.log(city_df['avg_population']))
print(f"\n  Corr(mean ΔGini, log population): {corr_pop_gini:+.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11: ΔGini time series (monthly trajectory)
# ──────────────────────────────────────────────────────────────────────────────
# Loads raw final_stats.csv to compute monthly delta_gini = treated − baseline
# for each (city, config, seed). Averaged over seeds and configs per city.
# Validates that pre-2020 deltas ≈ 0 (all configs identical before divergence).
print("\n" + "=" * 70)
print("SECTION 11: ΔGini time series (monthly trajectory)")
print("=" * 70)

STATS_FILE = SCRIPT_DIR.parent.parent / 'output' / 'final_stats.csv'
print(f"  Loading {STATS_FILE.name} (this may take a moment)...")

ts = pd.read_csv(STATS_FILE, low_memory=False)
ts['month'] = pd.to_datetime(ts['month'])

# Keep complete sims only
n_months_ts = ts.groupby('simulation_id')['month'].nunique()
ts = ts[ts['simulation_id'].isin(n_months_ts[n_months_ts == 360].index)].copy()

ts['interest_housing'] = ts['interest_housing'].map(
    {'baixa': 'low', 'media': 'medium', 'alta': 'high'})
ts['seed'] = ts['simulation_id'].str.split('__').str[-1].astype(int)

print(f"  {ts['simulation_id'].nunique()} complete sims, "
      f"{ts['processing_acps'].nunique()} cities, "
      f"{ts['month'].nunique()} months")

# Separate baseline
is_base = (ts['interest_housing'] == 'medium') & (ts['policy_melhorias'] == False)
baseline_ts = (
    ts[is_base][['processing_acps', 'seed', 'month', 'gini_index']]
    .rename(columns={'gini_index': 'gini_base'})
)
treated_ts = ts[~is_base].copy()
treated_ts['policy_config'] = treated_ts.apply(
    lambda r: f"{r['interest_housing']}_{'active' if r['policy_melhorias'] else 'inactive'}",
    axis=1
)

# Monthly delta per (city, config, seed)
ts_merged = treated_ts.merge(
    baseline_ts, on=['processing_acps', 'seed', 'month'], how='inner'
)
ts_merged['delta_gini'] = ts_merged['gini_index'] - ts_merged['gini_base']

# Average over seeds and configs → one value per city per month
ts_city = (
    ts_merged
    .groupby(['processing_acps', 'month'])['delta_gini']
    .mean()
    .reset_index()
)
ts_city['year'] = ts_city['month'].dt.year

# Also: average by config (pooled cities) to see which configs diverge most
ts_config_global = (
    ts_merged
    .groupby(['policy_config', 'month'])['delta_gini']
    .mean()
    .reset_index()
)
ts_config_global['year'] = ts_config_global['month'].dt.year

# Validation: pre-2020 deltas should be ~0 (configs are identical before 2020)
pre2020 = ts_city[ts_city['year'] < 2020]['delta_gini']
print(f"\n  Pre-2020 ΔGini (all cities): "
      f"mean={pre2020.mean():.8f}  std={pre2020.std():.8f}  (expect ≈ 0)")

# Annual mean ΔGini for representative cities (two each direction)
FOCAL_CITIES = ['SAO PAULO', 'PORTO ALEGRE', 'CURITIBA', 'FORTALEZA']
annual = (
    ts_city[ts_city['processing_acps'].isin(FOCAL_CITIES)]
    .groupby(['processing_acps', 'year'])['delta_gini']
    .mean()
    .unstack('processing_acps')
)
focal_present = [c for c in FOCAL_CITIES if c in annual.columns]

print(f"\n  Annual mean ΔGini (averaged over configs and seeds):")
header = f"  {'Year':6s}" + ''.join(f"  {c:>14s}" for c in focal_present)
print(header)
print("  " + "-" * len(header))
for year, row in annual.iterrows():
    if year < 2018:
        continue
    line = f"  {year:6d}" + ''.join(f"  {row.get(c, np.nan):+14.6f}" for c in focal_present)
    print(line)

# Annual mean ΔGini by policy_config (global, all cities)
annual_config = (
    ts_config_global
    .groupby(['policy_config', 'year'])['delta_gini']
    .mean()
    .unstack('policy_config')
)
print(f"\n  Annual mean ΔGini by policy config (all 27 cities pooled):")
print(annual_config[annual_config.index >= 2018].round(6).to_string())

# Save full time series
ts_city_out = ts_city[['processing_acps', 'month', 'year', 'delta_gini']]
ts_config_out = ts_merged[['processing_acps', 'policy_config', 'seed', 'month', 'delta_gini']]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12: Capital vs periphery paired comparison
# ──────────────────────────────────────────────────────────────────────────────
# Tests whether policy effects on Gini diverge between the capital municipality
# (primary beneficiary of MELHORIAS owner-occupier upgrades) and the peripheral
# municipalities of the same ACP (exposed to price spillovers but fewer MELHORIAS
# beneficiaries). For each city that has a periphery, computes:
#   gap = delta_gini_periphery − delta_regional_gini   (positive = periphery worse)
# Averaged over all non-baseline configs and all seeds.
print("\n" + "=" * 70)
print("SECTION 12: Capital vs periphery paired comparison")
print("=" * 70)

# Merge capital and periphery deltas on (city, policy_config, seed)
cap_slim = cap[['processing_acps', 'policy_config', 'seed',
                'delta_regional_gini', 'delta_regional_house_values',
                'delta_median_affordability']].copy()
per_slim = per[['processing_acps', 'policy_config', 'seed',
                'delta_gini_periphery', 'delta_house_values_periphery',
                'delta_median_affordability_periphery']].copy()

paired = cap_slim.merge(per_slim, on=['processing_acps', 'policy_config', 'seed'], how='inner')
paired['gap_gini']   = paired['delta_gini_periphery'] - paired['delta_regional_gini']
paired['gap_rent']   = (paired['delta_median_affordability_periphery']
                        - paired['delta_median_affordability'])
paired['gap_price']  = (paired['delta_house_values_periphery']
                        - paired['delta_regional_house_values'])

# City-level averages (over all configs × seeds)
city_paired = (
    paired
    .groupby('processing_acps')[
        ['delta_regional_gini', 'delta_gini_periphery',
         'gap_gini', 'gap_rent', 'gap_price']
    ]
    .mean()
    .reset_index()
    .sort_values('gap_gini')
)

n_cities_paired = len(city_paired)
n_periphery_worse = (city_paired['gap_gini'] > 0).sum()

print(f"\n  Cities with periphery: {n_cities_paired}")
print(f"  Cities where periphery ΔGini > capital ΔGini: {n_periphery_worse}/{n_cities_paired}")
print(f"\n  {'City':22s}  {'Capital ΔGini':>14s}  {'Periph ΔGini':>13s}  {'Gap (P-C)':>10s}  {'Gap rent':>9s}")
print("  " + "-" * 78)
for _, r in city_paired.iterrows():
    flag = " *" if r['gap_gini'] > 0 else ""
    print(f"  {r['processing_acps']:22s}  {r['delta_regional_gini']:+14.6f}  "
          f"{r['delta_gini_periphery']:+13.6f}  {r['gap_gini']:+10.6f}  "
          f"{r['gap_rent']:+9.6f}{flag}")

# Per-config breakdown: does the gap hold for each policy lever separately?
print(f"\n  Mean gap (periphery − capital) by policy config:")
config_gap = (
    paired
    .groupby('policy_config')[['gap_gini', 'gap_rent', 'gap_price']]
    .mean()
    .sort_values('gap_gini')
)
print(config_gap.round(6).to_string())

# Aggregate directional summary
print(f"\n  Summary:")
print(f"    Capital   mean ΔGini: {city_paired['delta_regional_gini'].mean():+.6f}  "
      f"(falls in {(city_paired['delta_regional_gini'] < 0).sum()}/{n_cities_paired} cities)")
print(f"    Periphery mean ΔGini: {city_paired['delta_gini_periphery'].mean():+.6f}  "
      f"(falls in {(city_paired['delta_gini_periphery'] < 0).sum()}/{n_cities_paired} cities)")
print(f"    Mean gap (P−C):       {city_paired['gap_gini'].mean():+.6f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7: Save outputs
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 7: Saving outputs")
print("=" * 70)

outputs = {
    'regression_primary.csv':    primary_df,
    'regression_secondary.csv':  secondary_df,
    'regression_split.csv':      split_df,
    'regression_quality.csv':    quality_df,
    'cross_city_heterogeneity.csv': city_df,
    'timeseries_delta_gini.csv': ts_city_out,
    'timeseries_by_config.csv':  ts_config_out,
    'capital_periphery_paired.csv': city_paired,
    'capital_periphery_by_config.csv': config_gap.reset_index(),
}
for fname, df in outputs.items():
    df.to_csv(OUT_DIR / fname, index=False)
    print(f"  {fname:<45s}  →  {OUT_DIR / fname}")

tex_path = OUT_DIR / 'regression_table.tex'
tex_path.write_text(tex_str)
print(f"  {'regression_table.tex':<45s}  →  {tex_path}")

print("\nDone.")
