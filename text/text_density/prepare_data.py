"""
prepare_data.py
===============
Reads output/final_stats.csv and output/regional_stats.csv from the PS3 project
root and produces the analysis-ready datasets needed for the density paper.

Run from the project root:
    /home/b/miniconda3/envs/ps3/bin/python text/text_density/prepare_data.py

Outputs (written to text/text_density/data/):
    acp_deltas.csv          delta panel at metropolitan-area level (Eq. 1–3)
    mun_deltas.csv          delta panel at municipality level
    capital_deltas.csv      capital municipality only
    periphery_deltas.csv    non-capital municipalities, population-weighted
    acp_population.csv      mean ACP population (baseline) for cross-city scatter
    capital_mun_ids.csv     capital mun_id identified for each ACP

Methodology reference: text/text_density/main.tex, Sections 3.1–3.5
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent       # …/PS3/
STATS_FILE   = PROJECT_ROOT / 'output' / 'final_stats.csv'
REGIONAL_FILE = PROJECT_ROOT / 'output' / 'regional_stats.csv'
OUT_DIR      = SCRIPT_DIR / 'data'
OUT_DIR.mkdir(exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
COMPLETE_SIM_MONTHS = 360    # 2010-01 to 2039-12 inclusive
WINDOW_MONTHS       = 60     # final 60 months = 2035-01 to 2039-12 (Eq. 2)
BASELINE_IH         = 'medium'   # after translation
BASELINE_PM         = False

# Portuguese → English label maps applied immediately after loading
IH_MAP = {'baixa': 'low', 'media': 'medium', 'alta': 'high'}
PM_MAP = {True: 'active', False: 'inactive'}    # for policy_config label only;
                                                  # policy_melhorias bool column kept as-is

# Columns to carry through from each source file
#
# Primary mechanism M (Eq. 3):
#   ACP level  → house_price         (final_stats)
#   Mun level  → regional_house_values (regional_stats)
#
# Causal chain: policy → house_price ↑ → estate_value ↑ → wealth ↑
#               → permanent_income ↑ (= wages + r × wealth) → Gini ↓
# This applies to BOTH policies:
#   MELHORIAS: upgrades quality 0.5→1.0 for bottom-38th-pct OWNER-OCCUPIERS
#              → direct house price rise for target households
#   INTEREST:  lower credit rate → homeownership access expands
#              → housing demand rises → price appreciation
#
# Secondary variable (INTEREST channel only):
#   median_affordability / affordability_median = median(rent / permanent_income)
#   among RENTERS. Valid for the credit-rate dimension; MELHORIAS targets owners
#   so this misses its main mechanism. Kept for robustness.
#
# NOT used:
#   affordability_ratio (regional) / affordable (final_stats)
#   = share of renters paying < 30% income OR holding a voucher.
#   Includes voucher holders in the numerator — mechanical policy endogeneity.
ACP_OUTCOME_COLS = [
    'gini_index',
    'house_price',            # PRIMARY mechanism M (ACP level)
    'affordability_median',   # secondary: median rent/income among renters (INTEREST channel)
    'house_quality',          # descriptive: direct target of MELHORIAS upgrade
    'house_vacancy',
    'unemployment',
    'pop',
]

MUN_OUTCOME_COLS = [
    'regional_gini',
    'regional_house_values',  # PRIMARY mechanism M (municipality level)
    'median_affordability',   # secondary: median rent/income among renters
    'pop',
]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: Load raw data
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: Loading data")
print("=" * 70)

print(f"  {STATS_FILE.name} ...")
stats = pd.read_csv(STATS_FILE, low_memory=False)
stats['month'] = pd.to_datetime(stats['month'])
print(f"    {len(stats):,} rows, {stats['simulation_id'].nunique()} simulation_ids")

print(f"  {REGIONAL_FILE.name} ...")
reg = pd.read_csv(REGIONAL_FILE, low_memory=False)
reg['month'] = pd.to_datetime(reg['month'])
print(f"    {len(reg):,} rows, {reg['simulation_id'].nunique()} simulation_ids")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1b: Translate Portuguese labels to English
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 1b: Translating labels to English")
print("=" * 70)

for frame, label in [(stats, 'final_stats'), (reg, 'regional_stats')]:
    frame['interest_housing'] = frame['interest_housing'].map(IH_MAP)

n_unmapped = stats['interest_housing'].isna().sum()
if n_unmapped:
    print(f"  WARNING: {n_unmapped} rows with unmapped interest_housing values")
else:
    print(f"  interest_housing: baixa→low, media→medium, alta→high  ✓")
    print(f"  policy_melhorias: kept as bool (True/False); "
          f"policy_config will use 'active'/'inactive'")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: Filter to complete simulations (360 months)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: Filtering to complete simulations (360 months)")
print("=" * 70)


def filter_complete(df, label):
    n_months = df.groupby('simulation_id')['month'].nunique()
    complete  = n_months[n_months == COMPLETE_SIM_MONTHS].index
    incomplete = n_months[n_months < COMPLETE_SIM_MONTHS]
    if len(incomplete):
        print(f"  [{label}] Dropping {len(incomplete)} incomplete simulations:")
        meta = (
            df[df['simulation_id'].isin(incomplete.index)]
            [['simulation_id', 'processing_acps', 'interest_housing', 'policy_melhorias']]
            .drop_duplicates()
        )
        for _, r in meta.sort_values('processing_acps').iterrows():
            nm = incomplete[r['simulation_id']]
            print(f"    {r['processing_acps']:20s}  ih={r['interest_housing']:6s}  "
                  f"melhorias={str(r['policy_melhorias']):5s}  months_done={nm}")
    print(f"  [{label}] Retained {len(complete)} complete simulations "
          f"(out of {len(n_months)})")
    return df[df['simulation_id'].isin(complete)].copy()


stats = filter_complete(stats, 'final_stats')
reg   = filter_complete(reg,   'regional_stats')

n_complete = stats['simulation_id'].nunique()
assert n_complete == reg['simulation_id'].nunique(), \
    "Mismatch in complete simulation count between final_stats and regional_stats"

print(f"\n  Complete simulations retained: {n_complete}")
print(f"  ACPs covered: {stats['processing_acps'].nunique()}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: Parse seed index from simulation_id
# ──────────────────────────────────────────────────────────────────────────────
# simulation_id format:
#   <timestamp>__INTEREST_HOUSING=<ih>__POLICY_MELHORIAS=<pm>__PROCESSING_ACPS=<city>__<seed>
# The seed (0, 1, 2) is the last element after splitting on '__'.

print("\n" + "=" * 70)
print("SECTION 3: Parsing seed index")
print("=" * 70)

stats['seed'] = stats['simulation_id'].str.split('__').str[-1].astype(int)
reg['seed']   = reg['simulation_id'].str.split('__').str[-1].astype(int)

seed_check = stats.groupby(['processing_acps', 'interest_housing',
                             'policy_melhorias'])['seed'].nunique()
print(f"  Seeds per (city, config): min={seed_check.min()}, max={seed_check.max()}")
if (seed_check < 3).any():
    print("  WARNING: some combos have fewer than 3 seeds:")
    print(seed_check[seed_check < 3].to_string())


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: Define the analysis window (final 60 months)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: Defining the analysis window")
print("=" * 70)

T       = stats['month'].max()
T_start = T - pd.DateOffset(months=WINDOW_MONTHS - 1)

print(f"  T (last month):   {T.strftime('%Y-%m')}")
print(f"  T_start:          {T_start.strftime('%Y-%m')}")
print(f"  Window:           {WINDOW_MONTHS} months ({T_start.strftime('%Y-%m')} "
      f"to {T.strftime('%Y-%m')})")

stats_w = stats[(stats['month'] >= T_start) & (stats['month'] <= T)].copy()
reg_w   = reg  [(reg['month']   >= T_start) & (reg['month']   <= T)].copy()

n_stats_months = stats_w.groupby('simulation_id')['month'].nunique()
n_reg_months   = reg_w.groupby('simulation_id')['month'].nunique()
print(f"  Months in window per sim: stats={n_stats_months.unique().tolist()}, "
      f"regional={n_reg_months.unique().tolist()}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: Compute simulation-level means over the window
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5: Averaging over the window (Eq. 2)")
print("=" * 70)

GROUP_ACP = ['simulation_id', 'processing_acps',
             'interest_housing', 'policy_melhorias', 'seed']

GROUP_MUN = ['simulation_id', 'processing_acps', 'mun_id',
             'interest_housing', 'policy_melhorias', 'seed']

stats_mean = (
    stats_w
    .groupby(GROUP_ACP)[ACP_OUTCOME_COLS]
    .mean()
    .reset_index()
)

reg_mean = (
    reg_w
    .groupby(GROUP_MUN)[MUN_OUTCOME_COLS]
    .mean()
    .reset_index()
)

print(f"  stats_mean: {len(stats_mean)} rows (expect {n_complete} = 1 per sim)")
print(f"  reg_mean:   {len(reg_mean)} rows "
      f"(expect {n_complete} x ~{reg['mun_id'].nunique() // n_complete} munis per sim)")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: Separate baseline and compute counterfactual deltas (Eq. 1)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6: Computing counterfactual deltas (Eq. 1)")
print("=" * 70)

# Baseline: POLICY_MELHORIAS=False, INTEREST_HOUSING=media
is_baseline_acp = ((stats_mean['policy_melhorias'] == BASELINE_PM) &
                   (stats_mean['interest_housing'] == BASELINE_IH))
is_baseline_reg = ((reg_mean['policy_melhorias']   == BASELINE_PM) &
                   (reg_mean['interest_housing']    == BASELINE_IH))

baseline_acp = (
    stats_mean[is_baseline_acp]
    [['processing_acps', 'seed'] + ACP_OUTCOME_COLS]
    .rename(columns={c: f'{c}_base' for c in ACP_OUTCOME_COLS})
)

baseline_reg = (
    reg_mean[is_baseline_reg]
    [['processing_acps', 'mun_id', 'seed'] + MUN_OUTCOME_COLS]
    .rename(columns={c: f'{c}_base' for c in MUN_OUTCOME_COLS})
)

print(f"  Baseline sims (ACP):    {len(baseline_acp)} "
      f"(expect {stats_mean['processing_acps'].nunique() * 3})")
print(f"  Baseline sims (mun):    {len(baseline_reg)}")

# Non-baseline configurations (5 out of 6)
treated_acp = stats_mean[~is_baseline_acp].copy()
treated_reg = reg_mean[~is_baseline_reg].copy()

print(f"  Treated sims (ACP):     {len(treated_acp)} "
      f"(expect cities x 5 configs x 3 seeds)")

# Match treated to baseline by city + seed, compute deltas
acp_merged = treated_acp.merge(
    baseline_acp, on=['processing_acps', 'seed'], how='inner'
)
mun_merged = treated_reg.merge(
    baseline_reg, on=['processing_acps', 'mun_id', 'seed'], how='inner'
)

n_unmatched_acp = len(treated_acp) - len(acp_merged)
n_unmatched_mun = len(treated_reg) - len(mun_merged)
if n_unmatched_acp:
    print(f"  WARNING: {n_unmatched_acp} treated ACP rows dropped (no baseline match)")
if n_unmatched_mun:
    print(f"  WARNING: {n_unmatched_mun} treated mun rows dropped (no baseline match)")

# ACP-level deltas
acp_merged['delta_gini']                 = acp_merged['gini_index']           - acp_merged['gini_index_base']
acp_merged['delta_house_price']          = acp_merged['house_price']          - acp_merged['house_price_base']
acp_merged['delta_affordability_median'] = acp_merged['affordability_median'] - acp_merged['affordability_median_base']
acp_merged['delta_house_quality']        = acp_merged['house_quality']        - acp_merged['house_quality_base']
acp_merged['delta_vacancy']              = acp_merged['house_vacancy']        - acp_merged['house_vacancy_base']
acp_merged['delta_unemployment']         = acp_merged['unemployment']         - acp_merged['unemployment_base']

# Municipal-level deltas
mun_merged['delta_regional_gini']           = mun_merged['regional_gini']        - mun_merged['regional_gini_base']
mun_merged['delta_regional_house_values']   = mun_merged['regional_house_values']- mun_merged['regional_house_values_base']
mun_merged['delta_median_affordability']    = mun_merged['median_affordability'] - mun_merged['median_affordability_base']

print(f"  acp_merged: {len(acp_merged)} rows")
print(f"  mun_merged: {len(mun_merged)} rows")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7: Policy configuration labels
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 7: Labelling policy configurations")
print("=" * 70)

# 5 non-baseline configurations (after translation):
#   low_inactive    high_inactive
#   low_active      medium_active   high_active
CONFIG_ORDER = [
    'low_inactive',
    'high_inactive',
    'low_active',
    'medium_active',
    'high_active',
]


def policy_label(row):
    pm = PM_MAP[row['policy_melhorias']]
    return f"{row['interest_housing']}_{pm}"


acp_merged['policy_config'] = acp_merged.apply(policy_label, axis=1)
mun_merged['policy_config'] = mun_merged.apply(policy_label, axis=1)

# Validate all expected configs are present
found_configs = sorted(acp_merged['policy_config'].unique())
print(f"  Configs found: {found_configs}")
missing = set(CONFIG_ORDER) - set(found_configs)
if missing:
    print(f"  WARNING: missing configs: {missing}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8: Identify capital municipality per ACP
# ──────────────────────────────────────────────────────────────────────────────
# Capital = the municipality with the highest average population in the baseline.
# This uses the within-window baseline mean population (pop_base after merge).
print("\n" + "=" * 70)
print("SECTION 8: Identifying capital municipalities")
print("=" * 70)

# Use the baseline regional data (before merging) for a clean population estimate
cap_pop = (
    baseline_reg
    .groupby(['processing_acps', 'mun_id'])['pop_base']
    .mean()
    .reset_index()
)
cap_idx = cap_pop.groupby('processing_acps')['pop_base'].idxmax()
capitals = (
    cap_pop.loc[cap_idx, ['processing_acps', 'mun_id']]
    .rename(columns={'mun_id': 'capital_mun_id'})
    .reset_index(drop=True)
)

print("  Capital municipalities (largest by mean population in window):")
for _, r in capitals.sort_values('processing_acps').iterrows():
    avg_pop = cap_pop[
        (cap_pop['processing_acps'] == r['processing_acps']) &
        (cap_pop['mun_id'] == r['capital_mun_id'])
    ]['pop_base'].values[0]
    n_munis = cap_pop[cap_pop['processing_acps'] == r['processing_acps']]['mun_id'].nunique()
    print(f"    {r['processing_acps']:20s}  capital_mun_id={r['capital_mun_id']}  "
          f"avg_pop={avg_pop:,.0f}  ({n_munis} munis in metro)")

# Tag municipalities in the merged panel
mun_merged = mun_merged.merge(capitals, on='processing_acps', how='left')
mun_merged['is_capital'] = (mun_merged['mun_id'] == mun_merged['capital_mun_id'])


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9: Capital and periphery delta panels
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 9: Building capital and periphery panels")
print("=" * 70)

# Capital: one row per (city, config, seed)
capital_deltas = mun_merged[mun_merged['is_capital']].copy()
print(f"  capital_deltas: {len(capital_deltas)} rows")

# Periphery: population-weighted mean of all non-capital municipalities
periphery = mun_merged[~mun_merged['is_capital']].copy()

# Compute weighted sums, then divide by total population
periphery = periphery.copy()
periphery['w_gini']         = periphery['delta_regional_gini']         * periphery['pop_base']
periphery['w_house_val']    = periphery['delta_regional_house_values'] * periphery['pop_base']
periphery['w_afford_med']   = periphery['delta_median_affordability']  * periphery['pop_base']

GROUP_PER = ['processing_acps', 'policy_config',
             'interest_housing', 'policy_melhorias', 'seed']

per_agg = (
    periphery
    .groupby(GROUP_PER)
    .agg(
        sum_w_gini       =('w_gini',       'sum'),
        sum_w_house_val  =('w_house_val',  'sum'),
        sum_w_afford_med =('w_afford_med', 'sum'),
        total_pop        =('pop_base',     'sum'),
        n_municipalities =('mun_id',       'nunique'),
    )
    .reset_index()
)

per_agg['delta_gini_periphery']                 = per_agg['sum_w_gini']       / per_agg['total_pop']
per_agg['delta_house_values_periphery']         = per_agg['sum_w_house_val']  / per_agg['total_pop']
per_agg['delta_median_affordability_periphery'] = per_agg['sum_w_afford_med'] / per_agg['total_pop']

periphery_deltas = per_agg.drop(
    columns=['sum_w_gini', 'sum_w_house_val', 'sum_w_afford_med']
)
print(f"  periphery_deltas: {len(periphery_deltas)} rows "
      f"(n_municipalities per city: "
      f"min={periphery_deltas['n_municipalities'].min()}, "
      f"max={periphery_deltas['n_municipalities'].max()})")

# Sanity: for single-municipality ACPs periphery would be empty — warn if any
acps_no_periphery = set(capitals['processing_acps']) - set(periphery_deltas['processing_acps'])
if acps_no_periphery:
    print(f"  WARNING: no periphery municipalities for: {acps_no_periphery}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10: ACP population for cross-city scatter (Section 3.5)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 10: Building ACP population table")
print("=" * 70)

acp_pop = (
    stats_mean[is_baseline_acp]
    .groupby('processing_acps')['pop']
    .mean()
    .reset_index()
    .rename(columns={'pop': 'avg_population'})
    .sort_values('avg_population', ascending=False)
)
print("  ACP population (baseline mean, analysis window):")
for _, r in acp_pop.iterrows():
    print(f"    {r['processing_acps']:20s}  {r['avg_population']:>12,.0f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11: Save outputs
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 11: Saving outputs")
print("=" * 70)

ACP_OUT_COLS = [
    'processing_acps', 'policy_config', 'interest_housing', 'policy_melhorias', 'seed',
    # outcomes
    'delta_gini', 'gini_index', 'gini_index_base',
    # primary mechanism M: housing wealth channel
    'delta_house_price', 'house_price', 'house_price_base',
    # secondary: rental affordability burden (INTEREST channel only)
    'delta_affordability_median', 'affordability_median', 'affordability_median_base',
    # descriptive
    'delta_house_quality', 'house_quality', 'house_quality_base',
    'delta_vacancy', 'house_vacancy', 'house_vacancy_base',
    'delta_unemployment', 'unemployment', 'unemployment_base',
    'pop',
]

MUN_OUT_COLS = [
    'processing_acps', 'mun_id', 'is_capital', 'capital_mun_id',
    'policy_config', 'interest_housing', 'policy_melhorias', 'seed',
    # outcomes
    'delta_regional_gini', 'regional_gini', 'regional_gini_base',
    # primary mechanism M
    'delta_regional_house_values', 'regional_house_values', 'regional_house_values_base',
    # secondary
    'delta_median_affordability', 'median_affordability', 'median_affordability_base',
    'pop_base',
]

CAP_OUT_COLS = [
    'processing_acps', 'mun_id',
    'policy_config', 'interest_housing', 'policy_melhorias', 'seed',
    'delta_regional_gini', 'regional_gini', 'regional_gini_base',
    'delta_regional_house_values', 'regional_house_values', 'regional_house_values_base',
    'delta_median_affordability', 'median_affordability', 'median_affordability_base',
    'pop_base',
]

PER_OUT_COLS = [
    'processing_acps', 'policy_config', 'interest_housing', 'policy_melhorias', 'seed',
    'delta_gini_periphery',
    'delta_house_values_periphery',
    'delta_median_affordability_periphery',
    'total_pop', 'n_municipalities',
]

outputs = {
    'acp_deltas.csv':       acp_merged[ACP_OUT_COLS],
    'mun_deltas.csv':       mun_merged[MUN_OUT_COLS],
    'capital_deltas.csv':   capital_deltas[CAP_OUT_COLS],
    'periphery_deltas.csv': periphery_deltas[PER_OUT_COLS],
    'acp_population.csv':   acp_pop,
    'capital_mun_ids.csv':  capitals,
}

for fname, df in outputs.items():
    path = OUT_DIR / fname
    df.to_csv(path, index=False)
    print(f"  {fname:<30s}  {len(df):>6d} rows  →  {path}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12: Validation summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 12: Validation summary")
print("=" * 70)

acp_out = outputs['acp_deltas.csv']
cap_out = outputs['capital_deltas.csv']
per_out = outputs['periphery_deltas.csv']

print(f"\n  Complete simulations used: {n_complete} / 486")
print(f"  Analysis window:           {T_start.strftime('%Y-%m')} – {T.strftime('%Y-%m')}")
print(f"  ACPs:                      {acp_out['processing_acps'].nunique()}")
print(f"  Policy configs (non-base): {acp_out['policy_config'].nunique()}")
print(f"  Seeds:                     {acp_out['seed'].nunique()}")
print(f"  ACP delta obs:             {len(acp_out)}")
print(f"    (expect max {27 * 5 * 3} = 405 if all 486 complete)")

print(f"\n  delta_gini (ACP):")
print(f"    mean  = {acp_out['delta_gini'].mean():.6f}")
print(f"    std   = {acp_out['delta_gini'].std():.6f}")
print(f"    min   = {acp_out['delta_gini'].min():.6f}")
print(f"    max   = {acp_out['delta_gini'].max():.6f}")
print(f"    pct negative (gini reduced) = "
      f"{(acp_out['delta_gini'] < 0).mean() * 100:.1f}%")

print(f"\n  delta_house_price (primary M, ACP):")
print(f"    mean  = {acp_out['delta_house_price'].mean():.6f}")
print(f"    std   = {acp_out['delta_house_price'].std():.6f}")

print(f"\n  delta_affordability_median (secondary M, ACP):")
print(f"    mean  = {acp_out['delta_affordability_median'].mean():.6f}")
print(f"    std   = {acp_out['delta_affordability_median'].std():.6f}")

print(f"\n  delta_house_quality (descriptive, ACP):")
print(f"    mean  = {acp_out['delta_house_quality'].mean():.6f}")
print(f"    std   = {acp_out['delta_house_quality'].std():.6f}")

print(f"\n  delta_regional_gini (capital):")
print(f"    mean  = {cap_out['delta_regional_gini'].mean():.6f}")
print(f"    std   = {cap_out['delta_regional_gini'].std():.6f}")

print(f"\n  delta_gini_periphery:")
print(f"    mean  = {per_out['delta_gini_periphery'].mean():.6f}")
print(f"    std   = {per_out['delta_gini_periphery'].std():.6f}")

# Per-city mean delta_gini to flag any anomalies
print(f"\n  Mean delta_gini by ACP (averaged across configs and seeds):")
city_means = (
    acp_out
    .groupby('processing_acps')['delta_gini']
    .mean()
    .sort_values()
)
for city, val in city_means.items():
    bar = '▼' if val < 0 else '▲'
    print(f"    {city:20s}  {val:+.6f}  {bar}")

print("\nDone.")
