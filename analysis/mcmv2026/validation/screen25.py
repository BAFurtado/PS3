"""Validation screen: floor to the end of the run, 2017-19 levels, held-out data.
usage (repo root): python analysis/mcmv2026/validation/screen25.py <runs root> [out dir, default <runs root>]
Runs: every stats.csv under the root (server layout PROCESSING_ACPS=<ACP>/<i>/ with conf.json, or local
<ACP>_s<seed>_<arm>/), avg/ folders skipped. Writes screen25_runs.csv, screen25_cities.csv, screen25_table2.csv.
- Floor: levels.explodes over the first 10 y and over the whole run, first year with u >= 30 %.
- Levels 2017-19 (first 120 months, last 36): bands of analysis/calibration/data/level_targets.csv; unemployment per
  city [min(Census 2010, PNAD-C min 2012-19), PNAD-C max 2012-19] from gate_bounds25.csv; implausibility as in the
  calibration (seed sd, discrepancy 0.10).
- Held out: GDP per resident year 1 and 2017-19 deflated over IBGE 2010 (apu_va25.csv, +-20 %); real GDP 2012-19
  vs IBGE (+-1.5 pp/yr); family PI Gini 2017-19 against the state PNAD-C Gini 2012-19 (reported, different concept).
- Table 2: pooled median, p10, p90 over runs, 2011-2024."""
import glob, json, os, sys
import numpy as np
import pandas as pd
sys.path[:0] = ['.', 'analysis/mcmv2026/validation']
import dashboard as D
from analysis.calibration import levels
from analysis.validation.housing_validation import compute_derived_monthly_indicators

K, DISC, HERE = 1000.0, 0.10, 'analysis/mcmv2026/validation'
root = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else root
gdp = pd.read_csv(f'{HERE}/apu_va25.csv').set_index('acp')
gb = pd.read_csv(f'{HERE}/gate_bounds25.csv').set_index('acp')
LEVELS = levels.HOUSING + ['nontradable_inflation', 'unemployment', 'household_income_pc']
T2 = ['housing_stock_gdp', 'price_wage', 'price_income', 'vacancy', 'consumption_gdp', 'housing_production_per_1000',
      'housing_stock_permanent_income', 'gini', 'zero_cons', 'unemployment']


def identify(d):
    c = os.path.join(d, 'conf.json')
    if os.path.isfile(c):
        p = json.load(open(c))['PARAMS']
        return p['PROCESSING_ACPS'][0], int(p['SEED'])
    acp, seed, _ = os.path.basename(os.path.normpath(d)).rsplit('_', 2)
    return acp.replace('_', ' '), int(seed[1:])


runs, monthly = [], []
for f in sorted(glob.glob(os.path.join(root, '**', 'stats.csv'), recursive=True)):
    d = os.path.dirname(f)
    if os.path.basename(d) == 'avg':
        continue
    acp, seed = identify(d)
    df = D.load('.', d)
    if len(df) < 120:
        print('incomplete', acp, seed, len(df), 'months')
        continue
    m10, mall = levels.level_moments(df.iloc[:120], 36), levels.level_moments(df, 36)
    p = df.price_level / df.price_level.iloc[0]
    gdp_pc = df.gdp_level / df['pop'] * K
    y = pd.to_datetime(df.month).dt.year
    real = (df.gdp_level / df.price_level).groupby(y).mean()
    u30 = np.where(df.unemployment.values[24:] >= 0.30)[0]
    runs.append(dict(acp=acp, seed=seed, months=len(df), **{k: m10[k] for k in LEVELS},
                     explodes_10y=levels.explodes(m10), explodes_run=levels.explodes(mall),
                     first_u30=int(y.iloc[24 + u30[0]]) if len(u30) else None,
                     max_u_run=mall['max_unemployment_after_m24'], u_last_year=df.unemployment.tail(12).mean(),
                     gdp_pc_y1=gdp_pc.iloc[:12].mean() / gdp.loc[acp, 'gdp_pc'],
                     gdp_pc_1719=(gdp_pc / p).iloc[84:120].mean() / gdp.loc[acp, 'gdp_pc'],
                     real_gdp_12_19=((real[2019] / real[2012]) ** (1 / 7) - 1) * 100,
                     gini_1719=df.gini_index.iloc[84:120].mean()))
    h = df.copy()
    h['run_id'] = f'{acp}_{seed}'
    monthly.append(compute_derived_monthly_indicators(h))
R = pd.DataFrame(runs).sort_values(['acp', 'seed'])
R.to_csv(os.path.join(out, 'screen25_runs.csv'), index=False, float_format='%.4f')

C = R.groupby('acp').mean(numeric_only=True)
sd = R.groupby('acp')[LEVELS].std().fillna(0)
C['n'] = R.groupby('acp').size()
C['explodes_10y'] = R.groupby('acp').explodes_10y.any()
C['explodes_run'] = R.groupby('acp').explodes_run.any()
C['first_u30'] = R.groupby('acp').first_u30.min()
bands = levels.load_targets(C.index)
for a in C.index:
    if a in gb.index:
        bands[a]['unemployment'] = (min(gb.loc[a, 'census_u'], gb.loc[a, 'pnad_min']), gb.loc[a, 'pnad_max'])
    for k in LEVELS:
        b = bands[a].get(k)
        C.loc[a, f'I_{k}'] = levels.implausibility(C.loc[a, k], b, sd.loc[a, k], DISC) if b else np.nan
I = [f'I_{k}' for k in LEVELS]
C['max_I'] = C[I].max(axis=1)
C['real_gdp_ibge'] = gb.real_gdp_12_19.reindex(C.index)
C['state_gini'] = gb.state_gini_12_19.reindex(C.index)
C['gdp_pc_ok'] = C.gdp_pc_y1.between(0.8, 1.2) & C.gdp_pc_1719.between(0.8, 1.2)
C['real_gdp_ok'] = (C.real_gdp_12_19 - C.real_gdp_ibge).abs() <= 1.5
C.to_csv(os.path.join(out, 'screen25_cities.csv'), float_format='%.4f')

M = pd.concat(monthly)
M = M[(M.month >= '2011-01-01') & (M.month <= '2024-12-31')]
pr = M.groupby('run_id')[T2].mean()
T = pd.DataFrame({'median': pr.median(), 'p10': pr.quantile(.1), 'p90': pr.quantile(.9)})
T.to_csv(os.path.join(out, 'screen25_table2.csv'), float_format='%.4f')

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 60)
print(f'{len(R)} runs, {len(C)} cities\n')
print('== Floor: hard constraints (any seed)')
print(C[['n', 'explodes_10y', 'explodes_run', 'first_u30', 'max_u_run', 'u_last_year']].round(3).to_string())
print(f"no explosion in 10 y: {(~C.explodes_10y).sum()}/{len(C)}, to the end of the run: {(~C.explodes_run).sum()}/{len(C)}")
print('\n== Levels 2017-19 (seed means) and implausibility')
print(C[LEVELS].round(3).to_string())
print(C[I + ['max_I']].round(2).to_string())
print('cities with I <= 2 per level:', {k: int((C[f'I_{k}'] <= 2).sum()) for k in LEVELS if C[f'I_{k}'].notna().any()})
print('cities with I = 0 (in band) per level:', {k: int((C[f'I_{k}'] == 0).sum()) for k in LEVELS if C[f'I_{k}'].notna().any()})
print('\n== Held out')
print(C[['gdp_pc_y1', 'gdp_pc_1719', 'gdp_pc_ok', 'real_gdp_12_19', 'real_gdp_ibge', 'real_gdp_ok', 'gini_1719',
         'state_gini']].round(3).to_string())
print(f"GDP per resident within 20 %: {C.gdp_pc_ok.sum()}/{len(C)}; real GDP 2012-19 within 1.5 pp: "
      f"{C.real_gdp_ok.sum()}/{len(C)}; corr real GDP with IBGE {C.real_gdp_12_19.corr(C.real_gdp_ibge):.2f}")
print('\n== Table 2 (2011-2024, pooled over runs)')
print(T.round(3).to_string())
