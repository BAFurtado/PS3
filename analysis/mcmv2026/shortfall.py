"""mcmv2026: d09 population against the immigration target (Census 2010-22 growth path) and the d25 - d09 city
effects. Same target as funding2026/population_shortfall.py. Municipal rows: d25 - d09 population and units 2026-34,
grouped by whether d25 adds units and whether d09 sits more than 5 % below target.
usage (repo root, env ps3_2): python analysis/mcmv2026/shortfall.py [root]"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path[:0] = ['.']
from analysis.output import columns_for
from world.population import census_growth
root = sys.argv[1] if len(sys.argv) > 1 else 'output/mcmv2026'
out = os.path.join(root, '_analysis')
A, B = '2026-01-01', '2034-12-31'


def reg(d):
    f = os.path.join(d, 'regional.csv')
    n = len(next(l for l in open(f) if l.strip()).split(';'))
    r = pd.read_csv(f, sep=';', header=None, names=columns_for('regional', n), usecols=['month', 'mun_id', 'pop', 'mcmv_units_bought'])
    r['month'] = pd.to_datetime(r.month); r['mun_id'] = r.mun_id.astype(str)
    return r


rows, muns = [], []
for done in sorted(glob.glob(os.path.join(root, 'd09', '*', '*', 'DONE'))):
    d = os.path.dirname(done); city, i = d.split(os.sep)[-2:]
    r = reg(d)
    first = r[r.month == r.month.min()].set_index('mun_id')['pop']
    g = pd.Series(census_growth(list(first.index)))
    r['target'] = r.mun_id.map(first) * r.mun_id.map(g) ** ((r.month - r.month.min()).dt.days / 365.25)
    c = r.groupby('month')[['pop', 'target']].sum()
    sh = lambda m: 100 * (c['pop'].loc[m] / c.target.loc[m] - 1)
    rows.append(dict(city=city, seed=int(i), short_2022=sh('2022-12-01'), short_2025=sh('2025-12-01')))
    o = reg(d.replace(os.sep + 'd09' + os.sep, os.sep + 'd25' + os.sep))
    win = lambda x: x[(x.month >= A) & (x.month <= B)]
    m = win(r).groupby('mun_id').agg(pop_ref=('pop', 'mean'), target=('target', 'mean'), units_ref=('mcmv_units_bought', 'sum'))
    m = m.join(win(o).groupby('mun_id').agg(pop_d25=('pop', 'mean'), units_d25=('mcmv_units_bought', 'sum')))
    muns.append(m.assign(city=city, seed=int(i)).reset_index())
S = pd.DataFrame(rows).groupby('city')[['short_2022', 'short_2025']].mean()
print('d09 population against target (% of target), seed means')
print(S.sort_values('short_2022').round(2).to_string())
print(f"below target in 2022: {(S.short_2022 < 0).sum()}/{len(S)}, beyond -5 %: {(S.short_2022 < -5).sum()}, "
      f"median {S.short_2022.median():.2f}\n")
C = pd.read_csv(os.path.join(out, 'city_effects.csv')).set_index('city')
C = C.drop(columns=[k for k in S if k in C]).join(S)
print('d09 population against target, 2025-12 (% of target), and d25 - d09 city effects 2026-34')
print(C[['short_2025', 'pop', 'families', 'dwellings', 'unemployment_pp', 'house_price', 'employment']].sort_values('short_2025').round(2).to_string())
print('\ncorrelation across 26 cities with short_2025 (pearson, spearman):')
for y in ('pop', 'families', 'dwellings', 'unemployment_pp', 'house_price', 'employment', 'real_gdp'):
    print(f'  {y:16s} {stats.pearsonr(C.short_2025, C[y])[0]:+.2f}  {stats.spearmanr(C.short_2025, C[y])[0]:+.2f}')
print(f'  pop vs unemployment_pp {stats.spearmanr(C["pop"], C.unemployment_pp)[0]:+.2f} (spearman)')
M = pd.concat(muns).groupby(['city', 'mun_id']).mean(numeric_only=True).drop(columns='seed')
M['short'] = 100 * (M.pop_ref / M.target - 1)
M['d_pop'] = M.pop_d25 - M.pop_ref; M['d_units'] = M.units_d25 - M.units_ref
G = M.assign(gets_units=M.d_units > 1, below_target=M.short < -5).groupby(['gets_units', 'below_target']).agg(
    municipalities=('d_pop', 'size'), d_units=('d_units', 'sum'), d_pop=('d_pop', 'sum'))
G['residents_per_unit'] = G.d_pop / G.d_units
print('\nmunicipalities, d25 - d09 2026-34:'); print(G.round(2).to_string())
C.to_csv(os.path.join(out, 'city_effects.csv'), float_format='%.5g')
