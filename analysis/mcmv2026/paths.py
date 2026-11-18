"""mcmv2026: yearly effect of d13 and d25 against d09, 2024-2034, pooled over 26 cities x 10 seeds with a 90 %
bootstrap over cities, from stats.csv. Then city ranking reliability: per-city effect per extra real (d25 - d09,
2026-34) in seeds 0-4 against seeds 5-9 (Spearman).
usage (repo root, env ps3_2): python analysis/mcmv2026/paths.py [root]
Writes <root>/_analysis/paths.csv and city_effects.csv."""
import glob, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path[:0] = ['.']
from analysis.output import columns_for

root = sys.argv[1] if len(sys.argv) > 1 else 'output/mcmv2026'
out = os.path.join(root, '_analysis')
VARS = ['real_gdp', 'employment', 'unemployment', 'pop', 'families', 'dwellings', 'vacancy', 'house_price',
        'house_rent', 'wages', 'beneficiaries', 'gini', 'ogu']


def one(d):
    s, c, i = d.split(os.sep)[-3:]
    f = os.path.join(d, 'stats.csv')
    n = len(next(l for l in open(f) if l.strip()).split(';'))
    x = pd.read_csv(f, sep=';', header=None, names=columns_for('stats', n))
    p = x.price_level
    y = pd.DataFrame({'year': x.month.astype(str).str[:4].astype(int), 'real_gdp': x.gdp_level / p,
                      'employment': x.firms_total_employment, 'unemployment': x.unemployment, 'pop': x['pop'],
                      'families': x.families_count, 'dwellings': x.number_domiciles, 'vacancy': x.house_vacancy,
                      'house_price': x.house_price / p, 'house_rent': x.house_rent / p,
                      'wages': x.families_total_wages / p, 'beneficiaries': x.beneficiary_families,
                      'gini': x.gini_index, 'ogu': x.money_ogu.diff() / p})
    agg = {v: 'mean' for v in VARS}
    agg['ogu'] = 'sum'
    g = y.groupby('year').agg(agg).reset_index()
    g['scenario'], g['city'], g['seed'] = s, c, int(i)
    return g


dirs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(root, 'd*', '*', '*', 'DONE')))
Y = pd.concat(Parallel(n_jobs=8)(delayed(one)(d) for d in dirs))
Y = Y[Y.year >= 2024]
ref = Y[Y.scenario == 'd09'].set_index(['city', 'seed', 'year'])[VARS]
rng = np.random.default_rng(0)
rows = []
for s in ('d13', 'd25'):
    a = Y[Y.scenario == s].set_index(['city', 'seed', 'year'])[VARS]
    idx = a.index.intersection(ref.index)
    D, B = (a.loc[idx] - ref.loc[idx]).groupby(['city', 'year']).mean(), ref.loc[idx].groupby(['city', 'year']).mean()
    cities = D.index.get_level_values(0).unique()
    def stat(cs):
        d = pd.concat([D.loc[c] for c in cs]).groupby(level=0).sum()
        b = pd.concat([B.loc[c] for c in cs]).groupby(level=0).sum()
        r = 100 * d / b
        for v in ('unemployment', 'vacancy', 'gini'):
            r[v] = 100 * d[v] / len(cs)  # pp, mean over cities
        r['beneficiaries'] = d.beneficiaries / len(cs)  # families per run
        r['ogu_gdp'] = 100 * d.ogu / b.real_gdp / 12
        return r
    pt = stat(cities)
    boot = [stat(cities[rng.integers(0, len(cities), len(cities))]) for _ in range(500)]
    lo = pd.concat(boot).groupby(level=0).quantile(.05)
    hi = pd.concat(boot).groupby(level=0).quantile(.95)
    for v in list(VARS) + ['ogu_gdp']:
        if v == 'ogu':
            continue
        for yr in pt.index:
            rows.append(dict(scenario=s, var=v, year=yr, effect=pt.loc[yr, v], p05=lo.loc[yr, v], p95=hi.loc[yr, v]))
P = pd.DataFrame(rows)
P.to_csv(os.path.join(out, 'paths.csv'), index=False, float_format='%.5g')
pd.set_option('display.width', 250)
for s in ('d25', 'd13'):
    t = P[P.scenario == s].pivot(index='var', columns='year', values='effect')
    sig = P[P.scenario == s].assign(sig=lambda x: np.where((x.p05 > 0) | (x.p95 < 0), '*', ' ')).pivot(
        index='var', columns='year', values='sig')
    print(f'==== {s} - d09 by year (% of d09; unemployment, vacancy, gini in pp; beneficiaries = families/run; '
          f'ogu_gdp = extra OGU % of GDP; * = 90 % interval excludes 0) ====')
    print((t.map(lambda v: f'{v:7.3f}') + sig).to_string(), '\n')

# City effects: d25 - d09 over 2026-34, per run, and split-half reliability
W = Y[Y.year >= 2026]
C = W.groupby(['scenario', 'city', 'seed'])[VARS].mean()
C['ogu'] = W.groupby(['scenario', 'city', 'seed']).ogu.sum()
d = (C.loc['d25'] - C.loc['d09']).dropna()
b = C.loc['d09'].loc[d.index]
rel = 100 * d[['real_gdp', 'employment', 'pop', 'families', 'dwellings', 'house_price', 'house_rent', 'wages']] / b[
    ['real_gdp', 'employment', 'pop', 'families', 'dwellings', 'house_price', 'house_rent', 'wages']]
rel['gdp_per_real'] = d.real_gdp * 108 / d.ogu  # 108 months of extra GDP per real of extra OGU
rel['unemployment_pp'] = 100 * d.unemployment
rel['extra_ogu_gdp'] = 100 * d.ogu / (b.real_gdp * 108)
rel = rel.reset_index()
rel['half'] = np.where(rel.seed % 10 < 5, 'h1', 'h2')
ce = rel.groupby('city').mean(numeric_only=True).drop(columns='seed')
ce['pop_2025'] = Y[(Y.scenario == 'd09') & (Y.year == 2025)].groupby('city')['pop'].mean()
ce.to_csv(os.path.join(out, 'city_effects.csv'), float_format='%.5g')
hh = rel.groupby(['half', 'city']).mean(numeric_only=True)
print('==== split-half reliability of city effects (Spearman, seeds 0-4 vs 5-9, cities with both halves) ====')
for v in ['real_gdp', 'gdp_per_real', 'employment', 'unemployment_pp', 'pop', 'families', 'dwellings', 'house_price',
          'house_rent', 'wages']:
    print(f'{v:16s} {hh.loc["h1"][v].corr(hh.loc["h2"][v], method="spearman"):6.2f}')
print('\n==== city effects d25 - d09, 2026-34 means (% of d09) ====')
print(ce.sort_values('pop_2025', ascending=False).round(3).to_string())
