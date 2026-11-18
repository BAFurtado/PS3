"""mcmv2026: why cities differ. d25 - d09 city effects 2026-34 (city_effects.csv, paths.py + shortfall.py; % of d09,
unemployment in pp) against city traits in d09 2025 (stats.csv, mean over seeds): log population, persons per
family, vacancy, unemployment, log real GDP per dwelling, shortfall below the Census path (short_2025), and the
dwellings effect itself. Each relation is computed in both seed halves (0-4, 5-9): a trait counts only if the sign
holds in both. Then OLS of families and population effects on the dwellings effect plus one trait, HC3, 26 cities.
usage (repo root, env ps3_2): python analysis/mcmv2026/city_traits.py [root]
Writes <root>/_analysis/city_traits.csv."""
import glob, os, sys
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
sys.path[:0] = ['.']
from analysis.output import columns_for

root = sys.argv[1] if len(sys.argv) > 1 else 'output/mcmv2026'
out = os.path.join(root, '_analysis')
COLS = ['month', 'pop', 'price_level', 'gdp_level', 'unemployment', 'house_vacancy', 'number_domiciles',
        'families_count']
VARS = ['real_gdp', 'employment', 'pop', 'families', 'dwellings', 'house_price', 'unemployment']


def read(f, cols):
    n = len(next(l for l in open(f) if l.strip()).split(';'))
    x = pd.read_csv(f, sep=';', header=None, names=columns_for('stats', n), usecols=cols)
    x['month'] = x.month.astype(str)
    return x


def run_vals(d):
    x = read(os.path.join(d, 'stats.csv'), COLS)
    w = (x.month >= '2026-01-01') & (x.month <= '2034-12-31')
    x['real_gdp'] = x.gdp_level / x.price_level
    pre = x[(x.month >= '2025-01-01') & (x.month <= '2025-12-31')]
    W = x[w]
    return dict(pop=W['pop'].mean(), families=W.families_count.mean(), dwellings=W.number_domiciles.mean(),
                unemployment=W.unemployment.mean(), real_gdp=W.real_gdp.mean()), dict(
        log_pop=np.log(pre['pop'].mean()), persons_per_family=(pre['pop'] / pre.families_count).mean(),
        vacancy=100 * pre.house_vacancy.mean(), unemp_2025=100 * pre.unemployment.mean(),
        log_gdp_per_dwelling=np.log((pre.real_gdp / pre.number_domiciles).mean()))


rows = []
for done in sorted(glob.glob(os.path.join(root, 'd09', '*', '*', 'DONE'))):
    d = os.path.dirname(done)
    city, i = d.split(os.sep)[-2:]
    e = d.replace(os.sep + 'd09' + os.sep, os.sep + 'd25' + os.sep)
    if not os.path.exists(os.path.join(e, 'DONE')):
        continue
    (b, tr), (a, _) = run_vals(d), run_vals(e)
    r = dict(city=city, seed=int(i), **tr)
    for k in ['pop', 'families', 'dwellings', 'real_gdp']:
        r['e_' + k] = 100 * (a[k] / b[k] - 1)
    r['e_unemployment_pp'] = 100 * (a['unemployment'] - b['unemployment'])
    rows.append(r)
R = pd.DataFrame(rows)
R['half'] = np.where(R.seed % 10 < 5, 'h1', 'h2')
sh = pd.read_csv(os.path.join(out, 'city_effects.csv')).set_index('city')[['short_2025']]
TRAITS = ['log_pop', 'persons_per_family', 'vacancy', 'unemp_2025', 'log_gdp_per_dwelling', 'short_2025']
EFF = ['e_dwellings', 'e_families', 'e_pop', 'e_unemployment_pp', 'e_real_gdp']
C = R.groupby('city').mean(numeric_only=True).join(sh)
H = {h: R[R.half == h].groupby('city').mean(numeric_only=True).join(sh) for h in ('h1', 'h2')}
res = []
for y in EFF:
    for x in TRAITS + (['e_dwellings'] if y != 'e_dwellings' else []):
        # trait from the full sample (traits are pre-2026, effects are noisy); effect from each half
        s1 = stats.spearmanr(C.loc[H['h1'].index, x], H['h1'][y])[0]
        s2 = stats.spearmanr(C.loc[H['h2'].index, x], H['h2'][y])[0]
        sa, pa = stats.spearmanr(C[x], C[y])
        res.append(dict(effect=y, trait=x, rho_all=sa, p_all=pa, rho_h1=s1, rho_h2=s2,
                        both_halves=np.sign(s1) == np.sign(s2) and min(abs(s1), abs(s2)) > .3))
T = pd.DataFrame(res)
T.to_csv(os.path.join(out, 'city_traits.csv'), index=False)
pd.set_option('display.width', 200)
print('split-half Spearman of the effects themselves:')
for y in EFF:
    both = H['h1'].index.intersection(H['h2'].index)
    print(f'  {y:20s} {stats.spearmanr(H["h1"].loc[both, y], H["h2"].loc[both, y])[0]:+.2f}')
print('\nSpearman, trait (2025, d09) against city effect (2026-34); both_halves = same sign and |rho| > .3 in each half')
print(T.round(2).to_string(index=False))
print(f'\nOLS across {len(C)} cities, HC3, effects in % (unemployment pp)')
Z = C.copy()
for y in ['e_families', 'e_pop', 'e_unemployment_pp']:
    for x in ['', ' + short_2025', ' + persons_per_family', ' + vacancy', ' + log_gdp_per_dwelling']:
        f = smf.ols(f'{y} ~ e_dwellings{x}', Z).fit(cov_type='HC3')
        print(f'  {y:18s} ~ e_dwellings{x:24s} R2 {f.rsquared:.2f}  ' +
              '  '.join(f'{k} {f.params[k]:+.3f} (p {f.pvalues[k]:.3f})' for k in f.params.index if k != 'Intercept'))
print('\ncity table (means over seeds):')
print(C[['e_dwellings', 'e_families', 'e_pop', 'e_unemployment_pp'] + TRAITS].sort_values('e_pop').round(2).to_string())
