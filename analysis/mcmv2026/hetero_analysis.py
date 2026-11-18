"""mcmv2026 heterogeneity: d25 - d09 and d13 - d09 at matched city and seed, by window (dose 2026-30, after 2031-34),
from <root>/_analysis/hetero/{dist,mun,sector}.csv (hetero_extract.py).
usage (repo root, env ps3_2): python analysis/mcmv2026/hetero_analysis.py [root]
Effect = mean over cities of the mean over seeds of the difference; '%' rows divide by the mean of d09, 'pp' rows are
x 100. 90 % bootstrap over cities (1,000 draws); * marks an interval that excludes 0. Writes hetero_effects.csv."""
import os
import sys

import numpy as np
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else 'output/mcmv2026'
H = os.path.join(root, '_analysis', 'hetero')
REF = 'd09'
PCT = ['families', 'persons', 'members_per_family', 'mean_pi_pc', 'median_pi_pc', 'mean_wage_pc', 'savings_pc',
       'rent_pc_renters', 'owner_value_pc', 'commute_dist']
PP = ['renter_share', 'zero_wage_share', 'benef_share', 'rent_burden_median', 'rent_burden_over30',
      'employment_rate', 'works_other_mun', 'gini_pi_pc', 'gini_wage_pc', 'bottom40_share', 'top10_share']
DRAWS = {}


def draws(n):
    """The point estimate (None) and 1,000 bootstrap draws of n cities, the same for every call with n cities."""
    if n not in DRAWS:
        rng = np.random.default_rng(0)
        DRAWS[n] = [None] + [rng.integers(0, n, n) for _ in range(1000)]
    return DRAWS[n]


def effect(t, keys, cols, scen):
    """t: one row per (scenario, city, seed, *keys). Returns effect, p05, p95 per keys x col."""
    a = t[t.scenario == scen].set_index(['city', 'seed'] + keys)[cols]
    b = t[t.scenario == REF].set_index(['city', 'seed'] + keys)[cols]
    idx = a.index.intersection(b.index)
    d, r = a.loc[idx] - b.loc[idx], b.loc[idx]
    dc = d.groupby(['city'] + keys).mean()
    rc = r.groupby(['city'] + keys).mean()
    cities = dc.index.get_level_values('city').unique()
    res = []
    for k in draws(len(cities)):
        cs = cities if k is None else cities[k]
        dd = pd.concat([dc.xs(c, level='city') for c in cs]).groupby(level=list(range(len(keys)))).mean()
        rr = pd.concat([rc.xs(c, level='city') for c in cs]).groupby(level=list(range(len(keys)))).mean()
        e = pd.DataFrame({c: 100 * dd[c] / rr[c] if c in PCT or c.endswith('_pct') else 100 * dd[c] for c in cols})
        res.append(e)
    pt = res[0]
    bs = pd.concat(res[1:], keys=range(len(res) - 1))
    lo = bs.groupby(level=list(range(1, len(keys) + 1))).quantile(.05)
    hi = bs.groupby(level=list(range(1, len(keys) + 1))).quantile(.95)
    star = ((lo > 0) | (hi < 0)).replace({True: '*', False: ' '})
    return pt, lo, hi, star


def show(title, pt, star):
    print(f'==== {title} ====')
    print((pt.map(lambda v: f'{v:8.3f}') + star).to_string(), '\n')


pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 40)
out = []

# 1. Families and people, by group
dist = pd.read_csv(os.path.join(H, 'dist.csv'))
cols = PCT + PP
for scen in ('d25', 'd13'):
    pt, lo, hi, star = effect(dist, ['window', 'group', 'value'], cols, scen)
    long = pt.stack().rename('effect').to_frame().join(lo.stack().rename('p05')).join(hi.stack().rename('p95'))
    long['scenario'] = scen
    out.append(long.reset_index().rename(columns={'level_3': 'var'}))
    if scen == 'd25':
        for w in ('dose', 'after'):
            show(f'{scen} - {REF}, families by group, {w} (% for {", ".join(PCT[:3])}..., pp for shares)',
                 pt.xs(w, level='window').T, star.xs(w, level='window').T)

# 2. Core and periphery: municipal rows summed per run
mun = pd.read_csv(os.path.join(H, 'mun.csv'))
mun['zone'] = np.where(mun.core, 'core', 'periphery')
mun['pop_growth'] = mun.pop_end - mun.pop_2025
mun['families_growth'] = mun.families_end - mun.families_2025
z = mun.groupby(['scenario', 'city', 'seed', 'window', 'zone'])[
    ['mcmv_units_bought_sum', 'pop', 'families_count', 'pop_growth', 'families_growth', 'jobs', 'construction_jobs',
     'licenses_sum']].sum().reset_index()
zt = mun.groupby(['scenario', 'city', 'seed', 'window'])[['mcmv_units_bought_sum', 'pop_growth',
                                                          'families_growth']].sum()
z = z.join(zt, on=['scenario', 'city', 'seed', 'window'], rsuffix='_metro')
for c in ('pop', 'families_count', 'jobs', 'construction_jobs'):
    z[c + '_pct'] = z[c]
for scen in ('d25', 'd13'):
    a = z[z.scenario == scen].set_index(['city', 'seed', 'window', 'zone'])
    b = z[z.scenario == REF].set_index(['city', 'seed', 'window', 'zone'])
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    d = (a.select_dtypes('number') - b.select_dtypes('number'))
    tab = d.groupby(['city', 'window', 'zone'])[['mcmv_units_bought_sum', 'pop_growth', 'families_growth']].mean(
    ).groupby(['window', 'zone']).sum()
    share = tab / tab.groupby('window').transform('sum')
    print(f'==== {scen} - {REF}: share of the extra units, residents and families by zone ====')
    print(share.round(3).to_string())
    bp = b.groupby(['city', 'window', 'zone'])['pop'].mean().groupby(['window', 'zone']).sum()
    print('d09 share of residents (pop):', (bp / bp.groupby('window').transform('sum')).round(3).to_dict(), '\n')
    pt, lo, hi, star = effect(z, ['window', 'zone'], ['pop_pct', 'families_count_pct', 'jobs_pct',
                                                      'construction_jobs_pct'], scen)
    show(f'{scen} - {REF} by zone, % of d09', pt, star)

# 3. Jobs by sector, metro
sec = pd.read_csv(os.path.join(H, 'sector.csv'))
sec = sec[sec.zone == 'all']
sec['employees_pct'] = sec.employees
sec['wages_pct'] = sec.wages
sec['revenue_pct'] = sec.revenue
for scen in ('d25',):
    pt, lo, hi, star = effect(sec, ['window', 'sector'], ['employees_pct', 'wages_pct', 'revenue_pct'], scen)
    show(f'{scen} - {REF} by sector, metro, % of d09', pt.unstack('window'), star.unstack('window'))
    ref = sec[sec.scenario == REF].groupby(['city', 'window', 'sector']).employees.mean().groupby(['window', 'sector']).mean()
    a = sec[sec.scenario == scen].groupby(['city', 'window', 'sector']).employees.mean().groupby(['window', 'sector']).mean()
    print('extra jobs per run by sector (d25 - d09):')
    print((a - ref).unstack('window').round(1).to_string(), '\n')

pd.concat(out).to_csv(os.path.join(root, '_analysis', 'hetero_effects.csv'), index=False, float_format='%.5g')
