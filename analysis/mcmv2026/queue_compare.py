"""MCMV queue order test: Goiânia, queue in area-code order (output/mcmv2026) against poorest first
(output/queue_test, TOTAL_TARGETING_POLICY True). Reads <root>/_analysis/hetero/{dist,stats}.csv from
hetero_extract.py.
usage (repo root, env ps3_2): python analysis/mcmv2026/queue_compare.py [LAST_SEED], seeds 0..LAST_SEED (3)
Prints (1) d09 levels by group, the share of beneficiaries above all, and (2) d25 - d09 effects by group and window,
seed means, with the seed range, for both queue orders."""
import sys

import numpy as np
import pandas as pd

CITY = 'GOIANIA'
SEEDS = list(range(int(sys.argv[1]) + 1 if len(sys.argv) > 1 else 4))
ROOTS = {'area order': 'output/mcmv2026', 'poorest first': 'output/queue_test'}
COLS = ['families', 'persons', 'members_per_family', 'renter_share', 'benef_share', 'median_pi_pc', 'zero_wage_share',
        'rent_burden_over30', 'works_other_mun']
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 200)


def load(kind):
    out = []
    for name, root in ROOTS.items():
        d = pd.read_csv(f'{root}/_analysis/hetero/{kind}.csv')
        d = d[(d.city == CITY) & d.seed.isin(SEEDS) & d.scenario.isin(['d09', 'd25'])]
        out.append(d.assign(queue=name))
    return pd.concat(out)


D = load('dist')
keys = ['queue', 'window', 'group', 'value']
print('== d09 levels, after 2030, seed means')
lv = D[(D.scenario == 'd09') & (D.window == 'after')].groupby(keys)[COLS].mean()
print(lv.unstack('queue').swaplevel(axis=1).sort_index(axis=1).round(4).to_string())

w = D.pivot_table(index=['queue', 'window', 'group', 'value', 'seed'], columns='scenario', values=COLS)
eff = pd.concat({c: w[(c, 'd25')] - w[(c, 'd09')] for c in COLS}, axis=1)
rel = ['families', 'persons', 'members_per_family', 'median_pi_pc']
for c in rel:
    eff[c] = 100 * eff[c] / w[(c, 'd09')]
for c in set(COLS) - set(rel):
    eff[c] = 100 * eff[c]
print('\n== d25 - d09 (% for families, persons, members, median PI; pp otherwise), seed mean [min, max]')
g = eff.groupby(level=['window', 'group', 'value', 'queue'])
summary = g.mean().round(3).astype(str) + ' [' + g.min().round(2).astype(str) + ', ' + g.max().round(2).astype(str) + ']'
print(summary[['families', 'members_per_family', 'renter_share', 'benef_share', 'median_pi_pc']].to_string())

S = load('stats')
num = [c for c in S.columns if c not in ('scenario', 'city', 'seed', 'window', 'queue') and pd.api.types.is_numeric_dtype(S[c])]
ws = S.pivot_table(index=['queue', 'window', 'seed'], columns='scenario', values=num)
ratio = ['pop', 'families_count', 'beneficiary_families', 'firms_total_employment', 'house_price', 'gdp_level']
rates = ['house_vacancy', 'unemployment']
es = pd.concat({**{c: 100 * (ws[(c, 'd25')] / ws[(c, 'd09')] - 1) for c in ratio},
                **{c + '_pp': 100 * (ws[(c, 'd25')] - ws[(c, 'd09')]) for c in rates}}, axis=1)
print('\n== metro totals, d25 / d09 - 1 (%), rates in pp, seed means')
print(es.groupby(level=['window', 'queue']).mean().round(3).to_string())


print('\n== after 2031-34: area order, poorest first, difference (poorest first - area order) and its t, seed means')
rows = {}
for (grp, val) in [('all', 'all'), ('tenure', 'owner'), ('tenure', 'renter'), ('quint', 'q1'), ('quint', 'q2'),
                   ('zone', 'core'), ('zone', 'periphery')]:
    for c in ('families', 'members_per_family', 'renter_share', 'benef_share'):
        x = eff.xs(('after', grp, val), level=['window', 'group', 'value'])[c]
        a, b = x.xs('area order', level='queue'), x.xs('poorest first', level='queue')
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        rows[(grp, val, c)] = dict(area=a.mean(), poorest=b.mean(), diff=b.mean() - a.mean(), t=(b.mean() - a.mean()) / se)
e2 = es.xs('after', level='window')
for c in e2.columns:
    a, b = e2.loc['area order', c], e2.loc['poorest first', c]
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    rows[('metro', 'total', c)] = dict(area=a.mean(), poorest=b.mean(), diff=b.mean() - a.mean(), t=(b.mean() - a.mean()) / se)
print(pd.DataFrame(rows).T.round(3).to_string())
