"""Residents per household, 2010 and 2022: Census (Sidra 156 for 2010, 4712 for 2022, summed over each area's
municipalities in input/ACPs_MUN_CODES.csv) against the d09 reference run (pop / families_count, yearly mean of
stats.csv, mean over seeds).
usage (repo root): python analysis/mcmv2026/households_census.py [sidra_dir] [root]
<sidra_dir> holds c2010.json (t/156/n6/all/v/2048,134/p/2010) and c2022.json (t/4712/n6/all/v/381,382/p/2022).
Writes <root>/_analysis/households_census.csv."""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path[:0] = ['.']
from analysis.output import columns_for

sidra = sys.argv[1] if len(sys.argv) > 1 else 'output/mcmv2026/_analysis/sidra'
root = sys.argv[2] if len(sys.argv) > 2 else 'output/mcmv2026'


def census(f, hh, res):
    r = pd.DataFrame(json.load(open(os.path.join(sidra, f)))[1:])
    r = r[r.D2C.isin([hh, res])]
    r['V'] = pd.to_numeric(r.V, errors='coerce')
    return r.pivot(index='D1C', columns='D2C', values='V').rename(columns={hh: 'hh', res: 'res'})


acp = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
acp['cod_mun'] = acp.cod_mun.astype(str)
C = {}
for yr, f, hh, res in [(2010, 'c2010.json', '2048', '134'), (2022, 'c2022.json', '381', '382')]:
    c = census(f, hh, res).join(acp.set_index('cod_mun').ACPs, how='inner').groupby('ACPs')[['hh', 'res']].sum()
    C[yr] = c.res / c.hh


def one(d):
    f = os.path.join(d, 'stats.csv')
    n = len(next(l for l in open(f) if l.strip()).split(';'))
    x = pd.read_csv(f, sep=';', header=None, names=columns_for('stats', n))
    x['year'] = x.month.astype(str).str[:4].astype(int)
    g = x[x.year.isin([2010, 2022])].groupby('year')[['pop', 'families_count']].mean()
    return g['pop'] / g.families_count


rows = []
for d in sorted(os.path.dirname(f) for f in glob.glob(os.path.join(root, 'd09', '*', '*', 'DONE'))):
    s = one(d)
    rows.append({'city': d.split(os.sep)[-2], 'seed': int(d.split(os.sep)[-1]), 'm2010': s[2010], 'm2022': s[2022]})
M = pd.DataFrame(rows).groupby('city')[['m2010', 'm2022']].mean()
T = M.join(pd.DataFrame({'c2010': C[2010], 'c2022': C[2022]}), how='inner')
T['m_change'] = T.m2022 / T.m2010 - 1
T['c_change'] = T.c2022 / T.c2010 - 1
T.round(4).to_csv(os.path.join(root, '_analysis', 'households_census.csv'))
print(T.round(3).sort_values('c_change').to_string())
print('\nmedian change, model %.3f census %.3f' % (T.m_change.median(), T.c_change.median()))
print('areas where model falls: %d of %d' % ((T.m_change < 0).sum(), len(T)))
print('model change / census change, median %.2f' % (T.m_change / T.c_change).median())
r = spearmanr(T.m_change, T.c_change)
print('rank corr of change across areas %.2f (p %.3f)' % (r.statistic, r.pvalue))
r = spearmanr(T.m2022, T.c2022)
print('rank corr of 2022 level %.2f, 2010 level %.2f' % (r.statistic, spearmanr(T.m2010, T.c2010).statistic))
