"""Household and population growth 2010-2022 by municipality, d09 (regional.csv) against the Census (Sidra 156, 4712).
usage (repo root): python analysis/mcmv2026/households_census_mun.py [sidra_dir]. Prints the SI Table B.2 rows."""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from joblib import Parallel, delayed
sys.path[:0] = ['.']
from analysis.output import columns_for
S = (sys.argv[1] if len(sys.argv) > 1 else 'output/mcmv2026/_analysis/sidra') + '/'
acp = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';'); acp['cod_mun'] = acp.cod_mun.astype(int)
def c(f, hh, res):
    r = pd.DataFrame(json.load(open(S + f))[1:]); r['V'] = pd.to_numeric(r.V, errors='coerce'); r['m'] = r.D1C.astype(int)
    return r.pivot(index='m', columns='D2C', values='V').rename(columns={hh: 'hh', res: 'res'})
C10, C22 = c('c2010.json', '2048', '134'), c('c2022.json', '381', '382')
def one(d):
    f = os.path.join(d, 'regional.csv')
    x = pd.read_csv(f, sep=';', header=None, names=columns_for('regional', 46), usecols=[0, 1, 3, 42])
    x['year'] = x.month.str[:4].astype(int)
    g = x[x.year.isin([2010, 2022])].groupby(['mun_id', 'year'])[['pop', 'families_count']].mean().unstack()
    g['city'] = d.split(os.sep)[-2]
    return g
dirs = sorted(os.path.dirname(f) for f in glob.glob('output/mcmv2026/d09/*/*/DONE'))
R = pd.concat(Parallel(n_jobs=8)(delayed(one)(d) for d in dirs))
R.columns = ['_'.join(map(str, k)) if isinstance(k, tuple) and k[1] != '' else k[0] for k in R.columns]
M = R.groupby(['city', R.index]).mean()
M.index.names = ['city', 'mun']
M = M.reset_index().set_index('mun')
M['m_pop'] = M.pop_2022 / M.pop_2010 - 1
M['m_hh'] = M.families_count_2022 / M.families_count_2010 - 1
M['c_pop'] = C22.res / C10.res - 1
M['c_hh'] = C22.hh / C10.hh - 1
M['m_pph'] = (M.pop_2022 / M.families_count_2022) / (M.pop_2010 / M.families_count_2010) - 1
M['c_pph'] = (C22.res / C22.hh) / (C10.res / C10.hh) - 1
M = M.dropna(subset=['c_hh'])
M['big'] = M.pop_2010 >= 20  # at least 2,000 residents in the Census (1 % sample)
print('municipalities', len(M), 'with >=20 agents', M.big.sum())
B = M[M.big]
for v in ['pop', 'hh', 'pph']:
    r = spearmanr(B['m_' + v], B['c_' + v]).statistic
    w = B.groupby('city').apply(lambda g: spearmanr(g['m_' + v], g['c_' + v]).statistic if len(g) >= 5 else np.nan)
    print(f'{v}: pooled rank {r:.2f}, within-area median rank {w.median():.2f} over {w.notna().sum()} areas; '
          f'median model {B["m_"+v].median():.3f} census {B["c_"+v].median():.3f}')
B['gap_hh'] = B.m_hh - B.c_hh
print(B.gap_hh.describe().round(3))
print('share of muns with model household growth below census', (B.gap_hh < 0).mean().round(2))
print('weighted (by 2010 pop) mean gap hh', np.average(B.gap_hh, weights=B.pop_2010).round(3))
