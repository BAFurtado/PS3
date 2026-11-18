"""mcmv2026: is the fall in the mean house price a fall in prices or a change in the mix of dwellings?
stats.csv house_price is the mean price over every dwelling (analysis/stats.py calculate_house_metrics), so new
quality-1 units lower it by composition alone. This script compares, d25 against d09, the mean price of dwellings
occupied by families outside MCMV (programme != 'mcmv'), which leaves the commissioned units out, with the mean over
all occupied dwellings. January snapshots 2027-31 (dose) and 2032-34 (after), prices divided by the run's
price_level. Pooled % of d09, 90 % bootstrap over cities.
usage (repo root, env ps3_2): python analysis/mcmv2026/price_composition.py [root]
Writes <root>/_analysis/price_composition.csv (windows and single Januaries 2025-34)."""
import datetime
import glob
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path[:0] = ['.']
from analysis.output import columns_for

root = sys.argv[1] if len(sys.argv) > 1 else 'output/mcmv2026'
out = os.path.join(root, '_analysis')
SNAPS = {'dose': ['2027', '2028', '2029', '2030', '2031'], 'after': ['2032', '2033', '2034']}
SNAPS.update({y: [y] for y in map(str, range(2025, 2035))})


def one(d):
    s, c, i = d.split(os.sep)[-3:]
    f = os.path.join(d, 'stats.csv')
    n = len(next(l for l in open(f) if l.strip()).split(';'))
    x = pd.read_csv(f, sep=';', header=None, names=columns_for('stats', n), usecols=['month', 'price_level'])
    p = dict(zip(x.month.astype(str).str[:7], x.price_level))
    fa = pd.read_parquet(os.path.join(d, 'families.parquet'), columns=['month', 'house_price', 'programme'],
                         filters=[('month', '>=', datetime.date(2025, 1, 1)), ('month', '<=', datetime.date(2034, 12, 31))])
    fa['m'] = fa.month.astype(str).str[:7]
    fa = fa[fa.m.str.endswith('-01')]
    fa['price'] = fa.house_price / fa.m.map(p)
    fa['year'] = fa.m.str[:4]
    rows = []
    for w, ys in SNAPS.items():
        f_ = fa[fa.year.isin(ys)]
        non = f_[f_.programme != 'mcmv']
        rows.append(dict(scenario=s, city=c, seed=int(i), window=w, all_sum=f_.price.sum(), all_n=len(f_),
                         non_sum=non.price.sum(), non_n=len(non)))
    return rows


dirs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(root, 'd*', '*', '*', 'DONE'))
              if f.split(os.sep)[-4] in ('d09', 'd25'))
R = pd.DataFrame([r for rr in Parallel(n_jobs=6)(delayed(one)(d) for d in dirs) for r in rr])
rng = np.random.default_rng(0)
res = []
for w in SNAPS:
    a = R[(R.scenario == 'd25') & (R.window == w)].groupby('city')[['all_sum', 'all_n', 'non_sum', 'non_n']].mean()
    b = R[(R.scenario == 'd09') & (R.window == w)].groupby('city')[['all_sum', 'all_n', 'non_sum', 'non_n']].mean()
    cities = a.index

    def stat(cs, k):
        aa, bb = a.loc[cs].sum(), b.loc[cs].sum()
        return 100 * ((aa[k + '_sum'] / aa[k + '_n']) / (bb[k + '_sum'] / bb[k + '_n']) - 1)
    for k, lab in (('all', 'occupied dwellings'), ('non', 'dwellings of families outside MCMV')):
        bs = [stat(cities[rng.integers(0, len(cities), len(cities))], k) for _ in range(500)]
        city = 100 * ((a[k + '_sum'] / a[k + '_n']) / (b[k + '_sum'] / b[k + '_n']) - 1)
        res.append(dict(window=w, measure=lab, effect=stat(cities, k), p05=np.quantile(bs, .05),
                        p95=np.quantile(bs, .95), n_neg=int((city < 0).sum())))
T = pd.DataFrame(res)
T.to_csv(os.path.join(out, 'price_composition.csv'), index=False, float_format='%.5g')
print(T[T.window.isin(['dose', 'after'])].round(3).to_string(index=False))
print(T[T.measure.str.startswith('dw')].round(3).to_string(index=False))
