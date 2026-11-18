"""Excess rent burden (FJP deficit component) in mcmv2026, d25 against d09.

A family is in excess burden when it rents, its permanent income is above zero and at most three 2010 minimum wages
(3 x R$510 = 1.53 model units, one unit = R$1,000 of 2010), and its rent exceeds 30 % of that income.
Small families have one or two members. January counts from families.parquet, windows as in hetero_analysis: dose = Jan 2027-2030, after = Jan 2032-2034.
Pooling as in screen.py: counts in % of the reference summed over areas, shares as the mean over areas of the
seed-mean difference in percentage points; 90 % interval from 400 bootstrap draws over areas.
Output: output/mcmv2026/_analysis/excess_burden.csv and excess_burden_city.csv.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

root = sys.argv[1] if len(sys.argv) > 1 else 'output/mcmv2026'
out = os.path.join(root, '_analysis')
THRESHOLD = 3 * 0.51
WINDOWS = {'dose': (2027, 2030), 'after': (2032, 2034)}
COUNTS = ['families', 'renters', 'renters_zero_income', 'renters_low', 'burden_any', 'excess_burden',
          'families_small', 'renters_low_small', 'excess_burden_small', 'excess_burden_large']


def one(d):
    s, c, i = d.split(os.sep)[-3:]
    x = pd.read_parquet(os.path.join(d, 'families.parquet'),
                        columns=['month', 'house_rent', 'permanent_income', 'programme', 'num_members'])
    x['year'] = pd.to_datetime(x.month).dt.year
    x = x[x.year >= 2027]
    renter = x.house_rent.notna()
    pi = x.permanent_income
    low = renter & (pi > 0) & (pi <= THRESHOLD)
    burden = renter & (pi > 0) & (x.house_rent > 0.3 * pi)
    x = x.assign(families=1, renters=renter, renters_zero_income=renter & (pi <= 0), renters_low=low,
                 burden_any=burden, excess_burden=low & burden)
    small = x.num_members <= 2
    x = x.assign(families_small=small, renters_low_small=x.renters_low & small,
                 excess_burden_small=x.excess_burden & small, excess_burden_large=x.excess_burden & ~small)
    cols = COUNTS
    y = x.groupby('year')[cols].sum()
    rows = []
    for w, (a, b) in WINDOWS.items():
        m = y.loc[a:b].mean()
        m['excess_share'] = m.excess_burden / m.families
        m['excess_share_renters'] = m.excess_burden / m.renters
        m['excess_share_low_renters'] = m.excess_burden / m.renters_low
        m['scenario'], m['city'], m['seed'], m['window'] = s, c, int(i), w
        rows.append(m)
    return rows


dirs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(root, 'd*', '*', '*', 'DONE'))
              if f.split(os.sep)[-4] in ('d09', 'd25'))
R = pd.DataFrame([r for rr in Parallel(n_jobs=6)(delayed(one)(d) for d in dirs) for r in rr])
SHARES = ['excess_share', 'excess_share_renters', 'excess_share_low_renters']
rng = np.random.default_rng(0)
res, city_rows = [], []
for w in WINDOWS:
    a = R[(R.scenario == 'd25') & (R.window == w)].set_index(['city', 'seed'])[COUNTS + SHARES]
    r = R[(R.scenario == 'd09') & (R.window == w)].set_index(['city', 'seed'])[COUNTS + SHARES]
    idx = a.index.intersection(r.index)
    a, r = a.loc[idx], r.loc[idx]
    D = a - r
    cities = D.index.get_level_values(0).unique()
    for v in COUNTS + SHARES:
        dc, rc = D[v].groupby(level=0).mean(), r[v].groupby(level=0).mean()

        def stat(cs):
            if v in SHARES:
                return 100 * dc.loc[cs].mean()
            return 100 * dc.loc[cs].sum() / rc.loc[cs].sum()
        bs = np.array([stat(cities[rng.integers(0, len(cities), len(cities))]) for _ in range(400)])
        res.append(dict(window=w, var=v, kind='pp' if v in SHARES else '% of ref', effect=stat(cities),
                        p05=np.quantile(bs, .05), p95=np.quantile(bs, .95), ref_pooled=rc.sum() if v in COUNTS
                        else rc.mean(), delta_per_run=dc.mean(), n_runs=len(idx)))
        for c in cities:
            city_rows.append(dict(window=w, var=v, city=c, ref=rc[c], delta=dc[c]))
S = pd.DataFrame(res)
S.to_csv(os.path.join(out, 'excess_burden.csv'), index=False, float_format='%.6g')
pd.DataFrame(city_rows).to_csv(os.path.join(out, 'excess_burden_city.csv'), index=False, float_format='%.6g')
pd.set_option('display.width', 200)
print(S.round(4).to_string(index=False))
