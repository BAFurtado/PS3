"""mcmv2026, one pass over the parquet: (1) employees by January, construction and all other sectors;
(2) families by size (1, 2, 3+) and wage status (zero wage or not), January 2027-2034.
usage (repo root, env ps3_2): python analysis/mcmv2026/yearly_extract.py [root] [cpus]
Writes <root>/_analysis/jobs_yearly.csv and families_size.csv."""
import glob, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
root = sys.argv[1] if len(sys.argv) > 1 else 'output/mcmv2026'
cpus = int(sys.argv[2]) if len(sys.argv) > 2 else 6
out = os.path.join(root, '_analysis')


def one(d):
    s, city, i = d.split(os.sep)[-3:]
    fi = pd.read_parquet(os.path.join(d, 'firms.parquet'), columns=['month', 'sector', 'number_employees'])
    fi['year'] = pd.to_datetime(fi.month.astype(str)).dt.year
    fi = fi[fi.year >= 2024]
    fi['group'] = np.where(fi.sector == 'Construction', 'construction', 'other')
    j = fi.groupby(['year', 'group']).number_employees.sum().unstack().reset_index()
    fa = pd.read_parquet(os.path.join(d, 'families.parquet'), columns=['month', 'num_members', 'total_wage'])
    fa['year'] = pd.to_datetime(fa.month.astype(str)).dt.year
    fa = fa[fa.year >= 2027]
    fa['size'] = np.where(fa.num_members <= 1, '1', np.where(fa.num_members == 2, '2', '3+'))
    fa['wage'] = np.where(fa.total_wage > 0, 'wage', 'zero')
    f = fa.groupby(['year', 'size', 'wage']).size().rename('families').reset_index()
    for x in (j, f):
        x['scenario'], x['city'], x['seed'] = s, city, int(i)
    return j, f


dirs = sorted(os.path.dirname(x) for x in glob.glob(os.path.join(root, 'd*', '*', '*', 'DONE')))
res = Parallel(n_jobs=cpus, verbose=2)(delayed(one)(d) for d in dirs)
pd.concat([r[0] for r in res]).to_csv(os.path.join(out, 'jobs_yearly.csv'), index=False)
pd.concat([r[1] for r in res]).to_csv(os.path.join(out, 'families_size.csv'), index=False)
print(len(res), 'runs')
