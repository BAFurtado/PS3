"""
HIRING_SAMPLE_SIZE grid: 20 (the default) against 10, 40 and 80, paired by seed, on the settings of runner_jobsearch_ab
'off' (ENMU base, runner_enmu.COMMON with LABOUR_FLOWS 'education', TRANSPORT_WEIGHTS trips, base network all through,
ON_THE_JOB_SEARCH off), the six ENMU areas, 2010-2021, seeds SEED + i. Saved: stats, regional, houses and agents parquet
every January, sales.csv. The 20 side has the settings of output/jobsearch_ab/off, whose runs can be linked in.

Rule fixed before the runs: the value is the one with the smallest mean over the six areas of kl_private + kl_transit
(employees' commute bands in January 2021 against Census 2022 by mode, runner_education_ab), each a mean over seeds.
Rates and indicators against 20 as runner_rent_ap comparar (2 seed sd, the six-area width, unemployment above 0.30),
and the run's wall time in minutes.

    python runner_sample_grid.py lote CPUS [cities=enmu|A,B] [runs=N] [part=K/N]
    python runner_sample_grid.py comparar       paired deltas of each value against 20, then the grid table

Output: output/sample_grid/<value>/<CITY>/<i>/. Local conf/params.py and conf/run.py are ignored.
"""
import glob
import os
import sys

import pandas as pd

import runner_education_ab as X

A = X.A
A.OUTPUT = 'output/sample_grid'
A.KEY = 'HIRING_SAMPLE_SIZE'
A.LEVELS = ['20', '10', '40', '80']
_build = A.build_jobs


def build_jobs(root, items, total_days=None):
    """runner_rent_ap jobs with the sample size as an integer"""
    jobs = _build(root, items, total_days)
    for job in jobs:
        job['params'][A.KEY] = int(job['params'][A.KEY])
    return jobs


def wall_minutes(path):
    return (os.path.getmtime(os.path.join(path, 'DONE')) - os.path.getmtime(os.path.join(path, 'conf.json'))) / 60


A.build_jobs = build_jobs
A.METRICS = X.A.METRICS + [('wall_minutes', wall_minutes)]
SHOWN = ['kl_private', 'kl_transit', 'w30_private', 'w30_transit', 'u_gap', 'wall_minutes']


def grid():
    """Mean over seeds per area and value, the observed bands, and the value the rule picks"""
    rows = []
    for level in A.LEVELS:
        for done in sorted(glob.glob(os.path.join(A.OUTPUT, level, '*', '*', 'DONE'))):
            p = os.path.dirname(done)
            city, i = p.split(os.sep)[-2:]
            r = {'k': int(level), 'city': city, 'seed': int(i)}
            for name in SHOWN:
                r[name] = dict(A.METRICS)[name](p)
            rows.append(r)
    t = pd.DataFrame(rows)
    t['kl'] = t.kl_private + t.kl_transit
    pd.set_option('display.width', 250)
    g = t.groupby(['city', 'k'])
    out = g.mean(numeric_only=True).drop(columns='seed').join(g.size().rename('n'))
    print('\nGrid, mean over seeds, January 2021 (kl: commute bands against Census 2022, lower is better)')
    print(out.round(3).to_string())
    obs = {c: (round(float(X.observed_bands(c, 'private')[0]), 3), round(float(X.observed_bands(c, 'transit')[0]), 3))
           for c in sorted(t.city.unique())}
    print('\nCensus 2022 within 30 min (private, transit):', obs)
    full = out.n.groupby('k').size() == t.city.nunique()
    best = out.kl.groupby('k').mean()[full]
    print('\nMean kl over areas by k (values with every area):', best.round(4).to_dict(), '-> k', best.idxmin())
    t.to_csv(os.path.join(A.OUTPUT, 'grid.csv'), sep=';', index=False)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'comparar':
        A.main()
        grid()
    else:
        A.main()
