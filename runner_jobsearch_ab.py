"""
ON_THE_JOB_SEARCH A/B: 'off' (only job seekers without a job meet the posts) against 'pme' (employees meet them too
with the PME 2010 search rate of their education level (flow search) and move to an offer worth more than the current job),
paired by seed. Settings, saved data and indicators as runner_education_ab.py (ENMU base with LABOUR_FLOWS
'education', 2010-2021, six areas, unemployment by level, AP correlations, commute bands), plus:
    j2j1-j2j4, j2j    job-to-job moves over employees per month, 2011-2021, by level and all (job_moves.csv), against
                      the PME 2010 rates (input/labour_flows_level_2010.csv, flow j2j)
    seek              employed job seekers over employees per month, all levels
    wgap3, wgap4      mean log last wage of employees at level 3 (4) minus level 1, January 2017-2021
    minutes           employees' mean one-way commute minutes, January 2021

    python runner_jobsearch_ab.py lote CPUS [cities=enmu|A,B] [runs=N] [part=K/N]
    python runner_jobsearch_ab.py comparar

Output: output/jobsearch_ab/<value>/<CITY>/<i>/. Local conf/params.py and conf/run.py are ignored.
"""
import os

import numpy as np
import pandas as pd

import runner_education_ab as X

A = X.A
A.OUTPUT = 'output/jobsearch_ab'
A.KEY = 'ON_THE_JOB_SEARCH'
A.LEVELS = ['off', 'pme']
_moves = {}


def moves(path):
    if path not in _moves:
        f = os.path.join(path, 'job_moves.csv')
        if not os.path.exists(f):
            _moves[path] = None
        else:
            m = pd.read_csv(f, sep=';', header=None, names=['month', 'level', 'employees', 'seekers', 'moves'])
            m['month'] = pd.to_datetime(m.month).dt.date
            _moves[path] = m[(m.month >= A.V.WINDOW[0]) & (m.month <= A.V.WINDOW[1])]
    return _moves[path]


def j2j(k=None, column='moves'):
    def f(path):
        m = moves(path)
        if m is None:
            return np.nan
        if k is not None:
            m = m[m.level == k]
        return float(m[column].sum() / m.employees.sum())
    return f


def wgap(k):
    def f(path):
        a = pd.read_parquet(os.path.join(path, 'agents.parquet'), columns=['month', 'age', 'qualification', 'firm_id',
                                                                           'last_wage'])
        a['month'] = pd.to_datetime(a.month).dt.date
        a = a[a.month.isin(X.MONTHS) & (a.age > 16) & (a.age < 70) & a.firm_id.notna() & (a.last_wage > 0)]
        level = pd.cut(a.qualification, [-1, 3, 8, 11, 99], labels=[1, 2, 3, 4]).astype(int)
        lw = np.log(a.last_wage).groupby(level).mean()
        return float(lw.get(k, np.nan) - lw.get(1, np.nan))
    return f


def minutes(path):
    a = X.agents(path)
    a = a[(a.month == X.LAST) & a.emp & (a.distance > 0)]
    return float(a.distance.mean())


A.METRICS = X.A.METRICS + [('j2j%d' % k, j2j(k)) for k in range(1, 5)] + \
    [('j2j', j2j()), ('seek', j2j(column='seekers')), ('wgap3', wgap(3)), ('wgap4', wgap(4)), ('minutes', minutes)]

if __name__ == '__main__':
    A.main()
