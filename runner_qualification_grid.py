"""
CB_QUALIFICATION grid under QUALIFICATION_SCORE 'log': alpha = 0, 0.1, 0.2, 0.35, 0.5, paired by seed, on the ENMU base
(runner_enmu.COMMON, LABOUR_FLOWS 'education', TRANSPORT_WEIGHTS trips, base network all through), 2010-2015, Goiania
and Fortaleza, seeds SEED + i. Saved: stats, regional, houses, agents and firms parquet every January, sales.csv.

alpha is the value with the smallest mean sector distance: over the Januaries 2013-2015, the total variation distance
between the education mix (Census levels 1-4, world/own_account.level) of each sector's employees in the model and in
the Census 2010 (employees aged 17-69 of the run's municipalities, input/own_account_2010.csv), weighted by the
model's sector employment. Other indicators as runner_education_ab.py, plus:
    edu4_<sector>   share of level 4 among the employees of Government, Financial and Business
    wgap4, wgap3    mean log last wage of employees at level 4 (3) minus level 1, against the Census 2010
                    (census_wage_level_enmu_2010.csv)

    python runner_qualification_grid.py lote CPUS [cities=A,B] [runs=N] [part=K/N]
    python runner_qualification_grid.py comparar      paired deltas of each alpha against 0, then the grid table

Output: output/qualification_grid/<alpha>/<CITY>/<i>/. Local conf/params.py and conf/run.py are ignored.
"""
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

import runner_education_ab as X

A = X.A
A.OUTPUT = 'output/qualification_grid'
A.KEY = 'CB_QUALIFICATION'
A.LEVELS = ['0', '0.1', '0.2', '0.35', '0.5']
A.BASE = dict(X.A.BASE, QUALIFICATION_SCORE='log')
A.V.R.conf.RUN['SAVE_DATA'] = ['house', 'sales', 'agents', 'firms']
CITIES = ['GOIANIA', 'FORTALEZA']
END = datetime.date(2016, 1, 1)
X.MONTHS = [datetime.date(y, 1, 1) for y in range(2013, 2016)]
X.LAST = datetime.date(2015, 1, 1)
A.V.WINDOW = (datetime.date(2011, 1, 1), datetime.date(2015, 12, 1))
SECTORS = ['Government', 'Financial', 'Business']
CENSUS_WAGE = os.path.join(X.PRIVATE, 'analysis/enmu/census_wage_level_enmu_2010.csv')
_staff = {}
_build = A.build_jobs


def build_jobs(root, items, total_days=None):
    """runner_rent_ap jobs with alpha as a number and the run ending at END"""
    jobs = _build(root, items, total_days or (END - A.V.R.conf.PARAMS['STARTING_DAY']).days)
    for job in jobs:
        job['params'][A.KEY] = float(job['params'][A.KEY])
    return jobs


A.build_jobs = build_jobs


def staff(path):
    """Employees aged 17-69 in X.MONTHS with level, sector and last wage"""
    if path not in _staff:
        a = pd.read_parquet(os.path.join(path, 'agents.parquet'),
                            columns=['month', 'age', 'qualification', 'firm_id', 'last_wage'])
        a['month'] = pd.to_datetime(a.month).dt.date
        a = a[a.month.isin(X.MONTHS) & (a.age > 16) & (a.age < 70) & a.firm_id.notna()].copy()
        f = pd.read_parquet(os.path.join(path, 'firms.parquet'), columns=['month', 'firm_id', 'sector'])
        f['month'] = pd.to_datetime(f.month).dt.date
        a = a.merge(f, on=['month', 'firm_id'], how='inner')
        a['level'] = pd.cut(a.qualification, [-1, 3, 8, 11, 99], labels=[1, 2, 3, 4]).astype(int)
        _staff[path] = a
    return _staff[path]


def census_mix(path):
    """Census 2010 employees' level shares by sector in the run's municipalities"""
    acps = json.load(open(os.path.join(path, 'conf.json')))['PARAMS']['PROCESSING_ACPS']
    m = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    muns = set(m[m.ACPs.isin(acps)].cod_mun.astype(int))
    t = pd.read_csv('input/own_account_2010.csv', sep=';')
    t = t[t.cod_mun.isin(muns) & (t.position == 'employee') & (t.sector != 'Unknown')]
    t = t.groupby(['sector', 'level']).persons.sum().unstack(fill_value=0.0)
    return t.div(t.sum(axis=1), axis=0)


def tvd(path):
    a = staff(path)
    c = census_mix(path)
    out = []
    for _, g in a.groupby('month'):
        m = g.groupby(['sector', 'level']).size().unstack(fill_value=0).reindex(columns=[1, 2, 3, 4], fill_value=0)
        m = m[m.index.isin(c.index)]
        n = m.sum(axis=1)
        d = 0.5 * (m.div(n, axis=0) - c.reindex(m.index)[[1, 2, 3, 4]]).abs().sum(axis=1)
        out.append(float((d * n).sum() / n.sum()))
    return float(np.mean(out))


def edu4(sector):
    def f(path):
        g = staff(path)
        g = g[g.sector == sector]
        return float((g.level == 4).mean()) if len(g) else np.nan
    return f


def wgap(k):
    def f(path):
        g = staff(path)
        g = g[g.last_wage > 0]
        lw = np.log(g.last_wage).groupby(g.level).mean()
        return float(lw.get(k, np.nan) - lw.get(1, np.nan))
    return f


def census_wgap(acp, k):
    c = pd.read_csv(CENSUS_WAGE).set_index(['acp', 'level']).log_gap
    return float(c.get((acp, k), np.nan))


A.METRICS = X.A.METRICS + [('tvd', tvd)] + [('edu4_' + s, edu4(s)) for s in SECTORS] + \
    [('wgap4', wgap(4)), ('wgap3', wgap(3))]


def grid():
    """Mean over seeds of the sector distance and the sorting indicators per area and alpha"""
    rows = []
    for level in A.LEVELS:
        for city in CITIES:
            for done in sorted(__import__('glob').glob(os.path.join(A.OUTPUT, level, city, '*', 'DONE'))):
                p = os.path.dirname(done)
                r = {'alpha': float(level), 'city': city, 'seed': int(p.split(os.sep)[-1])}
                for name in ['tvd'] + ['edu4_' + s for s in SECTORS] + ['wgap4', 'wgap3', 'u_grad', 'u_gap',
                                                                         'w30_private', 'w30_transit']:
                    r[name] = dict(A.METRICS)[name](p)
                rows.append(r)
    t = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    g = t.groupby(['city', 'alpha'])
    out = g.mean(numeric_only=True).drop(columns='seed').join(g.tvd.std().rename('tvd_sd')).join(g.size().rename('n'))
    print('\nGrid, mean over seeds of the Januaries 2013-2015 (tvd: sector distance to the Census 2010, lower is better)')
    print(out.round(3).to_string())
    print('\nCensus 2010 log wage gaps:', {c: (round(census_wgap(c, 4), 3), round(census_wgap(c, 3), 3)) for c in CITIES})
    best = out.groupby('alpha').tvd.mean()
    print('\nMean tvd over areas by alpha:', best.round(4).to_dict(), '-> alpha', best.idxmin())
    t.to_csv(os.path.join(A.OUTPUT, 'grid.csv'), sep=';', index=False)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'lote':
        if not any(a.startswith('cities=') for a in sys.argv):
            sys.argv.append('cities=' + ','.join(CITIES))
        if not any(a.startswith('runs=') for a in sys.argv):
            sys.argv.append('runs=2')
        A.main()
    elif len(sys.argv) > 1 and sys.argv[1] == 'comparar':
        A.main()
        grid()
    else:
        A.main()
