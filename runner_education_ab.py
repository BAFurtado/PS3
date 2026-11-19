"""
POSTING_EDUCATION A/B: 'off' (any applicant fills any vacancy) against 'census' (each vacancy draws its education level
from its sector's Census 2010 employees and is filled from applicants of that level), paired by seed. Both sides as
the ENMU base scenario (runner_enmu.COMMON, TRANSPORT_WEIGHTS trips, base network all through), 2010-2021, seeds
SEED + i, the settings of runner_decay_ab 'off', whose runs serve as the 'off' side where they exist. Saved: stats,
regional, houses and agents parquet every January, sales.csv.

Modes and comparison as runner_rent_ap.py (cities default enmu), plus, from the agents parquet of January 2017-2021
(active = the PARTICIPATION 'census' draw, levels as world/own_account.level):
    u1-u4, u_gap, u_grad     unemployment by education level, the sum over levels of |model - Census 2010| and the
                             top level's unemployment minus the mean of levels 1-3 (census_edu_enmu_2010.csv)
    rho_edu1, rho_min        Spearman across APs of unemployment with the share at level 1 and with employees' mean
                             commute minutes, January 2021 (Census 2010: census_access_ap_2010.csv)
    kl_*, w30_*              employees' commute bands against Census 2022 by mode, as runner_decay_ab

    python runner_education_ab.py lote CPUS [cities=enmu|A,B] [runs=N] [part=K/N]
    python runner_education_ab.py comparar

Output: output/education_ab/<value>/<CITY>/<i>/. Local conf/params.py and conf/run.py are ignored.
"""
import datetime
import json
import os

import numpy as np
import pandas as pd

import runner_enmu as E
import runner_rent_ap as A

A.OUTPUT = 'output/education_ab'
A.KEY = 'POSTING_EDUCATION'
A.LEVELS = ['off', 'census']
A.BASE = dict(E.COMMON, TRANSPORT_WEIGHTS='trips', TRANSPORT_SCHEDULE=None)
A.V.R.conf.RUN['SAVE_DATA'] = ['house', 'sales', 'agents']
PRIVATE = os.path.expanduser('~/MyModels/PS3/private')
CENSUS_EDU = os.path.join(PRIVATE, 'analysis/enmu/census_edu_enmu_2010.csv')
CENSUS_AP = os.path.join(PRIVATE, 'analysis/enmu/census_access_ap_2010.csv')
CENSUS_BANDS = os.path.join(PRIVATE, 'notes/text_bndes/item3_hiring_decay/hiring_decay_2022.csv')
MONTHS = [datetime.date(y, 1, 1) for y in range(2017, 2022)]
LAST = datetime.date(2021, 1, 1)
BANDS = [30, 60, 120]
_cache = {}


class _Agent:
    pass


def acp_of(path):
    return path.rstrip(os.sep).split(os.sep)[-2]


def agents(path):
    """Agents aged 17-69 in MONTHS with active, employed and level"""
    if path not in _cache:
        from world.participation import Participation
        a = pd.read_parquet(os.path.join(path, 'agents.parquet'),
                            columns=['month', 'id', 'age', 'gender', 'region_id', 'qualification', 'firm_id',
                                     'distance', 'has_car'])
        a['month'] = pd.to_datetime(a.month).dt.date
        a = a[a.month.isin(MONTHS) & (a.age > 16) & (a.age < 70)].copy()
        a['region_id'] = a.region_id.astype(str)
        seed = json.load(open(os.path.join(path, 'conf.json')))['PARAMS']['SEED']
        p = Participation(sorted({r[:7] for r in a.region_id}), seed)
        x = _Agent()

        def active(row):
            x.id, x.age, x.gender, x.region_id = row.id, row.age, row.gender, row.region_id
            return p.is_active(x)

        a['active'] = [active(r) for r in a.itertuples()]
        a['emp'] = a.firm_id.notna()
        a['level'] = pd.cut(a.qualification, [-1, 3, 8, 11, 99], labels=[1, 2, 3, 4]).astype(int)
        _cache[path] = a
    return _cache[path]


def census_u(acp):
    c = pd.read_csv(CENSUS_EDU)
    c = c[(c.acp == acp) & (c.edu < 5)].set_index('edu')
    return 1 - c.employed / c.active


def u_level(path):
    a = agents(path)
    act = a[a.active]
    return 1 - act.groupby('level').emp.mean()


def u_of(k):
    return lambda path: float(u_level(path).get(k, np.nan))


def u_gap(path):
    return float((u_level(path) - census_u(acp_of(path))).abs().sum())


def u_grad(path):
    u = u_level(path)
    return float(u[4] - u[[1, 2, 3]].mean())


def ap_rho(column):
    def f(path):
        a = agents(path)
        a = a[a.month == LAST]
        act = a[a.active]
        g = act.groupby('region_id').agg(n=('emp', 'size'), u=('emp', lambda e: 1 - e.mean()),
                                         edu1=('level', lambda v: (v == 1).mean()))
        g['minutes'] = a[a.emp & (a.distance > 0)].groupby('region_id').distance.mean()
        g = g[g.n >= 10]
        return float(g.u.rank().corr(g[column].rank()))
    return f


def commute_bands(path):
    a = agents(path)
    a = a[(a.month == LAST) & a.emp & (a.distance > 0)]
    out = {}
    for mode, g in (('private', a[a.has_car.astype(bool)]), ('transit', a[~a.has_car.astype(bool)])):
        s = np.bincount(np.digitize(g.distance, BANDS, right=True), minlength=4)[:4].astype(float)
        out[mode] = s / s.sum() if s.sum() else np.full(4, np.nan)
    return out


def observed_bands(acp, mode):
    t = pd.read_csv(CENSUS_BANDS, sep=';')
    t = t[(t.acp == acp) & (t.weights == 'trips') & (t.group == 'all') & (t['mode'] == mode)]
    return t.drop_duplicates('band').set_index('band').observed.reindex(['0-30', '31-60', '61-120', '120+']).to_numpy()


def kl(mode):
    def f(path):
        obs, model = observed_bands(acp_of(path), mode), commute_bands(path)[mode]
        return float(np.sum(obs * np.log(np.maximum(obs, 1e-12) / np.maximum(model, 1e-6))))
    return f


def within_30(mode):
    return lambda path: float(commute_bands(path)[mode][0])


A.METRICS = ([('u%d' % k, u_of(k)) for k in range(1, 5)] + [('u_gap', u_gap), ('u_grad', u_grad),
             ('rho_edu1', ap_rho('edu1')), ('rho_min', ap_rho('minutes'))] +
             [('kl_private', kl('private')), ('kl_transit', kl('transit')),
              ('w30_private', within_30('private')), ('w30_transit', within_30('transit'))])

if __name__ == '__main__':
    A.main()
