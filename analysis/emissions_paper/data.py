"""Tidy tables from a runner_emissions.py batch: <root>/<SCENARIO>/<CITY>/<seed>/ with stats.csv and firms.parquet.

    load(root) -> Batch with
        stats    scenario, city, seed, month, emissions, gdp_level       monthly, from stats.csv
        sectors  scenario, city, seed, year, sector, E, B, e             January snapshots of firms.parquet
        coords   city, lon, lat                                          median firm location

Only finished runs (DONE) are read, and only NO_POLICY and the tax scenarios. The tables are cached in
<root>/_figcache/ and rebuilt when the set of finished runs changes. Run from the repository root.
"""
import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from analysis.output import columns_for

REFERENCE = 'NO_POLICY'
STATS_COLUMNS = ['month', 'emissions', 'gdp_level']
INTENSITY = pd.read_csv('input/emissions_sectors.csv').set_index('isic_12').eco


@dataclass
class Batch:
    stats: pd.DataFrame
    sectors: pd.DataFrame
    coords: pd.DataFrame

    def pairs(self, scenario):
        """(city, seed) finished in both `scenario` and the reference."""
        have = lambda s: set(map(tuple, self.stats.loc[self.stats.scenario == s, ['city', 'seed']]
                                 .drop_duplicates().values))
        return have(scenario) & have(REFERENCE)


def wanted(scenario):
    return scenario == REFERENCE or scenario.startswith('TAX')


def finished_runs(root):
    runs = []
    for done in glob.glob(os.path.join(root, '*', '*', '*', 'DONE')):
        d = os.path.dirname(done)
        s, city, seed = d.split(os.sep)[-3:]
        if wanted(s):
            runs.append((s, city, int(seed), d))
    return sorted(runs)


def _emissions_param(run_dir):
    with open(os.path.join(run_dir, 'conf.json')) as f:
        return json.load(f)['PARAMS']['EMISSIONS_PARAM']


def read_run(scenario, city, seed, run_dir):
    raw = pd.read_csv(os.path.join(run_dir, 'stats.csv'), sep=';', header=None)
    raw.columns = columns_for('stats', raw.shape[1])
    stats = raw[STATS_COLUMNS].copy()
    stats['month'] = pd.to_datetime(stats.month)

    f = pd.read_parquet(os.path.join(run_dir, 'firms.parquet'),
                        columns=['month', 'sector', 'emissions', 'eco_eff', 'long', 'lat'])
    a = f.sector.map(INTENSITY).fillna(0.0)
    # The saved revenue is not the base the month's emissions were computed on (it moves after
    # create_externalities), so the base is recovered from the emissions themselves: E = eff * a * B / param
    with np.errstate(divide='ignore', invalid='ignore'):
        f['B'] = np.where((a > 0) & (f.eco_eff > 0), _emissions_param(run_dir) * f.emissions / (f.eco_eff * a), 0.0)
    f['eB'] = f.eco_eff * f.B
    f['year'] = pd.to_datetime(f.month).dt.year
    sectors = f.groupby(['year', 'sector']).agg(E=('emissions', 'sum'), B=('B', 'sum'), eB=('eB', 'sum')).reset_index()
    sectors['e'] = np.where(sectors.B > 0, sectors.eB / sectors.B.where(sectors.B > 0, 1), 1.0)
    sectors = sectors.drop(columns='eB')
    coords = (float(f.long.median()), float(f.lat.median()))

    for t in (stats, sectors):
        t.insert(0, 'seed', seed)
        t.insert(0, 'city', city)
        t.insert(0, 'scenario', scenario)
    return stats, sectors, coords


def load(root, n_jobs=6):
    runs = finished_runs(root)
    if not runs:
        raise SystemExit('No finished NO_POLICY or tax runs in {}'.format(root))
    cache = os.path.join(root, '_figcache')
    index = [list(r[:3]) for r in runs]
    if os.path.exists(os.path.join(cache, 'runs.json')):
        with open(os.path.join(cache, 'runs.json')) as f:
            if json.load(f) == index:
                return Batch(*(pd.read_parquet(os.path.join(cache, n + '.parquet'))
                               for n in ('stats', 'sectors', 'coords')))
    print('Reading {} runs from {}'.format(len(runs), root))
    out = Parallel(n_jobs=n_jobs)(delayed(read_run)(*r) for r in runs)
    stats = pd.concat([o[0] for o in out], ignore_index=True)
    sectors = pd.concat([o[1] for o in out], ignore_index=True)
    coords = (pd.DataFrame([(r[1], *o[2]) for r, o in zip(runs, out)], columns=['city', 'lon', 'lat'])
              .groupby('city', as_index=False).median())
    os.makedirs(cache, exist_ok=True)
    for name, t in (('stats', stats), ('sectors', sectors), ('coords', coords)):
        t.to_parquet(os.path.join(cache, name + '.parquet'))
    with open(os.path.join(cache, 'runs.json'), 'w') as f:
        json.dump(index, f)
    return Batch(stats, sectors, coords)
