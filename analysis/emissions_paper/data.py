"""Processed tables for the emissions paper, from a runner_emissions.py batch: <root>/<SCENARIO>/<CITY>/<seed>/ with
stats.csv, firms.parquet and, when saved, agents.parquet (January of each year).

    load(root) -> Batch with
        stats    scenario, city, seed, month, and STATS (emissions, GDP, labour, inequality, prices, taxes, firms)
                                                                                         monthly, from stats.csv
        sectors  scenario, city, seed, year, sector, E, B, e, firms, sold, revenue, profit, losing, balance,
                 employees, investment                                                   January snapshots
        cohort   scenario, city, seed, firm_id, year, sector, revenue, profit, eco_eff, employees, balance, sold
                 the firms alive in BASELINE_YEAR, followed from then on (empty rows once a firm exits)
        workers  scenario, city, seed, year, group, n, employed, same_sector, wage, wage_gini
                 the agents of BASELINE_YEAR grouped by the sector they worked in then ('none': not employed);
                 empty when the batch did not save agents
        coords   city, lon, lat                                                          median firm location

The population and seeds are shared by every scenario, and BASELINE_YEAR (January 2014) is before the policy start
(December 2014), so a firm or a worker of the cohort is the same agent in every scenario of a city and seed: the
cohort tables compare matched agents.

Only finished runs (DONE) are read. The tables are cached in <root>/_figcache/ and rebuilt when the set of finished
runs or VERSION changes. Run from the repository root.
"""
import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from analysis.output import columns_for

VERSION = 2
REFERENCE = 'NO_POLICY'
BASELINE_YEAR = 2014
INTENSITY = pd.read_csv('input/emissions_sectors.csv').set_index('isic_12').eco
# stats.csv column -> name in the stats table
STATS = {'emissions': 'emissions', 'gdp_level': 'gdp', 'unemployment': 'unemployment',
         'firms_total_employment': 'employment', 'gini_index': 'gini', 'families_total_income': 'income',
         'families_total_wages': 'wages', 'price_level': 'price_level', 'emissions_fund': 'tax_revenue',
         'firms_count': 'firms', 'firms_exited': 'firms_exited', 'firms_entered': 'firms_entered',
         'money_profits_out': 'profits_out'}
FIRM_COLUMNS = ['month', 'firm_id', 'sector', 'long', 'lat', 'total_balance', 'number_employees', 'amount_sold',
                'revenue', 'profit', 'emissions', 'eco_eff', 'innov_investment']
TABLES = ('stats', 'sectors', 'cohort', 'workers', 'coords')
KEYS = ['scenario', 'city', 'seed']


@dataclass
class Batch:
    stats: pd.DataFrame
    sectors: pd.DataFrame
    cohort: pd.DataFrame
    workers: pd.DataFrame
    coords: pd.DataFrame

    @property
    def scenarios(self):
        """Treated scenarios present, in the order of runner_emissions.SCENARIOS."""
        order = ['TAX', 'TAX_SUBSIDIES', 'TAX_RECYCLING', 'TAX_RECYCLING_SUBSIDIES']
        have = set(self.stats.scenario) - {REFERENCE}
        return [s for s in order if s in have] + sorted(have - set(order))

    def pairs(self, scenario):
        """(city, seed) finished in both `scenario` and the reference."""
        have = lambda s: set(map(tuple, self.stats.loc[self.stats.scenario == s, ['city', 'seed']]
                                 .drop_duplicates().values))
        return have(scenario) & have(REFERENCE)

    def paired(self, table, scenario):
        """Rows of `table` (a Batch attribute name or a DataFrame) for `scenario` and the reference, on the pairs
        finished in both."""
        t = getattr(self, table) if isinstance(table, str) else table
        keep = self.pairs(scenario)
        t = t[t.scenario.isin([scenario, REFERENCE])]
        return t[[k in keep for k in zip(t.city, t.seed)]]


def versus(t, scenario, on, values, how='pct'):
    """`scenario` against the reference, matched on `on` (e.g. ['city', 'seed', 'year']): 100 * (T / N - 1) for
    how='pct', T - N for how='diff'. Returns `on` plus one column per value; rows missing on either side are dropped.
    """
    a = t[t.scenario == scenario].set_index(on)[values]
    b = t[t.scenario == REFERENCE].set_index(on)[values]
    a, b = a.align(b, join='inner')
    out = 100 * (a / b - 1) if how == 'pct' else a - b
    return out.reset_index()


def gini(x):
    x = np.sort(np.asarray(x, float))
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.sum() <= 0:
        return np.nan
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


# ── One run ───────────────────────────────────────────────────────────────────

def finished_runs(root):
    runs = []
    for done in glob.glob(os.path.join(root, '*', '*', '*', 'DONE')):
        d = os.path.dirname(done)
        s, city, seed = d.split(os.sep)[-3:]
        runs.append((s, city, int(seed), d))
    return sorted(runs)


def _emissions_param(run_dir):
    with open(os.path.join(run_dir, 'conf.json')) as f:
        return json.load(f)['PARAMS']['EMISSIONS_PARAM']


def _stats(run_dir):
    raw = pd.read_csv(os.path.join(run_dir, 'stats.csv'), sep=';', header=None)
    raw.columns = columns_for('stats', raw.shape[1])
    s = raw[[c for c in ['month', *STATS] if c in raw.columns]].rename(columns=STATS)
    s['month'] = pd.to_datetime(s.month)
    return s


def _firms(run_dir):
    f = pd.read_parquet(os.path.join(run_dir, 'firms.parquet'), columns=FIRM_COLUMNS)
    f['year'] = pd.to_datetime(f.month).dt.year
    f['firm_id'] = f.firm_id.astype(str)
    return f.drop(columns='month')


def _sectors(f, param):
    a = f.sector.map(INTENSITY).fillna(0.0)
    # The saved revenue is not the base the month's emissions were computed on (it moves after
    # create_externalities), so the base is recovered from the emissions themselves: E = eff * a * B / param
    with np.errstate(divide='ignore', invalid='ignore'):
        B = np.where((a > 0) & (f.eco_eff > 0), param * f.emissions / (f.eco_eff * a), 0.0)
    g = f.assign(B=B, eB=f.eco_eff * B, losing=(f.profit < 0).astype(int)).groupby(['year', 'sector'])
    s = g.agg(E=('emissions', 'sum'), B=('B', 'sum'), eB=('eB', 'sum'), firms=('firm_id', 'size'),
              sold=('amount_sold', 'sum'), revenue=('revenue', 'sum'), profit=('profit', 'sum'),
              losing=('losing', 'sum'), balance=('total_balance', 'sum'), employees=('number_employees', 'sum'),
              investment=('innov_investment', 'sum')).reset_index()
    s['e'] = np.where(s.B > 0, s.eB / s.B.where(s.B > 0, 1), 1.0)
    return s.drop(columns='eB')


def _cohort(f):
    base = f.loc[f.year == BASELINE_YEAR, 'firm_id']
    c = f[(f.year >= BASELINE_YEAR) & f.firm_id.isin(base)]
    c = c[['firm_id', 'year', 'sector', 'revenue', 'profit', 'eco_eff', 'number_employees', 'total_balance',
           'amount_sold']].rename(columns={'number_employees': 'employees', 'total_balance': 'balance',
                                           'amount_sold': 'sold'})
    for col in ['revenue', 'profit', 'eco_eff', 'employees', 'balance', 'sold']:
        c[col] = c[col].astype('float32')
    return c


def _workers(run_dir, f):
    """The agents of BASELINE_YEAR by the sector they worked in then, followed year by year: how many are still in
    the population, employed, employed in the same sector, and their mean wage when employed. Plus the Gini of wages
    among all employed agents that year (wage_gini, the same in every group)."""
    path = os.path.join(run_dir, 'agents.parquet')
    if not os.path.exists(path):
        return pd.DataFrame()
    a = pd.read_parquet(path, columns=['month', 'id', 'firm_id', 'wage'])
    a['year'] = pd.to_datetime(a.month).dt.year
    a = a[a.year >= BASELINE_YEAR]
    sector = f.set_index(['year', 'firm_id']).sector
    fid = a.firm_id.astype(str).where(a.firm_id.notna())
    a['sector'] = sector.reindex(pd.MultiIndex.from_arrays([a.year, fid])).values
    a['employed'] = a.sector.notna()
    base = a[a.year == BASELINE_YEAR].set_index('id').sector.fillna('none').rename('group')
    c = a.join(base, on='id', how='inner')
    c['same_sector'] = c.employed & (c.sector == c.group)
    c['wage_emp'] = c.wage.where(c.employed)
    w = c.groupby(['year', 'group']).agg(n=('id', 'size'), employed=('employed', 'sum'),
                                         same_sector=('same_sector', 'sum'), wage=('wage_emp', 'mean')).reset_index()
    g = a[a.employed].groupby('year').wage.apply(gini).rename('wage_gini')
    return w.join(g, on='year')


def read_run(scenario, city, seed, run_dir):
    f = _firms(run_dir)
    out = {'stats': _stats(run_dir), 'sectors': _sectors(f, _emissions_param(run_dir)), 'cohort': _cohort(f),
           'workers': _workers(run_dir, f)}
    for t in out.values():
        if len(t):
            for k, v in zip(reversed(KEYS), (seed, city, scenario)):
                t.insert(0, k, v)
    out['coords'] = pd.DataFrame({'city': [city], 'lon': [f.long.median()], 'lat': [f.lat.median()]})
    return out


# ── Batch ─────────────────────────────────────────────────────────────────────

def load(root, n_jobs=6, refresh=False):
    """Every finished run of `root`, read in parallel and cached in <root>/_figcache/."""
    runs = finished_runs(root)
    if not runs:
        raise SystemExit('No finished runs in {}'.format(root))
    cache = os.path.join(root, '_figcache')
    index = {'version': VERSION, 'runs': [list(r[:3]) for r in runs]}
    if not refresh and os.path.exists(os.path.join(cache, 'runs.json')):
        with open(os.path.join(cache, 'runs.json')) as fh:
            if json.load(fh) == index:
                return Batch(*(pd.read_parquet(os.path.join(cache, n + '.parquet')) for n in TABLES))
    print('Reading {} runs from {}'.format(len(runs), root))
    out = Parallel(n_jobs=n_jobs)(delayed(read_run)(*r) for r in runs)
    tables = {}
    for n in TABLES:
        parts = [o[n] for o in out if len(o[n])]
        t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for k in ('scenario', 'city'):
            if k in t:
                t[k] = t[k].astype('category')
        tables[n] = t
    tables['coords'] = tables['coords'].groupby('city', as_index=False, observed=True).median()
    os.makedirs(cache, exist_ok=True)
    for n, t in tables.items():
        t.to_parquet(os.path.join(cache, n + '.parquet'))
    with open(os.path.join(cache, 'runs.json'), 'w') as fh:
        json.dump(index, fh)
    return Batch(*(tables[n] for n in TABLES))
