"""
Capitalisation test of the ENMU runs: does a large cut in commute time raise house prices where it falls?

One area (CITY), the settings of runner_enmu.py (COMMON, trips weights, acceptance off). From START-01 the network
switches at once to the state 'capit': the Rede Base, with the transit minutes from every treated AP to every
destination times FACTOR. Car minutes stay as they are. The treated APs are a random half of the area's APs (fixed
draw, TREATED_SEED), the other half are controls in the same run. The state is added to the matrix in memory
(world.transport.read_acp_matrix is wrapped before the workers fork), so no input file changes. The reference is
the base run of runner_enmu.py with the same seed (output/enmu2026_rede/base/CITY/i), and a capit run equals it
before START-01.

    python runner_capit.py verificar          the capit state against base: treated and control minutes
    python runner_capit.py lote CPUS [N]      seeds 0 to N-1 (default 6); rerun to resume
    python runner_capit.py analisar           identity before START-01, then house price, vacancy, residents and
                                              commute minutes by AP, treated against control, paired by seed

Output: output/enmu2026_capit/capit/CITY/<i>/, treated APs in output/enmu2026_capit/tratadas.csv, tables in
output/enmu2026_capit/analise/.
"""
import datetime
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd

import runner_enmu as E
import world.transport as T

R = E.R
CITY = 'FORTALEZA'
WEIGHTS = 'trips'
FACTOR = 0.5
TREATED_SEED = 0
START = 2024
OUTPUT = 'output/enmu2026_capit'
REFERENCE_ROOT = 'output/enmu2026_rede'
WINDOWS = {'2024-30': (2024, 2030), '2031-37': (2031, 2037), '2038-39': (2038, 2039)}

_read = T.read_acp_matrix


def treated_aps():
    """A random half of the APs of CITY's matrix (origins), fixed by TREATED_SEED"""
    wide = _read(CITY, WEIGHTS)
    aps = np.array(sorted(wide.index.get_level_values('ap_orig').unique()))
    rng = np.random.default_rng(TREATED_SEED)
    return set(rng.choice(aps, size=len(aps) // 2, replace=False).tolist())


TREATED = treated_aps()


def read_with_capit(acp, weights=None):
    wide = _read(acp, weights)
    if wide is not None and acp == CITY and weights == WEIGHTS:
        wide = wide.copy()
        treated = wide.index.get_level_values('ap_orig').isin(TREATED)
        wide['capit'] = wide['base'] * np.where(treated, FACTOR, 1.0)
    return wide


T.read_acp_matrix = read_with_capit


def capit_params(scenario, city, start=START):
    if scenario == 'base':
        return {}
    return {'TRANSPORT_FUTURE': 'capit', 'TRANSPORT_SCHEDULE': [[start, 'capit']]}


def setup():
    R.SCENARIOS.clear()
    R.REFERENCE.clear()
    R.SCENARIOS['capit'] = {'TRANSPORT_WEIGHTS': WEIGHTS, 'TRANSPORT_SCHEDULE': None}
    R.REFERENCE['capit'] = 'capit'
    E.city_params = capit_params


def verificar():
    wide = T.read_acp_matrix(CITY, WEIGHTS)
    t = wide.index.get_level_values('ap_orig').isin(TREATED)
    print('{}: {} APs, {} treated'.format(CITY, wide.index.get_level_values('ap_orig').nunique(), len(TREATED)))
    for name, mask in (('treated', t), ('control', ~t)):
        print('{:8s} base {:.1f} min, capit {:.1f} min, car {:.1f} min (mean over pairs)'.format(
            name, wide.loc[mask, 'base'].mean(), wide.loc[mask, 'capit'].mean(), wide.loc[mask, 'car'].mean()))
    print('Pairs with capit != base outside the treated origins:',
          int((wide.loc[~t, 'capit'] != wide.loc[~t, 'base']).sum()))


def identity(a, b, start):
    x, y = (pd.read_csv(os.path.join(p, 'stats.csv'), sep=';', header=None, dtype=str) for p in (a, b))
    x, y = (z[z[0] < str(start)].reset_index(drop=True) for z in (x, y))
    return len(x) == len(y) and x.equals(y)


def by_ap(d):
    """By AP and January: mean house price, vacancy share, residents, mean commute minutes of employed residents
    without and with a car"""
    h = pd.read_parquet(os.path.join(d, 'houses.parquet'), columns=['month', 'region_id', 'price', 'family_id'])
    h['year'] = pd.to_datetime(h.month).dt.year
    h = h[h.year >= START - 1]
    houses = h.groupby(['year', 'region_id']).agg(price=('price', 'mean'),
                                                  vacancy=('family_id', lambda x: x.isna().mean()),
                                                  houses=('price', 'size'))
    ag = pd.read_parquet(os.path.join(d, 'agents.parquet'), columns=['month', 'region_id', 'firm_id', 'distance',
                                                                     'has_car'])
    ag['year'] = pd.to_datetime(ag.month).dt.year
    ag = ag[ag.year >= START - 1]
    emp = ag[ag.firm_id.notna()]
    people = pd.DataFrame({
        'residents': ag.groupby(['year', 'region_id']).size(),
        'minutes_no_car': emp[~emp.has_car].groupby(['year', 'region_id']).distance.mean(),
        'minutes_car': emp[emp.has_car].groupby(['year', 'region_id']).distance.mean(),
    })
    return houses.join(people, how='outer')


def analisar():
    out = os.path.join(OUTPUT, 'analise')
    os.makedirs(out, exist_ok=True)
    rows, n_ok, n = [], 0, 0
    for done in sorted(glob.glob(os.path.join(OUTPUT, 'capit', CITY, '*', 'DONE'))):
        d = os.path.dirname(done)
        i = int(os.path.basename(d))
        ref = R.run_path(REFERENCE_ROOT, 'base', CITY, i)
        if not os.path.exists(os.path.join(ref, 'DONE')):
            print('seed {}: no base run in {}'.format(i, REFERENCE_ROOT))
            continue
        n += 1
        same = identity(d, ref, datetime.date(START, 1, 1))
        n_ok += same
        if not same:
            print('seed {}: differs from base before {}-01, left out'.format(i, START))
            continue
        x, y = by_ap(d), by_ap(ref)
        j = x.join(y, lsuffix='_capit', rsuffix='_base', how='inner').reset_index()
        j['seed'] = i
        rows.append(j)
    print('Identical to base before {}-01: {} of {}'.format(START, n_ok, n))
    if not rows:
        sys.exit('No complete pairs')
    a = pd.concat(rows)
    a['treated'] = a.region_id.isin(TREATED)
    a['price_pct'] = 100 * (a.price_capit / a.price_base - 1)
    a['vacancy_pt'] = 100 * (a.vacancy_capit - a.vacancy_base)
    a['residents_pct'] = 100 * (a.residents_capit / a.residents_base - 1)
    a['houses_diff'] = a.houses_capit - a.houses_base
    for m in ('minutes_no_car', 'minutes_car'):
        a[m + '_diff'] = a[m + '_capit'] - a[m + '_base']
    a.to_csv(os.path.join(out, 'ap_ano.csv'), sep=';', index=False)

    metrics = ['minutes_no_car_diff', 'minutes_car_diff', 'price_pct', 'vacancy_pt', 'residents_pct', 'houses_diff']
    lines = []
    for w, (y0, y1) in WINDOWS.items():
        s = a[(a.year >= y0) & (a.year <= y1)]
        # Mean over APs within each seed and group, then treated - control by seed
        g = s.groupby(['seed', 'treated'])[metrics].mean().unstack('treated')
        for m in metrics:
            t, c = g[(m, True)], g[(m, False)]
            dd = t - c
            k = dd.notna().sum()
            tstat = dd.mean() / (dd.std(ddof=1) / np.sqrt(k)) if k > 1 else np.nan
            lines.append((w, m, t.mean(), c.mean(), dd.mean(), tstat, k))
    tab = pd.DataFrame(lines, columns=['window', 'metric', 'treated', 'control', 'treated_minus_control', 't',
                                       'seeds'])
    tab.to_csv(os.path.join(out, 'tratadas_controle.csv'), sep=';', index=False)
    pd.set_option('display.width', 200)
    print('{} capit - base by AP, mean over APs of each group, then over seeds (FACTOR {}, {} treated APs)'.format(
        CITY, FACTOR, len(TREATED)))
    print(tab.round(3).to_string(index=False))

    # Across APs, 2038-39: price change against the minutes saved by its employed residents without a car
    s = a[a.year >= 2038].groupby('region_id')[['price_pct', 'minutes_no_car_diff', 'residents_base']].mean()
    s = s[s.residents_base >= 50]
    print('Spearman across APs, 2038-39, price change vs minutes saved without a car: {:.3f} ({} APs)'.format(
        s.price_pct.corr(s.minutes_no_car_diff, method='spearman'), len(s)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'verificar':
        verificar()
    elif mode == 'lote' and len(sys.argv) >= 3:
        setup()
        E.check_inputs([CITY])
        os.makedirs(OUTPUT, exist_ok=True)
        pd.Series(sorted(TREATED), name='region_id').to_csv(os.path.join(OUTPUT, 'tratadas.csv'), index=False)
        seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        items = [('capit', CITY, i) for i in range(seeds)]
        E.run(OUTPUT, items, int(sys.argv[2]), os.path.join(OUTPUT, 'queue'), dict(E.COMMON))
    elif mode == 'analisar':
        analisar()
    else:
        sys.exit(__doc__)
