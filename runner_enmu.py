"""
ENMU networks in six areas, 2010-2040: Brasilia, Fortaleza, Goiania, Manaus, Baixada Santista and Teresina. Every run
starts on the Rede Base of the ENMU zone matrices (2054, morning peak, tools/transport/partner_matrix.py). Scenarios:
    base        Rede Base for the whole run, no investment (the reference)
    money       Rede Base for the whole run, with the investment
    padrao      the investment, and the Rede Futura, Cenario Padrao, phased in with it
    otimizado   the investment, and the Rede Futura, Cenario Otimizado, phased in with it, with the Otimizado fare
The investment (PUBLIC_WORKS): the area's ENMU infrastructure capex in equal shares of GDP over YEARS years from
START, as a percent of its 2023 GDP (input/enmu_investment.csv, auxiliary/enmu_investment.py), money from outside
the area buying production from its construction firms. The network moves from Base to the future state in YEARS
equal yearly steps from START-01. The Otimizado fare: FARE_INCOME_SHARE times the area's ENMU fare ratio
Otimizado / Padrao, moving with the network. Every run of money, padrao and otimizado equals the base run of the same
area and seed before START-01. The matrices use trips weights. Teresina also runs the four scenarios with area weights
(base_area, money_area, padrao_area, otimizado_area).
Settings of every run: CAR_OWNERSHIP census2022, LOCATION_CHOICE logit, VACANCY_SCOPE region, TRANSPORT_MONEY fare,
RENT_LEVEL census_ap, HOUSING_BIDDING filatova, the other settings at the code defaults. Seeds SEED + i (i = 0-9,
the seeds of runner_mcmv2026). Saved: stats, regional, and the agents, families, houses and firms parquet every
January.

Modes (cwd = repository root):
    python runner_enmu.py teste CPUS             Teresina, the eight scenarios, seed 0, 2 years, investment and
                                                 networks from 2011-01, then the check
    python runner_enmu.py lote CPUS [OPTIONS]    the batch; rerun the same command to resume (finished runs skipped)
        cities=A,B     a comma-separated list of the six areas (default all)
        part=K/N       the K-th of N contiguous slices of the queue (default 1/1)
        seeds=N        the first N seeds only (default 10)
        area=0         without the area-weights scenarios of Teresina
        acceptance=X   COMMUTE_ACCEPTANCE X in every run, output in output/enmu2026_X
    python runner_enmu.py conferir [acceptance=X]   finished runs, identity with the reference before START-01

Output: output/enmu2026/<scenario>/<CITY>/<i>/; queue manifest and log in output/enmu2026/queue_<K>of<N>/.
Needs the private matrices input/transport/travel_times_<ACP>__<weights>.parquet. Local conf/params.py and
conf/run.py are ignored.
"""
import copy
import datetime
import logging
import os
import shutil
import sys

import pandas as pd

import runner_funding2026 as R
from world.transport import linear_phase_in

OUTPUT = 'output/enmu2026'
START = 2024
YEARS = 14
# Largest first within each seed
CITIES = ['BRASILIA', 'FORTALEZA', 'GOIANIA', 'MANAUS', 'BAIXADA SANTISTA', 'TERESINA']
AREA_CITIES = ['TERESINA']
FUTURES = ['padrao', 'otimizado']
COMMON = {
    'CAR_OWNERSHIP': 'census2022',
    'TRANSPORT_TIME': False,
    'LOCATION_CHOICE': 'logit',
    'VACANCY_SCOPE': 'region',
    'TRANSPORT_MONEY': 'fare',
    'RENT_LEVEL': 'census_ap',
    'HOUSING_BIDDING': 'filatova',
}
R.conf.RUN['SAVE_DATA'] = ['agents', 'family', 'house', 'firms']
logger = logging.getLogger('main')


def add_scenarios():
    """The scenarios of each weighting, the reference first; returns {weights: names}."""
    R.SCENARIOS.clear()
    R.REFERENCE.clear()
    names = {}
    for weights, suffix in (('trips', ''), ('area', '_area')):
        names[weights] = [s + suffix for s in ['base', 'money'] + FUTURES]
        for s in names[weights]:
            R.SCENARIOS[s] = {'TRANSPORT_WEIGHTS': weights, 'TRANSPORT_SCHEDULE': None}
            R.REFERENCE[s] = 'base' + suffix
    return names


def city_params(scenario, city, start=START):
    """The investment, network and fare settings of a scenario in a city"""
    name = scenario.replace('_area', '')
    if name == 'base':
        return {}
    row = pd.read_csv('input/enmu_investment.csv', sep=';').set_index('acp').loc[city]
    p = {'PUBLIC_WORKS': [start, start + YEARS - 1, 100 * row.infra_capex / YEARS / row.gdp_2023]}
    if name in FUTURES:
        p['TRANSPORT_FUTURE'] = name
        p['TRANSPORT_SCHEDULE'] = linear_phase_in(start, YEARS)
    if name == 'otimizado':
        p['FARE_INCOME_SHARE_FUTURE'] = R.conf.PARAMS['FARE_INCOME_SHARE'] * row.fare_otimizado / row.fare_padrao
    return p


def queue(names, cities, indices, area=True):
    """(scenario, city, i) seed by seed, the scenarios of a city together, so a batch stopped halfway holds
    complete sets."""
    return [(s, city, i)
            for i in indices
            for city in cities
            for s in names['trips'] + (names['area'] if area and city in AREA_CITIES else [])]


def check_inputs(cities):
    from world.transport import standard_path
    missing = [standard_path(c, w) for c in cities for w in ['trips'] + (['area'] if c in AREA_CITIES else [])
               if not os.path.exists(standard_path(c, w))]
    if missing:
        sys.exit('Missing matrices (unzip enmu_private_inputs.zip in the repository root): ' + ', '.join(missing))


def options(args):
    opts = {'cities': ','.join(CITIES), 'part': '1/1', 'seeds': str(R.RUNS), 'area': '1', 'acceptance': 'off'}
    for a in args:
        key, sep, v = a.partition('=')
        if not sep or key not in opts:
            sys.exit('Unknown option {!r}. Options: {}'.format(a, ', '.join(k + '=' for k in opts)))
        opts[key] = v
    cities = opts['cities'].split(',')
    unknown = set(cities) - set(CITIES)
    if unknown:
        sys.exit('Unknown areas {}. Areas: {}'.format(sorted(unknown), ', '.join(CITIES)))
    k, n = (int(x) for x in opts['part'].split('/'))
    if not 1 <= k <= n:
        sys.exit('part must be K/N with 1 <= K <= N')
    opts.update(cities=[c for c in CITIES if c in cities], k=k, n=n, seeds=int(opts['seeds']), area=opts['area'] != '0')
    return opts


def root_of(acceptance):
    return OUTPUT if acceptance == 'off' else '{}_{}'.format(OUTPUT, acceptance)


def run(root, items, cpus, manifest_dir, params, start=START):
    """Runs the unfinished items. Folders of unfinished runs are removed before their run is queued."""
    from main import save_jobs, _run_jobs_parallel
    jobs = []
    for s, city, i in items:
        path = R.run_path(root, s, city, i)
        if os.path.exists(os.path.join(path, 'DONE')):
            continue
        if os.path.exists(path):
            shutil.rmtree(path)
        p = copy.deepcopy(R.conf.PARAMS)
        p.update(copy.deepcopy(R.SCENARIOS[s]))
        p.update(params)
        p.update(city_params(s, city, start))
        p['PROCESSING_ACPS'] = [city]
        p['SEED'] = R.SEED + i
        jobs.append({'params': p, 'path': path})
    logger.info('%d of %d runs to do (finished ones skipped), %d at a time', len(jobs), len(items), cpus)
    os.makedirs(manifest_dir, exist_ok=True)
    save_jobs(manifest_dir, jobs, cpus)
    _run_jobs_parallel(jobs, cpus)


def differences(root, start):
    """Scenarios against their reference from `start` on: share of stats.csv rows that differ"""
    import glob
    import pandas as pd
    for done in sorted(glob.glob(os.path.join(root, '*', '*', '*', 'DONE'))):
        s, city, i = os.path.dirname(done).split(os.sep)[-3:]
        ref = R.run_path(root, R.REFERENCE[s], city, i)
        if R.REFERENCE[s] == s or not os.path.exists(os.path.join(ref, 'DONE')):
            continue
        a, b = (pd.read_csv(os.path.join(p, 'stats.csv'), sep=';', header=None, dtype=str)
                for p in (os.path.dirname(done), ref))
        a, b = (x[x[0] >= str(start)].reset_index(drop=True) for x in (a, b))
        print('{} {} {}: {} of {} rows from {} differ from {}'.format(
            s, city, i, int((a != b).any(axis=1).sum()), len(a), start, R.REFERENCE[s]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if R.LOCAL_CONF:
        print('Ignoring local overrides:', ', '.join(R.LOCAL_CONF))
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'conferir':
        add_scenarios()
        R.conferir(root_of(options(sys.argv[2:])['acceptance']), datetime.date(START, 1, 1))
    elif mode == 'teste' and len(sys.argv) == 3:
        root = OUTPUT + '_teste'
        names = add_scenarios()
        check_inputs(['TERESINA'])
        run(root, queue(names, ['TERESINA'], [0]), int(sys.argv[2]), root, dict(COMMON, TOTAL_DAYS=730), start=2011)
        R.conferir(root, datetime.date(2011, 1, 1))
        differences(root, datetime.date(2011, 1, 1))
    elif mode == 'lote' and len(sys.argv) >= 3:
        o = options(sys.argv[3:])
        names = add_scenarios()
        check_inputs(o['cities'])
        root = root_of(o['acceptance'])
        items = queue(names, o['cities'], list(range(o['seeds'])), o['area'])
        # Contiguous slices of the queue, sizes differing by at most one
        items = items[(o['k'] - 1) * len(items) // o['n']:o['k'] * len(items) // o['n']]
        manifest_dir = os.path.join(root, 'queue_{}of{}'.format(o['k'], o['n']))
        os.makedirs(manifest_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(manifest_dir, 'runner.log'), encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        logger.info('areas %s, part %d/%d, %d runs', ','.join(o['cities']), o['k'], o['n'], len(items))
        run(root, items, int(sys.argv[2]), manifest_dir, dict(COMMON, COMMUTE_ACCEPTANCE=o['acceptance']))
        R.conferir(root, datetime.date(START, 1, 1))
    else:
        sys.exit(__doc__)
