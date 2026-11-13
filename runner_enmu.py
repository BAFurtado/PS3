"""
ENMU networks in six areas, 2010-2040: Brasilia, Fortaleza, Goiania, Manaus, Baixada Santista and Teresina. Every run
starts on the Rede Base of the ENMU zone matrices (2054, morning peak, tools/transport/partner_matrix.py). Scenarios:
    base        Rede Base for the whole run (the reference)
    padrao      Rede Futura, Cenario Padrao, from 2024-01
    otimizado   Rede Futura, Cenario Otimizado, from 2024-01
Every run of padrao and otimizado equals the base run of the same area and seed before 2024-01. The matrices use
trips weights. Teresina also runs the three scenarios with area weights (base_area, padrao_area, otimizado_area).
Settings of every run: CAR_OWNERSHIP census2022, LOCATION_CHOICE logit, VACANCY_SCOPE region, TRANSPORT_MONEY fare,
the other settings at the code defaults. Seeds SEED + i (i = 0-9, the seeds of runner_mcmv2026). Saved: stats,
regional, and the agents, families, houses and firms parquet every January.

Modes (cwd = repository root):
    python runner_enmu.py teste CPUS             Teresina, the six scenarios, seed 0, 2 years, networks switching in
                                                 2011-01, then the check
    python runner_enmu.py lote CPUS [OPTIONS]    the batch; rerun the same command to resume (finished runs skipped)
        cities=A,B     a comma-separated list of the six areas (default all)
        part=K/N       the K-th of N contiguous slices of the queue (default 1/1)
    python runner_enmu.py conferir               finished runs, identity with the reference before 2024-01

Output: output/enmu2026/<scenario>/<CITY>/<i>/; queue manifest and log in output/enmu2026/queue_<K>of<N>/.
Needs the private matrices input/transport/travel_times_<ACP>__<weights>.parquet. Local conf/params.py and
conf/run.py are ignored.
"""
import datetime
import logging
import os
import sys

import runner_funding2026 as R

OUTPUT = 'output/enmu2026'
SWITCH = 2024
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
}
R.conf.RUN['SAVE_DATA'] = ['agents', 'family', 'house', 'firms']
logger = logging.getLogger('main')


def add_scenarios(switch=SWITCH):
    """The scenarios of each weighting, the reference first; returns {weights: names}."""
    R.SCENARIOS.clear()
    R.REFERENCE.clear()
    names = {}
    for weights, suffix in (('trips', ''), ('area', '_area')):
        base = 'base' + suffix
        R.SCENARIOS[base] = {'TRANSPORT_WEIGHTS': weights, 'TRANSPORT_SCHEDULE': None}
        R.REFERENCE[base] = base
        for future in FUTURES:
            R.SCENARIOS[future + suffix] = {'TRANSPORT_WEIGHTS': weights, 'TRANSPORT_SCHEDULE': [[switch, future]]}
            R.REFERENCE[future + suffix] = base
        names[weights] = [base] + [f + suffix for f in FUTURES]
    return names


def queue(names, cities, indices):
    """(scenario, city, i) seed by seed, the scenarios of a city together, so a batch stopped halfway holds
    complete sets."""
    return [(s, city, i)
            for i in indices
            for city in cities
            for s in names['trips'] + (names['area'] if city in AREA_CITIES else [])]


def check_inputs(cities):
    from world.transport import standard_path
    missing = [standard_path(c, w) for c in cities for w in ['trips'] + (['area'] if c in AREA_CITIES else [])
               if not os.path.exists(standard_path(c, w))]
    if missing:
        sys.exit('Missing matrices (unzip enmu_private_inputs.zip in the repository root): ' + ', '.join(missing))


def options(args):
    opts = {'cities': ','.join(CITIES), 'part': '1/1'}
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
    return [c for c in CITIES if c in cities], k, n


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
        R.conferir(OUTPUT, datetime.date(SWITCH, 1, 1))
    elif mode == 'teste' and len(sys.argv) == 3:
        root = OUTPUT + '_teste'
        names = add_scenarios(switch=2011)
        check_inputs(['TERESINA'])
        R.run(root, queue(names, ['TERESINA'], [0]), int(sys.argv[2]), root, dict(COMMON, TOTAL_DAYS=730))
        R.conferir(root, datetime.date(2011, 1, 1))
        differences(root, datetime.date(2011, 1, 1))
    elif mode == 'lote' and len(sys.argv) >= 3:
        cities, k, n = options(sys.argv[3:])
        names = add_scenarios()
        check_inputs(cities)
        items = queue(names, cities, list(range(R.RUNS)))
        # Contiguous slices of the queue, sizes differing by at most one
        items = items[(k - 1) * len(items) // n:k * len(items) // n]
        manifest_dir = os.path.join(OUTPUT, 'queue_{}of{}'.format(k, n))
        os.makedirs(manifest_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(manifest_dir, 'runner.log'), encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        logger.info('areas %s, part %d/%d, %d runs', ','.join(cities), k, n, len(items))
        R.run(OUTPUT, items, int(sys.argv[2]), manifest_dir, COMMON)
        R.conferir(OUTPUT, datetime.date(SWITCH, 1, 1))
    else:
        sys.exit(__doc__)
