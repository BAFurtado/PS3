"""
MCMV Faixa 1 in one metro pot that commissions units at the observed FAR price (MCMV_ALLOCATION 'metro',
MCMV_COMMISSION), from 2010, in the 26 capitals without Sao Paulo, 2010-2034. Three OGU doses in 2026-2030, all at
.09 % of municipal GDP from 2031:
    d09   .09 % (tendencial, the reference)
    d13   .13 % (PMCMV 2016)
    d25   .25 %
Every run of d13 and d25 equals the d09 run of the same city and seed before 2026-01. Seeds (SEED + i, 10 runs), the
capitals, the saved data (stats, regional, and firms, agents and families parquet every January) and every other
setting come from runner_funding2026. OFFER_SIZE_ON_PRICE stays at its default.

Modes (cwd = repository root):
    python runner_mcmv2026.py teste CPUS           every scenario in Palmas, 1 seed, 2 years, then the check
    python runner_mcmv2026.py lote CPUS [OPTIONS]  the batch; rerun the same command to resume (finished runs skipped)
        part=K/N                the K-th of N contiguous slices of the queue, for N servers (default 1/1)
    python runner_mcmv2026.py conferir             finished runs, identity with d09 before 2026-01

Output: output/mcmv2026/<scenario>/<CITY>/<i>/; the queue manifest and log go to output/mcmv2026/queue_<K>of<N>/.
Local conf/params.py and conf/run.py are ignored.
"""
import datetime
import logging
import os
import sys

import runner_funding2026 as R

OUTPUT = 'output/mcmv2026'
END = datetime.date(2035, 1, 1)
DOSES = {'d09': .09, 'd13': .13, 'd25': .25}
COMMON = {
    'MCMV_ALLOCATION': 'metro',
    'MCMV_COMMISSION': True,
    'OGU_AFTER_DOSE': (2030, .09),
    'TOTAL_DAYS': (END - R.conf.PARAMS['STARTING_DAY']).days,
}
logger = logging.getLogger('main')


def add_scenarios():
    """The three doses, d09 first; returns their names."""
    for name, dose in DOSES.items():
        R.SCENARIOS[name] = {'OGU_INVESTMENT': {'otimista': dose, 'tendencial': dose, 'pessimista': dose}}
        R.REFERENCE[name] = 'd09'
    return list(DOSES)


def queue(scenarios, cities, indices):
    """(scenario, city, i) seed by seed, largest city first, so a batch stopped halfway holds complete sets."""
    return [(s, city, i) for i in indices for city in cities for s in scenarios]


def part(args):
    k, n = 1, 1
    for a in args:
        key, sep, v = a.partition('=')
        if key != 'part' or not sep:
            sys.exit('Unknown option {!r}. Options: part='.format(a))
        k, n = (int(x) for x in v.split('/'))
    if not 1 <= k <= n:
        sys.exit('part must be K/N with 1 <= K <= N')
    return k, n


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if R.LOCAL_CONF:
        print('Ignoring local overrides:', ', '.join(R.LOCAL_CONF))
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    scenarios = add_scenarios()
    if mode == 'conferir':
        R.conferir(OUTPUT)
    elif mode == 'teste':
        root = OUTPUT + '_teste'
        R.conf.RUN['SAVE_DATA'] = []
        R.run(root, queue(scenarios, ['PALMAS'], [0]), int(sys.argv[2]), root, dict(COMMON, TOTAL_DAYS=730))
        R.conferir(root, datetime.date(2012, 1, 1))
    elif mode == 'lote':
        k, n = part(sys.argv[3:])
        items = queue(scenarios, R.CAPITALS, list(range(R.RUNS)))
        # Contiguous slices of the queue, sizes differing by at most one
        items = items[(k - 1) * len(items) // n:k * len(items) // n]
        manifest_dir = os.path.join(OUTPUT, 'queue_{}of{}'.format(k, n))
        os.makedirs(manifest_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(manifest_dir, 'runner.log'), encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        logger.info('part %d/%d, %d runs', k, n, len(items))
        R.run(OUTPUT, items, int(sys.argv[2]), manifest_dir, COMMON)
        R.conferir(OUTPUT)
    else:
        print(__doc__)
