"""
MCMV Faixa 1 in one metro pot that commissions units at the observed FAR price (MCMV_ALLOCATION 'metro',
MCMV_COMMISSION), from 2010, 2010-2034. Three OGU doses in 2026-2030, all at .09 % of municipal GDP from 2031:
    d09   .09 % (tendencial, the reference)
    d13   .13 % (PMCMV 2016)
    d25   .25 %
Every run of d13 and d25 equals the d09 run of the same city and seed before 2026-01. Seeds (SEED + i, 10 runs), the
capitals and every other setting come from runner_funding2026. OFFER_SIZE_ON_PRICE stays at its default.
City sets:
    26   the 26 capitals without Sao Paulo, d09 / d13 / d25; stats, regional, and firms, agents and families parquet
         every January
    sp   Sao Paulo, d09 / d25; stats, regional, and firms and families parquet every January (no agents parquet)

Modes (cwd = repository root):
    python runner_mcmv2026.py teste CPUS [cities=sp]  26: every scenario in Palmas, 1 seed, 2 years, then the check;
                                                      sp: d09 and d25 in Sao Paulo, seed 0, 2 years, the check and
                                                      the saved files
    python runner_mcmv2026.py lote CPUS [OPTIONS]     the batch; rerun the same command to resume (finished runs
                                                      skipped)
        cities=26|sp            city set (default 26)
        part=K/N                26: the K-th of N contiguous slices of the queue; sp: the K-th of N contiguous
                                slices of the seeds, so d09 and d25 of a seed run on the same server (N = 3 gives
                                seeds 0-2, 3-5, 6-9). Default 1/1
    python runner_mcmv2026.py conferir                finished runs, identity with d09 before 2026-01

Output: output/mcmv2026/<scenario>/<CITY>/<i>/; the queue manifest and log go to output/mcmv2026/queue_<K>of<N>/
(26) or output/mcmv2026/queue_sp_<K>of<N>/ (sp). Local conf/params.py and conf/run.py are ignored.
"""
import datetime
import glob
import json
import logging
import os
import sys

import runner_funding2026 as R

OUTPUT = 'output/mcmv2026'
END = datetime.date(2035, 1, 1)
DOSES = {'d09': .09, 'd13': .13, 'd25': .25}
SP_DOSES = ['d09', 'd25']
SP_SAVE_DATA = ['firms', 'family']
COMMON = {
    'MCMV_ALLOCATION': 'metro',
    'MCMV_COMMISSION': True,
    'OGU_AFTER_DOSE': (2030, .09),
    'TOTAL_DAYS': (END - R.conf.PARAMS['STARTING_DAY']).days,
}
logger = logging.getLogger('main')

# At import, so worker processes started by spawn (Windows) save the same files as the parent
if 'cities=sp' in sys.argv[2:]:
    R.conf.RUN['SAVE_DATA'] = SP_SAVE_DATA


def add_scenarios():
    """The three doses, d09 first; returns their names."""
    for name, dose in DOSES.items():
        R.SCENARIOS[name] = {'OGU_INVESTMENT': {'otimista': dose, 'tendencial': dose, 'pessimista': dose}}
        R.REFERENCE[name] = 'd09'
    return list(DOSES)


def queue(scenarios, cities, indices):
    """(scenario, city, i) seed by seed, largest city first, so a batch stopped halfway holds complete sets."""
    return [(s, city, i) for i in indices for city in cities for s in scenarios]


def options(args):
    """key=value options: city set, and part K, N."""
    opts = {'cities': '26', 'part': '1/1'}
    for a in args:
        key, sep, v = a.partition('=')
        if not sep or key not in opts:
            sys.exit('Unknown option {!r}. Options: {}'.format(a, ', '.join(k + '=' for k in opts)))
        opts[key] = v
    if opts['cities'] not in ('26', 'sp'):
        sys.exit('cities must be 26 or sp')
    k, n = (int(x) for x in opts['part'].split('/'))
    if not 1 <= k <= n:
        sys.exit('part must be K/N with 1 <= K <= N')
    return opts['cities'], k, n


def saved_files(root, city='SAO PAULO'):
    """SAVE_DATA recorded in each finished run of `city` (conf.json) and the parquet files the run wrote."""
    for done in sorted(glob.glob(os.path.join(root, '*', city, '*', 'DONE'))):
        d = os.path.dirname(done)
        with open(os.path.join(d, 'conf.json')) as f:
            save = json.load(f)['RUN']['SAVE_DATA']
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, '*.parquet')))
        mb = sum(os.path.getsize(os.path.join(d, x)) for x in os.listdir(d)) / 2 ** 20
        print('{}: SAVE_DATA {}, files {}, {:.0f} MB'.format(os.path.relpath(d, root), save, files, mb))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if R.LOCAL_CONF:
        print('Ignoring local overrides:', ', '.join(R.LOCAL_CONF))
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    scenarios = add_scenarios()
    if mode == 'conferir':
        R.conferir(OUTPUT)
    elif mode == 'teste':
        cities, _, _ = options(sys.argv[3:])
        if cities == 'sp':
            root = OUTPUT + '_teste_sp'
            R.run(root, queue(SP_DOSES, ['SAO PAULO'], [0]), int(sys.argv[2]), root, dict(COMMON, TOTAL_DAYS=730))
            R.conferir(root, datetime.date(2012, 1, 1))
            saved_files(root)
        else:
            root = OUTPUT + '_teste'
            R.conf.RUN['SAVE_DATA'] = []
            R.run(root, queue(scenarios, ['PALMAS'], [0]), int(sys.argv[2]), root, dict(COMMON, TOTAL_DAYS=730))
            R.conferir(root, datetime.date(2012, 1, 1))
    elif mode == 'lote':
        cities, k, n = options(sys.argv[3:])
        if cities == 'sp':
            # Contiguous slices of the seeds, sizes differing by at most one
            seeds = list(range(R.RUNS))[(k - 1) * R.RUNS // n:k * R.RUNS // n]
            items = queue(SP_DOSES, ['SAO PAULO'], seeds)
            manifest_dir = os.path.join(OUTPUT, 'queue_sp_{}of{}'.format(k, n))
        else:
            items = queue(scenarios, R.CAPITALS, list(range(R.RUNS)))
            # Contiguous slices of the queue, sizes differing by at most one
            items = items[(k - 1) * len(items) // n:k * len(items) // n]
            manifest_dir = os.path.join(OUTPUT, 'queue_{}of{}'.format(k, n))
        os.makedirs(manifest_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(manifest_dir, 'runner.log'), encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        logger.info('cities %s, part %d/%d, %d runs', cities, k, n, len(items))
        R.run(OUTPUT, items, int(sys.argv[2]), manifest_dir, COMMON)
        R.conferir(OUTPUT)
        if cities == 'sp':
            saved_files(OUTPUT)
    else:
        print(__doc__)
