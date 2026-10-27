"""
Funding from 2026, the ogu result at other values of OFFER_SIZE_ON_PRICE (default 5), the weight of vacancy in the
house price formula, in the 26 capitals without Sao Paulo, 2010-2039. Scenarios, seeds (SEED + i, 5 runs) and every
other setting come from runner_funding2026; value 5 is the base and ogu pair already in output/funding2026.

For each value V the batch runs base_oV and ogu_oV (V with p for the decimal point: 0 -> o0, 2.5 -> o2p5), and ogu_oV is
compared with base_oV before 2026-01. Value 0 switches the vacancy factor off.

Modes (cwd = repository root):
    python runner_oferta26.py teste CPUS [valores=...]           every scenario in Palmas, 1 seed, 2 years, then the check
    python runner_oferta26.py lote CPUS [OPTIONS]                the batch; rerun the same command to resume
        valores=0,1.5           values of OFFER_SIZE_ON_PRICE (default 0)
        part=K/N                the K-th of N contiguous slices of the queue, for N servers (default 1/1)
    python runner_oferta26.py conferir                           finished runs, identity with the reference before 2026-01

Output: output/funding2026_oferta26/<scenario>/<CITY>/<i>/ with stats.csv and regional.csv only (no parquet); the
queue manifest and log go to output/funding2026_oferta26/queue_<K>of<N>/. Local conf/params.py and conf/run.py are
ignored.
"""
import datetime
import logging
import os
import sys

import runner_funding2026 as R

R.conf.RUN.update({'SAVE_DATA': []})

OUTPUT = 'output/funding2026_oferta26'
logger = logging.getLogger('main')


def tag(v):
    return 'o' + ('%g' % v).replace('.', 'p')


def add_scenarios(values):
    """base_oV and ogu_oV for each value; returns the scenario names, base first."""
    names = []
    for v in values:
        b, o = 'base_' + tag(v), 'ogu_' + tag(v)
        R.SCENARIOS[b] = {'OFFER_SIZE_ON_PRICE': v}
        R.SCENARIOS[o] = {'OGU_INVESTMENT': R.OGU_HIGH, 'OFFER_SIZE_ON_PRICE': v}
        R.REFERENCE[b] = R.REFERENCE[o] = b
        names += [b, o]
    return names


def queue(scenarios, cities, indices):
    """(scenario, city, i) seed by seed, largest city first, so a batch stopped halfway holds complete sets."""
    return [(s, city, i) for i in indices for city in cities for s in scenarios]


def options(args):
    """key=value options of the lote and teste modes."""
    opts = {'valores': '0', 'part': '1/1'}
    for a in args:
        k, sep, v = a.partition('=')
        if not sep or k not in opts:
            sys.exit('Unknown option {!r}. Options: {}'.format(a, ', '.join(k + '=' for k in opts)))
        opts[k] = v
    values = [float(x) for x in opts['valores'].split(',')]
    if 5 in values:
        sys.exit('value 5 is the default, already in ' + R.OUTPUT)
    k, n = (int(x) for x in opts['part'].split('/'))
    if not 1 <= k <= n:
        sys.exit('part must be K/N with 1 <= K <= N')
    return values, k, n


def conferir(root=OUTPUT, start=R.START):
    """Scenario names come from the folders, so any value already run is checked against its base_oV."""
    if os.path.isdir(root):
        for d in os.listdir(root):
            if d.startswith(('base_o', 'ogu_o')):
                R.REFERENCE[d] = 'base_' + d.split('_', 1)[1]
    R.conferir(root, start)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if R.LOCAL_CONF:
        print('Ignoring local overrides:', ', '.join(R.LOCAL_CONF))
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'conferir':
        conferir()
    elif mode == 'teste':
        values, _, _ = options(sys.argv[3:])
        root = OUTPUT + '_teste'
        R.run(root, queue(add_scenarios(values), ['PALMAS'], [0]), int(sys.argv[2]), root, {'TOTAL_DAYS': 730})
        conferir(root, datetime.date(2012, 1, 1))
    elif mode == 'lote':
        values, k, n = options(sys.argv[3:])
        items = queue(add_scenarios(values), R.CAPITALS, list(range(R.RUNS)))
        # Contiguous slices of the queue, sizes differing by at most one
        items = items[(k - 1) * len(items) // n:k * len(items) // n]
        manifest_dir = os.path.join(OUTPUT, 'queue_{}of{}'.format(k, n))
        os.makedirs(manifest_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(manifest_dir, 'runner.log'), encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        logger.info('values %s, part %d/%d', ','.join('%g' % v for v in values), k, n)
        R.run(OUTPUT, items, int(sys.argv[2]), manifest_dir)
        conferir()
    else:
        print(__doc__)
