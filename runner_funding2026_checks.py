"""
Funding from 2026, two checks on the ogu result, 2010-2039, in five capitals that span its GDP return per real:
Brasilia, Belo Horizonte, Goiania, Salvador and Florianopolis. Scenarios, seeds (SEED + i, 5 runs) and every other
setting come from runner_funding2026.

Batches:
    painel   base and ogu again, saving agents and families once a year, so the people alive before 2026 can be
             followed across scenarios at a seed. Each run must equal its twin in output/funding2026 in every
             stats.csv row.
    oferta   base and ogu with OFFER_SIZE_ON_PRICE 2.5 and 7.5 (default 5), the weight of vacancy in the house price
             formula. Scenarios base_o25, ogu_o25, base_o75, ogu_o75.
Both batches save firms, agents and families annually; stats.csv carries the house_sales columns (price paid and
list price of negotiated sales).

Modes (cwd = repository root):
    python runner_funding2026_checks.py teste CPUS             every scenario in Palmas, 1 seed, 2 years, then the checks
    python runner_funding2026_checks.py lote CPUS [OPTIONS]    the batch; rerun the same command to resume
        batches=painel,oferta   batches (default both)
        part=K/N                the K-th of N contiguous slices of the queue, for N servers (default 1/1)
    python runner_funding2026_checks.py conferir               finished runs, identity with output/funding2026 (painel)
                                                               and with the reference before 2026-01 (oferta)

Output: output/funding2026_painel/<scenario>/<CITY>/<i>/ and output/funding2026_oferta/<scenario>/<CITY>/<i>/; the
queue manifest and log go to output/funding2026_checks/queue_<K>of<N>/. Local conf/params.py and conf/run.py are
ignored.
"""
import datetime
import logging
import os
import sys

import runner_funding2026 as R

R.conf.RUN.update({
    'SAVE_DATA': ['firms', 'agents', 'family'],
    'SAVE_DATA_PERIDIOCITY': 'ANNUALLY',
})

CITIES = ['BELO HORIZONTE', 'BRASILIA', 'SALVADOR', 'GOIANIA', 'FLORIANOPOLIS']
OUTPUT = {'painel': 'output/funding2026_painel', 'oferta': 'output/funding2026_oferta'}
MANIFEST = 'output/funding2026_checks'
OFFER_SIZES = {'o25': 2.5, 'o75': 7.5}

for _k, _v in OFFER_SIZES.items():
    R.SCENARIOS['base_' + _k] = {'OFFER_SIZE_ON_PRICE': _v}
    R.SCENARIOS['ogu_' + _k] = {'OGU_INVESTMENT': R.OGU_HIGH, 'OFFER_SIZE_ON_PRICE': _v}
    R.REFERENCE['base_' + _k] = R.REFERENCE['ogu_' + _k] = 'base_' + _k
BATCHES = {'painel': ['base', 'ogu'],
           'oferta': [s + '_' + k for k in OFFER_SIZES for s in ('base', 'ogu')]}

logger = logging.getLogger('main')


def queue(batches, cities, indices):
    """(batch, scenario, city, i) seed by seed, largest city first, so a batch stopped halfway holds complete sets."""
    return [(b, s, city, i)
            for i in indices
            for city in cities
            for b in batches
            for s in BATCHES[b]]


def run(items, cpus, manifest_dir, params=None):
    from main import save_jobs, _run_jobs_parallel
    jobs = []
    for b in BATCHES:
        jobs += R.build_jobs(OUTPUT[b], [(s, city, i) for bb, s, city, i in items if bb == b], params)
    logger.info('%d of %d runs to do (finished ones skipped), %d at a time', len(jobs), len(items), cpus)
    os.makedirs(manifest_dir, exist_ok=True)
    save_jobs(manifest_dir, jobs, cpus)
    _run_jobs_parallel(jobs, cpus)


def conferir_painel(root, twin_root=R.OUTPUT):
    """Every finished painel run against the run with the same scenario, city and seed in twin_root, all rows. Columns
    added to stats.csv after the twin ran sit at the end of the row, so the comparison covers the twin's columns."""
    import glob
    import pandas as pd
    same, differ, missing = 0, [], 0
    for done in sorted(glob.glob(os.path.join(root, '*', '*', '*', 'DONE'))):
        d = os.path.dirname(done)
        s, city, i = d.split(os.sep)[-3:]
        twin = R.run_path(twin_root, s, city, i)
        if not os.path.exists(os.path.join(twin, 'DONE')):
            missing += 1
            continue
        a, b = (pd.read_csv(os.path.join(x, 'stats.csv'), sep=';', header=None, dtype=str) for x in (d, twin))
        if a.iloc[:, :b.shape[1]].equals(b):
            same += 1
        else:
            differ.append((s, city, i))
    print('painel: {} identical to {} in every stats.csv row, {} differ, {} without a twin'.format(
        same, twin_root, len(differ), missing))
    for s, city, i in differ:
        print('  DIFFERS:', s, city, i)


def conferir(start=R.START):
    for b in BATCHES:
        print('==', b)
        R.conferir(OUTPUT[b], start)
    conferir_painel(OUTPUT['painel'])


def options(args):
    """key=value options of the lote mode."""
    opts = {'batches': ','.join(BATCHES), 'part': '1/1'}
    for a in args:
        k, sep, v = a.partition('=')
        if not sep or k not in opts:
            sys.exit('Unknown option {!r}. Options: {}'.format(a, ', '.join(k + '=' for k in opts)))
        opts[k] = v
    batches = opts['batches'].split(',')
    if [b for b in batches if b not in BATCHES]:
        sys.exit('batches among {}'.format(', '.join(BATCHES)))
    k, n = (int(x) for x in opts['part'].split('/'))
    if not 1 <= k <= n:
        sys.exit('part must be K/N with 1 <= K <= N')
    return batches, k, n


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if R.LOCAL_CONF:
        print('Ignoring local overrides:', ', '.join(R.LOCAL_CONF))
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'conferir':
        conferir()
    elif mode == 'teste':
        OUTPUT = {b: v + '_teste' for b, v in OUTPUT.items()}
        run(queue(list(BATCHES), ['PALMAS'], [0]), int(sys.argv[2]), MANIFEST + '_teste', {'TOTAL_DAYS': 730})
        for b in BATCHES:
            print('==', b)
            R.conferir(OUTPUT[b], datetime.date(2012, 1, 1))
        for b in BATCHES:
            for s in BATCHES[b]:
                d = R.run_path(OUTPUT[b], s, 'PALMAS', 0)
                print(s, sorted(f for f in os.listdir(d) if f.endswith('.parquet')))
    elif mode == 'lote':
        batches, k, n = options(sys.argv[3:])
        items = queue(batches, CITIES, list(range(R.RUNS)))
        items = items[(k - 1) * len(items) // n:k * len(items) // n]
        manifest_dir = os.path.join(MANIFEST, 'queue_{}of{}'.format(k, n))
        os.makedirs(manifest_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(manifest_dir, 'runner.log'), encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        logger.info('batches %s, part %d/%d', ','.join(batches), k, n)
        run(items, int(sys.argv[2]), manifest_dir)
        conferir()
    else:
        print(__doc__)
