"""
Emissions policy: every scenario x capital x seed in one resumable queue, 2010-2039.

Scenarios (each differs from NO_POLICY only after the eco policy starts, STARTING_DAY + ECO_POLICY_DAYS, 2014-12):
    NO_POLICY                 control: every policy flag off for the whole run (DiD / event-study counterfactual)
    TAX                       flat emission tax only
    SUBSIDIES                 flat eco-investment subsidies only
    TARGETED_SUBSIDIES        sector-weighted subsidies (TARGETED_SECTORS: Agriculture, Transport, Utilities)
    TAX_SUBSIDIES             tax + flat subsidies
    TAX_TARGETED_SUBSIDIES    tax + sector-weighted subsidies
    TAX_RECYCLING             tax with revenue recycled to bottom-quartile households
    TAX_RECYCLING_SUBSIDIES   recycling + flat subsidies
    TAX_RECYCLING_TARGETED    recycling + sector-weighted subsidies

Seeds are paired: run i of every scenario and capital uses seed SEED + i, so a treated run and the NO_POLICY run at
the same capital and i differ only by the policy. Population: one per capital, shared by every scenario and seed,
generated once from SEED into StoragedAgents/. Its sha256 is recorded in <OUTPUT>/populations/; a batch refuses to
start, and warns at the end, if the cached file no longer matches (other runners, e.g. runner_funding2026 with
FORCE_NEW_POPULATION, overwrite the same file -- do not run them on the same machine at the same time).
City sets: '26' = the 26 capitals without Sao Paulo, 'sp' = Sao Paulo.

Modes (cwd = repository root):
    python runner_emissions.py teste CPUS             every scenario in Palmas, 1 seed, 2 years with the policy from
                                                      day 360, then the check
    python runner_emissions.py lote CPUS [OPTIONS]    the batch; rerun the same command to resume (finished runs,
                                                      marked DONE, are skipped)
        cities=26|sp            city set (default 26)
        part=K/N                the K-th of N contiguous slices of the queue, for N servers (default 1/1)
    python runner_emissions.py conferir [FOLDER]      finished runs per scenario and city, identity with NO_POLICY
                                                      before the policy start

Output: output/emissions/<scenario>/<CITY>/<i>/ with stats.csv, regional.csv and firms.parquet saved every January
(no agents or families); the queue manifest and log of each batch go to output/emissions/queue_<cities>_<K>of<N>/.
Local conf/params.py and conf/run.py are ignored: every value comes from conf/default plus the overrides below.
"""
import copy
import datetime
import hashlib
import json
import logging
import os
import shutil
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import conf  # noqa: E402

# Module level on purpose: worker processes re-import this module (spawn on Windows), so they get the same conf
LOCAL_CONF = [f for f in ('conf/params.py', 'conf/run.py') if os.path.exists(f)]
for _n in ('params', 'run'):
    conf.load_conf_module('default.{}'.format(_n), key=_n)

SEED = 1000
RUNS = 10
OUTPUT = 'output/emissions'

_OFF = {'TAX_EMISSION': 0, 'ECO_INVESTMENT_SUBSIDIES': 0, 'TARGETED_SUBSIDIES': False, 'CARBON_TAX_RECYCLING': False}
SCENARIOS = {
    'NO_POLICY': _OFF,
    'TAX': {**_OFF, 'TAX_EMISSION': 1},
    'SUBSIDIES': {**_OFF, 'ECO_INVESTMENT_SUBSIDIES': 0.2},
    'TARGETED_SUBSIDIES': {**_OFF, 'ECO_INVESTMENT_SUBSIDIES': 0.2, 'TARGETED_SUBSIDIES': True},
    'TAX_SUBSIDIES': {**_OFF, 'TAX_EMISSION': 1, 'ECO_INVESTMENT_SUBSIDIES': 0.2},
    'TAX_TARGETED_SUBSIDIES': {**_OFF, 'TAX_EMISSION': 1, 'ECO_INVESTMENT_SUBSIDIES': 0.2,
                               'TARGETED_SUBSIDIES': True},
    'TAX_RECYCLING': {**_OFF, 'TAX_EMISSION': 1, 'CARBON_TAX_RECYCLING': True},
    'TAX_RECYCLING_SUBSIDIES': {**_OFF, 'TAX_EMISSION': 1, 'ECO_INVESTMENT_SUBSIDIES': 0.2,
                                'CARBON_TAX_RECYCLING': True},
    'TAX_RECYCLING_TARGETED': {**_OFF, 'TAX_EMISSION': 1, 'ECO_INVESTMENT_SUBSIDIES': 0.2,
                               'TARGETED_SUBSIDIES': True, 'CARBON_TAX_RECYCLING': True},
}
REFERENCE = 'NO_POLICY'

# Largest first within each seed, so the longest runs do not start last
CAPITALS = ['RIO DE JANEIRO', 'BELO HORIZONTE', 'PORTO ALEGRE', 'BRASILIA', 'RECIFE', 'FORTALEZA', 'SALVADOR',
            'CURITIBA', 'BELEM', 'GOIANIA', 'MANAUS', 'VITORIA', 'NATAL', 'SAO LUIS', 'MACEIO', 'JOAO PESSOA',
            'TERESINA', 'FLORIANOPOLIS', 'CAMPO GRANDE', 'CUIABA', 'ARACAJU', 'MACAPA', 'PORTO VELHO', 'BOA VISTA',
            'RIO BRANCO', 'PALMAS']
CITY_SETS = {'26': CAPITALS, 'sp': ['SAO PAULO']}

conf.RUN.update({
    'OUTPUT_PATH': OUTPUT,
    'KEEP_RANDOM_SEED': False,
    'SEED': SEED,
    'FORCE_NEW_POPULATION': False,
    'SAVE_DATA': ['firms'],
    'SAVE_DATA_PERIDIOCITY': 'ANNUALLY',
    'PLOT_EACH_RUN': False,
    'SAVE_PLOTS_FIGURES': False,
    'PRINT_STATISTICS_AND_RESULTS_DURING_PROCESS': False,
})

logger = logging.getLogger('main')


def check_defaults():
    """Settings the scenarios rely on without overriding them."""
    assert conf.PARAMS['TARGETED_SECTORS'] == ['Agriculture', 'Transport', 'Utilities'], \
        'conf/default/params.py TARGETED_SECTORS drifted from what this runner assumes'
    assert conf.PARAMS['CARBON_RECYCLING_QUANTILE'] == 0.25, \
        'conf/default/params.py CARBON_RECYCLING_QUANTILE drifted from what this runner assumes'


def policy_start(params=None):
    """First day of the month the eco policy starts in; stats.csv rows before it are policy-free."""
    p = {**conf.PARAMS, **(params or {})}
    d = p['STARTING_DAY'] + datetime.timedelta(p['ECO_POLICY_DAYS'])
    return datetime.date(d.year, d.month, 1)


def run_path(root, scenario, city, i):
    return os.path.join(root, scenario, city, str(i))


def queue(cities, indices):
    """(scenario, city, i) seed by seed, largest city first, so a batch stopped halfway holds complete sets of
    scenarios."""
    return [(s, city, i) for i in indices for city in cities for s in SCENARIOS]


# ── Population ────────────────────────────────────────────────────────────────

def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _population(root, city, params=None):
    """(Simulation for generating, cached population file, record file) of a city."""
    from simulation import Simulation
    p = copy.deepcopy(conf.PARAMS)
    p.update(params or {})
    p['PROCESSING_ACPS'] = [city]
    p['SEED'] = SEED
    gen_dir = os.path.join(root, 'populations', city)
    os.makedirs(gen_dir, exist_ok=True)
    sim = Simulation(p, gen_dir)
    return sim, '{}.agents'.format(sim.output.save_name), os.path.join(root, 'populations', city + '.json')


def prepare_populations(root, cities, params=None):
    """One population per city, shared by every scenario and seed. Generated from SEED the first time a batch in
    `root` meets the city (overwriting any older cache) and recorded by hash; later batches only check the hash."""
    for city in cities:
        sim, agents_file, record = _population(root, city, params)
        if os.path.exists(record):
            with open(record) as f:
                rec = json.load(f)
            if not os.path.exists(agents_file) or _sha256(agents_file) != rec['sha256']:
                sys.exit('Population of {} changed since {} ({}). Finished runs used the recorded one; restore it or '
                         'start a new OUTPUT.'.format(city, rec['created'], agents_file))
            logger.info('Population of %s: cached, matches %s', city, record)
            continue
        logger.info('Population of %s: generating from seed %d', city, SEED)
        if os.path.exists(agents_file):
            os.remove(agents_file)
        sim.generate()
        with open(record, 'w') as f:
            json.dump({'file': agents_file, 'sha256': _sha256(agents_file), 'seed': SEED,
                       'created': datetime.datetime.now().isoformat()}, f, indent=4)


def check_populations(root, cities, params=None):
    """Warn if a population file changed while the batch ran."""
    for city in cities:
        _, agents_file, record = _population(root, city, params)
        with open(record) as f:
            rec = json.load(f)
        if not os.path.exists(agents_file) or _sha256(agents_file) != rec['sha256']:
            logger.error('Population of %s changed during the batch: runs of %s may not share one population',
                         city, city)


# ── Batch ─────────────────────────────────────────────────────────────────────

def build_jobs(root, items, params=None):
    """Jobs for the unfinished items. Folders of unfinished runs are removed before their run is queued."""
    jobs = []
    for s, city, i in items:
        path = run_path(root, s, city, i)
        if os.path.exists(os.path.join(path, 'DONE')):
            continue
        if os.path.exists(path):
            shutil.rmtree(path)
        p = copy.deepcopy(conf.PARAMS)
        p.update(copy.deepcopy(SCENARIOS[s]))
        p.update(params or {})
        p['PROCESSING_ACPS'] = [city]
        p['SEED'] = SEED + i
        jobs.append({'params': p, 'path': path})
    return jobs


def run(root, items, cpus, manifest_dir, params=None):
    from main import save_jobs, _run_jobs_parallel
    cities = list(dict.fromkeys(city for _, city, _ in items))
    prepare_populations(root, cities, params)
    jobs = build_jobs(root, items, params)
    logger.info('%d of %d runs to do (finished ones skipped), %d at a time', len(jobs), len(items), cpus)
    save_jobs(manifest_dir, jobs, cpus)
    _run_jobs_parallel(jobs, cpus)
    check_populations(root, cities, params)
    summary(root, items)


def summary(root, items):
    missing = Counter((city, s) for s, city, i in items
                      if not os.path.exists(os.path.join(run_path(root, s, city, i), 'DONE')))
    logger.info('%d/%d runs finished', len(items) - sum(missing.values()), len(items))
    for (city, s), n in sorted(missing.items()):
        logger.error('  [FAIL] %-20s %-26s %d run(s) missing', city, s, n)
    if missing:
        logger.error('Rerun the same command to resume.')


def conferir(root, start=None):
    """Finished runs per scenario and city, their last month, and whether each run equals the NO_POLICY run with the
    same city and seed in every stats.csv row before `start` (default: the policy start)."""
    import glob
    import pandas as pd
    start = start or policy_start()
    rows = []
    for done in sorted(glob.glob(os.path.join(root, '*', '*', '*', 'DONE'))):
        d = os.path.dirname(done)
        s, city, i = d.split(os.sep)[-3:]
        if s not in SCENARIOS:
            continue
        last = pd.read_csv(os.path.join(d, 'stats.csv'), sep=';', header=None, usecols=[0]).iloc[-1, 0]
        rows.append((s, city, int(i), last))
    if not rows:
        print('No finished runs in', root)
        return
    t = pd.DataFrame(rows, columns=['scenario', 'city', 'run', 'last_month'])
    pd.set_option('display.width', 200)
    print(t.groupby('scenario').agg(cities=('city', 'nunique'), runs=('run', 'count'),
                                    last_month_min=('last_month', 'min')).to_string())
    print('\nTotal finished:', len(t))

    def before(s, city, i):
        x = pd.read_csv(os.path.join(run_path(root, s, city, i), 'stats.csv'), sep=';', header=None, dtype=str)
        return x[x[0] < str(start)].reset_index(drop=True)

    done = set(zip(t.scenario, t.city, t.run))
    differ, compared = [], 0
    for s, city, i in sorted(done):
        if s == REFERENCE or (REFERENCE, city, i) not in done:
            continue
        compared += 1
        if not before(s, city, i).equals(before(REFERENCE, city, i)):
            differ.append((s, city, i))
    print('identical to {} before {}: {} of {} compared'.format(REFERENCE, start, compared - len(differ), compared))
    for s, city, i in differ:
        print('  DIFFERS:', s, city, i)


def options(args):
    """key=value options of the lote mode."""
    opts = {'cities': '26', 'part': '1/1'}
    for a in args:
        k, sep, v = a.partition('=')
        if not sep or k not in opts:
            sys.exit('Unknown option {!r}. Options: {}'.format(a, ', '.join(k + '=' for k in opts)))
        opts[k] = v
    if opts['cities'] not in CITY_SETS:
        sys.exit('cities must be one of {}'.format(', '.join(CITY_SETS)))
    k, n = (int(x) for x in opts['part'].split('/'))
    if not 1 <= k <= n:
        sys.exit('part must be K/N with 1 <= K <= N')
    return opts['cities'], k, n


def log_to(manifest_dir):
    os.makedirs(manifest_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(manifest_dir, 'runner.log'), encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    logging.getLogger().addHandler(handler)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if LOCAL_CONF:
        print('Ignoring local overrides:', ', '.join(LOCAL_CONF))
    check_defaults()
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'conferir':
        conferir(sys.argv[2] if len(sys.argv) > 2 else OUTPUT)
    elif mode == 'teste':
        # Two years, with the policy from day 360 so every scenario's policy code runs
        root = OUTPUT + '_teste'
        test_params = {'TOTAL_DAYS': 730, 'ECO_POLICY_DAYS': 360}
        log_to(root)
        run(root, queue(['PALMAS'], [0]), int(sys.argv[2]), root, test_params)
        conferir(root, policy_start(test_params))
    elif mode == 'lote':
        cities, k, n = options(sys.argv[3:])
        items = queue(CITY_SETS[cities], list(range(RUNS)))
        # Contiguous slices of the queue, sizes differing by at most one
        items = items[(k - 1) * len(items) // n:k * len(items) // n]
        manifest_dir = os.path.join(OUTPUT, 'queue_{}_{}of{}'.format(cities, k, n))
        log_to(manifest_dir)
        logger.info('cities %s, part %d/%d, %d scenarios, policy from %s', cities, k, n, len(SCENARIOS),
                    policy_start())
        run(OUTPUT, items, int(sys.argv[2]), manifest_dir)
        conferir(OUTPUT)
    else:
        print(__doc__)
