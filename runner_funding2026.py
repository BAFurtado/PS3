"""
Funding from 2026: every scenario x city x seed in one queue, 2010-2039.

Groups and scenarios (each scenario changes only the money; the main and both scenarios differ from base only from
2026-01, the melhorias pair differ from each other only from 2026-01):
    main       base      tendencial
               ogu       MCMV subsidy (OGU) at .25 of municipal GDP from 2026, credit lines tendencial
               credito   FGTS and SBPE lines otimista, OGU tendencial (.09)
    melhorias  melh_base MCMV off, melhorias on, OGU tendencial
               melh_ogu  MCMV off, melhorias on, OGU .25 from 2026
    both       both      FUNDS_AVAILABILITY otimista (OGU .25 and otimista credit lines)
City sets: '26' = the 26 capitals without Sao Paulo (main and both), with melhorias in Macapa, Belem and Goiania;
'sp' = Sao Paulo (main and both). Run i of every scenario and city uses seed SEED + i. Each run builds its population
from its own seed.

Modes (cwd = repository root):
    python runner_funding2026.py teste CPUS             every scenario in Palmas, 1 seed, 2 years, then the checks
    python runner_funding2026.py lote CPUS [OPTIONS]    the batch; rerun the same command to resume (finished runs,
                                                        marked DONE, are skipped)
        cities=26|sp            city set (default 26)
        groups=main,melhorias,both   groups, queued in this order (default all)
        part=K/N                the K-th of N contiguous slices of the queue, for N servers (default 1/1)
    python runner_funding2026.py conferir [FOLDER]      finished runs per scenario and city, identity before 2026-01

Output: output/funding2026/<scenario>/<CITY>/<i>/, the same tree on every server; the queue manifest and the log of
each batch go to output/funding2026/queue_<cities>_<K>of<N>/. Local conf/params.py and conf/run.py are ignored: every
value comes from conf/default plus the overrides below.
"""
import copy
import datetime
import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import conf  # noqa: E402

LOCAL_CONF = [f for f in ('conf/params.py', 'conf/run.py') if os.path.exists(f)]
for _n in ('params', 'run'):
    conf.load_conf_module('default.{}'.format(_n), key=_n)

SEED = 1000
RUNS = 5
START = datetime.date(2026, 1, 1)
OUTPUT = 'output/funding2026'

OGU_HIGH = {'otimista': .25, 'tendencial': .25, 'pessimista': .04}
OGU_TENDENCIAL = {'otimista': .09, 'tendencial': .09, 'pessimista': .04}
SCENARIOS = {
    'base': {},
    'ogu': {'OGU_INVESTMENT': OGU_HIGH},
    'credito': {'FUNDS_AVAILABILITY': 'otimista', 'OGU_INVESTMENT': OGU_TENDENCIAL},
    'melh_base': {'POLICY_MCMV': False, 'POLICY_MELHORIAS': True},
    'melh_ogu': {'POLICY_MCMV': False, 'POLICY_MELHORIAS': True, 'OGU_INVESTMENT': OGU_HIGH},
    'both': {'FUNDS_AVAILABILITY': 'otimista'},
}

# Largest first within each seed, so the longest runs do not start last
CAPITALS = ['RIO DE JANEIRO', 'BELO HORIZONTE', 'PORTO ALEGRE', 'BRASILIA', 'RECIFE', 'FORTALEZA', 'SALVADOR',
            'CURITIBA', 'BELEM', 'GOIANIA', 'MANAUS', 'VITORIA', 'NATAL', 'SAO LUIS', 'MACEIO', 'JOAO PESSOA',
            'TERESINA', 'FLORIANOPOLIS', 'CAMPO GRANDE', 'CUIABA', 'ARACAJU', 'MACAPA', 'PORTO VELHO', 'BOA VISTA',
            'RIO BRANCO', 'PALMAS']
MELHORIAS_CITIES = ['BELEM', 'GOIANIA', 'MACAPA']
CITY_SETS = {
    '26': {'main': CAPITALS, 'melhorias': MELHORIAS_CITIES, 'both': CAPITALS},
    'sp': {'main': ['SAO PAULO'], 'melhorias': [], 'both': ['SAO PAULO']},
}
GROUPS = {'main': ['base', 'ogu', 'credito'], 'melhorias': ['melh_base', 'melh_ogu'], 'both': ['both']}
# Scenario whose runs every other scenario of the group must equal before START
REFERENCE = {'base': 'base', 'ogu': 'base', 'credito': 'base', 'both': 'base',
             'melh_base': 'melh_base', 'melh_ogu': 'melh_base'}

conf.RUN.update({
    'OUTPUT_PATH': OUTPUT,
    'KEEP_RANDOM_SEED': False,
    'SEED': SEED,
    'FORCE_NEW_POPULATION': True,
    'SAVE_DATA': ['firms'],
    'PLOT_EACH_RUN': False,
    'SAVE_PLOTS_FIGURES': False,
    'PRINT_STATISTICS_AND_RESULTS_DURING_PROCESS': False,
})

logger = logging.getLogger('main')


def run_path(root, scenario, city, i):
    return os.path.join(root, scenario, city, str(i))


def queue(cities, groups, indices):
    """(scenario, city, i) group by group, seed by seed within a group, so a batch stopped halfway holds complete
    sets of scenarios."""
    return [(s, city, i)
            for g in groups
            for i in indices
            for city in CITY_SETS[cities][g]
            for s in GROUPS[g]]


def build_jobs(root, items, params=None):
    """Jobs for the unfinished items. Folders of unfinished runs are removed before their run is queued."""
    params = params or {}
    jobs = []
    for s, city, i in items:
        path = run_path(root, s, city, i)
        if os.path.exists(os.path.join(path, 'DONE')):
            continue
        if os.path.exists(path):
            shutil.rmtree(path)
        p = copy.deepcopy(conf.PARAMS)
        p.update(copy.deepcopy(SCENARIOS[s]))
        p.update(params)
        p['PROCESSING_ACPS'] = [city]
        p['SEED'] = SEED + i
        jobs.append({'params': p, 'path': path})
    return jobs


def run(root, items, cpus, manifest_dir, params=None):
    from main import save_jobs, _run_jobs_parallel
    jobs = build_jobs(root, items, params)
    logger.info('%d of %d runs to do (finished ones skipped), %d at a time', len(jobs), len(items), cpus)
    save_jobs(manifest_dir, jobs, cpus)
    _run_jobs_parallel(jobs, cpus)


def conferir(root, start=START):
    """Finished runs per scenario and city, their last month, and whether each run equals the run of its reference
    scenario with the same city and seed in every stats.csv row before `start`."""
    import glob
    import pandas as pd
    rows = []
    for done in sorted(glob.glob(os.path.join(root, '*', '*', '*', 'DONE'))):
        d = os.path.dirname(done)
        s, city, i = d.split(os.sep)[-3:]
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
        ref = REFERENCE[s]
        if ref == s or (ref, city, i) not in done:
            continue
        compared += 1
        if not before(s, city, i).equals(before(ref, city, i)):
            differ.append((s, city, i))
    print('identical to the reference scenario before {}: {} of {} compared'.format(
        start, compared - len(differ), compared))
    for s, city, i in differ:
        print('  DIFFERS:', s, city, i)


def options(args):
    """key=value options of the lote mode."""
    opts = {'cities': '26', 'groups': ','.join(GROUPS), 'part': '1/1'}
    for a in args:
        k, sep, v = a.partition('=')
        if not sep or k not in opts:
            sys.exit('Unknown option {!r}. Options: {}'.format(a, ', '.join(k + '=' for k in opts)))
        opts[k] = v
    groups = opts['groups'].split(',')
    unknown = [g for g in groups if g not in GROUPS]
    if opts['cities'] not in CITY_SETS or unknown:
        sys.exit('cities must be one of {}; groups among {}'.format(', '.join(CITY_SETS), ', '.join(GROUPS)))
    k, n = (int(x) for x in opts['part'].split('/'))
    if not 1 <= k <= n:
        sys.exit('part must be K/N with 1 <= K <= N')
    return opts['cities'], groups, k, n


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if LOCAL_CONF:
        print('Ignoring local overrides:', ', '.join(LOCAL_CONF))
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'conferir':
        conferir(sys.argv[2] if len(sys.argv) > 2 else OUTPUT)
    elif mode == 'teste':
        root = OUTPUT + '_teste'
        for g in GROUPS:
            CITY_SETS['26'][g] = ['PALMAS']
        conf.RUN['SAVE_DATA'] = []
        run(root, queue('26', list(GROUPS), [0]), int(sys.argv[2]), root, {'TOTAL_DAYS': 730})
        conferir(root, datetime.date(2012, 1, 1))
    elif mode == 'lote':
        cities, groups, k, n = options(sys.argv[3:])
        items = queue(cities, groups, list(range(RUNS)))
        # Contiguous slices of the queue, sizes differing by at most one
        items = items[(k - 1) * len(items) // n:k * len(items) // n]
        manifest_dir = os.path.join(OUTPUT, 'queue_{}_{}of{}'.format(cities, k, n))
        os.makedirs(manifest_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(manifest_dir, 'runner.log'), encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        logger.info('cities %s, groups %s, part %d/%d', cities, ','.join(groups), k, n)
        run(OUTPUT, items, int(sys.argv[2]), manifest_dir)
        conferir(OUTPUT)
    else:
        print(__doc__)
