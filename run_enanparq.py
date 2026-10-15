"""ENANPARQ 9 (Oct 2026): Belo Horizonte, five arms x N paired seeds, on tag ps3-model-2026 plus the world/funds.py fix.

Arms (each run 2010-2039, all with MCMV on as in the tag's defaults):
    tendencial   FUNDS_AVAILABILITY tendencial, no municipal policy (reference)
    otimista     FUNDS_AVAILABILITY otimista (more OGU subsidy for MCMV and larger FGTS/SBPE lines from 2026)
    buy          tendencial + POLICIES buy  (municipality buys new houses for the poorest renters, as MCMV faixa 1)
    rent         tendencial + POLICIES rent (24 months of rent paid up front)
    wage         tendencial + POLICIES wage (pot split equally among the poorest 20 %)
POLICY_COEFFICIENT 0.2 for buy/rent/wage (PS2 standard run: 20 % of the money, poorest 20 % eligible).
Seeds 20261015 + i, shared by the five arms, so each arm is compared with tendencial of the same seed.

Modes (cwd = repository root):
    python run_enanparq.py populacao              regenerate the shared BH population once (serial, ~5 min)
    python run_enanparq.py teste CPUS             five arms x 1 seed x 2 years, then prints a check table
    python run_enanparq.py lote N CPUS [FIRST]    five arms x seeds FIRST..FIRST+N-1 (FIRST defaults to 0); rerun the
                                                  same command to resume, a larger N or a FIRST to add seeds
    python run_enanparq.py conferir [PASTA]       count finished runs and their last month

Local conf/params.py and conf/run.py are ignored: every value comes from conf/default plus the overrides below.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import conf  # noqa: E402

locais = [f for f in ('conf/params.py', 'conf/run.py') if os.path.exists(f)]
for n in ('params', 'run'):
    conf.load_conf_module('default.{}'.format(n), key=n)
if locais:
    print('Ignoring local overrides (defaults of the tag are used):', ', '.join(locais))

SEMENTE = 20261015
SAIDA = 'output/enanparq_server'
BRACOS = {
    'tendencial': {'FUNDS_AVAILABILITY': 'tendencial', 'POLICIES': 'no_policy', 'POLICY_COEFFICIENT': 0},
    'otimista': {'FUNDS_AVAILABILITY': 'otimista', 'POLICIES': 'no_policy', 'POLICY_COEFFICIENT': 0},
    'buy': {'FUNDS_AVAILABILITY': 'tendencial', 'POLICIES': 'buy', 'POLICY_COEFFICIENT': 0.2},
    'rent': {'FUNDS_AVAILABILITY': 'tendencial', 'POLICIES': 'rent', 'POLICY_COEFFICIENT': 0.2},
    'wage': {'FUNDS_AVAILABILITY': 'tendencial', 'POLICIES': 'wage', 'POLICY_COEFFICIENT': 0.2},
}

conf.PARAMS['PROCESSING_ACPS'] = ['BELO HORIZONTE']
conf.RUN.update({
    'OUTPUT_PATH': SAIDA,
    'KEEP_RANDOM_SEED': False,
    'SEED': SEMENTE,
    'FORCE_NEW_POPULATION': False,
    'SAVE_DATA': ['firms', 'house', 'family'],
    'SAVE_DATA_PERIDIOCITY': 'ANNUALLY',
    'PLOT_EACH_RUN': False,
    'SAVE_PLOTS_FIGURES': False,
    'PRINT_STATISTICS_AND_RESULTS_DURING_PROCESS': False,
})


def conferir(pasta):
    import glob
    import pandas as pd
    feitos = sorted(glob.glob(os.path.join(pasta, '*', '*', 'DONE')))
    linhas = []
    for done in feitos:
        d = os.path.dirname(done)
        ultimo = pd.read_csv(os.path.join(d, 'stats.csv'), sep=';', header=None, usecols=[0]).iloc[-1, 0]
        linhas.append((os.path.basename(os.path.dirname(d)), int(os.path.basename(d)), ultimo))
    if not linhas:
        print('No finished runs in', pasta)
        return
    t = pd.DataFrame(linhas, columns=['arm', 'run', 'last_month'])
    print(t.groupby('arm').agg(runs=('run', 'count'), last_month_min=('last_month', 'min')).to_string())
    print('\nTotal finished:', len(t))
    # The arms differ only from 2026: runs of the same seed must be identical before
    quebrados = []
    for i, g in t.groupby('run'):
        antes = [pd.read_csv(os.path.join(pasta, a, str(i), 'stats.csv'), sep=';', header=None)
                 for a in g.arm]
        antes = [x[x[0] < '2026-01-01'].reset_index(drop=True) for x in antes]
        if any(not x.equals(antes[0]) for x in antes[1:]):
            quebrados.append(i)
    print('identical up to 2025-12:', 'all seeds' if not quebrados else 'NOT for runs {}'.format(quebrados))


def tabela_teste(pasta):
    import glob
    import pandas as pd
    from analysis.output import OUTPUT_DATA_SPEC
    cols = ['rent_default', 'families_helped', 'amount_subsidised', 'number_domiciles']
    out = {}
    for d in sorted(glob.glob(os.path.join(pasta, '*', '0'))):
        s = pd.read_csv(os.path.join(d, 'stats.csv'), sep=';', header=None,
                        names=OUTPUT_DATA_SPEC['stats']['columns'])
        out[os.path.basename(os.path.dirname(d))] = s[cols].tail(12).mean()
    pd.set_option('display.width', 200)
    print('\nLast 12 months of the test (buy/rent/wage must show more families_helped than tendencial):')
    print(pd.DataFrame(out).T.to_string())


def rodar(pasta, indices, cpus):
    """Arms x seeds, queued seed by seed so that a batch stopped halfway still holds complete pairs. Run i of every arm
    uses seed SEMENTE + i and lives in <arm>/<i>/; finished runs (DONE) are skipped, so the same command resumes."""
    from main import ensure_population_exists, save_jobs, _run_jobs_parallel, main_plotting
    params = {}
    for b, o in BRACOS.items():
        params[b] = copy.deepcopy(conf.PARAMS)
        params[b].update(o)
        # Population parameters are the same in every arm: the first call generates it if missing, the rest reuse it
        ensure_population_exists(params[b], os.path.join(pasta, main_plotting.conf_to_str(o)))
    jobs = []
    for i in indices:
        for b, o in BRACOS.items():
            caminho = os.path.join(pasta, main_plotting.conf_to_str(o), str(i))
            if os.path.exists(os.path.join(caminho, 'DONE')):
                continue
            p = copy.deepcopy(params[b])
            p['SEED'] = SEMENTE + i
            jobs.append({'params': p, 'path': caminho})
    print('{} runs to do ({} arms x seeds {}..{}, finished ones skipped), {} at a time'.format(
        len(jobs), len(BRACOS), indices[0], indices[-1], cpus))
    save_jobs(pasta, jobs, cpus)
    _run_jobs_parallel(jobs, cpus)


if __name__ == '__main__':
    modo = sys.argv[1] if len(sys.argv) > 1 else ''
    if modo == 'conferir':
        conferir(sys.argv[2] if len(sys.argv) > 2 else os.path.join(SAIDA, 'lote'))
    elif modo == 'populacao':
        from main import ensure_population_exists
        conf.RUN['FORCE_NEW_POPULATION'] = True
        p = copy.deepcopy(conf.PARAMS)
        p.update(BRACOS['tendencial'])
        ensure_population_exists(p, os.path.join(SAIDA, 'populacao'))
        print('Population generated.')
    elif modo == 'teste':
        conf.PARAMS['TOTAL_DAYS'] = 730
        conf.RUN['SAVE_DATA'] = []
        pasta = os.path.join(SAIDA, 'teste')
        rodar(pasta, [0], int(sys.argv[2]))
        tabela_teste(pasta)
    elif modo == 'lote':
        n, cpus = int(sys.argv[2]), int(sys.argv[3])
        primeira = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        rodar(os.path.join(SAIDA, 'lote'), list(range(primeira, primeira + n)), cpus)
        conferir(os.path.join(SAIDA, 'lote'))
    else:
        print(__doc__)
