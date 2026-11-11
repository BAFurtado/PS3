"""
Morris elementary-effects screen of the behavioural parameters the SI lists as set in the code, in the d09 setting
of runner_mcmv2026 (metro pot, commissioned units, OGU tendencial), 2010-2019, stats and regional only.

Design: r trajectories of k + 1 points on a p-level grid of the unit cube (Morris 1991, the B* formula of Saltelli
et al. 2004, ch. 4), chosen from a larger set of candidates by the greedy spread rule of Campolongo et al. (2007).
Each factor is uniform on [low, high] of FACTORS, integer factors rounded. Every point runs with the same seeds
(SEED + i, i < SEEDS) in every city, so the effects between two points of a trajectory share their random numbers.

Modes (cwd = repository root):
    python runner_morris.py desenho              writes analysis/morris/design.csv and problem.json (once; the batch
                                                 on every server reads these files)
    python runner_morris.py teste CPUS           the default point and the first point of the design in Aracaju,
                                                 seed 0, 2 years, then the check
    python runner_morris.py lote CPUS [OPTIONS]  the batch; rerun the same command to resume (finished runs skipped)
        cities=GOIANIA,ARACAJU  cities (default both)
        part=K/N                the K-th of N contiguous slices of the queue, for N servers (default 1/1)
    python runner_morris.py conferir             finished runs per city, and the default point against
                                                 output/mcmv2026/d09 before 2020-01

Output: output/morris2026/<CITY>/<point>/<i>/, point = base (the code's values) or p000 ... ; the queue manifest and
log go to output/morris2026/queue_<K>of<N>/. Local conf/params.py and conf/run.py are ignored.
"""
import copy
import datetime
import json
import logging
import os
import shutil
import sys

import numpy as np

import runner_funding2026 as R
import runner_mcmv2026 as M

OUTPUT = 'output/morris2026'
DESIGN_DIR = 'analysis/morris'
END = datetime.date(2020, 1, 1)
CITIES = ['GOIANIA', 'ARACAJU']
SEEDS = 2
TRAJECTORIES = 10
LEVELS = 4
CANDIDATES = 100
DESIGN_SEED = 20261111
D09 = {'OGU_INVESTMENT': {'otimista': .09, 'tendencial': .09, 'pessimista': .09}}

# name: (code value, low, high, integer)
FACTORS = {
    'WEALTH_TARGET_MONTHS': (4.5, 2.25, 6.75, False),
    'WEALTH_ADJUSTMENT': (1 / 12, 1 / 24, 1 / 8, False),
    'PRIVATE_TRANSIT_COST': (.25, .125, .375, False),
    'PUBLIC_TRANSIT_COST': (.05, .025, .075, False),
    'SAVINGS_BUFFER_MONTHS': (2, 1, 3, False),
    'SIZE_MARKET': (5, 2.5, 7.5, True),
    'INTERMEDIATE_SIZE_MARKET': (10, 5, 15, True),
    'INVENTORY_TARGET_RATIO': (.2, .1, .3, False),
    'EXPORTS_PRICE_ELASTICITY': (1., .5, 2., False),
    'MARKUP': (.1, .05, .15, False),
    'PRICE_RUGGEDNESS': (.1, .05, .15, False),
    'PRICE_MARKUP_CAP': (.0875, .04375, .13125, False),
    'PRICE_DEMAND_RESPONSE': (.1, .05, .15, False),
    'FIRM_CAPITAL_MONTHS': (3, 1, 5, True),
    'CONSTRUCTION_CAPITAL_MONTHS': (12, 6, 18, True),
    'CONSTRUCTION_ACC_CASH_FLOW': (12, 6, 18, True),
    'FIRM_EXIT_MONTHS': (6, 3, 9, True),
    'FIRE_UNPAID_MONTHS': (3, 1, 5, True),
    'PCT_DISTANCE_HIRING': (.2, .1, .3, False),
    'CB_COMMUTING': (.2, .1, .3, False),
    'HIRING_SAMPLE_SIZE': (20, 10, 30, True),
    'ON_MARKET_DECAY_FACTOR': (-.02, -.03, -.01, False),
    'MAX_OFFER_DISCOUNT': (.65, .5, .8, False),
    'MAX_OFFER_PREMIUM': (1.3, 1.15, 1.45, False),
    'NEIGHBORHOOD_EFFECT': (.2, .1, .3, False),
    'VACANCY_PRICE_REFERENCE': (.08, .04, .12, False),
    'OFFER_SIZE_ON_PRICE': (5, .8, 7.1, False),
    'MIN_DOWN_PAYMENT_FRACTION': (.2, .1, .3, False),
    'HOUSING_FINANCIAL_WEIGHT': (60, 30, 90, False),
    'LIQUIDITY_BUFFER_MONTHS': (6, 3, 9, False),
    'PERCENTAGE_ENTERING_ESTATE_MARKET': (.005, .0025, .0075, False),
    'INITIAL_RENTAL_SHARE': (.4, .2, .6, False),
    'MAX_RENT_TO_INCOME_RATIO': (.3, .15, .45, False),
    'CAPPED_TOP_VALUE': (1.3, 1.15, 1.45, False),
    'CAPPED_LOW_VALUE': (.7, .5, .9, False),
    'EXPECTED_LICENSES_PER_REGION': (1.5, .75, 2.25, False),
    'LICENSE_MIN_CITY_MONTHLY': (10, 5, 15, False),
    'MAX_HOUSE_STOCK': (36, 18, 54, True),
    'CONSTRUCTION_FIRM_MARKUP_MULTIPLIER': (5, 2.5, 7.5, False),
    'LOT_COST': (.15, .075, .225, False),
    'MAX_LOAN_BANK_PERCENT': (.6, .3, .9, False),
    'QLI_GROWTH_RATE': (.002, .001, .003, False),
    'QLI_GDP_NORM': (3.5, 1.75, 5.25, False),
}
COMMON = dict(M.COMMON, TOTAL_DAYS=(END - R.conf.PARAMS['STARTING_DAY']).days)
R.conf.RUN['SAVE_DATA'] = []
logger = logging.getLogger('main')


def trajectory(rng, k, p):
    """One Morris trajectory, (k + 1) x k on the unit grid {0, 1/(p-1), ..., 1}, step delta = p / (2 (p - 1))."""
    delta = p / (2 * (p - 1))
    B = np.tril(np.ones((k + 1, k)), -1)
    J = np.ones((k + 1, k))
    D = np.diag(rng.choice([-1, 1], k))
    x = rng.integers(0, p // 2, k) / (p - 1)
    P = np.eye(k)[rng.permutation(k)]
    return (J * x + delta / 2 * ((2 * B - J) @ D + J)) @ P


def choose(cands, r):
    """Greedy pick of r trajectories that maximise the summed distance between them (Campolongo et al. 2007)."""
    def dist(a, b):
        return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).sum()
    n = len(cands)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = dist(cands[i], cands[j])
    first = np.unravel_index(d.argmax(), d.shape)
    picked = list(first)
    while len(picked) < r:
        rest = [i for i in range(n) if i not in picked]
        picked.append(max(rest, key=lambda i: d[i, picked].sum()))
    return [cands[i] for i in picked]


def to_values(u):
    """Unit-cube row to parameter values, integers rounded."""
    out = {}
    for x, (name, (_, lo, hi, integer)) in zip(u, FACTORS.items()):
        v = lo + x * (hi - lo)
        out[name] = int(round(v)) if integer else float(v)
    return out


def desenho():
    import pandas as pd
    os.makedirs(DESIGN_DIR, exist_ok=True)
    path = os.path.join(DESIGN_DIR, 'design.csv')
    if os.path.exists(path):
        sys.exit('{} exists; the batch reads it. Remove it by hand to draw a new design.'.format(path))
    rng = np.random.default_rng(DESIGN_SEED)
    k = len(FACTORS)
    trajs = choose([trajectory(rng, k, LEVELS) for _ in range(CANDIDATES)], TRAJECTORIES)
    rows = []
    for t, T in enumerate(trajs):
        for j, u in enumerate(T):
            rows.append(dict(point='p{:03d}'.format(t * (k + 1) + j), trajectory=t, step=j,
                             **{'u_' + n: x for n, x in zip(FACTORS, u)}, **to_values(u)))
    pd.DataFrame(rows).to_csv(path, index=False)
    with open(os.path.join(DESIGN_DIR, 'problem.json'), 'w') as f:
        json.dump(dict(factors={n: dict(code=c, low=lo, high=hi, integer=i) for n, (c, lo, hi, i) in FACTORS.items()},
                       trajectories=TRAJECTORIES, levels=LEVELS, candidates=CANDIDATES, design_seed=DESIGN_SEED,
                       seeds=SEEDS, cities=CITIES, end=str(END)), f, indent=1)
    print('{} points ({} trajectories x {} factors + 1) written to {}'.format(len(rows), TRAJECTORIES, k, path))


def points():
    """{point: overrides}, the code's values first."""
    import pandas as pd
    path = os.path.join(DESIGN_DIR, 'design.csv')
    if not os.path.exists(path):
        sys.exit('No design: run "python runner_morris.py desenho" first and copy analysis/morris/ to every server.')
    d = pd.read_csv(path)
    out = {'base': {}}
    for row in d.itertuples(index=False):
        r = row._asdict()
        out[r['point']] = {n: (int(r[n]) if FACTORS[n][3] else float(r[n])) for n in FACTORS}
    return out


def queue(pts, cities, seeds):
    """(city, point, i): seed by seed, then point by point, Goiania and Aracaju alternating."""
    return [(c, p, i) for i in seeds for p in pts for c in cities]


def build_jobs(root, items, pts, params):
    jobs = []
    for city, point, i in items:
        path = os.path.join(root, city, point, str(i))
        if os.path.exists(os.path.join(path, 'DONE')):
            continue
        if os.path.exists(path):
            shutil.rmtree(path)
        p = copy.deepcopy(R.conf.PARAMS)
        p.update(copy.deepcopy(D09))
        p.update(params)
        p.update(pts[point])
        p['PROCESSING_ACPS'] = [city]
        p['SEED'] = R.SEED + i
        jobs.append({'params': p, 'path': path})
    return jobs


def run(root, items, pts, cpus, manifest_dir, params):
    from main import save_jobs, _run_jobs_parallel
    jobs = build_jobs(root, items, pts, params)
    logger.info('%d of %d runs to do (finished ones skipped), %d at a time', len(jobs), len(items), cpus)
    save_jobs(manifest_dir, jobs, cpus)
    _run_jobs_parallel(jobs, cpus)


def conferir(root, end=END):
    """Finished runs per city and point count; the base point against the d09 run of the same city and seed."""
    import glob
    import pandas as pd
    done = sorted(glob.glob(os.path.join(root, '*', '*', '*', 'DONE')))
    t = pd.DataFrame([os.path.dirname(d).split(os.sep)[-3:] for d in done], columns=['city', 'point', 'run'])
    if t.empty:
        print('No finished runs in', root)
        return
    print(t.groupby('city').agg(points=('point', 'nunique'), runs=('run', 'count')).to_string())
    for city, _, i in t[t.point == 'base'].itertuples(index=False):
        ref = os.path.join(M.OUTPUT, 'd09', city, i, 'stats.csv')
        if not os.path.exists(ref):
            print('base {} {}: no d09 run to compare'.format(city, i))
            continue
        a = pd.read_csv(os.path.join(root, city, 'base', i, 'stats.csv'), sep=';', header=None, dtype=str)
        b = pd.read_csv(ref, sep=';', header=None, dtype=str)
        b = b[b[0] < str(end)].reset_index(drop=True).iloc[:len(a)]
        same = (a.values == b.values).all(axis=1)
        print('base {} {}: {} of {} months identical to d09{}'.format(
            city, i, same.sum(), len(a), '' if same.all() else ', first difference ' + a[0][~same].iloc[0]))


def options(args):
    opts = {'cities': ','.join(CITIES), 'part': '1/1'}
    for a in args:
        key, sep, v = a.partition('=')
        if not sep or key not in opts:
            sys.exit('Unknown option {!r}. Options: {}'.format(a, ', '.join(k + '=' for k in opts)))
        opts[key] = v
    k, n = (int(x) for x in opts['part'].split('/'))
    if not 1 <= k <= n:
        sys.exit('part must be K/N with 1 <= K <= N')
    return opts['cities'].split(','), k, n


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if R.LOCAL_CONF:
        print('Ignoring local overrides:', ', '.join(R.LOCAL_CONF))
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'desenho':
        desenho()
    elif mode == 'conferir':
        conferir(OUTPUT)
    elif mode == 'teste':
        root = OUTPUT + '_teste'
        pts = points()
        two = datetime.date(2012, 1, 1)
        run(root, queue(['base', 'p000'], ['ARACAJU'], [0]), pts, int(sys.argv[2]), root,
            dict(COMMON, TOTAL_DAYS=(two - R.conf.PARAMS['STARTING_DAY']).days))
        conferir(root, two)
    elif mode == 'lote':
        cities, k, n = options(sys.argv[3:])
        pts = points()
        items = queue(list(pts), cities, list(range(SEEDS)))
        items = items[(k - 1) * len(items) // n:k * len(items) // n]
        manifest_dir = os.path.join(OUTPUT, 'queue_{}of{}'.format(k, n))
        os.makedirs(manifest_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(manifest_dir, 'runner.log'), encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        logger.info('cities %s, part %d/%d, %d runs', cities, k, n, len(items))
        run(OUTPUT, items, pts, int(sys.argv[2]), manifest_dir, COMMON)
        conferir(OUTPUT)
    else:
        print(__doc__)
