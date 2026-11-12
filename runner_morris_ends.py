"""
Phase 2 of the screen of the behavioural parameters set in the code: the d25 - d09 effect of runner_mcmv2026 with
one factor at a time at the low and at the high end of its phase-1 range (runner_morris.FACTORS), the other factors
at the code values, plus four joint settings. 2010-2034, d09 and d25 at every setting, paired by seed.

Factors: the 19 that phase 1 (output/morris2026) found influential, mu* >= 0.2 band widths and >= 2 x seed noise
on a validation indicator in Goiania or Aracaju. Joint settings move a group of factors together, each to the end
whose phase-1 mu has the sign named below (mean over the two cities); factors whose sign differs between the two
cities stay at the code values:
    joint_housing_vac_up / _down   housing and construction factors, the ends that raise / lower vacancy
    joint_goods_u_up / _down       goods and labour factors, the ends that raise / lower unemployment
The code values are the mcmv2026 runs of the same city and seed (output/mcmv2026/{d09,d25}/<CITY>/<i>), so each
city runs on the server that ran mcmv2026 for it.

Saved: stats, regional and the families parquet every January.

Modes (cwd = repository root):
    python runner_morris_ends.py teste CPUS           Aracaju, two settings, d09 and d25, seed 0, 2 years, then the
                                                      check
    python runner_morris_ends.py lote CPUS [OPTIONS]  the batch; rerun the same command to resume (finished runs
                                                      skipped)
        cities=GOIANIA,ARACAJU  cities (default both)
    python runner_morris_ends.py conferir             finished runs per city and dose, d25 against d09 before 2026-01,
                                                      families parquet present

Output: output/morris2026_ends/<CITY>/<setting>/<dose>/<i>/; the queue manifest and log go to
output/morris2026_ends/queue/. Local conf/params.py and conf/run.py are ignored.
"""
import copy
import datetime
import glob
import logging
import os
import shutil
import sys

import runner_funding2026 as R
import runner_mcmv2026 as M
import runner_morris as MO

OUTPUT = 'output/morris2026_ends'
CITIES = ['GOIANIA', 'ARACAJU']
SEEDS = 5
DOSES = {'d09': .09, 'd25': .25}
SPLIT = datetime.date(2026, 1, 1)

HOUSING = ['VACANCY_PRICE_REFERENCE', 'OFFER_SIZE_ON_PRICE', 'LOT_COST', 'CONSTRUCTION_CAPITAL_MONTHS',
           'CONSTRUCTION_ACC_CASH_FLOW', 'MARKUP', 'MAX_HOUSE_STOCK', 'NEIGHBORHOOD_EFFECT', 'WEALTH_TARGET_MONTHS',
           'QLI_GROWTH_RATE', 'INITIAL_RENTAL_SHARE']
GOODS = ['SIZE_MARKET', 'INVENTORY_TARGET_RATIO', 'PRICE_DEMAND_RESPONSE', 'PRICE_MARKUP_CAP',
         'EXPORTS_PRICE_ELASTICITY', 'PRICE_RUGGEDNESS', 'FIRE_UNPAID_MONTHS', 'PCT_DISTANCE_HIRING']
# Sign of phase-1 mu: +1 the high end raises the indicator, -1 it lowers it, 0 the two cities disagree
SIGN_VACANCY = {'VACANCY_PRICE_REFERENCE': 1, 'OFFER_SIZE_ON_PRICE': 1, 'LOT_COST': -1,
                'CONSTRUCTION_CAPITAL_MONTHS': 1, 'CONSTRUCTION_ACC_CASH_FLOW': 1, 'MARKUP': 1, 'MAX_HOUSE_STOCK': 1,
                'NEIGHBORHOOD_EFFECT': 0, 'WEALTH_TARGET_MONTHS': 0, 'QLI_GROWTH_RATE': 1,
                'INITIAL_RENTAL_SHARE': 0}
SIGN_UNEMPLOYMENT = {'SIZE_MARKET': 1, 'INVENTORY_TARGET_RATIO': -1, 'PRICE_DEMAND_RESPONSE': 1,
                     'PRICE_MARKUP_CAP': 1, 'EXPORTS_PRICE_ELASTICITY': 1, 'PRICE_RUGGEDNESS': -1,
                     'FIRE_UNPAID_MONTHS': 0, 'PCT_DISTANCE_HIRING': -1}

R.conf.RUN['SAVE_DATA'] = ['family']
R.conf.RUN['SAVE_DATA_PERIDIOCITY'] = 'ANNUALLY'
logger = logging.getLogger('main')


def end_value(name, end):
    """The factor's low (end 0) or high (end 1) value, integers rounded as in phase 1."""
    _, lo, hi, integer = MO.FACTORS[name]
    v = hi if end else lo
    return int(round(v)) if integer else float(v)


def joint(signs, up):
    return {n: end_value(n, (s > 0) == up) for n, s in signs.items() if s}


def settings():
    """{setting: overrides}: <FACTOR>_lo, <FACTOR>_hi, then the four joint settings."""
    out = {}
    for n in HOUSING + GOODS:
        out[n + '_lo'] = {n: end_value(n, 0)}
        out[n + '_hi'] = {n: end_value(n, 1)}
    out['joint_housing_vac_up'] = joint(SIGN_VACANCY, True)
    out['joint_housing_vac_down'] = joint(SIGN_VACANCY, False)
    out['joint_goods_u_up'] = joint(SIGN_UNEMPLOYMENT, True)
    out['joint_goods_u_down'] = joint(SIGN_UNEMPLOYMENT, False)
    return out


def run_path(root, city, setting, dose, i):
    return os.path.join(root, city, setting, dose, str(i))


def queue(names, cities, seeds):
    """(city, setting, dose, i): seed by seed, setting by setting, cities alternating, d09 and d25 together."""
    return [(c, s, d, i) for i in seeds for s in names for c in cities for d in DOSES]


def build_jobs(root, items, sets):
    jobs = []
    for city, s, dose, i in items:
        path = run_path(root, city, s, dose, i)
        if os.path.exists(os.path.join(path, 'DONE')):
            continue
        if os.path.exists(path):
            shutil.rmtree(path)
        p = copy.deepcopy(R.conf.PARAMS)
        p.update(M.COMMON)
        p['OGU_INVESTMENT'] = {k: DOSES[dose] for k in ('otimista', 'tendencial', 'pessimista')}
        p.update(sets[s])
        p['PROCESSING_ACPS'] = [city]
        p['SEED'] = R.SEED + i
        jobs.append({'params': p, 'path': path})
    return jobs


def run(root, items, sets, cpus, manifest_dir, total_days=None):
    from main import save_jobs, _run_jobs_parallel
    jobs = build_jobs(root, items, sets)
    if total_days:
        for j in jobs:
            j['params']['TOTAL_DAYS'] = total_days
    logger.info('%d of %d runs to do (finished ones skipped), %d at a time', len(jobs), len(items), cpus)
    save_jobs(manifest_dir, jobs, cpus)
    _run_jobs_parallel(jobs, cpus)


def conferir(root, split=SPLIT):
    """Finished runs per city and dose, last month, d25 against d09 of the same setting and seed before `split`, and
    runs without a families parquet."""
    import pandas as pd
    rows = []
    for done in sorted(glob.glob(os.path.join(root, '*', '*', '*', '*', 'DONE'))):
        d = os.path.dirname(done)
        city, s, dose, i = d.split(os.sep)[-4:]
        last = pd.read_csv(os.path.join(d, 'stats.csv'), sep=';', header=None, usecols=[0]).iloc[-1, 0]
        rows.append((city, s, dose, int(i), last, os.path.exists(os.path.join(d, 'families.parquet'))))
    if not rows:
        print('No finished runs in', root)
        return
    t = pd.DataFrame(rows, columns=['city', 'setting', 'dose', 'run', 'last_month', 'families'])
    print(t.groupby(['city', 'dose']).agg(settings=('setting', 'nunique'), runs=('run', 'count'),
                                          last_month_min=('last_month', 'min'), families=('families', 'sum'))
          .to_string())

    def before(city, s, dose, i):
        x = pd.read_csv(os.path.join(run_path(root, city, s, dose, i), 'stats.csv'), sep=';', header=None,
                        dtype=str)
        return x[x[0] < str(split)].reset_index(drop=True)

    done = set(zip(t.city, t.setting, t.dose, t.run))
    pairs = [(c, s, i) for c, s, d, i in done if d == 'd25' and (c, s, 'd09', i) in done]
    differ = [p for p in sorted(pairs) if not before(p[0], p[1], 'd25', p[2]).equals(before(p[0], p[1], 'd09', p[2]))]
    print('d25 identical to d09 before {}: {} of {} pairs'.format(split, len(pairs) - len(differ), len(pairs)))
    for p in differ:
        print('  DIFFERS:', *p)
    for p in t[~t.families].itertuples(index=False):
        print('  NO FAMILIES PARQUET:', p.city, p.setting, p.dose, p.run)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if R.LOCAL_CONF:
        print('Ignoring local overrides:', ', '.join(R.LOCAL_CONF))
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    sets = settings()
    if mode == 'conferir':
        conferir(OUTPUT)
    elif mode == 'teste':
        root = OUTPUT + '_teste'
        two = datetime.date(2012, 1, 1)
        run(root, queue([HOUSING[0] + '_lo', 'joint_housing_vac_up'], ['ARACAJU'], [0]), sets, int(sys.argv[2]),
            root, (two - R.conf.PARAMS['STARTING_DAY']).days)
        conferir(root, two)
    elif mode == 'lote':
        opts = dict(a.partition('=')[::2] for a in sys.argv[3:])
        unknown = set(opts) - {'cities'}
        if unknown:
            sys.exit('Unknown option(s) {}. Options: cities='.format(', '.join(sorted(unknown))))
        cities = opts.get('cities', ','.join(CITIES)).split(',')
        items = queue(list(sets), cities, list(range(SEEDS)))
        manifest_dir = os.path.join(OUTPUT, 'queue')
        os.makedirs(manifest_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(manifest_dir, 'runner.log'), encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        logger.info('cities %s, %d settings, %d runs', cities, len(sets), len(items))
        run(OUTPUT, items, sets, int(sys.argv[2]), manifest_dir)
        conferir(OUTPUT)
    else:
        print(__doc__)
