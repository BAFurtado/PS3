"""Phase 2 of the screen (runner_morris_ends.py): the d25 - d09 effect at each factor end and joint setting, against
the code values and against the spread of the same effect across the 27 areas.
usage (repo root, env ps3_2): python analysis/morris/ends.py [ends root, default output/morris2026_ends]
                                                                     [mcmv2026 root, default output/mcmv2026]
Effects, d25 against d09 at matched setting and seed, ratio of seed means:
    dwellings, families, residents, real GDP     % of d09, mean of 2026-34
    unemployment                                 pp, mean of 2026-34
    price_all_after                              % of d09, mean house price over every dwelling / price level, 2031-34
    price_nonmcmv_after                          % of d09, dwellings of families outside MCMV, January 2032-34
                                                 (as price_composition.py)
se: sd over seeds of the per-seed paired effect / sqrt(seeds).
Code values: the mcmv2026 runs of the same city, seeds 0 to SEEDS - 1. 27-area reference: every mcmv2026 area, all
seeds, same definitions (p10, median, p90, width = p90 - p10, areas with a negative effect).
Criteria fixed 2026-11-12 before the runs:
    1. dwellings and families stay positive;
    2. shift from the code values over the 27-area width, |shift| / width > 1 flagged;
    3. pooled over settings, the share with a negative price effect, against the 27-area share;
    4. a flag names a condition of the result, not a verdict on the model.
Writes <ends root>/_analysis/{areas27.csv, ends_runs.csv, ends_effects.csv}."""
import datetime
import glob
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path[:0] = ['.']
from analysis.output import columns_for  # noqa: E402

SEEDS = 5
COLS = ['month', 'pop', 'price_level', 'gdp_level', 'unemployment', 'number_domiciles', 'families_count',
        'house_price']
REL = ['dwellings', 'families', 'residents', 'real_gdp', 'price_all_after', 'price_nonmcmv_after']
ABS = ['unemployment']
VARS = REL + ABS
POSITIVE = ['dwellings', 'families']
PRICE = ['price_all_after', 'price_nonmcmv_after']


def stats(d):
    f = os.path.join(d, 'stats.csv')
    n = len(next(line for line in open(f) if line.strip()).split(';'))
    x = pd.read_csv(f, sep=';', header=None, names=columns_for('stats', n), usecols=COLS)
    x['month'] = x.month.astype(str)
    return x


def run_metrics(d):
    """Levels of one run; price_nonmcmv as the sum and count of January prices, so seeds pool as in
    price_composition.py."""
    x = stats(d)
    w = (x.month >= '2026-01-01') & (x.month <= '2034-12-31')
    a = (x.month >= '2031-01-01') & (x.month <= '2034-12-31')
    p = dict(zip(x.month.str[:7], x.price_level))
    fa = pd.read_parquet(os.path.join(d, 'families.parquet'), columns=['month', 'house_price', 'programme'],
                         filters=[('month', '>=', datetime.date(2032, 1, 1)),
                                  ('month', '<=', datetime.date(2034, 12, 31))])
    m = fa.month.astype(str).str[:7]
    fa = fa[m.str.endswith('-01') & (fa.programme != 'mcmv')]
    price = fa.house_price / m[fa.index].map(p)
    return dict(dwellings=x.number_domiciles[w].mean(), families=x.families_count[w].mean(),
                residents=x['pop'][w].mean(), real_gdp=(x.gdp_level / x.price_level)[w].mean(),
                unemployment=100 * x.unemployment[w].mean(),
                price_all_after=(x.house_price / x.price_level)[a].mean(),
                nonmcmv_sum=price.sum(), nonmcmv_n=len(price), last_month=x.month.iloc[-1])


def effect(a, b):
    """d25 rows a, d09 rows b, aligned by seed: {var: (effect, se)}."""
    out = {}
    for v in VARS:
        if v == 'price_nonmcmv_after':
            ea = a.nonmcmv_sum.mean() / a.nonmcmv_n.mean()
            eb = b.nonmcmv_sum.mean() / b.nonmcmv_n.mean()
            per = 100 * ((a.nonmcmv_sum / a.nonmcmv_n) / (b.nonmcmv_sum / b.nonmcmv_n) - 1)
            e = 100 * (ea / eb - 1)
        elif v in REL:
            e = 100 * (a[v].mean() / b[v].mean() - 1)
            per = 100 * (a[v] / b[v] - 1)
        else:
            e = a[v].mean() - b[v].mean()
            per = a[v] - b[v]
        out[v] = (e, per.std() / np.sqrt(len(per)) if len(per) > 1 else np.nan)
    return out


def collect(dirs, keys, cpus):
    rows = Parallel(n_jobs=cpus)(delayed(run_metrics)(d) for d in dirs)
    return pd.DataFrame([dict(k, **r) for k, r in zip(keys, rows)])


def paired(R, by):
    E = []
    for key, g in R.groupby(by):
        a = g[g.dose == 'd25'].set_index('seed')
        b = g[g.dose == 'd09'].set_index('seed')
        idx = a.index.intersection(b.index)
        if not len(idx):
            continue
        for v, (e, se) in effect(a.loc[idx], b.loc[idx]).items():
            E.append(dict(zip(by, key if isinstance(key, tuple) else (key,)), var=v, effect=e, se=se, seeds=len(idx)))
    return pd.DataFrame(E)


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else 'output/morris2026_ends'
    mroot = sys.argv[2] if len(sys.argv) > 2 else 'output/mcmv2026'
    cpus = 6
    out = os.path.join(root, '_analysis')
    os.makedirs(out, exist_ok=True)
    pd.set_option('display.width', 250)

    # 27-area reference and code values, from mcmv2026
    cache = os.path.join(out, 'areas27_runs.csv')
    if os.path.exists(cache):
        A = pd.read_csv(cache)
    else:
        dirs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(mroot, 'd*', '*', '*', 'DONE'))
                      if f.split(os.sep)[-4] in ('d09', 'd25'))
        keys = [dict(dose=d.split(os.sep)[-3], city=d.split(os.sep)[-2], seed=int(d.split(os.sep)[-1])) for d in dirs]
        A = collect(dirs, keys, cpus)
        A.to_csv(cache, index=False, float_format='%.8g')
    areas = paired(A, ['city'])
    ref = areas.groupby('var').effect.agg(p10=lambda e: e.quantile(.1), median='median', p90=lambda e: e.quantile(.9),
                                          lo='min', hi='max', n_neg=lambda e: int((e < 0).sum()), n='count')
    ref['width'] = ref.p90 - ref.p10
    areas.to_csv(os.path.join(out, 'areas27.csv'), index=False, float_format='%.5g')
    print('27-area reference, d25 - d09 (city effects, all seeds)')
    print(ref.round(3).to_string(), '\n')

    # Phase-2 runs
    done = sorted(glob.glob(os.path.join(root, '*', '*', '*', '*', 'DONE')))
    if not done:
        sys.exit('No finished runs in ' + root)
    dirs = [os.path.dirname(f) for f in done]
    keys = [dict(zip(['city', 'setting', 'dose', 'seed'], d.split(os.sep)[-4:])) for d in dirs]
    for k in keys:
        k['seed'] = int(k['seed'])
    R = collect(dirs, keys, cpus)
    R.to_csv(os.path.join(out, 'ends_runs.csv'), index=False, float_format='%.8g')
    cities = sorted(R.city.unique())
    seeds = sorted(R.seed.unique())
    base = paired(A[A.city.isin(cities) & A.seed.isin(seeds)], ['city']).set_index(['city', 'var'])
    E = paired(R, ['city', 'setting'])
    E['base'] = [base.loc[(c, v), 'effect'] for c, v in zip(E.city, E['var'])]
    E['base_se'] = [base.loc[(c, v), 'se'] for c, v in zip(E.city, E['var'])]
    E['shift'] = E.effect - E.base
    E['shift_over_width'] = E['shift'] / E['var'].map(ref.width)
    E['beyond_width'] = E.shift_over_width.abs() > 1
    E['sign_lost'] = E['var'].isin(POSITIVE) & (E.effect <= 0)
    E['outside_areas'] = (E.effect < E['var'].map(ref.lo)) | (E.effect > E['var'].map(ref.hi))
    E.to_csv(os.path.join(out, 'ends_effects.csv'), index=False, float_format='%.5g')

    print(R.groupby(['city', 'dose']).agg(settings=('setting', 'nunique'), runs=('seed', 'count'),
                                          last=('last_month', 'min')).to_string(), '\n')
    print('Code values (seeds {}), d25 - d09'.format(','.join(map(str, seeds))))
    print(base.effect.unstack().round(3).to_string(), '\n')
    print('Criterion 1, dwellings and families positive: {} of {} settings x cities lose the sign'.format(
        int(E.sign_lost.sum()), int(E['var'].isin(POSITIVE).sum())))
    print(E[E.sign_lost][['city', 'setting', 'var', 'effect', 'se']].round(3).to_string(index=False), '\n')
    print('Criterion 2, |shift| over the 27-area width (share of settings x cities above 1; largest)')
    for v, g in E.groupby('var'):
        top = g.loc[g.shift_over_width.abs().idxmax()]
        print('  {:20s} width {:6.3f}  above 1: {:2d}/{:2d}  largest {:+.2f} ({} {})'.format(
            v, ref.width[v], int(g.beyond_width.sum()), len(g), top.shift_over_width, top.city, top.setting))
    print(E[E.beyond_width][['city', 'setting', 'var', 'base', 'effect', 'se', 'shift_over_width']]
          .round(3).to_string(index=False), '\n')
    print('Criterion 3, share of settings with a negative effect, against the 27 areas')
    for v in PRICE + ['residents', 'real_gdp']:
        g = E[E['var'] == v]
        print('  {:20s} settings {:3d}/{:3d} negative   27 areas {:2d}/{:2d}   mean over settings {:+.3f}'.format(
            v, int((g.effect < 0).sum()), len(g), ref.n_neg[v], ref.n[v], g.effect.mean()))
