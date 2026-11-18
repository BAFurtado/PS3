"""
RENT_LEVEL A/B with transport off: 'census' (one Census rent factor per ACP) against 'census_ap' (the same ACP level
divided among its areas by each area's Census mean rent, world/house_values.py), paired by seed. Settings as
runner_vacancy_ab (transport off, 2010-2021, seeds SEED + i), with VACANCY_SCOPE 'region' as in the ENMU runs.
Saved: stats, regional and the houses parquet every January.

Modes (cwd = repository root):
    python runner_rent_ap.py teste CPUS                  Palmas, both levels, seed 0, 2 years
    python runner_rent_ap.py lote CPUS [OPTIONS]         rerun the same command to resume
        cities=enmu|local|big|all|A,B   city set (default enmu, the six ENMU areas; the others as
                                        runner_vacancy_ab)
        runs=N                     seeds 0 to N-1 (default 3)
        part=K/N                   the K-th of N contiguous slices of the queue (default 1/1)
    python runner_rent_ap.py comparar                    paired deltas over 2011-2021, and the area-level fit

comparar, rule fixed before the runs: a delta is reported when beyond 2 seed sd (rates in points; price, rent,
population and domiciles in percent). The p10-p90 width across areas is printed too; with six areas it is a weak
reference. Area-level fit: Spearman across areas between the mean rent of rented houses and the Census 2010
mean rent, in 2011-01 and 2021-01. Location of building: share of the houses added 2011-2021 in the areas whose
Census rent is in the ACP's top fifth.

Output: output/rent_ap/<level>/<CITY>/<i>/. Local conf/params.py and conf/run.py are ignored.
"""
import datetime
import glob
import logging
import os
import sys

import runner_vacancy_ab as V

OUTPUT = 'output/rent_ap'
LEVELS = ['census', 'census_ap']
ENMU = ['BRASILIA', 'FORTALEZA', 'BAIXADA SANTISTA', 'GOIANIA', 'MANAUS', 'TERESINA']
V.COMMON['VACANCY_SCOPE'] = 'region'
logger = logging.getLogger('main')


def queue(cities, indices):
    return [(level, city, i) for i in indices for city in cities for level in LEVELS]


def build_jobs(root, items, total_days=None):
    """runner_vacancy_ab jobs with the scope slot carrying RENT_LEVEL"""
    jobs = V.build_jobs(root, items, total_days)
    for job in jobs:
        level = job['path'].split(os.sep)[-3]
        job['params']['RENT_LEVEL'] = level
        job['params']['VACANCY_SCOPE'] = 'region'
    return jobs


def area_fit(path, month):
    """Spearman across areas of the rented houses' mean rent with the Census mean rent; share of houses added after
    2010-01 in the top fifth of areas by Census rent within the ACP"""
    import pandas as pd
    census = pd.read_csv('input/rent_AP_2010.csv', sep=';', dtype={'AREAP': str}).set_index('AREAP').mean_rent
    h = pd.read_parquet(os.path.join(path, 'houses.parquet'),
                        columns=['month', 'id', 'rent', 'family_id', 'region_id'])
    h['region_id'] = h.region_id.astype(str)
    now = h[h.month == month]
    rented = now[(now.rent > 0) & now.family_id.notna()]
    m = rented.groupby('region_id').rent.mean()
    c = census.reindex(m.index)
    fit = float(m.rank().corr(c.rank()))
    first = set(h[h.month == h.month.min()].id)
    new = now[~now.id.isin(first)]
    top = set(c[c >= c.quantile(.8)].index)
    share = float(new.region_id.isin(top).mean()) if len(new) else float('nan')
    return fit, share


def comparar(root=OUTPUT):
    import numpy as np
    import pandas as pd
    rows = []
    for done in sorted(glob.glob(os.path.join(root, LEVELS[0], '*', '*', 'DONE'))):
        city, i = done.split(os.sep)[-3:-1]
        other = V.run_path(root, LEVELS[1], city, i)
        if not os.path.exists(os.path.join(other, 'DONE')):
            continue
        r = {'city': city, 'seed': int(i)}
        for level, path in ((LEVELS[0], os.path.dirname(done)), (LEVELS[1], other)):
            s = V.stats(path)
            w = s[(s.month >= V.WINDOW[0]) & (s.month <= V.WINDOW[1])]
            for c in V.INDICATORS:
                r['%s_%s' % (level, c)] = w[c].mean()
            r['%s_max_unemployment' % level] = s.unemployment.max()
            r['%s_fit_2011' % level], _ = area_fit(path, datetime.date(2011, 1, 1))
            r['%s_fit_2021' % level], r['%s_new_top' % level] = area_fit(path, datetime.date(2021, 1, 1))
        rows.append(r)
    if not rows:
        print('No finished pairs in', root)
        return
    t = pd.DataFrame(rows)
    a, b = LEVELS
    cols = V.INDICATORS + ['max_unemployment', 'fit_2011', 'fit_2021', 'new_top']
    for c in cols:
        t['d_' + c] = 100 * (t[b + '_' + c] / t[a + '_' + c] - 1) if c in V.RELATIVE else t[b + '_' + c] - t[a + '_' + c]
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 60)
    g = t.groupby('city')
    print('Pairs per area:', g.size().to_dict(), '\n')
    level = g[[a + '_' + c for c in cols]].mean()
    delta = g[['d_' + c for c in cols]].mean()
    sd = g[['d_' + c for c in cols]].std()
    width = level.quantile(.9) - level.quantile(.1)
    width[[a + '_' + c for c in V.RELATIVE + ['fit_2011', 'fit_2021', 'new_top']]] = np.nan
    summary = pd.DataFrame({a + '_p10': level.quantile(.1).values, a + '_p90': level.quantile(.9).values,
                            'width': width.values, 'delta_mean': delta.mean().values,
                            'delta_min': delta.min().values, 'delta_max': delta.max().values,
                            'areas_beyond_width': np.where(np.isnan(width.values), np.nan,
                                                           (delta.abs() > width.values).sum().values),
                            'areas_beyond_2sd': (delta.abs() > 2 * sd.fillna(np.inf)).sum().values},
                           index=cols)
    print('Spread across areas under %s (window means) and the paired delta %s - %s, mean over seeds per area' % (a, b, a))
    print('(deltas of', ', '.join(V.RELATIVE), 'in percent):')
    print(summary.round(4).to_string())
    print('\nPer area, levels under both and delta (mean over seeds):')
    both = pd.concat([g[[a + '_' + c, b + '_' + c]].mean() for c in ['fit_2011', 'fit_2021', 'new_top']], axis=1)
    print(both.round(3).to_string())
    print(delta.round(4).to_string())
    exploded = t[(t[a + '_max_unemployment'] > .3) | (t[b + '_max_unemployment'] > .3)]
    print('\nRuns with unemployment above 0.30 in some month:', 'none' if exploded.empty else
          exploded[['city', 'seed', a + '_max_unemployment', b + '_max_unemployment']].to_string(index=False))
    t.to_csv(os.path.join(root, 'comparar.csv'), sep=';', index=False)
    print('\nwrote', os.path.join(root, 'comparar.csv'))


def options(args):
    opts = {'cities': 'enmu', 'runs': '3', 'part': '1/1'}
    for x in args:
        key, sep, v = x.partition('=')
        if not sep or key not in opts:
            sys.exit('Unknown option {!r}. Options: {}'.format(x, ', '.join(k + '=' for k in opts)))
        opts[key] = v
    k, n = (int(x) for x in opts['part'].split('/'))
    if not 1 <= k <= n:
        sys.exit('part must be K/N with 1 <= K <= N')
    return opts['cities'], int(opts['runs']), k, n


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if V.R.LOCAL_CONF:
        print('Ignoring local overrides:', ', '.join(V.R.LOCAL_CONF))
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'comparar':
        comparar()
    elif mode == 'teste' and len(sys.argv) == 3:
        root = OUTPUT + '_teste'
        V.run(root, build_jobs(root, queue(['PALMAS'], [0]), total_days=730), int(sys.argv[2]),
              os.path.join(root, 'queue'))
        for level in LEVELS:
            p = V.run_path(root, level, 'PALMAS', 0)
            s = V.stats(p)
            print(level, 'DONE' if os.path.exists(os.path.join(p, 'DONE')) else 'NOT DONE', 'last month',
                  s.month.iloc[-1], 'vacancy %.3f price %.2f' % (s.house_vacancy.iloc[-1], s.house_price.iloc[-1]),
                  'fit 2011 %.2f' % area_fit(p, datetime.date(2011, 1, 1))[0])
    elif mode == 'lote' and len(sys.argv) >= 3:
        name, runs, k, n = options(sys.argv[3:])
        items = queue(ENMU if name == 'enmu' else V.city_set(name), range(runs))
        size = -(-len(items) // n)
        jobs = build_jobs(OUTPUT, items[(k - 1) * size:k * size])
        V.run(OUTPUT, jobs, int(sys.argv[2]), os.path.join(OUTPUT, 'queue_{}_{}of{}'.format(name, k, n)))
    else:
        sys.exit(__doc__)
