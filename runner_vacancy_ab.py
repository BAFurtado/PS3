"""
VACANCY_SCOPE A/B with transport off: 'acp' (the vacancy of all houses in the run) against 'region' (each region's
vacancy shrunk toward its municipality's, world/location.regional_vacancy), paired by seed. 27 areas, 2010-2021,
seeds SEED + i (i = 0-2, the seeds of runner_mcmv2026). Transport off: no TRANSPORT_WEIGHTS (Brasilia keeps the
partner parquet when it exists, as in tag .3), LOCATION_CHOICE off, COMMUTE_ACCEPTANCE off, TRANSPORT_MONEY at its
default. Every other setting at the code default. Saved: stats, regional and the houses parquet every January.

City sets:
    local   the 25 areas without Sao Paulo and Rio de Janeiro
    big     Sao Paulo and Rio de Janeiro
    all     the 27 areas
    A,B,..  a comma-separated list of ACPs (the 27 areas or Baixada Santista)

Modes (cwd = repository root):
    python runner_vacancy_ab.py teste CPUS                  Palmas, both scopes, seed 0, 2 years
    python runner_vacancy_ab.py lote CPUS [OPTIONS]         the batch; rerun the same command to resume (finished
                                                            runs skipped)
        cities=local|big|all|A,B   city set (default local)
        part=K/N                   the K-th of N contiguous slices of the queue (default 1/1)
    python runner_vacancy_ab.py comparar                    paired deltas over 2011-2021 against the 27-area spread

Output: output/vacancy_ab/<scope>/<CITY>/<i>/; queue manifest and log in output/vacancy_ab/queue_<set>_<K>of<N>/.
Local conf/params.py and conf/run.py are ignored.
"""
import copy
import datetime
import glob
import logging
import os
import shutil
import sys

import runner_funding2026 as R

OUTPUT = 'output/vacancy_ab'
END = datetime.date(2022, 1, 1)
WINDOW = (datetime.date(2011, 1, 1), datetime.date(2021, 12, 1))
SCOPES = ['acp', 'region']
RUNS = 3
BIG = ['SAO PAULO', 'RIO DE JANEIRO']
# ENMU area outside the 27, accepted in a comma-separated list
EXTRA = ['BAIXADA SANTISTA']
COMMON = {
    'TRANSPORT_WEIGHTS': None,
    'TRANSPORT_TIME': False,
    'TRANSPORT_SCHEDULE': None,
    'LOCATION_CHOICE': 'off',
    'COMMUTE_ACCEPTANCE': 'off',
    'TOTAL_DAYS': (END - R.conf.PARAMS['STARTING_DAY']).days,
}
INDICATORS = ['unemployment', 'house_vacancy', 'house_price', 'house_rent', 'pct_zero_consumption', 'gini_index',
              'inflation', 'gdp_growth_rate', 'pop', 'number_domiciles']
# Scale-dependent indicators: deltas in percent of 'acp', judged against seed noise only
RELATIVE = ['house_price', 'house_rent', 'pop', 'number_domiciles']
R.conf.RUN['SAVE_DATA'] = ['house']
logger = logging.getLogger('main')


def areas():
    """The 27 areas, largest first: Sao Paulo, then the runner_funding2026 order"""
    return ['SAO PAULO'] + [c for c in R.CAPITALS if c != 'SAO PAULO']


def city_set(name):
    if name == 'local':
        return [c for c in areas() if c not in BIG]
    if name == 'big':
        return BIG
    if name == 'all':
        return areas()
    cities = name.split(',')
    unknown = set(cities) - set(areas()) - set(EXTRA)
    if unknown:
        sys.exit('Unknown ACPs: {}'.format(sorted(unknown)))
    return cities


def run_path(root, scope, city, i):
    return os.path.join(root, scope, city, str(i))


def queue(cities, indices):
    """(scope, city, i) seed by seed, both scopes of a city together, so a batch stopped halfway holds complete
    pairs."""
    return [(scope, city, i) for i in indices for city in cities for scope in SCOPES]


def build_jobs(root, items, total_days=None):
    """Jobs for the unfinished items. Folders of unfinished runs are removed before their run is queued."""
    jobs = []
    for scope, city, i in items:
        path = run_path(root, scope, city, i)
        if os.path.exists(os.path.join(path, 'DONE')):
            continue
        if os.path.exists(path):
            shutil.rmtree(path)
        p = copy.deepcopy(R.conf.PARAMS)
        p.update(COMMON)
        p['VACANCY_SCOPE'] = scope
        p['PROCESSING_ACPS'] = [city]
        p['SEED'] = R.SEED + i
        if total_days:
            p['TOTAL_DAYS'] = total_days
        jobs.append({'params': p, 'path': path})
    return jobs


def run(root, jobs, cpus, manifest):
    from main import save_jobs, _run_jobs_parallel
    os.makedirs(manifest, exist_ok=True)
    logger.info('%d runs to do (finished ones skipped), %d at a time', len(jobs), cpus)
    save_jobs(manifest, jobs, cpus)
    _run_jobs_parallel(jobs, cpus)


def options(args):
    opts = {'cities': 'local', 'part': '1/1'}
    for a in args:
        key, sep, v = a.partition('=')
        if not sep or key not in opts:
            sys.exit('Unknown option {!r}. Options: {}'.format(a, ', '.join(k + '=' for k in opts)))
        opts[key] = v
    k, n = (int(x) for x in opts['part'].split('/'))
    if not 1 <= k <= n:
        sys.exit('part must be K/N with 1 <= K <= N')
    return opts['cities'], k, n


def stats(path):
    import pandas as pd
    from analysis.output import OUTPUT_DATA_SPEC
    cols = OUTPUT_DATA_SPEC['stats']['columns']
    s = pd.read_csv(os.path.join(path, 'stats.csv'), sep=';', header=None)
    s.columns = cols[:s.shape[1]] + ['x%d' % k for k in range(s.shape[1] - len(cols))]
    s['month'] = pd.to_datetime(s['month']).dt.date
    return s


def price_dispersion(path, month):
    """Between-region sd of the regions' mean log price per size unit, houses parquet at `month`"""
    import numpy as np
    import pandas as pd
    h = pd.read_parquet(os.path.join(path, 'houses.parquet'), columns=['month', 'price', 'size', 'region_id'])
    h = h[(h.month == month) & (h['size'] > 0) & (h.price > 0)]
    if h.empty:
        return float('nan')
    return float(h.assign(lp=np.log(h.price / h['size'])).groupby('region_id').lp.mean().std())


def comparar(root=OUTPUT):
    """Per area: mean over seeds of the 2011-2021 window mean under 'acp', and the paired delta region - acp with its
    seed sd. Rule fixed before the runs: a rate's delta is reported when it exceeds the p10-p90 width of that rate
    across the 27 areas under 'acp'. RELATIVE indicators: delta in percent of 'acp', reported when beyond 2 seed sd.
    Explosions: unemployment above 0.30 in any month."""
    import numpy as np
    import pandas as pd
    rows = []
    for done in sorted(glob.glob(os.path.join(root, 'acp', '*', '*', 'DONE'))):
        city, i = done.split(os.sep)[-3:-1]
        other = run_path(root, 'region', city, i)
        if not os.path.exists(os.path.join(other, 'DONE')):
            continue
        r = {'city': city, 'seed': int(i)}
        for scope, path in (('acp', os.path.dirname(done)), ('region', other)):
            s = stats(path)
            w = s[(s.month >= WINDOW[0]) & (s.month <= WINDOW[1])]
            for c in INDICATORS:
                r['%s_%s' % (scope, c)] = w[c].mean()
            r['%s_max_unemployment' % scope] = s.unemployment.max()
            r['%s_price_sd' % scope] = price_dispersion(path, datetime.date(2021, 1, 1))
        rows.append(r)
    if not rows:
        print('No finished pairs in', root)
        return
    t = pd.DataFrame(rows)
    cols = INDICATORS + ['price_sd', 'max_unemployment']
    for c in cols:
        if c in RELATIVE:
            t['d_' + c] = 100 * (t['region_' + c] / t['acp_' + c] - 1)
        else:
            t['d_' + c] = t['region_' + c] - t['acp_' + c]
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 60)
    g = t.groupby('city')
    print('Pairs per area:', g.size().to_dict(), '\n')
    level = g[['acp_' + c for c in cols]].mean()
    delta = g[['d_' + c for c in cols]].mean()
    sd = g[['d_' + c for c in cols]].std()
    width = level.quantile(.9) - level.quantile(.1)
    width[['acp_' + c for c in RELATIVE]] = np.nan
    summary = pd.DataFrame({'acp_p10': level.quantile(.1).values, 'acp_p90': level.quantile(.9).values,
                            'width': width.values, 'delta_mean': delta.mean().values,
                            'delta_min': delta.min().values, 'delta_max': delta.max().values,
                            'areas_beyond_width': np.where(np.isnan(width.values), np.nan,
                                                           (delta.abs() > width.values).sum().values),
                            'areas_beyond_2sd': (delta.abs() > 2 * sd.fillna(np.inf)).sum().values},
                           index=cols)
    print('27-area spread under acp (window means) and the paired delta region - acp, mean over seeds per area')
    print('(deltas of', ', '.join(RELATIVE), 'in percent of acp):')
    print(summary.round(4).to_string())
    print('\nPer area, delta region - acp (mean over seeds):')
    print(delta.round(4).to_string())
    exploded = t[(t.acp_max_unemployment > .3) | (t.region_max_unemployment > .3)]
    print('\nRuns with unemployment above 0.30 in some month:', 'none' if exploded.empty else
          exploded[['city', 'seed', 'acp_max_unemployment', 'region_max_unemployment']].to_string(index=False))
    t.to_csv(os.path.join(root, 'comparar.csv'), sep=';', index=False)
    print('\nwrote', os.path.join(root, 'comparar.csv'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if R.LOCAL_CONF:
        print('Ignoring local overrides:', ', '.join(R.LOCAL_CONF))
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'comparar':
        comparar()
    elif mode == 'teste' and len(sys.argv) == 3:
        root = OUTPUT + '_teste'
        run(root, build_jobs(root, queue(['PALMAS'], [0]), total_days=730), int(sys.argv[2]),
            os.path.join(root, 'queue'))
        for scope in SCOPES:
            p = run_path(root, scope, 'PALMAS', 0)
            s = stats(p)
            print(scope, 'DONE' if os.path.exists(os.path.join(p, 'DONE')) else 'NOT DONE', 'last month',
                  s.month.iloc[-1], 'vacancy %.3f price %.2f' % (s.house_vacancy.iloc[-1], s.house_price.iloc[-1]),
                  'files', sorted(os.listdir(p)))
    elif mode == 'lote' and len(sys.argv) >= 3:
        name, k, n = options(sys.argv[3:])
        items = queue(city_set(name), range(RUNS))
        size = -(-len(items) // n)
        jobs = build_jobs(OUTPUT, items[(k - 1) * size:k * size])
        run(OUTPUT, jobs, int(sys.argv[2]), os.path.join(OUTPUT, 'queue_{}_{}of{}'.format(name, k, n)))
    else:
        sys.exit(__doc__)
