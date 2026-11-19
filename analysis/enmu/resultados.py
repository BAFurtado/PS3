"""
Paired effects of the ENMU batch (runner_enmu.py): every scenario against its reference, seed by seed, in each area.

    python analysis/enmu/resultados.py [ROOT]          ROOT default output/enmu2026_rede

Effects (same area, same seed):
    rede_padrao    padrao - base          the Padrao network
    rede_otimizado otimizado - base       the Otimizado network and fare
With the _area scenarios (Teresina) the same effects with area weights, suffix _area.
Windows: 2024-2030 (network phase-in, first half), 2031-2037 (second half), 2038-2039 (network complete).
Monthly stats.csv rows are averaged over the window; the parquet files hold January snapshots, averaged over the
Januaries of the window. Levels in percent of the reference, rates in percentage points, counts as differences.
Each cell is the mean over seeds of the paired difference and its t (mean / (sd / sqrt(n))); |t| 2.26 is the 5 %
two-sided value at 10 seeds.

Output in ROOT/analise/: efeitos.csv (long), efeitos_<CITY>.txt (tables), ap_precos.csv and ap_spearman.csv (house price change by AP against the minutes its employed
residents save).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from analysis.output import OUTPUT_DATA_SPEC  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'output/enmu2026_rede'
OUT = os.path.join(ROOT, 'analise')
STATS_COLUMNS = OUTPUT_DATA_SPEC['stats']['columns']
WINDOWS = {'2024-30': (2024, 2030), '2031-37': (2031, 2037), '2038-39': (2038, 2039)}
EFFECTS = {'rede_padrao': ('padrao', 'base'), 'rede_otimizado': ('otimizado', 'base')}
AP_YEARS = (2038, 2039)

# Metric: (source, kind). kind 'pct' = percent of the reference, 'pt' = rate difference x 100, 'diff' = difference
METRICS = {
    'gdp_level': ('stats', 'pct'),
    'unemployment': ('stats', 'pt'),
    'price_level': ('stats', 'pct'),
    'workers_median_wage': ('stats', 'pct'),
    'families_total_wages': ('stats', 'pct'),
    'gini_index': ('stats', 'diff'),
    'pop': ('stats', 'pct'),
    'own_account_workers': ('stats', 'pct'),
    'house_price': ('stats', 'pct'),
    'house_rent': ('stats', 'pct'),
    'house_vacancy': ('stats', 'pt'),
    'number_domiciles': ('stats', 'diff'),
    'house_sales_new': ('stats', 'diff'),
    'constr_employees': ('construction', 'pct'),
    'constr_wages_paid': ('construction', 'pct'),
    'constr_balance': ('construction', 'pct'),
    'minutes_no_car': ('agents', 'diff'),
    'minutes_car': ('agents', 'diff'),
    'minutes_all': ('agents', 'diff'),
    'car_share_employees': ('agents', 'pt'),
    'transit_employees_16_69': ('agents', 'pct'),
}


def runs(root):
    out = {}
    for done in glob.glob(os.path.join(root, '*', '*', '*', 'DONE')):
        d = os.path.dirname(done)
        s, city, i = d.split(os.sep)[-3:]
        out[(s, city, int(i))] = d
    return out


def read_stats(d):
    t = pd.read_csv(os.path.join(d, 'stats.csv'), sep=';', header=None, names=STATS_COLUMNS)
    t['year'] = pd.to_datetime(t.month).dt.year
    return t


def run_metrics(d):
    """{(metric, window): value} and the stats table of one run"""
    st = read_stats(d)
    co = pd.read_parquet(os.path.join(d, 'construction.parquet'),
                         columns=['month', 'number_employees', 'wages_paid', 'total_balance'])
    co['year'] = pd.to_datetime(co.month).dt.year
    co = co.groupby('year').agg(constr_employees=('number_employees', 'sum'),
                                constr_wages_paid=('wages_paid', 'sum'), constr_balance=('total_balance', 'sum'))
    ag = pd.read_parquet(os.path.join(d, 'agents.parquet'), columns=['month', 'age', 'firm_id', 'distance', 'has_car'])
    ag['year'] = pd.to_datetime(ag.month).dt.year
    ag = ag[ag.year >= 2024]
    emp = ag[ag.firm_id.notna()]
    working_age = ag[(ag.age >= 16) & (ag.age <= 69)].groupby('year').size()
    a = pd.DataFrame({
        'minutes_no_car': emp[~emp.has_car].groupby('year').distance.mean(),
        'minutes_car': emp[emp.has_car].groupby('year').distance.mean(),
        'minutes_all': emp.groupby('year').distance.mean(),
        'car_share_employees': emp.groupby('year').has_car.mean(),
        'transit_employees_16_69': emp[~emp.has_car & (emp.age >= 16) & (emp.age <= 69)].groupby('year').size()
        / working_age,
    })
    values = {}
    for w, (y0, y1) in WINDOWS.items():
        s = st[(st.year >= y0) & (st.year <= y1)]
        for m, (src, _) in METRICS.items():
            if src == 'stats':
                values[(m, w)] = s[m].mean()
            elif src == 'construction':
                values[(m, w)] = co.loc[y0:y1, m].mean()
            else:
                values[(m, w)] = a.loc[y0:y1, m].mean()
    return values, st


def paired(a, b, kind):
    if kind == 'pct':
        return 100 * (a / b - 1)
    if kind == 'pt':
        return 100 * (a - b)
    return a - b


def summarise(x):
    x = np.asarray([v for v in x if np.isfinite(v)])
    n = len(x)
    mean = x.mean() if n else np.nan
    sd = x.std(ddof=1) if n > 1 else np.nan
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    return mean, sd, t, n


def ap_prices(d, years):
    """Mean house price and mean commute minutes of employed residents by AP (region_id), Januaries of `years`"""
    h = pd.read_parquet(os.path.join(d, 'houses.parquet'), columns=['month', 'region_id', 'price'])
    h = h[pd.to_datetime(h.month).dt.year.isin(years)]
    ag = pd.read_parquet(os.path.join(d, 'agents.parquet'), columns=['month', 'region_id', 'firm_id', 'distance'])
    ag = ag[pd.to_datetime(ag.month).dt.year.isin(years) & ag.firm_id.notna()]
    return pd.DataFrame({'price': h.groupby('region_id').price.mean(),
                         'minutes': ag.groupby('region_id').distance.mean(),
                         'employees': ag.groupby('region_id').size() / len(years)})


def main():
    r = runs(ROOT)
    if not r:
        sys.exit('No finished runs in ' + ROOT)
    os.makedirs(OUT, exist_ok=True)
    print('Runs:', len(r), 'in', ROOT)
    cache, stats = {}, {}
    for k, d in sorted(r.items()):
        cache[k], stats[k] = run_metrics(d)

    rows = []
    for (s, city, i) in sorted(r):
        for suffix in ('', '_area'):
            for eff, (a, b) in EFFECTS.items():
                if s != a + suffix or (b + suffix, city, i) not in r:
                    continue
                va, vb = cache[(s, city, i)], cache[(b + suffix, city, i)]
                for (m, w), x in va.items():
                    rows.append((city, eff + suffix, m, w, i, paired(x, vb[(m, w)], METRICS[m][1])))
    long = pd.DataFrame(rows, columns=['city', 'effect', 'metric', 'window', 'seed', 'delta'])
    summary = (long.groupby(['city', 'effect', 'metric', 'window']).delta
               .apply(lambda x: pd.Series(summarise(x), index=['mean', 'sd', 't', 'n'])).unstack().reset_index())
    # Reference levels, so the deltas can be read against them
    ref_rows = []
    for (s, city, i), v in cache.items():
        for (m, w), x in v.items():
            ref_rows.append((s, city, m, w, x))
    levels = (pd.DataFrame(ref_rows, columns=['scenario', 'city', 'metric', 'window', 'value'])
              .groupby(['scenario', 'city', 'metric', 'window']).value.mean().reset_index())
    summary.to_csv(os.path.join(OUT, 'efeitos.csv'), sep=';', index=False)
    levels.to_csv(os.path.join(OUT, 'niveis.csv'), sep=';', index=False)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.max_rows', 500)
    order = list(METRICS)
    for city in sorted(summary.city.unique()):
        c = summary[summary.city == city].copy()
        c['cell'] = c.apply(lambda x: '{:+.2f} ({:+.1f})'.format(x['mean'], x['t']), axis=1)
        tab = c.pivot_table(index='metric', columns=['effect', 'window'], values='cell', aggfunc='first')
        tab = tab.reindex([m for m in order if m in tab.index])
        base = levels[(levels.city == city) & (levels.scenario == 'base')].pivot(index='metric', columns='window',
                                                                                  values='value')
        base = base.reindex(tab.index)[list(WINDOWS)]
        text = ['== {} (n = {} seeds)'.format(city, int(c.n.max())),
                'Reference levels, base:', base.to_string(float_format=lambda v: '{:.4g}'.format(v)), '']
        for eff in [e for e in tab.columns.get_level_values(0).unique()]:
            text += ['-- ' + eff, tab[eff][[w for w in WINDOWS if w in tab[eff].columns]].to_string(), '']
        text = '\n'.join(text)
        print(text)
        with open(os.path.join(OUT, 'efeitos_{}.txt'.format(city.replace(' ', '_'))), 'w') as f:
            f.write(text)

    # House price by AP against the minutes its employed residents save, network effects, seeds averaged
    aps, sp = [], []
    for (s, city, i), d in sorted(r.items()):
        for suffix in ('', '_area'):
            for fut in ('padrao', 'otimizado'):
                if s != fut + suffix or ('base' + suffix, city, i) not in r:
                    continue
                x = ap_prices(d, AP_YEARS)
                y = ap_prices(r[('base' + suffix, city, i)], AP_YEARS)
                j = x.join(y, lsuffix='_fut', rsuffix='_ref', how='inner')
                j['price_pct'] = 100 * (j.price_fut / j.price_ref - 1)
                j['minutes_diff'] = j.minutes_fut - j.minutes_ref
                j = j.reset_index().assign(city=city, effect='rede_' + fut + suffix, seed=i)
                aps.append(j)
    if aps:
        aps = pd.concat(aps)
        m = aps.groupby(['city', 'effect', 'region_id'])[['price_pct', 'minutes_diff', 'employees_ref']].mean()
        m.reset_index().to_csv(os.path.join(OUT, 'ap_precos.csv'), sep=';', index=False)
        for (city, eff), g in m.groupby(level=[0, 1]):
            g = g[g.employees_ref >= 20]
            rho = g.price_pct.corr(g.minutes_diff, method='spearman')
            sp.append((city, eff, len(g), rho, g.price_pct.std(), g.minutes_diff.mean()))
        sp = pd.DataFrame(sp, columns=['city', 'effect', 'aps', 'spearman_price_minutes', 'sd_price_pct',
                                       'mean_minutes_diff'])
        sp.to_csv(os.path.join(OUT, 'ap_spearman.csv'), sep=';', index=False)
        print('== AP house price change vs minutes saved by employed residents, {}-{} (APs with >= 20 employees)'
              .format(*AP_YEARS))
        print(sp.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
