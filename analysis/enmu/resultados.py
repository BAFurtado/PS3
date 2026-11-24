"""
Paired effects of the ENMU batch (runner_enmu.py): every scenario against its reference, seed by seed, in each area.

    python analysis/enmu/resultados.py [ROOT]          ROOT default output/enmu2026_rede_census2022

Effects (same area, same seed):
    rede_padrao    padrao - base          the Padrao network
    rede_otimizado otimizado - base       the Otimizado network and fare
With the _area scenarios (Teresina) the same effects with area weights, suffix _area.
Windows: 2024-2030 (network phase-in, first half), 2031-2037 (second half), 2038-2039 (network complete).
Monthly stats.csv rows are averaged over the window; the parquet files hold January snapshots, averaged over the
Januaries of the window. Levels in percent of the reference, rates in percentage points, counts as differences.
Each cell is the mean over seeds of the paired difference and its t (mean / (sd / sqrt(n))); |t| 2.26 is the 5 %
two-sided value at 10 seeds.

Commute minutes: minutes_no_car_network is the change the future matrix alone gives on the reference run's home-job
pairs (phi of each January times future minus base minutes, transit commuters; car minutes are the same in every
state), and minutes_no_car_resorting the rest of the effect (who works where and lives where). minutes_q1 to
minutes_q5: mean minutes of the employed by quintile of their family's permanent income per member, quintiles drawn
within each run and January. transport_* are the Transport sector's firms. families_savings, money_imports,
money_profits_out and firms_total_profit trace where the fare money goes.
Missing parquet files (runs saved with fewer files, runner_enmu.py save=) leave their metrics empty.

Output in ROOT/analise/: efeitos.csv (long), efeitos_<CITY>.txt (tables), ap_precos.csv and ap_spearman.csv (house
price change by AP against the minutes its employed residents save).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from analysis.output import OUTPUT_DATA_SPEC  # noqa: E402
import world.transport as T  # noqa: E402
from runner_enmu import START, YEARS  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'output/enmu2026_rede_census2022'
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
    'families_savings': ('stats', 'pct'),
    'money_imports': ('stats', 'pct'),
    'money_profits_out': ('stats', 'pct'),
    'firms_total_profit': ('stats', 'pct'),
    'transport_revenue': ('sector', 'pct'),
    'transport_employees': ('sector', 'pct'),
    'transport_wages_paid': ('sector', 'pct'),
    'constr_employees': ('construction', 'pct'),
    'constr_wages_paid': ('construction', 'pct'),
    'constr_balance': ('construction', 'pct'),
    'minutes_no_car': ('agents', 'diff'),
    'minutes_car': ('agents', 'diff'),
    'minutes_all': ('agents', 'diff'),
    'minutes_no_car_network': ('network', 'diff'),
    'minutes_no_car_resorting': ('network', 'diff'),
    'minutes_q1': ('agents', 'diff'),
    'minutes_q2': ('agents', 'diff'),
    'minutes_q3': ('agents', 'diff'),
    'minutes_q4': ('agents', 'diff'),
    'minutes_q5': ('agents', 'diff'),
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


def read_parquet(d, name, columns):
    """The January snapshots of one file with a year column, or None when the run did not save it"""
    path = os.path.join(d, name + '.parquet')
    if not os.path.exists(path):
        return None
    t = pd.read_parquet(path, columns=columns)
    t['year'] = pd.to_datetime(t.month).dt.year
    return t


def employed(d):
    """Employed agents from 2024 with their family's permanent income per member and their firm's region"""
    ag = read_parquet(d, 'agents', ['month', 'age', 'region_id', 'firm_id', 'family_id', 'distance', 'has_car'])
    if ag is None:
        return None, None
    ag = ag[ag.year >= 2024]
    fa = read_parquet(d, 'families', ['month', 'id', 'permanent_income', 'num_members'])
    if fa is not None:
        fa = fa.assign(income_pc=fa.permanent_income / fa.num_members)[['month', 'id', 'income_pc']]
        ag = ag.merge(fa.rename(columns={'id': 'family_id'}), on=['month', 'family_id'], how='left')
    firms = [t for t in (read_parquet(d, n, ['month', 'firm_id', 'region_id']) for n in ('firms', 'construction'))
             if t is not None]
    if firms:
        f = pd.concat(firms).drop_duplicates(['month', 'firm_id']).rename(columns={'region_id': 'firm_region'})
        ag = ag.merge(f[['month', 'firm_id', 'firm_region']], on=['month', 'firm_id'], how='left')
    return ag, ag[ag.firm_id.notna()]


def run_metrics(d):
    """{(metric, window): value} and the stats table of one run"""
    st = read_stats(d)
    frames = {}
    co = read_parquet(d, 'construction', ['month', 'number_employees', 'wages_paid', 'total_balance'])
    if co is not None:
        frames['construction'] = co.groupby('year').agg(
            constr_employees=('number_employees', 'sum'), constr_wages_paid=('wages_paid', 'sum'),
            constr_balance=('total_balance', 'sum'))
    fi = read_parquet(d, 'firms', ['month', 'sector', 'revenue', 'number_employees', 'wages_paid'])
    if fi is not None:
        frames['sector'] = fi[fi.sector == 'Transport'].groupby('year').agg(
            transport_revenue=('revenue', 'sum'), transport_employees=('number_employees', 'sum'),
            transport_wages_paid=('wages_paid', 'sum'))
    ag, emp = employed(d)
    if ag is not None:
        working_age = ag[(ag.age >= 16) & (ag.age <= 69)].groupby('year').size()
        a = pd.DataFrame({
            'minutes_no_car': emp[~emp.has_car].groupby('year').distance.mean(),
            'minutes_car': emp[emp.has_car].groupby('year').distance.mean(),
            'minutes_all': emp.groupby('year').distance.mean(),
            'car_share_employees': emp.groupby('year').has_car.mean(),
            'transit_employees_16_69': emp[~emp.has_car & (emp.age >= 16) & (emp.age <= 69)].groupby('year').size()
            / working_age,
        })
        if 'income_pc' in emp.columns:
            q = emp.groupby('year').income_pc.transform(
                lambda x: pd.qcut(x.rank(method='first'), 5, labels=False) + 1)
            for k in range(1, 6):
                a['minutes_q%d' % k] = emp[q == k].groupby('year').distance.mean()
        frames['agents'] = a
    values = {}
    for w, (y0, y1) in WINDOWS.items():
        s = st[(st.year >= y0) & (st.year <= y1)]
        for m, (src, _) in METRICS.items():
            if src == 'stats':
                values[(m, w)] = s[m].mean()
            elif src in frames and m in frames[src].columns:
                values[(m, w)] = frames[src].loc[y0:y1, m].mean()
            else:
                values[(m, w)] = np.nan
    return values, st


_matrices = {}


def network_shift(d, city, weights, future):
    """{window: mean change of transit commuters' minutes} that the future network gives on the home-job pairs of
    the reference run d, January by January (phi of the phase-in times future minus base). Car commuters and
    own-account pool members (minutes 0) count with no change."""
    if (city, weights) not in _matrices:
        _matrices[(city, weights)] = T.read_acp_matrix(city, weights)
    wide = _matrices[(city, weights)]
    _, emp = employed(d)
    if emp is None or wide is None or 'firm_region' not in emp.columns:
        return {w: np.nan for w in WINDOWS}
    emp = emp[~emp.has_car]
    pairs = list(zip(emp.region_id, emp.firm_region))
    delta = (wide[future] - wide['base']).reindex(pairs).to_numpy()
    schedule = T.linear_phase_in(START, YEARS)
    phi = {y: max([p for when, p in schedule if when <= y], default=0.0) for y in emp.year.unique()}
    shift = pd.Series(np.where(emp.distance.to_numpy() > 0, delta, 0.0) * emp.year.map(phi).to_numpy(),
                      index=emp.index).groupby(emp.year).mean()
    return {w: shift.loc[y0:y1].mean() for w, (y0, y1) in WINDOWS.items()}


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
                shift = network_shift(r[(b + suffix, city, i)], city, 'area' if suffix else 'trips', a)
                for (m, w), x in va.items():
                    if m == 'minutes_no_car_network':
                        delta = shift[w]
                    elif m == 'minutes_no_car_resorting':
                        delta = va[('minutes_no_car', w)] - vb[('minutes_no_car', w)] - shift[w]
                    else:
                        delta = paired(x, vb[(m, w)], METRICS[m][1])
                    rows.append((city, eff + suffix, m, w, i, delta))
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
                if not all(os.path.exists(os.path.join(p, 'houses.parquet')) for p in (d, r[('base' + suffix, city, i)])):
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
