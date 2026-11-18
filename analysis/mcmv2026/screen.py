"""mcmv2026: every stats.csv indicator, d25 - d09, so the figures rest on the whole set, not on a chosen few.
usage (repo root, env ps3_2): python analysis/mcmv2026/screen.py [root]
Writes <root>/_analysis/screen_all.csv, screen_city.csv and screen_runs.csv (window means per run).

Per run: monthly values averaged over each window (dose 2026-30, after 2031-34). Money columns are divided by the
run's price_level; cumulative money_* counters enter as monthly differences. Effect per city and seed = d25 - d09 at
the same seed. Kind 'share' (reference mean inside [0, 1] or a rate by name): effect in percentage points, mean over
runs. Kind 'level': pooled effect = sum of differences / sum of reference levels, in %. 90 % interval: bootstrap over
the 26 cities. n_pos / n_neg: cities whose 10-seed mean differs from 0 by more than two standard errors.
split_half: Spearman of the city effects in seeds 0-4 against seeds 5-9."""
import glob
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path[:0] = ['.']
from analysis.output import columns_for

root = sys.argv[1] if len(sys.argv) > 1 else 'output/mcmv2026'
out = os.path.join(root, '_analysis')
WINDOWS = {'dose': (2026, 2030), 'after': (2031, 2034)}
MONEY = ['gdp_level', 'families_median_permanent_income', 'families_wages_received', 'families_savings',
         'amount_subsidised', 'firms_total_profit', 'firms_median_profit', 'firms_median_wage_paid',
         'firms_wage_per_worker', 'workers_median_wage', 'house_price', 'house_rent', 'fpm', 'bank', 'emissions_fund',
         'equally', 'locally', 'credit_stock', 'bank_balance', 'families_total_permanent_income',
         'families_total_income', 'firms_investment', 'own_account_earnings', 'families_total_wages', 'taxes_products',
         'families_rent_received', 'beneficiary_permanent_income', 'beneficiary_total_income', 'house_sales_paid',
         'house_sales_new_paid', 'ext_amount_sold', 'ext_imports', 'ext_exports', 'ext_net_position',
         'ext_public_funding', 'household_imports']
SHARE_NAMES = ('unemployment', 'gini_index', 'house_vacancy', 'pct_', 'share_', 'perc_', 'p_delinquent',
               'loan_approval_rate', 'rent_burden_decis', 'affordability_median', 'firms_unmet_share',
               'firms_utilisation', 'firms_profit_share_paid', 'inflation', 'gdp_growth_rate', 'average_qli')
SKIP = {'month', 'gdp_change', 'wage_census_factor'}


def one(d):
    s, c, i = d.split(os.sep)[-3:]
    f = os.path.join(d, 'stats.csv')
    n = len(next(l for l in open(f) if l.strip()).split(';'))
    x = pd.read_csv(f, sep=';', header=None, names=columns_for('stats', n))
    x['year'] = x.month.astype(str).str[:4].astype(int)
    x = x[x.year >= 2026]
    p = x.price_level
    for col in x.columns:
        if col.startswith('money_'):
            x[col] = x[col].diff().fillna(0) / p
        elif col in MONEY:
            x[col] = x[col] / p
    rows = []
    for w, (a, b) in WINDOWS.items():
        m = x[(x.year >= a) & (x.year <= b)].drop(columns=['month', 'year']).mean(numeric_only=True)
        m['scenario'], m['city'], m['seed'], m['window'] = s, c, int(i), w
        rows.append(m)
    return rows


dirs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(root, 'd*', '*', '*', 'DONE'))
              if f.split(os.sep)[-4] in ('d09', 'd25'))
R = pd.DataFrame([r for rr in Parallel(n_jobs=8)(delayed(one)(d) for d in dirs) for r in rr])
R.to_csv(os.path.join(out, 'screen_runs.csv'), index=False, float_format='%.6g')
cols = [c for c in R.columns if c not in SKIP | {'scenario', 'city', 'seed', 'window'}]
rng = np.random.default_rng(0)
res, city_rows = [], []
for w in WINDOWS:
    a = R[(R.scenario == 'd25') & (R.window == w)].set_index(['city', 'seed'])[cols]
    r = R[(R.scenario == 'd09') & (R.window == w)].set_index(['city', 'seed'])[cols]
    idx = a.index.intersection(r.index)
    a, r = a.loc[idx], r.loc[idx]
    D = a - r
    cities = D.index.get_level_values(0).unique()
    for v in cols:
        ref = r[v]
        if ref.abs().sum() == 0 and D[v].abs().sum() == 0:
            continue
        share = v.startswith(SHARE_NAMES) or (ref.between(0, 1).all() and not v.startswith('money_') and
                                              v not in MONEY and ref.max() > 0)
        dc, rc = D[v].groupby(level=0).mean(), ref.groupby(level=0).mean()
        nc = pd.Series(1, index=dc.index)

        def stat(cs):
            if share:
                return 100 * dc.loc[cs].sum() / nc.loc[cs].sum()
            den = rc.loc[cs].sum()
            return np.nan if den == 0 else 100 * dc.loc[cs].sum() / den
        pt = stat(cities)
        bs = np.array([stat(cities[rng.integers(0, len(cities), len(cities))]) for _ in range(400)])
        # city effects (pp or % of the city's reference), per seed
        if share:
            ce = 100 * D[v]
        else:
            base = r[v].groupby(level=0).transform('mean')
            ce = 100 * D[v] / base.where(base != 0)
        g = ce.groupby(level=0)
        mean, se = g.mean(), g.std() / np.sqrt(g.size())
        h = ce.reset_index()
        h['half'] = h.seed % 10 < 5
        hh = h.groupby(['half', 'city'])[v].mean()
        sh = hh.loc[True].corr(hh.loc[False], method='spearman')
        res.append(dict(window=w, var=v, kind='share' if share else 'level', effect=pt,
                        p05=np.nanquantile(bs, .05), p95=np.nanquantile(bs, .95),
                        n_pos=int((mean > 2 * se).sum()), n_neg=int((mean < -2 * se).sum()), split_half=sh,
                        ref_mean=ref.mean()))
        for c in cities:
            city_rows.append(dict(window=w, var=v, city=c, effect=mean.get(c), se=se.get(c),
                                  h1=hh.loc[True].get(c), h2=hh.loc[False].get(c)))
S = pd.DataFrame(res)
S['sig'] = (S.p05 > 0) | (S.p95 < 0)
S.to_csv(os.path.join(out, 'screen_all.csv'), index=False, float_format='%.5g')
pd.DataFrame(city_rows).to_csv(os.path.join(out, 'screen_city.csv'), index=False, float_format='%.5g')
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 400)
for w in WINDOWS:
    t = S[S.window == w].copy()
    t['snr'] = t.effect.abs() / ((t.p95 - t.p05) / 3.29)
    print(f'==== {w}: {t.sig.sum()} of {len(t)} indicators with the 90 % interval off zero ====')
    print(t.sort_values('snr', ascending=False)[['var', 'kind', 'effect', 'p05', 'p95', 'n_pos', 'n_neg',
                                                  'split_half', 'snr']].round(3).to_string(index=False))
