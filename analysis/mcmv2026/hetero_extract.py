"""Heterogeneity extraction from mcmv2026 (d09, d13, d25; 26 capitals x 10 seeds), adapted from
funding2026/hetero_extract.py: two windows per run, dose (stats/regional 2026-01..2030-12, snapshots 2027-01..2031-01)
and after (2031-01..2034-12, snapshots 2032-01..2034-01); rows carry `window`. Extra groups in dist.csv: tenure
(renter, owner), benef (beneficiary, other), quintile (q1..q5 of the metro's person-weighted permanent income per
person in each snapshot).
usage (repo root, env ps3_2): python analysis/mcmv2026/hetero_extract.py [root] [cpus]
root default output/mcmv2026. Writes <root>/_analysis/hetero/{stats,mun,sector,dist}.csv, one row per run
(and municipality, sector x zone, or zone x group). Windows: stats.csv and regional.csv months 2026-01..2039-12,
parquet snapshots 2027-01..2039-01 (annual, end of January). Money is deflated by the run's price_level.
Zones: core = the capital municipality, periphery = the other municipalities of the metro. Neighbourhood income:
tercile of the metro's areas of weighting (AP) by Census 2010 income per person (input/income_per_person_AP_2010.csv).
Ledger columns that accumulate (money_*, ext_net_position, fpm, ...) enter as their monthly flow over the window."""
import datetime
import glob
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path[:0] = ['.']
from analysis.output import columns_for

root = sys.argv[1] if len(sys.argv) > 1 else 'output/mcmv2026'
cpus = int(sys.argv[2]) if len(sys.argv) > 2 else 6
out = os.path.join(root, '_analysis', 'hetero')
os.makedirs(out, exist_ok=True)
WINDOWS = {'dose': ('2026-01-01', '2030-12-31', datetime.date(2027, 1, 1), datetime.date(2031, 1, 1)),
           'after': ('2031-01-01', '2034-12-31', datetime.date(2032, 1, 1), datetime.date(2034, 1, 1))}
CORE = {'ARACAJU': 2800308, 'BELEM': 1501402, 'BELO HORIZONTE': 3106200, 'BOA VISTA': 1400100, 'BRASILIA': 5300108,
        'CAMPO GRANDE': 5002704, 'CUIABA': 5103403, 'CURITIBA': 4106902, 'FLORIANOPOLIS': 4205407,
        'FORTALEZA': 2304400, 'GOIANIA': 5208707, 'JOAO PESSOA': 2507507, 'MACAPA': 1600303, 'MACEIO': 2704302,
        'MANAUS': 1302603, 'NATAL': 2408102, 'PALMAS': 1721000, 'PORTO ALEGRE': 4314902, 'PORTO VELHO': 1100205,
        'RECIFE': 2611606, 'RIO BRANCO': 1200401, 'RIO DE JANEIRO': 3304557, 'SALVADOR': 2927408,
        'SAO LUIS': 2111300, 'SAO PAULO': 3550308, 'TERESINA': 2211001, 'VITORIA': 3205309}
CUMULATIVE = ['locally', 'fpm', 'ext_amount_sold', 'ext_net_position', 'ext_public_funding', 'money_public_transfers',
              'money_public_taxes_out', 'money_bank_profit_out', 'money_ogu', 'money_fgts_sbpe',
              'money_fgts_sbpe_repaid', 'money_immigrants', 'money_exports', 'money_imports', 'money_firm_entry',
              'money_social_transfers', 'money_profits_out', 'loan_requested', 'loan_approved', 'denied_recursos_fgts']
MONEY = ['gdp_level', 'gdp_change', 'families_median_permanent_income', 'families_wages_received', 'families_savings',
         'amount_subsidised', 'firms_total_profit', 'firms_median_profit', 'firms_median_wage_paid',
         'firms_wage_per_worker', 'workers_median_wage', 'house_price', 'house_rent', 'bank', 'credit_stock',
         'bank_balance', 'money_total', 'money_households', 'money_firms', 'money_public', 'money_bank',
         'money_deposits', 'families_total_permanent_income', 'families_total_income', 'families_total_wages',
         'own_account_earnings', 'taxes_products', 'families_rent_received', 'beneficiary_permanent_income',
         'beneficiary_total_income', 'firms_profit_share_paid', 'firms_investment', 'ext_imports', 'ext_exports',
         'household_imports'] + CUMULATIVE[:17]
SKIP = ['month', 'equally', 'emissions_fund', 'perc_fgts_used', 'perc_sbpe_used', 'firms_median_innovation_investment',
        'wage_census_factor', 'money_unexplained', 'money_bank_interest', 'money_import_tax', 'money_firm_writeoff',
        'ext_recycled', 'household_no_stock', 'rent_burden_decis_10'] + [c for c in columns_for('stats', 144)
                                                                         if c.startswith('denied_') and c != 'denied_recursos_fgts']
REG_MONEY = ['gdp_region', 'regional_house_values', 'qli_gdp_pc', 'qli_spend_pc', 'gdp_percapita', 'treasure',
             'median_permanent_income', 'mcmv_money_topup', 'families_total_permanent_income',
             'beneficiary_permanent_income']
REG_MEAN = ['commuting', 'pop', 'gdp_region', 'regional_gini', 'regional_house_values', 'regional_unemployment',
            'qli_index', 'gdp_percapita', 'treasure', 'affordability_ratio', 'median_permanent_income',
            'median_affordability', 'mcmv_eligible', 'families_count', 'families_total_permanent_income',
            'beneficiary_families']
REG_SUM = ['licenses', 'mcmv_units_bought', 'mcmv_money_topup']

ap_income = pd.read_csv('input/income_per_person_AP_2010.csv').set_index('AREAP').income_per_person


def read_csv(d, kind):
    f = os.path.join(d, kind + '.csv')
    n = len(next(line for line in open(f) if line.strip()).split(';'))
    x = pd.read_csv(f, sep=';', header=None, names=columns_for(kind, n))
    x['month'] = x.month.astype(str)
    return x


def gini(x):
    x = np.sort(np.asarray(x, float))
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.sum() <= 0:
        return np.nan
    n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


def dist_row(f, a):
    """Distribution indicators over one snapshot's families f (with pi_pc, wage_pc, rent, renter) and agents a."""
    r = {}
    pi = f.pi_pc.values
    w = np.repeat(pi, f.num_members.clip(lower=1).astype(int).values)  # person-weighted
    q = np.quantile(w, [0.1, 0.5, 0.9]) if len(w) else [np.nan] * 3
    s = np.sort(w)
    r.update(gini_pi_pc=gini(w), gini_wage_pc=gini(np.repeat(f.wage_pc.values,
                                                              f.num_members.clip(lower=1).astype(int).values)),
             top10_share=s[int(0.9 * len(s)):].sum() / s.sum() if len(s) else np.nan,
             bottom40_share=s[:int(0.4 * len(s))].sum() / s.sum() if len(s) else np.nan,
             p90_p10=q[2] / q[0] if q[0] > 0 else np.nan, median_pi_pc=q[1],
             mean_pi_pc=np.average(pi, weights=f.num_members.clip(lower=1)),
             mean_wage_pc=np.average(f.wage_pc, weights=f.num_members.clip(lower=1)),
             families=len(f), persons=f.num_members.sum(), members_per_family=f.num_members.mean(),
             renter_share=f.renter.mean(), zero_wage_share=(f.total_wage <= 0).mean(),
             benef_share=f.benef.mean(), savings_pc=np.average(f.savings_pc, weights=f.num_members.clip(lower=1)))
    rt = f[f.renter & (f.total_wage > 0)]
    burden = rt.house_rent / rt.total_wage
    r.update(rent_pc_renters=rt.rent_pc.mean(), rent_burden_median=burden.median(),
             rent_burden_over30=(burden > 0.3).mean(),
             owner_value_pc=f.loc[~f.renter, 'value_pc'].mean())
    wa = a[a.age.between(18, 64)]
    emp = wa[wa.firm_id.notna()]
    r.update(employment_rate=wa.firm_id.notna().mean(), commute_dist=emp.distance.mean(),
             works_other_mun=(emp.firm_mun != emp.home_mun).mean(),
             agent_money=a.money.mean())
    for lab, lo, hi in (('low', 0, 4), ('mid', 5, 11), ('high', 12, 99)):
        x = wa[wa.qualification.between(lo, hi)]
        r['employment_rate_educ_' + lab] = x.firm_id.notna().mean() if len(x) else np.nan
    return r


def one(d):
    out = [], [], [], []
    for window, (START, END, s0, s1) in WINDOWS.items():
        r = one_window(d, window, START, END, (s0, s1))
        for acc, x in zip(out, r):
            acc.append(x)
    return [out[0], pd.concat(out[1]), pd.concat(out[2]), pd.concat(out[3])]


def one_window(d, window, START, END, SNAP):
    s_name, city, i = d.split(os.sep)[-3:]
    key = dict(scenario=s_name, city=city, seed=int(i), window=window)
    core = CORE[city]
    # ---- stats.csv sweep
    st = read_csv(d, 'stats')
    p = st.set_index('month').price_level
    w = (st.month >= START) & (st.month <= END)
    pre = st[st.month < START].iloc[-1]
    W = st[w]
    row = dict(key)
    for c in st.columns:
        if c in SKIP:
            continue
        x = st[c].astype(float)
        if c in CUMULATIVE:
            fl = x.diff()
            if c in MONEY:
                fl = fl / st.price_level
            row[c] = fl[w].mean()
        elif c in MONEY:
            row[c] = (W[c] / W.price_level).mean()
        else:
            row[c] = W[c].mean()
    row['gdp_pc'] = row['gdp_level'] / row['pop']
    row['domiciles_added'] = W.number_domiciles.iloc[-1] - pre.number_domiciles
    row['families_added'] = W.families_count.iloc[-1] - pre.families_count
    row['pop_end'] = W['pop'].iloc[-1]
    row['consumption_household_pc'] = (W.demand_household - W.unmet_household).mean() / row['pop']
    row['import_share_household'] = (W.household_imports / W.demand_household).mean()
    row['trade_balance_gdp'] = ((W.ext_exports - W.ext_imports) / W.gdp_level).mean()
    row['imports_gdp'] = (W.ext_imports / W.gdp_level).mean()
    row['exports_gdp'] = (W.ext_exports / W.gdp_level).mean()
    row['taxes_gdp'] = (W.taxes_products / W.gdp_level).mean()
    row['wage_share_gdp'] = (W.families_total_wages / W.gdp_level).mean()
    row['profit_share_gdp'] = (W.firms_total_profit / W.gdp_level).mean()
    row['credit_gdp'] = (W.credit_stock / W.gdp_level).mean()
    row['price_income_ratio'] = (W.house_price / (W.families_total_income / W.families_count)).mean() / 12
    stats_row = row

    # ---- regional.csv by municipality
    rg = read_csv(d, 'regional')
    rg['p'] = rg.month.map(p)
    for c in REG_MONEY:
        rg[c] = rg[c] / rg.p
    R = rg[(rg.month >= START) & (rg.month <= END)]
    m = R.groupby('mun_id')[REG_MEAN].mean().join(R.groupby('mun_id')[REG_SUM].sum(), rsuffix='_sum')
    m.columns = [c if c not in REG_SUM or c in REG_MEAN else c + '_sum' for c in m.columns]
    m = m.rename(columns={'licenses': 'licenses_sum', 'mcmv_units_bought': 'mcmv_units_sum'})
    pre_r = rg[rg.month < START].groupby('mun_id').last()
    last_r = R.groupby('mun_id').last()
    m['pop_end'] = last_r['pop']
    m['pop_2025'] = pre_r['pop']
    m['families_end'] = last_r.families_count
    m['families_2025'] = pre_r.families_count

    # ---- parquet snapshots
    flt = [('month', '>=', SNAP[0]), ('month', '<=', SNAP[1])]
    fa = pd.read_parquet(os.path.join(d, 'families.parquet'), filters=flt)
    AG = ['month', 'region_id', 'id', 'age', 'qualification', 'firm_id', 'family_id', 'money', 'distance']
    if os.path.exists(os.path.join(d, 'agents.parquet')):
        ag = pd.read_parquet(os.path.join(d, 'agents.parquet'), columns=AG, filters=flt)
    else:  # runs without agents (Sao Paulo): no neighbourhood groups, agent indicators NaN
        ag = pd.DataFrame({c: pd.Series(dtype='float64') for c in AG})
    fi = pd.read_parquet(os.path.join(d, 'firms.parquet'), filters=flt)
    fa['mun_id'] = fa.mun_id.astype('int64')
    fi['mun_id'] = fi.mun_id.astype('int64')
    for x in (fa, ag, fi):
        x['month'] = x.month.astype(str)
        x['p'] = x.month.map(p)
    mem = fa.num_members.clip(lower=1)
    fa['renter'] = fa.house_rent.notna()
    fa['pi_pc'] = fa.permanent_income / mem / fa.p
    fa['wage_pc'] = fa.total_wage / mem / fa.p
    fa['rent_pc'] = fa.house_rent / mem / fa.p
    fa['value_pc'] = fa.house_price / mem / fa.p
    fa['savings_pc'] = fa.savings / mem / fa.p
    fa['benef'] = fa.programme != ''
    ag['money'] = ag.money / ag.p
    ag['home_ap'] = ag.region_id.astype('int64')
    ag['home_mun'] = ag.home_ap // 1000000
    firm_mun = fi.drop_duplicates(['month', 'firm_id']).set_index(['month', 'firm_id']).mun_id
    ag['firm_mun'] = pd.Series(list(zip(ag.month, ag.firm_id))).map(firm_mun).values
    # family's AP = AP of its members (one home)
    fam_ap = ag.groupby(['month', 'family_id']).home_ap.first()
    fa['ap'] = pd.Series(list(zip(fa.month, fa.id))).map(fam_ap).values
    aps = pd.Series(fa.ap.dropna().unique()).astype('int64')
    inc = ap_income.reindex(aps.values)
    terc = pd.Series(pd.qcut(inc.rank(method='first'), 3, labels=['poor', 'middle', 'rich']).astype(str).values,
                     index=aps.values) if len(aps) else pd.Series(dtype=str)
    fa['nbhd'] = fa.ap.map(terc)
    fa['tenure'] = np.where(fa.renter, 'renter', 'owner')
    fa['benefg'] = np.where(fa.benef, 'beneficiary', 'other')
    fa['quint'] = 'q?'
    for mo, g in fa.groupby('month'):
        rk = g.pi_pc.rank(method='first', pct=True)
        fa.loc[g.index, 'quint'] = 'q' + np.ceil(rk * 5).clip(1, 5).astype(int).astype(str)
    fam_tag = fa.set_index(['month', 'id'])[['tenure', 'benefg', 'quint']]
    for c in ('tenure', 'benefg', 'quint'):
        ag[c] = pd.Series(list(zip(ag.month, ag.family_id))).map(fam_tag[c]).values
    ag['nbhd'] = ag.home_ap.map(terc)
    fa['zone'] = np.where(fa.mun_id == core, 'core', 'periphery')
    ag['zone'] = np.where(ag.home_mun == core, 'core', 'periphery')

    dist = []
    for grp_col, grp_vals in (('all', ['all']), ('zone', ['core', 'periphery']), ('nbhd', ['poor', 'middle', 'rich']),
                              ('tenure', ['renter', 'owner']), ('benefg', ['beneficiary', 'other']),
                              ('quint', ['q1', 'q2', 'q3', 'q4', 'q5'])):
        for g in grp_vals:
            fs = fa if grp_col == 'all' else fa[fa[grp_col] == g]
            as_ = ag if grp_col == 'all' else ag[ag[grp_col] == g]
            if fs.empty:
                continue
            rr = [dist_row(fs[fs.month == mo], as_[as_.month == mo]) for mo in sorted(fs.month.unique())]
            dist.append(dict(key, group=grp_col, value=g, **pd.DataFrame(rr).mean().to_dict()))

    # municipality-level family and agent indicators
    fm = fa.groupby(['month', 'mun_id']).agg(renter_share=('renter', 'mean'), mean_pi_pc=('pi_pc', 'mean'),
                                             mean_wage_pc=('wage_pc', 'mean'), benef_share=('benef', 'mean'),
                                             rent_pc=('rent_pc', 'mean')).groupby('mun_id').mean()
    wa = ag[ag.age.between(18, 64)]
    am = wa.assign(emp=wa.firm_id.notna()).groupby(['month', 'home_mun']).agg(
        employment_rate=('emp', 'mean')).groupby('home_mun').mean()
    jobs = fi.groupby(['month', 'mun_id']).number_employees.sum().groupby('mun_id').mean().rename('jobs')
    cjobs = fi[fi.sector == 'Construction'].groupby(['month', 'mun_id']).number_employees.sum().groupby(
        'mun_id').mean().rename('construction_jobs')
    m = m.join(fm).join(am).join(jobs).join(cjobs)
    m['core'] = m.index == core
    for k, v in key.items():
        m[k] = v
    mun = m.reset_index().rename(columns={'index': 'mun_id'})

    # ---- firms by sector and zone
    fi['zone'] = np.where(fi.mun_id == core, 'core', 'periphery')
    for c in ('revenue', 'profit', 'wages_paid', 'input_cost', 'total_balance', 'price'):
        fi[c] = fi[c] / fi.p
    fi['active'] = fi.number_employees > 0
    rows = []
    for zone, z in (('all', fi), ('core', fi[fi.zone == 'core']), ('periphery', fi[fi.zone == 'periphery'])):
        if z.empty:
            continue
        g = z.groupby(['month', 'sector']).agg(
            employees=('number_employees', 'sum'), firms=('firm_id', 'size'), active_firms=('active', 'sum'),
            revenue=('revenue', 'sum'), profit=('profit', 'sum'), wages=('wages_paid', 'sum'),
            input_cost=('input_cost', 'sum'), produced=('amount_produced', 'sum'), sold=('amount_sold', 'sum'),
            stocks=('stocks', 'sum'), unmet=('unmet_quantity', 'sum'), emissions=('emissions', 'sum'),
            balance=('total_balance', 'sum'))
        g['price_w'] = (z.price * z.amount_sold).groupby([z.month, z.sector]).sum() / g.sold
        g = g.groupby('sector').mean()
        g['zone'] = zone
        rows.append(g.reset_index())
    sec = pd.concat(rows)
    for k, v in key.items():
        sec[k] = v
    return stats_row, mun, sec, pd.DataFrame(dist)


if __name__ == '__main__':
    dirs = sorted(os.path.dirname(x) for x in glob.glob(os.path.join(root, '*', '*', '*', 'DONE'))
                  if x.split(os.sep)[-4] in ('d09', 'd13', 'd25'))
    if len(sys.argv) > 3:
        dirs = [x for x in dirs if sys.argv[3] in x][:2]
    res = Parallel(n_jobs=cpus, verbose=5)(delayed(one)(x) for x in dirs)
    pd.DataFrame([x for r in res for x in r[0]]).to_csv(os.path.join(out, 'stats.csv'), index=False, float_format='%.6g')
    pd.concat([r[1] for r in res]).to_csv(os.path.join(out, 'mun.csv'), index=False, float_format='%.6g')
    pd.concat([r[2] for r in res]).to_csv(os.path.join(out, 'sector.csv'), index=False, float_format='%.6g')
    pd.concat([r[3] for r in res]).to_csv(os.path.join(out, 'dist.csv'), index=False, float_format='%.6g')
    print(len(res), 'runs')
