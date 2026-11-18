"""mcmv2026 batch (metro pot + commission from 2010; OGU .09/.13/.25 % of GDP in 2026-30, .09 after; 26 capitals x 10
seeds, 2010-34): one row per run, then d13 - d09 and d25 - d09 at matched city and seed.
usage (repo root, env ps3_2): python analysis/mcmv2026/effects.py [root] [cpus]
root default output/mcmv2026. Writes <root>/_analysis/{runs,effects,ledger}.csv and prints
  1. macro effects per dose, median over cities and pooled, in the dose window 2026-30 and after it 2031-34,
  2. dose response: effect of d25 over effect of d13 against the ratio of their extra OGU money,
  3. where the money goes: ledger columns, cumulative 2026-01..2034-12, per real of extra OGU money (pooled,
     90 % bootstrap over cities),
  4. units handed over in the core municipality, share by period.
Money is deflated by each run's price_level except in the ledger (nominal, as in funding2026/ledger.py)."""
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
out = os.path.join(root, '_analysis')
os.makedirs(out, exist_ok=True)
REF = 'd09'
WINDOWS = {'dose': ('2026-01-01', '2030-12-31'), 'after': ('2031-01-01', '2034-12-31'),
           'all': ('2026-01-01', '2034-12-31')}
CORE = {'ARACAJU': 2800308, 'BELEM': 1501402, 'BELO HORIZONTE': 3106200, 'BOA VISTA': 1400100, 'BRASILIA': 5300108,
        'CAMPO GRANDE': 5002704, 'CUIABA': 5103403, 'CURITIBA': 4106902, 'FLORIANOPOLIS': 4205407,
        'FORTALEZA': 2304400, 'GOIANIA': 5208707, 'JOAO PESSOA': 2507507, 'MACAPA': 1600303, 'MACEIO': 2704302,
        'MANAUS': 1302603, 'NATAL': 2408102, 'PALMAS': 1721000, 'PORTO ALEGRE': 4314902, 'PORTO VELHO': 1100205,
        'RECIFE': 2611606, 'RIO BRANCO': 1200401, 'RIO DE JANEIRO': 3304557, 'SALVADOR': 2927408,
        'SAO LUIS': 2111300, 'SAO PAULO': 3550308, 'TERESINA': 2211001, 'VITORIA': 3205309}
FLOW = ['money_ogu', 'money_imports', 'money_public_taxes_out', 'money_profits_out', 'money_bank_profit_out',
        'money_exports', 'money_public_transfers', 'money_social_transfers', 'money_immigrants', 'money_firm_entry',
        'money_fgts_sbpe', 'money_fgts_sbpe_repaid', 'money_unexplained']
STOCK = ['money_total', 'money_households', 'money_firms', 'money_public', 'money_bank']
PERIODS = {'2010-25': ('2010-01-01', '2025-12-31'), '2026-30': ('2026-01-01', '2030-12-31'),
           '2031-34': ('2031-01-01', '2034-12-31')}


def read(d, kind):
    f = os.path.join(d, kind + '.csv')
    n = len(next(line for line in open(f) if line.strip()).split(';'))
    x = pd.read_csv(f, sep=';', header=None, names=columns_for(kind, n))
    x['month'] = x.month.astype(str)
    return x


def one(d):
    s, city, i = d.split(os.sep)[-3:]
    df = read(d, 'stats')
    p = df.price_level
    r = dict(scenario=s, city=city, seed=int(i), last_month=df.month.iloc[-1],
             u_last_year=df.unemployment.tail(12).mean())
    for wn, (a, b) in WINDOWS.items():
        w = (df.month >= a) & (df.month <= b)
        W, pw = df[w], p[w]
        benef = W.beneficiary_permanent_income / pw
        total = W.families_total_permanent_income / pw
        r.update({f'{k}_{wn}': v for k, v in dict(
            real_gdp=(W.gdp_level / pw).mean(), unemployment=W.unemployment.mean(),
            employment=W.firms_total_employment.mean(), pop=W['pop'].mean(), families=W.families_count.mean(),
            house_price=(W.house_price / pw).mean(), house_rent=(W.house_rent / pw).mean(),
            vacancy=W.house_vacancy.mean(), gini=W.gini_index.mean(), wages=(W.families_total_wages / pw).mean(),
            total_pi=total.mean(), benef_pi=benef.mean(), nonbenef_pi=(total - benef).mean(),
            benef_families=W.beneficiary_families.mean(), gdp_sum=(W.gdp_level / pw).sum(),
            ogu_real=((df.money_ogu.diff() / p)[w]).sum()).items()})
    # Ledger: nominal cumulative columns, change from 2025-12 to the last month
    k = df[df.month < '2026-01-01'].index[-1]
    w = df.month >= '2026-01-01'
    for x in FLOW + STOCK:
        r['L_' + x] = df[x].iloc[-1] - df[x].iloc[k]
    r['L_gdp'] = df.gdp_level[w].sum()
    r['L_wages'] = df.families_total_wages[w].sum()
    r['L_rent_received'] = df.families_rent_received[w].sum()
    r['L_taxes_products'] = df.taxes_products[w].sum()
    rg = read(d, 'regional')
    core = rg.mun_id.astype(str).str[:6] == str(CORE[city])[:6]
    for pn, (a, b) in PERIODS.items():
        m = (rg.month >= a) & (rg.month <= b)
        r[f'units_{pn}'] = rg.mcmv_units_bought[m].sum()
        r[f'units_core_{pn}'] = rg.mcmv_units_bought[m & core].sum()
    m = rg.month >= '2026-01-01'
    r['pop_core_share'] = rg['pop'][m & core].sum() / rg['pop'][m].sum()
    r['mcmv_money_residual_2034'] = rg[rg.month == rg.month.max()].mcmv_money_residual.sum()
    return r


dirs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(root, 'd*', '*', '*', 'DONE')))
R = pd.DataFrame(Parallel(n_jobs=cpus)(delayed(one)(d) for d in dirs))
R.to_csv(os.path.join(out, 'runs.csv'), index=False, float_format='%.6g')
print(R.groupby('scenario').agg(runs=('seed', 'size'), cities=('city', 'nunique'),
                                last=('last_month', 'min')).to_string(), '\n')

VARS = [c for c in R.columns if c not in ('scenario', 'city', 'seed', 'last_month')]
b = R[R.scenario == REF].set_index(['city', 'seed'])[VARS]
E = []
for s in ('d13', 'd25'):
    a = R[R.scenario == s].set_index(['city', 'seed'])[VARS]
    idx = a.index.intersection(b.index)
    e = (a.loc[idx] - b.loc[idx]).add_suffix('_d').join(b.loc[idx].add_suffix('_ref'))
    e['scenario'] = s
    E.append(e.reset_index())
E = pd.concat(E, ignore_index=True)
E.to_csv(os.path.join(out, 'effects.csv'), index=False, float_format='%.6g')

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
REL = ['real_gdp', 'employment', 'wages', 'pop', 'families', 'total_pi', 'nonbenef_pi', 'house_price', 'house_rent']
ABS = ['unemployment', 'vacancy', 'gini', 'benef_families']


def table(e, wn):
    """Median over cities of the city mean, % of reference (REL) or absolute (ABS); pooled = sum over all pairs;
    share of cities with paired |t| > 2 and the same sign as the median."""
    rows = {}
    g = e.groupby('city')
    for c in REL + ABS:
        dcol, rcol = f'{c}_{wn}_d', f'{c}_{wn}_ref'
        m = g[dcol].mean()
        t = m / (g[dcol].std() / np.sqrt(g.size()))
        val = 100 * m / g[rcol].mean() if c in REL else m
        pooled = 100 * m.sum() / g[rcol].mean().sum() if c in REL else m.mean()
        rows[c] = dict(unit='%' if c in REL else 'abs', median=val.median(), pooled=pooled,
                       cities_sig=(np.sign(val) == np.sign(val.median())).where(t.abs() > 2, False).mean())
    return pd.DataFrame(rows).T


for s in ('d13', 'd25'):
    e = E[E.scenario == s]
    for wn in ('dose', 'after'):
        inj = 100 * e[f'ogu_real_{wn}_d'].sum() / e[f'gdp_sum_{wn}_ref'].sum()
        print(f'==== {s} - {REF}, {WINDOWS[wn][0][:4]}-{WINDOWS[wn][1][:4]}: extra OGU {inj:.3f} % of GDP ====')
        print(table(e, wn).to_string(float_format=lambda v: f'{v:.3f}'), '\n')

print('==== dose response, 2026-30: effect d25 / effect d13 (pooled), extra money ratio first ====')
d13, d25 = E[E.scenario == 'd13'], E[E.scenario == 'd25']
d25 = d25[d25.city.isin(d13.city.unique())]
dr = {'extra_ogu': d25.ogu_real_dose_d.sum() / d13.ogu_real_dose_d.sum()}
for c in REL + ABS:
    dr[c] = d25[f'{c}_dose_d'].sum() / d13[f'{c}_dose_d'].sum()
print(pd.Series(dr).round(2).to_string(), '\n')

# Ledger per real of extra OGU money, cumulative 2026-34, pooled with a bootstrap over cities
L = [c for c in E.columns if c.startswith('L_') and c.endswith('_d')]
rng = np.random.default_rng(0)
led = {}
for s in ('d13', 'd25'):
    cs = E[E.scenario == s].groupby('city')[L].mean()
    cols = [c for c in L if c != 'L_money_ogu_d']
    pooled = cs[cols].sum() / cs['L_money_ogu_d'].sum()
    boot = pd.DataFrame([(lambda x: x[cols].sum() / x['L_money_ogu_d'].sum())(cs.iloc[rng.integers(0, len(cs), len(cs))])
                         for _ in range(2000)])
    led[s] = pd.DataFrame({'per_real': pooled, 'p05': boot.quantile(.05), 'p95': boot.quantile(.95)})
    flows = [f'L_{x}_d' for x in FLOW]
    print(f'{s}: ledger identity, change in money_total / sum of flows =',
          round(cs['L_money_total_d'].sum() / cs[flows].sum(axis=1).sum(), 5))
led = pd.concat(led, axis=1)
led.index = [i[2:-2] for i in led.index]
led.to_csv(os.path.join(out, 'ledger.csv'), float_format='%.6g')
print('per real of extra OGU money, cumulative 2026-34, pooled over cities x seeds, 90 % bootstrap over cities')
print(led.round(3).to_string(), '\n')

print('==== units handed over: core share by period (pooled over cities and seeds), and core population share ====')
for s, g in R.groupby('scenario'):
    gc = g.groupby('city').mean(numeric_only=True)
    sh = {p: gc[f'units_core_{p}'].sum() / gc[f'units_{p}'].sum() for p in PERIODS}
    print(s, {k: round(v, 3) for k, v in sh.items()}, 'pop core 2026-34', round(g.pop_core_share.mean(), 3),
          'units/run', {p: round(g[f'units_{p}'].mean(), 1) for p in PERIODS})
gy = R[R.city == 'GOIANIA']
print('GOIANIA core share 2010-25 by dose:', (gy.groupby('scenario').units_core_2010_25 if False else
      gy.groupby('scenario')[['units_core_2010-25', 'units_2010-25']].sum().pipe(
          lambda x: (x.iloc[:, 0] / x.iloc[:, 1]).round(3)).to_dict()), '(observed 2009-16: 0.52)')
print('\nscreen: runs with unemployment > 30 % in the last year:')
print(R[R.u_last_year > .30].groupby(['scenario', 'city']).size().to_string())
