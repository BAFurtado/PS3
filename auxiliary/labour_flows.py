"""Monthly worker flows between employment and unemployment, 2010, for LABOUR_FLOWS 'data'
(input/labour_flows_2010.csv).

Source: IBGE, Pesquisa Mensal de Emprego (PME), microdados 2010, months 01-12 (local copy, PME below; downloaded from
ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Mensal_de_Emprego/Microdados/2010/, layout Documentacao/Layout/INPUT.txt).
Six metropolitan regions (V035): Recife, Salvador, Belo Horizonte, Rio de Janeiro, São Paulo, Porto Alegre.
A person seen in two consecutive months of the 4-8-4 rotation: same region, control number, series number and order
number (V035, V040, V050, V201), interview number one apart (V072), same sex and year of birth (V203, V224).
Condition in the reference week VD1: 1 employed, 2 unemployed, 3 not in the labour force. Position in the main job VD15:
1 employee (public and domestic included), 2 own-account, 3 employer, 4 unpaid. Ages 17-69 in the first month (V234).
Weights: first month's V215 (pesoexp1).

Rows: region;flow;rate;persons, region the V035 code or 'all'; flows (month t -> t+1):
  e_u     employed -> unemployed, over the employed
  ee_u    employee -> unemployed, over employees
  u_e     unemployed -> employed, over the unemployed
  u_ee    unemployed -> employee, over the unemployed
  u_oa    unemployed -> own-account, over the unemployed
  e_n     employed -> not in the labour force, over the employed
  u_n     unemployed -> not in the labour force, over the unemployed
  u       unemployment rate, first months
persons: weighted persons of the origin state (sum over the 11 month pairs).

Usage: python auxiliary/labour_flows.py
"""
import os

import pandas as pd

PME = os.path.expanduser('~/MyModels/pme2010/raw')
COLS = {'region': (0, 2), 'control': (2, 10), 'series': (10, 15), 'interview': (24, 25), 'order': (101, 103),
        'sex': (103, 104), 'birth_year': (108, 112), 'age': (112, 116), 'weight': (131, 138), 'vd1': (416, 417),
        'vd15': (430, 431)}
KEY = ['region', 'control', 'series', 'order', 'sex', 'birth_year']


def month(m):
    path = os.path.join(PME, f'PMEnova_{m:02d}2010', f'PMEnova.{m:02d}2010.txt')
    df = pd.read_fwf(path, colspecs=list(COLS.values()), names=list(COLS), dtype=str, encoding='latin-1')
    for c in ['region', 'interview', 'age', 'vd1', 'vd15']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['weight'] = pd.to_numeric(df.weight, errors='coerce') / 10
    df['state'] = state(df)
    return df.drop_duplicates(KEY, keep=False)


def state(df):
    s = df.vd1.map({1: 'E', 2: 'U', 3: 'N'})
    return s.where(~((df.vd1 == 1) & (df.vd15 == 1)), 'EE').where(~((df.vd1 == 1) & (df.vd15 == 2)), 'OA')


def pairs():
    months = {m: month(m) for m in range(1, 13)}
    out = []
    for m in range(1, 12):
        a, b = months[m], months[m + 1]
        a = a[(a.age >= 17) & (a.age <= 69)]
        p = a.merge(b, on=KEY, suffixes=('', '_1'))
        p = p[p.interview_1 == p.interview + 1]
        out.append(p[['region', 'weight', 'state', 'state_1']].rename(columns={'state': 's0', 'state_1': 's1'}))
    return pd.concat(out)


def rates(p):
    w = p.groupby(['s0', 's1']).weight.sum().unstack(fill_value=0.0)
    emp = ['EE', 'OA', 'E']
    e = w.loc[w.index.intersection(emp)].sum()
    u = w.loc['U']
    n_e, n_ee, n_u = e.sum(), w.loc['EE'].sum(), u.sum()
    u_e = u.reindex(emp).fillna(0.0).sum()
    rows = [('e_u', e.get('U', 0.0) / n_e, n_e), ('ee_u', w.loc['EE'].get('U', 0.0) / n_ee, n_ee),
            ('u_e', u_e / n_u, n_u), ('u_ee', u.get('EE', 0.0) / n_u, n_u), ('u_oa', u.get('OA', 0.0) / n_u, n_u),
            ('e_n', e.get('N', 0.0) / n_e, n_e), ('u_n', u.get('N', 0.0) / n_u, n_u), ('u', n_u / (n_u + n_e), n_u + n_e)]
    return pd.DataFrame(rows, columns=['flow', 'rate', 'persons'])


def main():
    p = pairs()
    out = [rates(p).assign(region='all')]
    for r, g in p.groupby('region'):
        out.append(rates(g).assign(region=str(int(r))))
    df = pd.concat(out)[['region', 'flow', 'rate', 'persons']]
    df.persons = df.persons.round(0)
    df.to_csv('input/labour_flows_2010.csv', sep=';', index=False, float_format='%.6f')
    print(df.pivot(index='region', columns='flow', values='rate').round(4).to_string())


if __name__ == '__main__':
    main()
