"""Sector-to-sector moves of job seekers, 2010, for HIRING_RULE 'applications' (input/pme_sector_transitions_2010.csv).

Source: IBGE, Pesquisa Mensal de Emprego (PME), microdados 2010, months 01-12, six metropolitan regions, the person
match, states, ages and weights of auxiliary/labour_flows.py, with three more columns of the layout
(Documentacao/Layout/INPUT.txt, 1-based @ positions): V408A activity of the main job @183 (2), V446A activity of the
last job of persons without work @355 (2). Activities are CNAE-Domiciliar 1.0 divisions (Metodologia/CBO E CNAE - PME
para CD.xls), mapped to PS3's 12 sectors as auxiliary/own_account.py maps CNAE 2.0.

Rows: kind;origin;destination;persons (weighted, 11 month pairs):
  j2j   employee at t and t+1, job at t+1 held up to 30 days (V427 1): sector at t -> sector at t+1
  u2e   unemployed at t with a last job (V446A), employee at t+1: last sector -> sector at t+1

Run from the repository root."""
import pandas as pd

import auxiliary.labour_flows as L

OUT = 'input/pme_sector_transitions_2010.csv'
L.COLS.update({'v408a': (182, 184), 'v446a': (354, 356)})


def sector(code):
    """CNAE-Domiciliar 1.0 division -> PS3 sector"""
    try:
        d = int(code)
    except (TypeError, ValueError):
        return None
    if d == 0:
        return None
    if d <= 5:
        return 'Agriculture'
    if d <= 14:
        return 'Mining'
    if d <= 37:
        return 'Manufacturing'
    if d <= 41 or d == 90:
        return 'Utilities'
    if d == 45:
        return 'Construction'
    if d <= 53:
        return 'Trade'
    if d <= 63:
        return 'OtherServices' if d == 55 else 'Transport'
    if d == 64 or d == 72:
        return 'Business'
    if d <= 67:
        return 'Financial'
    if d == 70:
        return 'RealEstate'
    if d == 75:
        return 'Government'
    return 'OtherServices'


if __name__ == '__main__':
    months = {m: L.month(m) for m in range(1, 13)}
    for df in months.values():
        df['sec'] = df.v408a.map(sector)
        df['sec_last'] = df.v446a.map(sector)
    rows = []
    for m in range(1, 12):
        a, b = months[m], months[m + 1]
        a = a[(a.age >= 17) & (a.age <= 69)]
        p = a.merge(b, on=L.KEY, suffixes=('', '_1'))
        p = p[p.interview_1 == p.interview + 1]
        new = pd.to_numeric(p.v427_1, errors='coerce') == 1
        j = p[(p.state == 'EE') & (p.state_1 == 'EE') & new & p.sec.notna() & p.sec_1.notna()]
        rows.append(j.groupby(['sec', 'sec_1']).weight.sum().rename('persons').reset_index().assign(kind='j2j'))
        u = p[(p.state == 'U') & (p.state_1 == 'EE') & p.sec_last.notna() & p.sec_1.notna()]
        rows.append(u.groupby(['sec_last', 'sec_1']).weight.sum().rename('persons').reset_index()
                    .rename(columns={'sec_last': 'sec'}).assign(kind='u2e'))
    t = pd.concat(rows).groupby(['kind', 'sec', 'sec_1']).persons.sum().reset_index()
    t.columns = ['kind', 'origin', 'destination', 'persons']
    t.to_csv(OUT, sep=';', index=False, float_format='%.1f')
