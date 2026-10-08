"""Yearly union formation and separation rates by sex and five-year age group, Brazil 2010 (input/union_rates_2010.csv),
and the share of households with a couple in each ACP (input/couple_households_2010.csv).

Sources:
- Census 2010 sample microdata, persons (local fixed-width files): V0601 sex, V6036 age, V0637 lives with a spouse or
  partner (1 yes, 2 no but has lived, 3 never), V0640 civil status (1 married), V0010 weight. Unions include
  consensual ones.
- IBGE Estatísticas do Registro Civil 2010 (Sidra): table 2759, registered marriages by sex and age group; table 1695,
  divorces granted in first instance or by deed, by age group of the husband and of the wife.

Separation: divorces / legally married persons of the sex and age group. Divorces end legal marriages only, so this is
a lower bound for the dissolution of all unions.
Formation, per person not in a union: the larger of
- the Census cross-section, (dU/da + separation x U) / (1 - U), U the share in a union by age (central differences
  over the groups), and
- registered marriages / persons not in a union, a lower bound, since consensual unions are not registered.
Widowhood is left out of the cross-section, so the registered rate binds from the mid-forties on.

Rows: sex;age;in_union;formation;separation   (sex male / female, age = lower bound of the group, 15 to 75+; in_union =
Census share living with a spouse or partner)

Couples: persons who are the spouse or partner of the household's responsible person (V0502 2 or 3) / responsible
persons (V0502 1), weighted, over the ACP's municipalities (input/ACPs_MUN_CODES.csv).
Rows: acp;share

Usage: python auxiliary/union_rates.py
"""
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

MICRODATA = '/home/furtado/MyModels/censo2010/data/amostra/*/Amostra_Pessoas_*.txt'
SIDRA = 'https://apisidra.ibge.gov.br/values'
GROUPS = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
OPEN_GROUPS = (65, 70, 75)


def census_stock():
    """{(sex, group): [in union, not in union, legally married]} weighted persons aged 15+, and {municipality:
    [responsible persons, their spouses]} weighted"""
    acc = defaultdict(lambda: np.zeros(3))
    heads = defaultdict(lambda: np.zeros(2))
    for path in sorted(glob.glob(MICRODATA)):
        with open(path, encoding='latin-1') as f:
            for line in f:
                relation = line[53:55]
                if relation in ('01', '02', '03'):
                    heads[int(line[0:7])][0 if relation == '01' else 1] += int(line[28:44]) / 1e13
                age = int(line[61:64])
                if age < 15:
                    continue
                w = int(line[28:44]) / 1e13
                key = ('male' if line[57] == '1' else 'female', max(g for g in GROUPS if g <= age))
                acc[key][0 if line[189] == '1' else 1] += w
                if line[193] == '1':
                    acc[key][2] += w
    return acc, heads


def sidra(path, age_dim, totals):
    """{age group label: count} for one Sidra query, the other classifications at their totals"""
    rows = requests.get(f'{SIDRA}/{path}', timeout=120).json()[1:]
    out = defaultdict(float)
    for r in rows:
        if all(r[d] == 'Total' for d in totals) and r['V'] not in ('-', '...', 'X'):
            out[r[age_dim]] += float(r['V'])
    return out


def label(group):
    if group == 15:
        return ['15 a 19 anos', 'Menos de 20 anos', 'Menos de 15 anos']
    if group == 75:
        return ['75 anos ou mais']
    return [f'{group} a {group + 4} anos']


def main():
    stock, heads = census_stock()
    acps = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    couples = []
    for acp, muns in acps.groupby('ACPs').cod_mun:
        h = sum((heads[m] for m in muns if m in heads), np.zeros(2))
        if h[0] > 0:
            couples.append((acp, h[1] / h[0]))
    couples = pd.DataFrame(couples, columns=['acp', 'share'])
    couples.to_csv('input/couple_households_2010.csv', sep=';', index=False, float_format='%.4f')
    print(couples.to_string(index=False))
    marriages = {
        'male': sidra('t/2759/n1/all/v/221/p/2010/c236/0/c247/0/c248/0/c245/all/c246/0', 'D7N',
                      ['D4N', 'D5N', 'D6N', 'D8N']),
        'female': sidra('t/2759/n1/all/v/221/p/2010/c236/0/c247/0/c248/0/c245/0/c246/all', 'D8N',
                        ['D4N', 'D5N', 'D6N', 'D7N'])}
    divorces = {
        'male': sidra('t/1695/n1/all/v/393/p/2010/c345/0/c274/all/c269/0/c275/0', 'D5N', ['D4N', 'D6N', 'D7N']),
        'female': sidra('t/1695/n1/all/v/393/p/2010/c345/0/c274/0/c269/0/c275/all', 'D7N', ['D4N', 'D5N', 'D6N'])}
    rows = []
    for sex in ('male', 'female'):
        union = np.array([stock[(sex, g)][0] for g in GROUPS])
        single = np.array([stock[(sex, g)][1] for g in GROUPS])
        married = np.array([stock[(sex, g)][2] for g in GROUPS])
        share = union / (union + single)
        separation = np.array([sum(divorces[sex].get(k, 0) for k in label(g)) for g in GROUPS]) / married
        # Registered marriages at 65+ come as one group: shared by the persons not in a union
        registered = np.array([sum(marriages[sex].get(k, 0) for k in label(g)) for g in GROUPS], dtype=float)
        open_ = [GROUPS.index(g) for g in OPEN_GROUPS]
        registered[open_] = marriages[sex].get('65 anos ou mais', 0) * single[open_] / single[open_].sum()
        cross_section = (np.gradient(share, 5.0) + separation * share) / (1 - share)
        formation = np.maximum(cross_section, registered / single)
        rows += [(sex, g, u, f, s) for g, u, f, s in zip(GROUPS, share, formation, separation)]
    out = pd.DataFrame(rows, columns=['sex', 'age', 'in_union', 'formation', 'separation'])
    out.to_csv('input/union_rates_2010.csv', sep=';', index=False, float_format='%.5f')
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
