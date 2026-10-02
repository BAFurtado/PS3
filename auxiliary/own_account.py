"""Employed people aged 17-69 and the own-account workers and employees among them, by municipality, education level
and sector, for OWN_ACCOUNT 'firms' and 'pool' and POSTING_EDUCATION 'census' (input/own_account_2010.csv).

Source: IBGE, Censo Demográfico 2010, microdados da amostra, persons file of each state (local copy, CENSUS below).
Employed: V6910 = 1 (ocupadas na semana de referência). Own-account: V6930 = 4 (conta própria, trabalho principal).
Education level V6400: 1 sem instrução e fundamental incompleto, 2 fundamental completo e médio incompleto, 3 médio
completo e superior incompleto, 4 superior completo; 5 (não determinado) left out. Sector: nível 12 of the
input-output matrix from the CNAE Domiciliar 2.0 division of the main job (V6471), grouped as
auxiliary/sector_shares_ibge12.py groups CNAE 2.0 sections (Trade = G, Business = J, Government = O, OtherServices = I,
M, N, P, Q, R, S, T, U); division 00 (atividades mal definidas) is 'Unknown'. Weights V0010.

Employees: V6930 = 1, 2, 3 (com carteira, militares e estatutários, sem carteira; domestic workers included), for
POSTING_EDUCATION.

Rows: cod_mun;level;position;sector;persons, position 'employed' (sector 'all'), 'own_account' or 'employee'.

input/own_account_income_2010.csv: monthly work income in the main job (V6513, R$ of July 2010) of the employed aged
17-69 with income, and of the own-account workers among them, by municipality and sector (weights V0010), for
OWN_ACCOUNT 'pool': cod_mun;sector;work_income;own_account_income.

Usage: python auxiliary/own_account.py
"""
import os

import pandas as pd

CENSUS = os.path.expanduser('~/MyModels/censo2010/data/amostra')
FILES = {11: ['RO/Amostra_Pessoas_11.txt'], 12: ['AC/Amostra_Pessoas_12.txt'], 13: ['AM/Amostra_Pessoas_13.txt'],
         14: ['RR/Amostra_Pessoas_14.txt'], 15: ['PA/Amostra_Pessoas_15.txt'], 16: ['AP/Amostra_Pessoas_16.txt'],
         17: ['TO/Amostra_Pessoas_17.txt'], 21: ['MA/Amostra_Pessoas_21.txt'], 22: ['PI/Amostra_Pessoas_22.txt'],
         23: ['CE/Amostra_Pessoas_23.txt'], 24: ['RN/Amostra_Pessoas_24.txt'], 25: ['PB/Amostra_Pessoas_25.txt'],
         26: ['PE/Amostra_Pessoas_26.txt'], 27: ['AL/Amostra_Pessoas_27.txt'], 28: ['SE/Amostra_Pessoas_28.txt'],
         29: ['BA/Amostra_Pessoas_29.txt'], 31: ['MG/Amostra_Pessoas_31.txt'], 32: ['ES/Amostra_Pessoas_32.txt'],
         33: ['RJ/Amostra_Pessoas_33.txt'], 35: ['SP1/Amostra_Pessoas_35_outras.txt', 'SP2-RM/Amostra_Pessoas_35_RMSP.txt'],
         41: ['PR/Amostra_Pessoas_41.txt'], 42: ['SC/Amostra_Pessoas_42.txt'], 43: ['RS/Amostra_Pessoas_43.txt'],
         50: ['MS/Amostra_Pessoas_50.txt'], 51: ['MT/Amostra_Pessoas_51.txt'], 52: ['GO/Amostra_Pessoas_52.txt'],
         53: ['DF/Amostra_Pessoas_53.txt']}
# CNAE 2.0 division upper bounds -> nível 12 sector
DIVISIONS = [(3, 'Agriculture'), (9, 'Mining'), (33, 'Manufacturing'), (39, 'Utilities'), (43, 'Construction'),
             (48, 'Trade'), (53, 'Transport'), (56, 'OtherServices'), (63, 'Business'), (66, 'Financial'),
             (68, 'RealEstate'), (83, 'OtherServices'), (84, 'Government'), (99, 'OtherServices')]


def sector(cnae):
    try:
        d = int(cnae[:2])
    except ValueError:
        return 'Unknown'
    if d == 0:
        return 'Unknown'
    return next(s for top, s in DIVISIONS if d <= top)


def main():
    muns = set(pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';').cod_mun.astype(str))
    counts, income = {}, {}
    for uf, files in FILES.items():
        if not any(m.startswith(str(uf)) for m in muns):
            continue
        for name in files:
            with open(os.path.join(CENSUS, name), encoding='latin-1') as f:
                for line in f:
                    mun = line[0:7]
                    if mun not in muns or line[391] != '1':
                        continue
                    age, level = int(line[61:64]), line[157]
                    if not 17 <= age <= 69 or level not in '1234':
                        continue
                    w = float(line[28:44]) / 1e13
                    key = (int(mun), int(level), 'employed', 'all')
                    counts[key] = counts.get(key, 0.0) + w
                    if line[393] in '1234':
                        position = 'own_account' if line[393] == '4' else 'employee'
                        key = (int(mun), int(level), position, sector(line[203:208]))
                        counts[key] = counts.get(key, 0.0) + w
                    try:
                        work = float(line[218:224])
                    except ValueError:
                        work = 0.0
                    if work > 0:
                        key = (int(mun), sector(line[203:208]))
                        total, own = income.get(key, (0.0, 0.0))
                        income[key] = (total + w * work, own + w * work * (line[393] == '4'))
        print(uf, len(counts))
    df = pd.DataFrame([(*k, v) for k, v in counts.items()], columns=['cod_mun', 'level', 'position', 'sector', 'persons'])
    df = df.sort_values(['cod_mun', 'level', 'position', 'sector'])
    df['persons'] = df.persons.round(2)
    df.to_csv('input/own_account_2010.csv', sep=';', index=False)
    inc = pd.DataFrame([(m, s, t, o) for (m, s), (t, o) in income.items()],
                       columns=['cod_mun', 'sector', 'work_income', 'own_account_income']).sort_values(['cod_mun', 'sector'])
    inc[['work_income', 'own_account_income']] = inc[['work_income', 'own_account_income']].round(0)
    inc.to_csv('input/own_account_income_2010.csv', sep=';', index=False)


if __name__ == '__main__':
    main()
