"""ENMU investment and fares by ACP -> input/enmu_investment.csv.

Source: the ENMU project bank (Portal Mobilidade Brasil export, input/bndes/ProjetosSIG_*.csv, R$ million) and the
IBGE municipal GDP of 2023 (Sidra table 5938, variable 37) summed over the ACP's municipalities
(input/ACPs_MUN_CODES.csv).
    infra_capex     nominal infrastructure investment of the first cycle, all projects of the region, Cenario Padrao
                    (the Otimizado value is the same)
    gdp_2023        R$ million
    fare_padrao, fare_otimizado   median over the projects of 'Tarifa publica media predominante'
Usage (repository root): python auxiliary/enmu_investment.py
"""
import glob

import pandas as pd
import requests

REGIONS = {'RIDEDF': 'BRASILIA', 'RMF': 'FORTALEZA', 'RMG': 'GOIANIA', 'RMM': 'MANAUS', 'RMBS': 'BAIXADA SANTISTA',
           'RIDEGT': 'TERESINA'}
INFRA = 'Valor nominal do ciclo inicial dos investimentos em infraestrutura'
FARE = 'Tarifa pública média predominante'
OUT = 'input/enmu_investment.csv'


def number(s):
    if s.dtype != object:
        return s
    return pd.to_numeric(s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')


def gdp_2023(codes):
    url = 'https://apisidra.ibge.gov.br/values/t/5938/n6/{}/v/37/p/2023'.format(','.join(str(c) for c in codes))
    rows = requests.get(url, timeout=120).json()[1:]
    return sum(float(r['V']) for r in rows) / 1e3


if __name__ == '__main__':
    t = pd.read_csv(sorted(glob.glob('input/bndes/ProjetosSIG_*.csv'))[-1], sep=';', encoding='utf-8-sig', decimal=',')
    t = t[t['Sigla RM'].isin(REGIONS)]
    t[INFRA], t[FARE] = number(t[INFRA]), number(t[FARE])
    muns = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    rows = []
    for code, acp in REGIONS.items():
        p, o = (t[(t['Sigla RM'] == code) & (t['Cenário'] == s)] for s in ('Padrão', 'Otimizado'))
        rows.append({'acp': acp, 'enmu_region': code, 'projects': len(p), 'infra_capex': round(p[INFRA].sum(), 2),
                     'gdp_2023': round(gdp_2023(muns[muns.ACPs == acp].cod_mun.unique()), 2),
                     'fare_padrao': p[FARE].median(), 'fare_otimizado': o[FARE].median()})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, sep=';', index=False)
    print(out.to_string())
