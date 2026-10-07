"""Resident population by municipality in the 2010 and 2022 Censuses, for the population targets
(input/census_population_2010_2022.csv).

Source: IBGE, Censo Demográfico 2010, Sidra table 202 (var 93, população residente, total of situação and sexo), and
Censo Demográfico 2022, Sidra table 4709 (var 93). Municipalities created after 2010 have no 2010 value and are left
out.

Usage: python auxiliary/census_population.py
"""
import pandas as pd
import requests

URL_2010 = 'https://apisidra.ibge.gov.br/values/t/202/n6/all/v/93/p/2010/c1/0/c2/0'
URL_2022 = 'https://apisidra.ibge.gov.br/values/t/4709/n6/all/v/93/p/2022'


def fetch(url, name):
    rows = requests.get(url, timeout=600).json()[1:]
    s = pd.DataFrame(rows)
    s = s[s.V.str.isdigit()]
    return s.set_index(s.D1C.astype(int)).V.astype(int).rename(name)


def main():
    out = pd.concat([fetch(URL_2010, 'pop_2010'), fetch(URL_2022, 'pop_2022')], axis=1, join='inner')
    out.index.name = 'cod_mun'
    out.sort_index().to_csv('input/census_population_2010_2022.csv', sep=';')
    print(len(out), 'municipalities;', out.sum().to_dict())


if __name__ == '__main__':
    main()
