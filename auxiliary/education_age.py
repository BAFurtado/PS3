"""Education levels for EDUCATION 'census': people by age group and level per municipality
(input/education_age_2010.csv) and people aged 10+ by level per weighting area (input/education_AP_2010.csv).

Source: IBGE, Censo Demográfico 2010, Resultados Gerais da Amostra, Sidra table 3572: persons aged 10 or more by
"nível de instrução" and age group (var 140), all households, both sexes, all colours. Levels: 1 sem instrução e
fundamental incompleto, 2 fundamental completo e médio incompleto, 3 médio completo e superior incompleto,
4 superior completo; "não determinado" is left out. Age groups 10-14, 15-17, 18-19, five-year groups 20-24 to 55-59,
60-69 and 70+, labelled by their lower bound. Weighting areas: Sidra table 1554, persons aged 10 or more by "nível de
instrução" (var 140), territorial level N18, same levels.

Usage: python auxiliary/education_age.py
"""
import pandas as pd
import requests

LEVELS = {'9493': 1, '9494': 2, '9495': 3, '99713': 4}
AGES = {'1142': 10, '2792': 15, '110992': 18, '1144': 20, '1145': 25, '1146': 30, '1147': 35, '1148': 40, '1149': 45,
        '1150': 50, '1151': 55, '3520': 60, '3244': 70}
URL = ('https://servicodados.ibge.gov.br/api/v3/agregados/3572/periodos/2010/variaveis/140'
       '?localidades=N6[{codes}]&classificacao=1[0]|2[0]|86[0]|12056[0]|1568[{levels}]|58[{ages}]')
URL_AP = ('https://servicodados.ibge.gov.br/api/v3/agregados/1554/periodos/2010/variaveis/140'
          '?localidades=N18[{areas}]&classificacao=1568[{levels}]')
CHUNK = 20


def fetch(codes):
    url = URL.format(codes=','.join(str(c) for c in codes), levels=','.join(LEVELS), ages=','.join(AGES))
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    return r.json()


def fetch_areas(state):
    r = requests.get(URL_AP.format(areas=f'N3[{state}]', levels=','.join(LEVELS)), timeout=600)
    r.raise_for_status()
    rows = []
    for res in r.json()[0]['resultados']:
        level = LEVELS[list(res['classificacoes'][0]['categoria'])[0]]
        for s in res['series']:
            rows.append((s['localidade']['id'], level, pd.to_numeric(s['serie']['2010'], errors='coerce')))
    return rows


def main():
    codes = sorted(pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';').cod_mun.unique())
    rows = []
    for i in range(0, len(codes), CHUNK):
        for res in fetch(codes[i:i + CHUNK])[0]['resultados']:
            cats = {c['id']: list(c['categoria'])[0] for c in res['classificacoes']}
            for s in res['series']:
                v = pd.to_numeric(s['serie']['2010'], errors='coerce')
                rows.append((int(s['localidade']['id']), AGES[cats['58']], LEVELS[cats['1568']], v))
    df = pd.DataFrame(rows, columns=['cod_mun', 'age_group', 'level', 'pop']).fillna(0.0)
    df = df.groupby(['cod_mun', 'age_group', 'level'], as_index=False)['pop'].sum()
    missing = set(codes) - set(df.cod_mun)
    if missing:
        raise SystemExit(f'no data for {sorted(missing)}')
    df.to_csv('input/education_age_2010.csv', sep=';', index=False, float_format='%.0f')
    rows = []
    for state in sorted({str(c)[:2] for c in codes}):
        rows += fetch_areas(state)
    ap = pd.DataFrame(rows, columns=['area', 'level', 'pop']).fillna(0.0)
    ap = ap[ap.area.str[:7].isin(set(map(str, codes)))]
    ap = ap.pivot_table(index='area', columns='level', values='pop', aggfunc='sum').fillna(0.0)
    missing = set(map(str, codes)) - {a[:7] for a in ap.index}
    if missing:
        print(f'no weighting areas for {sorted(missing)}')
    ap.reset_index().to_csv('input/education_AP_2010.csv', sep=';', index=False, float_format='%.0f')
    adults = df[df.age_group.between(18, 60)].groupby('level')['pop'].sum()
    print(f'{df.cod_mun.nunique()} municipalities, {len(ap)} weighting areas; levels 1-4 at 18-69: {(adults / adults.sum()).round(3).tolist()}')


if __name__ == '__main__':
    main()
