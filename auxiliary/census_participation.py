"""People, economically active people and employed people by municipality, sex and age group, for who is in the labour force
'census' (input/participation_2010.csv).

Source: IBGE, Censo Demográfico 2010, Resultados Gerais da Amostra, Sidra table 3573: persons aged 10 or more by
"condição de atividade e de ocupação na semana de referência" (classification 12049: total, "Economicamente ativas",
"Economicamente ativas - ocupadas"), sex and age group (16-17, 18-19, the five-year groups 20-24 to 55-59, 60-69).
`age_group` is the group's first age.

Usage: python auxiliary/census_participation.py
"""
import pandas as pd
import requests

AGES = {114534: 16, 110992: 18, 1144: 20, 1145: 25, 1146: 30, 1147: 35, 1148: 40, 1149: 45, 1150: 50, 1151: 55,
        3520: 60}
CONDITIONS = {0: 'pop', 99497: 'active', 99498: 'employed'}
SEXES = {4: 'male', 5: 'female'}
URL = ('https://servicodados.ibge.gov.br/api/v3/agregados/3573/periodos/2010/variaveis/140'
       '?localidades=N6[{codes}]&classificacao=12049[{conditions}]|2[{sexes}]|58[{ages}]|455[0]|184[0]')
CHUNK = 20


def fetch(codes):
    url = URL.format(codes=','.join(str(c) for c in codes),
                     conditions=','.join(str(c) for c in CONDITIONS),
                     sexes=','.join(str(c) for c in SEXES),
                     ages=','.join(str(c) for c in AGES))
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    return r.json()


def main():
    codes = sorted(pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';').cod_mun.unique())
    rows = []
    for i in range(0, len(codes), CHUNK):
        for res in fetch(codes[i:i + CHUNK])[0]['resultados']:
            cats = {c['id']: int(list(c['categoria'])[0]) for c in res['classificacoes']}
            for s in res['series']:
                v = pd.to_numeric(s['serie']['2010'], errors='coerce')
                rows.append((int(s['localidade']['id']), SEXES[cats['2']], AGES[cats['58']],
                             CONDITIONS[cats['12049']], v))
    df = pd.DataFrame(rows, columns=['cod_mun', 'gender', 'age_group', 'col', 'v']).fillna(0.0)
    df = df.pivot_table(index=['cod_mun', 'gender', 'age_group'], columns='col', values='v',
                        aggfunc='sum').reset_index()
    missing = set(codes) - set(df.cod_mun)
    if missing:
        raise SystemExit(f'no data for {sorted(missing)}')
    df[['cod_mun', 'gender', 'age_group', 'pop', 'active', 'employed']].to_csv(
        'input/participation_2010.csv', sep=';', index=False, float_format='%.0f')
    print(f'{df.cod_mun.nunique()} municipalities; participation {df.active.sum() / df["pop"].sum():.3f}, '
          f'unemployment {1 - df.employed.sum() / df.active.sum():.3f} overall')


if __name__ == '__main__':
    main()
