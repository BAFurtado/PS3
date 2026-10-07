"""People aged 17-69 and those of them employed, per municipality
(input/nonemployment_2010.csv).

Source: IBGE, Censo Demográfico 2010, Resultados Gerais da Amostra, Sidra table 1572: persons aged 10 or more by age and
"condição de atividade e de ocupação na semana de referência" (var 140), ages 17, 18, 19 and the five-year groups
20-24 to 65-69, total and "Economicamente ativas - ocupadas". The same ages as the model's unemployment statistic
(16 < age < 70, everyone without a job).

Usage: python auxiliary/census_nonemployment.py
"""
import pandas as pd
import requests

AGES = '6574,6575,6576,93087,93088,93089,93090,93091,93092,93093,93094,93095,93096'
TOTAL, EMPLOYED = '0', '104806'
URL = ('https://servicodados.ibge.gov.br/api/v3/agregados/1572/periodos/2010/variaveis/140'
       '?localidades=N6[{codes}]&classificacao=287[{ages}]|12251[{total},{employed}]')
CHUNK = 20


def fetch(codes):
    url = URL.format(codes=','.join(str(c) for c in codes), ages=AGES, total=TOTAL, employed=EMPLOYED)
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    return r.json()


def main():
    codes = sorted(pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';').cod_mun.unique())
    rows = []
    for i in range(0, len(codes), CHUNK):
        for res in fetch(codes[i:i + CHUNK])[0]['resultados']:
            cond = list(res['classificacoes'][1]['categoria'])[0]
            for s in res['series']:
                v = pd.to_numeric(s['serie']['2010'], errors='coerce')
                rows.append((int(s['localidade']['id']), 'employed' if cond == EMPLOYED else 'pop_17_69', v))
    df = pd.DataFrame(rows, columns=['cod_mun', 'col', 'v']).fillna(0.0)
    df = df.pivot_table(index='cod_mun', columns='col', values='v', aggfunc='sum').reset_index()
    missing = set(codes) - set(df.cod_mun)
    if missing:
        raise SystemExit(f'no data for {sorted(missing)}')
    df[['cod_mun', 'pop_17_69', 'employed']].to_csv('input/nonemployment_2010.csv', sep=';', index=False,
                                                     float_format='%.0f')
    print(f'{len(df)} municipalities; non-employment {1 - df.employed.sum() / df.pop_17_69.sum():.3f} overall')


if __name__ == '__main__':
    main()
