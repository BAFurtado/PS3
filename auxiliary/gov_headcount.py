"""Public jobs per municipality and year by place of residence, for the public headcount
(input/gov_headcount_census.csv, same columns as input/qtde_vinc_gov_rais_stable_from_2020_onwards.csv).

Level, 2010: IBGE, Censo Demográfico 2010, Resultados Gerais da Amostra, Sidra table 1577: employed persons whose main
job is "Empregado - militar e funcionário público estatutário", by municipality of residence, times the national ratio
of RAIS public jobs (input/qtde_vinc_gov_rais_stable_from_2020_onwards.csv, 2010) to that Census count. Nationally the
two count the same public sector, so the ratio covers what the Census category leaves out (public employees under CLT,
temporaries, misreporting); by municipality RAIS counts jobs where the employer is registered, which in state capitals
includes state staff who live and work elsewhere.
Path: each ACP's own log-linear trend of its RAIS public jobs 2010-2020 (sum over its municipalities), applied to its
municipalities' 2010 levels, held at the 2020 value afterwards, as the RAIS file is. Municipalities with no Census
figure keep their RAIS series.

Usage: python auxiliary/gov_headcount.py
"""
import numpy as np
import pandas as pd
import requests

RAIS = 'input/qtde_vinc_gov_rais_stable_from_2020_onwards.csv'
SIDRA = ('https://servicodados.ibge.gov.br/api/v3/agregados/1577/periodos/2010/variaveis/916'
         '?localidades=N{level}[{codes}]&classificacao=11913[96168]')
CHUNK = 50


def census(codes):
    out = {}
    for i in range(0, len(codes), CHUNK):
        url = SIDRA.format(level=6, codes=','.join(str(c) for c in codes[i:i + CHUNK]))
        for s in requests.get(url, timeout=300).json()[0]['resultados'][0]['series']:
            v = s['serie']['2010']
            if v not in ('-', '...', 'X'):
                out[int(s['localidade']['id'])] = float(v)
    return pd.Series(out)


def main():
    rais = pd.read_csv(RAIS)
    acps = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    acps['codemun'] = acps.cod_mun // 10
    national = requests.get(SIDRA.format(level=1, codes='all'), timeout=300).json()
    national = float(national[0]['resultados'][0]['series'][0]['serie']['2010'])
    ratio = rais[rais.ano == 2010].qtde_vinc_ativos.sum() / national
    level = census(sorted(acps.cod_mun.unique()))
    level.index = level.index // 10
    years = sorted(rais.ano.unique())
    rows = []
    for acp, group in acps.groupby('ACPs'):
        series = rais[rais.codemun.isin(group.codemun) & rais.ano.between(2010, 2020)].groupby('ano').qtde_vinc_ativos.sum()
        growth = np.polyfit(series.index - 2010, np.log(series.values), 1)[0] if (series > 0).all() else 0.0
        for mun in group.codemun.unique():
            if mun in level.index:
                base = level[mun] * ratio
                rows += [(mun, y, base * np.exp(growth * (min(y, 2020) - 2010))) for y in years]
            else:
                own = rais[rais.codemun == mun]
                rows += list(own[['codemun', 'ano', 'qtde_vinc_ativos']].itertuples(index=False, name=None))
    out = pd.DataFrame(rows, columns=['codemun', 'ano', 'qtde_vinc_ativos']).drop_duplicates(['codemun', 'ano'])
    out.sort_values(['codemun', 'ano']).to_csv('input/gov_headcount_census.csv', index=False, float_format='%.1f')
    print(f'national RAIS / Census ratio {ratio:.3f}, {out.codemun.nunique()} municipalities, '
          f'{(~acps.codemun.isin(level.index)).sum()} kept on RAIS')


if __name__ == '__main__':
    main()
