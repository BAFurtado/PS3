"""SINAPI average residential construction cost per m² by state, July 2010 (input/sinapi_2010.csv).

Source: IBGE, Sinapi, Sidra table 2296: variable 48 (custo médio m², moeda corrente, published from January 2013) and
variables 49 (número-índice) and 1196 (variação percentual no mês). The index is rebased in 2012 and the monthly change
is published from the rebase on, so July 2010 = January 2013 cost / [old index (last old-base month) / old index (July
2010) x product of (1 + monthly change) from the first new-base month to January 2013]. Normal finish
standard, materials and labour on site; no land, projects, licences, administration, financing or builder / developer
profit.

Rows: uf;sinapi   (R$ / m², 'BR' = Brazil)

Usage: python auxiliary/sinapi_2010.py
"""
import pandas as pd
import requests

URL = 'https://apisidra.ibge.gov.br/values/t/2296/n3/all/n1/all/v/{v}/p/{p}'
UF = {11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO', 21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN',
      25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA', 31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP', 41: 'PR', 42: 'SC',
      43: 'RS', 50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF', 1: 'BR'}


def get(v, p):
    d = pd.DataFrame(requests.get(URL.format(v=v, p=p), timeout=120).json()[1:])
    d['uf'] = d.D1C.astype(int).map(UF)
    d['V'] = pd.to_numeric(d.V, errors='coerce')
    return d.pivot(index='uf', columns='D3C', values='V')


def main():
    level = get(48, '201301')['201301']
    months = list(pd.period_range('2010-07', '2013-01', freq='M').strftime('%Y%m'))
    index = get(49, ','.join(months))[months]
    change = get(1196, ','.join(months))[months]
    old = [m for m in months if (index[m] > 1e6).all()]
    new = months[len(old):]
    assert (index[new] < 1e6).all().all() and change[new].notna().all().all()
    growth = index[old[-1]] / index['201007'] * (1 + change[new] / 100).prod(axis=1)
    print('last old-base month', old[-1], '| BR growth July 2010 - January 2013', round(growth['BR'], 4))
    out = (level / growth).rename('sinapi').round(2)
    out.to_frame().to_csv('input/sinapi_2010.csv', sep=';')
    print(out.sort_values().to_string())


if __name__ == '__main__':
    main()
