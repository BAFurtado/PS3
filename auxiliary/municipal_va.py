"""Value added of the market sectors by municipality, 2010, for PRODUCTIVITY_LEVEL 'municipal' (input/municipal_va_2010.csv).

Sources:
- IBGE, Produto Interno Bruto dos Municípios, Sidra 5938, 2010, R$ thousand: value added of agriculture (513), industry
  (517) and services other than public administration, defence, education, health and social security (6575).
- IBGE, Censo Demográfico 2010, Sidra 1378, variable 93: population.
- Imputed rent, which the model's households do not buy (HOUSEHOLD_REAL_ESTATE False), is taken out at its national
  share of the same value added: output of product 68002 "Aluguel imputado" (Tabelas de Recursos e Usos 2015, nível 68,
  tabela 1, the product has little intermediate consumption) over the value added of the eleven activities other than
  public administration (nível 12, tabela 2).
`va_market` = (agriculture + industry + services) x (1 - imputed share), R$ a year.

Usage: python auxiliary/municipal_va.py
"""
import pandas as pd
import requests

from sector_productivity import sheet

VA = 'https://servicodados.ibge.gov.br/api/v3/agregados/5938/periodos/2010/variaveis/{var}?localidades=N6[{codes}]'
POP = 'https://servicodados.ibge.gov.br/api/v3/agregados/1378/periodos/2010/variaveis/93?localidades=N6[{codes}]'
VARIABLES = {513: 'agriculture', 517: 'industry', 6575: 'services'}
CHUNK = 100


def series(url, codes, scale):
    out = {}
    for i in range(0, len(codes), CHUNK):
        for s in requests.get(url.format(codes=','.join(str(c) for c in codes[i:i + CHUNK])),
                              timeout=300).json()[0]['resultados'][0]['series']:
            out[int(s['localidade']['id'])] = pd.to_numeric(s['serie']['2010'], errors='coerce') * scale
    return pd.Series(out)


def imputed_share():
    va = sheet('12_2000_2015', '12_tab2_2015.xls', 'VA')
    market_va = float(va[va.iloc[:, 0].astype(str).str.strip().str.startswith('Valor adicionado bruto')].iloc[0, 1:12]
                      .astype(float).sum())
    production = sheet('68_2010_2015', '68_tab1_2015.xls', 'producao')
    row = production[production.iloc[:, 0].astype(str).str.startswith('68002')].iloc[0]
    return float(pd.to_numeric(row.iloc[2:], errors='coerce').max()) / market_va


def main():
    codes = sorted(pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';').cod_mun.unique())
    df = pd.DataFrame(index=pd.Index(codes, name='cod_mun'))
    df['pop'] = series(POP, codes, 1.0)
    for var, name in VARIABLES.items():
        df[name] = series(VA.replace('{var}', str(var)), codes, 1000.0)
    share = imputed_share()
    df['va_market'] = df[list(VARIABLES.values())].sum(axis=1, min_count=1) * (1 - share)
    df = df.dropna()
    df[['pop', 'va_market']].to_csv('input/municipal_va_2010.csv', sep=';', float_format='%.0f')
    print(f'{len(df)} municipalities, imputed rent share {share:.4f}, market value added per resident '
          f'R$ {df.va_market.sum() / df["pop"].sum():,.0f} a year')


if __name__ == '__main__':
    main()
