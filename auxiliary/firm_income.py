"""Where firms' value added goes, from the national accounts, for firms' wage share and payout.

Outputs:
- input/firm_income_2015.csv, per sector (the model's twelve, IBGE nível 12 order):
  `wage_share` = (remunerations + gross mixed income) / value added of the activity; Real Estate net of imputed rent,
  whose value added is taken as the output of product 68002 "Aluguel imputado" (little intermediate consumption).
  The products firms invest in follow the FBCF column of input/final_demand.csv (MIP 2015), as public investment does.
- input/investment_rate_2015.csv: gross fixed capital formation / gross operating surplus of non-financial and
  financial corporations together.

Sources: IBGE, Sistema de Contas Nacionais 2015, Tabelas de Recursos e Usos, nível 12, tabela 2 (sheet VA) and nível 68, tabela 1 (sheet producao); Contas Econômicas Integradas 2015 (B.2 and P.51 by sector).
<https://ftp.ibge.gov.br/Contas_Nacionais/Sistema_de_Contas_Nacionais/2015/tabelas_xls/>

Usage: python auxiliary/firm_income.py
"""
import io
import zipfile

import pandas as pd
import requests

from sector_productivity import SECTORS, row, sheet

CEI = ('https://ftp.ibge.gov.br/Contas_Nacionais/Sistema_de_Contas_Nacionais/2015/tabelas_xls/'
       'contas_economicas_integradas/contas_economicas_integradas_2000a2015_xls.zip')
# Uses-side columns of the CEI sheet: non-financial corporations, financial corporations
CORPORATIONS = (9, 8)


def investment_rate():
    raw = zipfile.ZipFile(io.BytesIO(requests.get(CEI, timeout=300).content)).read('CEI2015.xls')
    d = pd.read_excel(io.BytesIO(raw), sheet_name='CEI', header=None)
    code = d[10].astype(str).str.strip()
    total = lambda c: sum(float(d[code == c].iloc[0, col]) for col in CORPORATIONS)  # noqa: E731
    return total('P.51') / total('B.2')


def main():
    va = sheet('12_2000_2015', '12_tab2_2015.xls', 'VA')
    added = row(va, 'Valor adicionado').iloc[1:13].astype(float).values
    labour = (row(va, 'Remunerações').iloc[1:13].astype(float).values
              + row(va, 'Rendimento misto').iloc[1:13].astype(float).values)
    production = sheet('68_2010_2015', '68_tab1_2015.xls', 'producao')
    imputed_row = production[production.iloc[:, 0].astype(str).str.startswith('68002')].iloc[0]
    imputed = float(pd.to_numeric(imputed_row.iloc[2:], errors='coerce').max())
    df = pd.DataFrame({'sector': SECTORS, 'value_added': added, 'labour': labour})
    df.loc[df.sector == 'RealEstate', 'value_added'] -= imputed
    df['wage_share'] = df.labour / df.value_added

    df[['sector', 'wage_share']].to_csv('input/firm_income_2015.csv', sep=';', index=False,
                                                      float_format='%.6f')
    rate = investment_rate()
    pd.DataFrame({'year': [2015], 'investment_rate': [rate]}).to_csv('input/investment_rate_2015.csv', sep=';',
                                                                      index=False, float_format='%.6f')
    print(df.round(3).to_string(index=False))
    print(f'corporate investment / gross operating surplus {rate:.3f}')


if __name__ == '__main__':
    main()
