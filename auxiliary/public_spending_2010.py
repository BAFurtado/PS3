"""Public investment per unit of public payroll, 2010 (input/public_spending_2010.csv).

Source: IBGE, Contas Econômicas Integradas 2010 (Sistema de Contas Nacionais, referência 2010), general government
(S.13): gross fixed capital formation (P.51) over compensation of employees paid (D.1); and, for reference,
intermediate consumption (P.2) over D.1.

Usage: python auxiliary/public_spending_2010.py <download dir>
"""
import os
import sys
import zipfile

import pandas as pd
import requests

CEI = ('https://ftp.ibge.gov.br/Contas_Nacionais/Sistema_de_Contas_Nacionais/2011/tabelas_xls/'
       '03_00_contas_economicas_integradas_2010_xls.zip')


def main():
    d = sys.argv[1]
    path = os.path.join(d, 'CEI2010.xls')
    if not os.path.exists(path):
        z = os.path.join(d, 'cei2010.zip')
        open(z, 'wb').write(requests.get(CEI, timeout=300).content)
        zipfile.ZipFile(z).extractall(d)
    cei = pd.read_excel(path, 'CEI', header=None).iloc[8:]
    cei = cei[[10, 7]].rename(columns={10: 'code', 7: 'U_S13'}).dropna(subset=['code'])
    cei['code'] = cei.code.astype(str).str.strip()
    c = cei.drop_duplicates('code').set_index('code').U_S13.astype(float)
    out = pd.DataFrame([('investment_per_payroll', c['P.51'] / c['D.1']),
                        ('intermediate_per_payroll', c['P.2'] / c['D.1'])], columns=['ratio', 'value'])
    out.to_csv('input/public_spending_2010.csv', sep=';', index=False, float_format='%.5f')
    print(out.to_string(index=False), c[['D.1', 'P.2', 'P.51']].to_dict())


if __name__ == '__main__':
    main()
