"""Tax rates and the municipal share of product taxes, 2010 (input/product_tax_2010.csv, input/taxes_2010.csv,
input/tax_shares_2010.csv).

Sources:
- IBGE, Matriz de Insumo-Produto 2010, nível 12: tabela 02 (usos a preços de consumidor), 05 / 06 (destino dos impostos
  sobre produtos nacionais / importados). Per product: net taxes over its domestic uses (exports are untaxed), and over
  all its uses.
- IBGE, Contas Econômicas Integradas 2010. Labour: employer contributions (D.12) + households' contributions (D.613) +
  household income tax (D.5, S.14) x the wage share of households' primary income (D.11 / (D.11 + B.3 + D.4 received)),
  over compensation (D.1). Rent: household income tax over that primary income. Firms: corporate income tax (D.5,
  S.11 + S.12) over their gross operating surplus (B.2). Salary share: wages and salaries (D.11) over compensation.
- STN, Finbra, municipal revenue 2010 (Ipeadata series RISSM, RIPTUM, RICMSM) and IBGE PIB dos Municípios 2010 (Sidra
  5938, v543 net taxes on products): per ACP, (ISS + cota-parte ICMS) / net taxes on products; IPTU over the value of
  the housing stock, households x Census mean rent x 12 / RENTAL_YIELD (input/rent_AP_2010.csv).

Usage: python auxiliary/taxes_2010.py <download dir>
"""
import os
import sys
import time
import zipfile

import pandas as pd
import requests

MIP = ('https://ftp.ibge.gov.br/Contas_Nacionais/Matriz_de_Insumo_Produto/2010/'
       'Matriz_de_Insumo_Produto_2010_Nivel_12_20161019.xls')
CEI = ('https://ftp.ibge.gov.br/Contas_Nacionais/Sistema_de_Contas_Nacionais/2011/tabelas_xls/'
       '03_00_contas_economicas_integradas_2010_xls.zip')
IPEA = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{}')"
SIDRA = 'https://servicodados.ibge.gov.br/api/v3/agregados/5938/periodos/2010/variaveis/543?localidades=N6[all]'
SECTORS = ['Agriculture', 'Mining', 'Manufacturing', 'Utilities', 'Construction', 'Trade', 'Transport', 'Business',
           'Financial', 'RealEstate', 'OtherServices', 'Government']
RENTAL_YIELD = 0.0664


def download(d):
    mip = os.path.join(d, 'mip2010_12.xls')
    if not os.path.exists(mip):
        open(mip, 'wb').write(requests.get(MIP, timeout=300).content)
        z = os.path.join(d, 'cei2010.zip')
        open(z, 'wb').write(requests.get(CEI, timeout=300).content)
        zipfile.ZipFile(z).extractall(d)
    return mip, os.path.join(d, 'CEI2010.xls')


def product_taxes(mip):
    x = pd.ExcelFile(mip)
    rows = slice(5, 17)
    use = pd.read_excel(x, '02', header=None).iloc[rows, 2:].astype(float).reset_index(drop=True)
    tax = (pd.read_excel(x, '05', header=None).iloc[rows, 2:].astype(float).reset_index(drop=True)
           + pd.read_excel(x, '06', header=None).iloc[rows, 2:].astype(float).reset_index(drop=True))
    # tax columns: 2 total, 16 exports; use columns: 15 exports, 22 total
    return pd.DataFrame({'sector': SECTORS,
                         'domestic': ((tax[2] - tax[16]) / (use[22] - use[15])).fillna(0.0).values,
                         'sales': (tax[2] / use[22]).fillna(0.0).values})


def national_rates(path):
    cei = pd.read_excel(path, 'CEI', header=None).iloc[8:]
    cei = cei[[10, 4, 6, 8, 9, 15]].rename(columns={10: 'code', 4: 'U_S1', 6: 'U_S14', 8: 'U_S12', 9: 'U_S11',
                                                     15: 'R_S14'}).dropna(subset=['code'])
    cei['code'] = cei.code.astype(str).str.strip()
    c = cei.drop_duplicates('code').set_index('code').astype(float)
    comp, wages = c.loc['D.1', 'U_S1'], c.loc['D.11', 'U_S1']
    primary = wages + c.loc['B.3', 'R_S14'] + c.loc['D.4', 'R_S14']
    income_tax = c.loc['D.5', 'U_S14'] / primary
    return pd.DataFrame([
        ('labour', (c.loc['D.12', 'U_S1'] + c.loc['D.613', 'U_S14'] + income_tax * wages) / comp),
        ('rent', income_tax),
        ('firm', (c.loc['D.5', 'U_S11'] + c.loc['D.5', 'U_S12']) / (c.loc['B.2', 'U_S11'] + c.loc['B.2', 'U_S12'])),
        ('salary_share', wages / comp)], columns=['tax', 'rate'])


def ipea(code, d_dir, tries=5):
    cache = os.path.join(d_dir, f'{code}.json')
    if not os.path.exists(cache):
        for i in range(tries):
            r = requests.get(IPEA.format(code), timeout=600)
            if r.ok and r.text.strip():
                open(cache, 'w').write(r.text)
                break
            time.sleep(30 * (i + 1))
    d = pd.DataFrame(pd.read_json(cache, orient='index').loc['value'].iloc[0])
    d = d[(d.NIVNOME == 'Municípios') & d.VALDATA.str.startswith('2010')]
    return d.set_index(d.TERCODIGO.astype(int)).VALVALOR


def municipal_shares(d_dir):
    mun = pd.DataFrame({'iss': ipea('RISSM', d_dir), 'iptu': ipea('RIPTUM', d_dir), 'icms_quota': ipea('RICMSM', d_dir)})
    s = requests.get(SIDRA, timeout=600).json()[0]['resultados'][0]['series']
    mun['product_taxes'] = pd.Series({int(x['localidade']['id']): 1000 * float(x['serie']['2010'])
                                      for x in s if x['serie']['2010'] not in ('-', '...', 'X')})
    rent = pd.read_csv('input/rent_AP_2010.csv', sep=';')
    rent['cod_mun'] = rent.AREAP.astype(str).str[:7].astype(int)
    rent['stock'] = rent.households * rent.mean_rent * 12 / RENTAL_YIELD
    mun['stock'] = rent.groupby('cod_mun').stock.sum()
    mun = mun.fillna(0.0)
    acps = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';').drop_duplicates('cod_mun')
    acps = acps[acps.cod_mun.isin(mun.index)]
    by = mun.loc[acps.cod_mun].groupby(acps.ACPs.values).sum()
    national = mun.sum()
    out = pd.DataFrame({'local_product_share': (by.iss + by.icms_quota) / by.product_taxes,
                        'iptu_rate': by.iptu / by.stock})
    # The Federal District collects state and municipal taxes and has no municipal accounts: national values
    fallback = {'local_product_share': (national.iss + national.icms_quota) / national.product_taxes,
                'iptu_rate': float(out.iptu_rate[out.iptu_rate > 0].median())}
    for col, value in fallback.items():
        out.loc[(out[col] <= 0) | out[col].isna(), col] = value
        out.loc['BRASILIA', col] = value
    out.index.name = 'acp'
    return out


def main():
    mip, cei = download(sys.argv[1])
    product_taxes(mip).to_csv('input/product_tax_2010.csv', sep=';', index=False, float_format='%.5f')
    national_rates(cei).to_csv('input/taxes_2010.csv', sep=';', index=False, float_format='%.5f')
    municipal_shares(sys.argv[1]).to_csv('input/tax_shares_2010.csv', sep=';', float_format='%.5f')
    for f in ('product_tax_2010', 'taxes_2010', 'tax_shares_2010'):
        print(pd.read_csv(f'input/{f}.csv', sep=';').head(12).to_string(index=False))


if __name__ == '__main__':
    main()
