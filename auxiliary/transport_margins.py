"""Transport margin by product, for FREIGHT 'margins' (input/transport_margins.csv).

Source: IBGE, Sistema de Contas Nacionais, Tabelas de Recursos e Usos 2015 (the year of the input-output matrix),
nível 12, tabela 1, sheet oferta: "Margem de transporte" over "Oferta total a preço básico" by product (R$ million).
Transport's own row carries the negative total of the margins and is set to 0, as are products with no margin.
<https://ftp.ibge.gov.br/Contas_Nacionais/Sistema_de_Contas_Nacionais/2015/tabelas_xls/tabelas_de_recursos_e_usos/>
The twelve products are, in order, the model's sectors.

Usage: python auxiliary/transport_margins.py
"""
import io
import zipfile

import pandas as pd
import requests

URL = ("https://ftp.ibge.gov.br/Contas_Nacionais/Sistema_de_Contas_Nacionais/2015/tabelas_xls/"
       "tabelas_de_recursos_e_usos/nivel_12_2000_2015_xls.zip")
SECTORS = ['Agriculture', 'Mining', 'Manufacturing', 'Utilities', 'Construction', 'Trade', 'Transport', 'Business',
           'Financial', 'RealEstate', 'OtherServices', 'Government']


def main():
    archive = zipfile.ZipFile(io.BytesIO(requests.get(URL, timeout=300).content))
    member = [m for m in archive.namelist() if m.endswith('12_tab1_2015.xls')][0]
    supply = pd.read_excel(archive.open(member), sheet_name='oferta', header=None)
    header = supply.iloc[3].astype(str).str.replace('\n', ' ').str.split().str.join(' ')
    margin_col = header[header.str.startswith('Margem de transporte')].index[0]
    basic_col = header[header.str.startswith('Oferta total a preço b')].index[0]
    rows = supply[supply.iloc[:, 0].astype(str).str.fullmatch(r'\d{2}')].iloc[:12]
    margin = pd.to_numeric(rows[margin_col]).values
    basic = pd.to_numeric(rows[basic_col]).values
    df = pd.DataFrame({'sector': SECTORS, 'transport_margin': margin, 'supply_basic': basic})
    df['margin'] = (df.transport_margin / df.supply_basic).clip(lower=0.0)
    df.to_csv('input/transport_margins.csv', sep=';', index=False, float_format='%.6f')
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
