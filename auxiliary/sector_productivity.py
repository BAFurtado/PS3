"""Output per job by sector relative to the national mean, for SECTOR_PRODUCTIVITY (input/sector_productivity.csv).

Source: IBGE, Sistema de Contas Nacionais, Tabelas de Recursos e Usos 2015 (the year of the input-output matrix),
nível 12, tabela 2, sheet VA: "Valor da produção" (R$ million) and "Fator trabalho (ocupações)" by activity; nível 68,
tabela 1, sheet producao: output of product 68002 "Aluguel imputado", taken out of Atividades imobiliárias since it has
no workers. <https://ftp.ibge.gov.br/Contas_Nacionais/Sistema_de_Contas_Nacionais/2015/tabelas_xls/tabelas_de_recursos_e_usos/>
The twelve activities are, in order, the model's sectors.

Usage: python auxiliary/sector_productivity.py
"""
import io
import zipfile

import pandas as pd
import requests

URL = ("https://ftp.ibge.gov.br/Contas_Nacionais/Sistema_de_Contas_Nacionais/2015/tabelas_xls/"
       "tabelas_de_recursos_e_usos/nivel_{}_xls.zip")
SECTORS = ['Agriculture', 'Mining', 'Manufacturing', 'Utilities', 'Construction', 'Trade', 'Transport', 'Business',
           'Financial', 'RealEstate', 'OtherServices', 'Government']


def sheet(level, table, name):
    archive = zipfile.ZipFile(io.BytesIO(requests.get(URL.format(level), timeout=300).content))
    member = [m for m in archive.namelist() if m.endswith(table)][0]
    return pd.read_excel(archive.open(member), sheet_name=name, header=None)


def row(df, label):
    return df[df.iloc[:, 0].astype(str).str.strip().str.startswith(label)].iloc[0]


def main():
    va = sheet('12_2000_2015', '12_tab2_2015.xls', 'VA')
    output = row(va, 'Valor da produção').iloc[1:13].astype(float).values
    jobs = row(va, 'Fator trabalho').iloc[1:13].astype(float).values
    production = sheet('68_2010_2015', '68_tab1_2015.xls', 'producao')
    imputed_row = production[production.iloc[:, 0].astype(str).str.startswith('68002')].iloc[0]
    # Only Atividades imobiliárias produces it, so its column equals the product total
    imputed = float(pd.to_numeric(imputed_row.iloc[2:], errors='coerce').max())
    df = pd.DataFrame({'sector': SECTORS, 'output': output, 'jobs': jobs})
    df.loc[df.sector == 'RealEstate', 'output'] -= imputed
    df['productivity'] = df.output / df.jobs / (df.output.sum() / df.jobs.sum())
    df.to_csv('input/sector_productivity.csv', sep=';', index=False, float_format='%.6f')
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
