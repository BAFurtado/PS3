"""Value added per own-account worker relative to the other workers of each sector, and the shares of value added that
are and are not own-account income (input/own_account_productivity_2010.csv).

relative_productivity = (gross mixed income / own-account workers) / ((value added - gross mixed income) / other
employed), by nível-12 sector, 2010. Gross mixed income is the income of unincorporated household businesses, here
counted as own-account workers' value added; the employers of such businesses are left among the other employed. Real
Estate's value added is net of imputed rent (output of product 68002, as auxiliary/firm_income.py).
wage_share_firms = remunerations / (value added - gross mixed income): the labour share of the value added that is not
own-account income, the share firms pay when own-account workers are not among their staff.

Sources: IBGE, Sistema de Contas Nacionais, Tabelas de Recursos e Usos 2010, nível 12, tabela 2 (sheet VA): value
added, remunerations and gross mixed income by activity; nível 68, tabela 1 (sheet producao): imputed rent.
IBGE, Censo Demográfico 2010, microdados da amostra, all states (local copy, auxiliary/own_account.py): employed persons (V6910 = 1) and own-account workers (V6930 = 4) by sector of the main job
(V6471, grouped as in auxiliary/own_account.py; 'Unknown' left out), weights V0010, all ages.

Usage: python auxiliary/own_account_productivity.py
"""
import os

import pandas as pd

from own_account import CENSUS, FILES, sector
from sector_productivity import SECTORS, row, sheet


def census_counts():
    counts = {}
    for files in FILES.values():
        for name in files:
            with open(os.path.join(CENSUS, name), encoding='latin-1') as f:
                for line in f:
                    if line[391] != '1':
                        continue
                    s = sector(line[203:208])
                    w = float(line[28:44]) / 1e13
                    e, o = counts.get(s, (0.0, 0.0))
                    counts[s] = (e + w, o + w * (line[393] == '4'))
    return pd.DataFrame([(s, e, o) for s, (e, o) in counts.items()], columns=['sector', 'employed', 'own_account'])


def main():
    va = sheet('12_2000_2015', '12_tab2_2010.xls', 'VA')
    df = pd.DataFrame({'sector': SECTORS,
                       'value_added': row(va, 'Valor adicionado').iloc[1:13].astype(float).values,
                       'remunerations': row(va, 'Remunerações').iloc[1:13].astype(float).values,
                       'mixed_income': row(va, 'Rendimento misto bruto').iloc[1:13].astype(float).values})
    production = sheet('68_2010_2015', '68_tab1_2010.xls', 'producao')
    imputed_row = production[production.iloc[:, 0].astype(str).str.startswith('68002')].iloc[0]
    df.loc[df.sector == 'RealEstate', 'value_added'] -= float(pd.to_numeric(imputed_row.iloc[2:], errors='coerce').max())
    df = df.merge(census_counts(), on='sector', how='left').fillna(0.0)
    own = df.mixed_income / df.own_account
    other = (df.value_added - df.mixed_income) / (df.employed - df.own_account)
    df['relative_productivity'] = (own / other).where(df.own_account > 0, 1.0)
    df['wage_share_firms'] = df.remunerations / (df.value_added - df.mixed_income)
    df.to_csv('input/own_account_productivity_2010.csv', sep=';', index=False, float_format='%.6f')
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
