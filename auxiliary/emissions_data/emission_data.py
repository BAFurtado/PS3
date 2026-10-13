"""Sector emission intensities for the model's 12 sectors, from Alvarenga Junior (2024).

emission_intensity.csv holds the gross emission intensities of the 42 industries of the Alves-Passoni & Freitas
input-output tables, in tCO2e per R$ million of gross output at 2010 prices, for 2000, 2010 and 2019, without
(sem_LUC) and with (com_LUC) land-use change. Each industry belongs to one sector of the IBGE nível 12
classification the model uses (SECTOR_SHARES 'ibge12'): Trade = G, Business = J, Government = O and public P/Q,
OtherServices = I, M, N, R, S, T and private P/Q. A sector's intensity is the mean of its industries' intensities
weighted by their output (valor_producao_42_setores.csv).

Writes input/emissions_sectors.csv (2019, without land-use change), which the firms read, and the 2019 aggregates
with and without land-use change next to this script.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
YEAR = 2019

# GIC 42-industry code -> model sector (IBGE nível 12)
SECTOR = {
    1: 'Agriculture',
    2: 'Mining', 3: 'Mining', 4: 'Mining',
    **{code: 'Manufacturing' for code in range(5, 29)},
    29: 'Utilities',
    30: 'Construction',
    31: 'Trade',
    32: 'Transport',
    33: 'OtherServices',   # accommodation and food (I)
    34: 'Business',        # information and communication (J)
    35: 'Financial',
    36: 'RealEstate',
    37: 'OtherServices',   # business and household services (M, N, R, S, T)
    38: 'Government',      # public administration (O)
    39: 'Government',      # public education
    40: 'OtherServices',   # private education
    41: 'Government',      # public health
    42: 'OtherServices',   # private health
}


def sector_intensities(column):
    coefs = pd.read_csv(HERE / 'emission_intensity.csv', sep=';', decimal=',', encoding='latin-1')
    output = pd.read_csv(HERE / 'valor_producao_42_setores.csv', sep=';', decimal=',', encoding='latin-1').dropna()
    data = coefs[['GIC_code', column]].merge(output[['GIC_code', 'Demanda Total']].astype({'GIC_code': int}),
                                             on='GIC_code', validate='one_to_one')
    assert sorted(data.GIC_code) == sorted(SECTOR), 'every one of the 42 industries needs a sector'
    data['isic_12'] = data.GIC_code.map(SECTOR)
    data['emissions'] = data[column] * data['Demanda Total']
    sectors = data.groupby('isic_12')[['emissions', 'Demanda Total']].sum()
    sectors['eco'] = sectors['emissions'] / sectors['Demanda Total']
    return sectors.reset_index()


if __name__ == '__main__':
    no_luc = sector_intensities(f'sem_LUC_{YEAR}')
    with_luc = sector_intensities(f'com_LUC_{YEAR}')
    no_luc.to_csv(HERE / f'emissions_12_sectors_{YEAR}.csv', index=False)
    with_luc.to_csv(HERE / f'emissions_12_sectors_{YEAR}_with_LUC.csv', index=False)
    no_luc[['isic_12', 'eco']].to_csv(ROOT / 'input' / 'emissions_sectors.csv', index=False)
    print(no_luc[['isic_12', 'eco']].to_string(index=False))
