"""Rent paid by renting households per weighting area (AP), Census 2010 (input/rent_AP_2010.csv).

Source: IBGE, Censo Demográfico 2010, microdados da amostra, household file of each state (local copy, CENSUS below).
Permanent private households renting (V0201 = 3) with a positive rent (V2011, R$ of July 2010 a month), weights V0010;
households: all permanent private households with a weight. Areas: V0011 (área de ponderação, the model's AREAP);
municipalities with one area use the municipality code followed by zeros, as in the other AREAP inputs.

Rows: AREAP;households;renters;mean_rent;median_rent

Usage: python auxiliary/rent_ap.py
"""
import os

import numpy as np
import pandas as pd

CENSUS = os.path.expanduser('~/MyModels/censo2010/data/amostra')
DIRS = {11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO', 21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN',
        25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA', 31: 'MG', 32: 'ES', 33: 'RJ', 35: None, 41: 'PR', 42: 'SC',
        43: 'RS', 50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'}
COLS = [(0, 7, 'mun'), (7, 20, 'ap'), (28, 44, 'w'), (57, 58, 'tenure'), (58, 64, 'rent')]


def files(uf):
    if uf == 35:
        return [os.path.join(CENSUS, d, f) for d in ('SP1', 'SP2-RM') for f in os.listdir(os.path.join(CENSUS, d))
                if f.startswith('Amostra_Domicilios')]
    return [os.path.join(CENSUS, DIRS[uf], f'Amostra_Domicilios_{uf}.txt')]


def wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return x[o][np.searchsorted(c, c[-1] / 2)]


def main():
    rows = []
    for uf in DIRS:
        for f in files(uf):
            d = pd.read_fwf(f, colspecs=[(a, b) for a, b, _ in COLS], names=[n for *_, n in COLS], dtype=str,
                            encoding='latin-1')
            d['w'] = d.w.astype(float) / 1e13
            d = d[d.w > 0]
            d['rent'] = pd.to_numeric(d.rent, errors='coerce')
            for ap, g in d.groupby('ap'):
                r = g[(g.tenure == '3') & (g.rent > 0)]
                rows.append({'AREAP': ap, 'households': g.w.sum(), 'renters': r.w.sum(),
                             'mean_rent': np.average(r.rent, weights=r.w) if len(r) else np.nan,
                             'median_rent': wmedian(r.rent.values, r.w.values) if len(r) else np.nan})
            print(f, len(rows))
    pd.DataFrame(rows).to_csv('input/rent_AP_2010.csv', sep=';', index=False, float_format='%.6g')


if __name__ == '__main__':
    main()
