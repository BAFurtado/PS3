"""Age profile and residual dispersion of work income per ACP, for the wage split (input/wage_dispersion_2010.csv).

Source: IBGE, Censo Demográfico 2010, microdados da amostra, persons file of each state (local copy, CENSUS below).
Employed (V6910 = 1) aged 17-69 with positive work income in the main job (V6513, R$ of July 2010), weights V0010.
Weighted least squares of log work income per ACP:
  (1) on education level (V6400, 1-4; 5 não determinado as its own level) and age (V6036) and age squared:
      age_b1, age_b2 are the age and age-squared coefficients;
  (2) on (1) plus sector (CNAE Domiciliar 2.0 division of the main job, V6471, 2 digits) and weighting area (V0011):
      resid_sd is the standard deviation of the residual, the dispersion among workers alike in education, age,
      sector and area.
(2) absorbs the area effects by weighted demeaning within area. ACPs with fewer than MIN_OBS sample persons, and any ACP
missing from the file, use the row 'BRASIL' (medians of the ACP rows, weighted by sample persons).

Rows: acp;age_b1;age_b2;resid_sd;obs

Usage: python auxiliary/wage_dispersion.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from own_account import CENSUS, FILES  # noqa: E402

COLS = [(0, 7, 'mun'), (7, 20, 'ap'), (28, 44, 'w'), (61, 64, 'age'), (157, 158, 'edu'), (203, 205, 'cnae'),
        (218, 224, 'inc'), (390, 391, 'v6900'), (391, 392, 'v6910')]
MIN_OBS = 2000


def read(path, codes):
    d = pd.read_fwf(path, colspecs=[(a, b) for a, b, _ in COLS], names=[n for *_, n in COLS], dtype=str,
                    encoding='latin-1')
    d['mun'] = d.mun.astype(int)
    d = d[d.mun.isin(codes) & (d.v6910 == '1')]
    d['inc'] = pd.to_numeric(d.inc, errors='coerce')
    d['age'] = d.age.astype(int)
    d = d[(d.inc > 0) & d.age.between(17, 69)]
    d['w'] = d.w.astype(float) / 1e13
    return d[['mun', 'ap', 'w', 'age', 'edu', 'cnae', 'inc']]


def dummies(s):
    return pd.get_dummies(s, drop_first=True).values.astype(float)


def wls(X, y, w):
    sw = np.sqrt(w)
    b = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    return b, y - X @ b


def within(a, groups, w):
    """Weighted deviation from the group mean, column by column"""
    a = np.asarray(a, float).reshape(len(w), -1)
    g = pd.factorize(groups)[0]
    sw = np.bincount(g, weights=w)
    means = np.column_stack([np.bincount(g, weights=a[:, j] * w) / sw for j in range(a.shape[1])])
    return a - means[g]


def fit(d):
    y, w = np.log(d.inc.values), d.w.values
    X = np.hstack([dummies(d.edu), d.age.values[:, None], (d.age.values ** 2)[:, None]])
    k = X.shape[1] - 2
    b, _ = wls(np.hstack([np.ones((len(d), 1)), X]), y, w)
    _, r = wls(within(np.hstack([X, dummies(d.cnae)]), d.ap.values, w), within(y, d.ap.values, w)[:, 0], w)
    return b[k + 1], b[k + 2], np.sqrt(np.average(r ** 2, weights=w))


def wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return x[o][np.searchsorted(c, c[-1] / 2)]


def main():
    muns = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    acp_of = muns.drop_duplicates('cod_mun').set_index('cod_mun').ACPs
    parts = []
    for uf, files in FILES.items():
        codes = set(acp_of.index[acp_of.index // 100000 == uf])
        if not codes:
            continue
        for f in files:
            parts.append(read(os.path.join(CENSUS, f), codes))
            print(f, len(parts[-1]))
    d = pd.concat(parts)
    d['acp'] = d.mun.map(acp_of)
    rows = []
    for acp, g in d.groupby('acp'):
        if len(g) >= MIN_OBS:
            b1, b2, sd = fit(g)
            rows.append({'acp': acp, 'age_b1': b1, 'age_b2': b2, 'resid_sd': sd, 'obs': len(g)})
    out = pd.DataFrame(rows)
    n = out.obs.values.astype(float)
    out.loc[len(out)] = {'acp': 'BRASIL', **{c: wmedian(out[c].values, n) for c in ('age_b1', 'age_b2', 'resid_sd')},
                         'obs': int(n.sum())}
    out.to_csv('input/wage_dispersion_2010.csv', sep=';', index=False, float_format='%.6g')
    print(out.to_string())


if __name__ == '__main__':
    main()
