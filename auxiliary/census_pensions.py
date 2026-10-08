"""Official pensions (aposentadoria or pensão de instituto de previdência oficial: RGPS and RPPS) by municipality, 2010,
for input/census_pensions_2010.csv: residents, pensioners and their pension income a month in 2010 R$.

Source: IBGE, Censo Demográfico 2010, sample microdata, persons (Amostra_Pessoas_<UF>.txt, layout of the IBGE
dictionary): V0656 = 1 receives an official pension; V6591 is the person's income from all sources other than work, one
value, also covering Bolsa Família / PETI (V0657), other social programmes (V0658) and other sources (V0659). A pensioner
without those three sources has V6591 as the pension. A pensioner with any of them is given the mean pension of the
pensioners without them, in the same state and age group (under 40, 40-54, 55-64, 65-74, 75 or more), at most V6591.
Weights V0010.

Usage: python auxiliary/census_pensions.py [<sample directory>]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/MyModels/censo2010/data/amostra')
OUT = 'input/census_pensions_2010.csv'
COLS = [(0, 7), (28, 44), (61, 64), (317, 318), (318, 319), (319, 320), (320, 321), (321, 327)]
NAMES = ['cod_mun', 'w', 'age', 'pen', 'pbf', 'prog', 'oth', 'v6591']
AGES = [-1, 39, 54, 64, 74, 200]

residents, pensioners, clean_val = [], [], []
clean_mean, mixed = {}, []
for path in sorted(glob.glob(os.path.join(ROOT, '*', 'Amostra_Pessoas_*.txt'))):
    sums = {}
    for ch in pd.read_fwf(path, colspecs=COLS, names=NAMES, header=None, chunksize=1_000_000, dtype=str,
                          encoding='latin-1'):
        ch['cod_mun'] = ch.cod_mun.astype(int)
        ch['w'] = ch.w.astype(float) / 1e13
        ch['age'] = pd.to_numeric(ch.age, errors='coerce').fillna(0)
        ch['v6591'] = pd.to_numeric(ch.v6591, errors='coerce').fillna(0.0)
        for c in ('pen', 'pbf', 'prog', 'oth'):
            ch[c] = ch[c].fillna('').str.strip() == '1'
        ch['ageg'] = pd.cut(ch.age, AGES, labels=False)
        residents.append(ch.groupby('cod_mun').w.sum())
        p = ch[ch.pen]
        pensioners.append(p.groupby('cod_mun').w.sum())
        clean = p[~p.pbf & ~p.prog & ~p.oth & (p.v6591 > 0)]
        clean_val.append((clean.v6591 * clean.w).groupby(clean.cod_mun).sum())
        for g, x in clean.groupby('ageg'):
            s = sums.setdefault(g, [0.0, 0.0])
            s[0] += (x.v6591 * x.w).sum()
            s[1] += x.w.sum()
        m = p[~p.index.isin(clean.index)]
        mixed.append(m[['cod_mun', 'w', 'ageg', 'v6591']].assign(uf=os.path.basename(path)))
    clean_mean[os.path.basename(path)] = {g: v / n for g, (v, n) in sums.items() if n > 0}
    print(path, flush=True)

m = pd.concat(mixed)
m['mean'] = [clean_mean[u].get(g, np.nan) for u, g in zip(m.uf, m.ageg)]
m['val'] = np.minimum(m['mean'].fillna(0.0), m.v6591)
out = pd.DataFrame({'pop': pd.concat(residents).groupby(level=0).sum()})
out['pensioners'] = pd.concat(pensioners).groupby(level=0).sum()
out['pension_val'] = pd.concat(clean_val).groupby(level=0).sum().add(
    (m.val * m.w).groupby(m.cod_mun).sum(), fill_value=0.0)
out = out.fillna(0.0)
out.index.name = 'cod_mun'
out.round(2).to_csv(OUT, sep=';')
print(out.sum().round(0), len(out))
