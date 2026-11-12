"""Car ownership by ACP and work-income decile, Census 2010 sample microdata.

A person aged 16-70 owns a car when their household answers yes to question 2.22, "automóvel para uso particular"
(V0222 in the household file). Persons with work income in their main job (V6513 > 0) fall into the weighted
deciles of that income within their ACP (group 1-10). Persons without work income form group 0.

Writes input/car_ownership_2010.csv: acp;group;p_car;n (n = sample persons).
Usage: python auxiliary/car_ownership.py [census sample directory]
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

DATA = '/home/furtado/MyModels/censo2010/data/amostra'
OUT = 'input/car_ownership_2010.csv'
# (start, end) 1-based inclusive, from dic_CENSO_domicilios_2010 and dic_CENSO_pessoas_2010
HOUSE = {'mun': (1, 7), 'ctrl': (21, 28), 'car': (103, 103)}
PERSON = {'mun': (1, 7), 'ctrl': (21, 28), 'w': (29, 44), 'age': (62, 64), 'inc': (219, 224)}


def files(data):
    for d in sorted(os.listdir(data)):
        path = os.path.join(data, d)
        if not os.path.isdir(path):
            continue
        houses = [f for f in os.listdir(path) if f.startswith('Amostra_Domicilios_')]
        for h in houses:
            yield os.path.join(path, h), os.path.join(path, h.replace('Domicilios', 'Pessoas'))


def cut(line, cols):
    return {k: line[a - 1:b] for k, (a, b) in cols.items()}


def read(data, mun_acp):
    rows = []
    for house_file, person_file in files(data):
        car = {}
        with open(house_file, encoding='latin-1') as f:
            for line in f:
                if line[0:7] in mun_acp:
                    r = cut(line, HOUSE)
                    car[(r['mun'], r['ctrl'])] = r['car'] == '1'
        if not car:
            continue
        with open(person_file, encoding='latin-1') as f:
            for line in f:
                if line[0:7] not in mun_acp:
                    continue
                r = cut(line, PERSON)
                key = (r['mun'], r['ctrl'])
                if key not in car:
                    continue
                rows.append((mun_acp[r['mun']], float(r['w']) / 1e13, int(r['age']),
                             float(r['inc']) if r['inc'].strip() else 0.0, car[key]))
        print(os.path.basename(person_file), len(rows), flush=True)
    return pd.DataFrame(rows, columns=['acp', 'w', 'age', 'inc', 'car'])


def deciles(d):
    """Group 0 without work income, 1-10 weighted work-income deciles within the ACP"""
    d = d.copy()
    d['group'] = 0
    earn = d.inc > 0
    e = d[earn].sort_values('inc')
    share = e.w.cumsum() / e.w.sum()
    d.loc[e.index, 'group'] = np.minimum((share.values * 10).astype(int), 9) + 1
    return d


def main(data=DATA):
    acp = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    mun_acp = dict(zip(acp.cod_mun.astype(str), acp.ACPs))
    d = read(data, mun_acp)
    d = d[d.age.between(16, 70)]
    out = defaultdict(list)
    for name, g in d.groupby('acp'):
        g = deciles(g)
        for k, h in g.groupby('group'):
            out['acp'].append(name)
            out['group'].append(int(k))
            out['p_car'].append(round(float(np.average(h.car, weights=h.w)), 4))
            out['n'].append(len(h))
    pd.DataFrame(out).to_csv(OUT, sep=';', index=False)
    print('wrote', OUT)


if __name__ == '__main__':
    main(*sys.argv[1:])
