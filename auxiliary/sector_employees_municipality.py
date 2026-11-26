"""Employees by municipality of work and sector, from the Census, for SECTOR_DECK 'municipality'
(input/sector_employees_municipality.csv).

Source: IBGE, Censo Demográfico 2010, microdados da amostra, persons file of each state (local copy, as in
auxiliary/own_account.py). Employees as in auxiliary/sector_shares_census.py: employed (V6910 = 1) aged 17-69, position
in the main job employee (V6930 = 1, 2, 3), weights V0010, CNAE Domiciliar 2.0 division of the main job (V6471),
division 00 left out, P and Q split into Government and OtherServices by the state's share of 'Administração pública'
(auxiliary/sector_shares_ibge12.public_shares). Residents of the ACP municipalities only. Municipality of work: the
residence for V0660 = 1 (own home) and 2 (this municipality), V6604 for 3 (another municipality); V0660 4-5 (abroad,
several municipalities) are left out.

Usage: python auxiliary/sector_employees_municipality.py [cache_dir]
"""
import os
import sys
import tempfile
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from own_account import CENSUS, FILES, sector  # noqa: E402
from sector_shares_ibge12 import public_shares  # noqa: E402


def employees():
    """Weighted employees by municipality of work and sector, P and Q apart"""
    muns = set(pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';').cod_mun.astype(str))
    counts = defaultdict(float)
    for uf, files in FILES.items():
        if not any(m.startswith(str(uf)) for m in muns):
            continue
        for name in files:
            with open(os.path.join(CENSUS, name), encoding='latin-1') as f:
                for line in f:
                    mun = line[0:7]
                    if mun not in muns or line[391] != '1' or line[393] not in '123':
                        continue
                    if not 17 <= int(line[61:64]) <= 69:
                        continue
                    place = line[327]
                    if place in '12' and place != ' ':
                        work = mun
                    elif place == '3':
                        work = line[335:342]
                    else:
                        continue
                    cnae = line[203:208]
                    s = sector(cnae)
                    if s == 'Unknown':
                        continue
                    if s == 'OtherServices' and cnae[:2] in ('85', '86', '87', '88'):
                        s = 'P' if cnae[:2] == '85' else 'Q'
                    counts[(int(work), s)] += float(line[28:44]) / 1e13
        print(uf, len(counts), flush=True)
    return pd.Series(counts).unstack(fill_value=0.0)


def main(cache_dir):
    staff = employees()
    share = public_shares(cache_dir)
    state = staff.index.map(lambda c: c // 100000)
    public = staff.P * [share(s, 'P') for s in state] + staff.Q * [share(s, 'Q') for s in state]
    staff['Government'] += public
    staff['OtherServices'] += staff.P + staff.Q - public
    staff = staff.drop(columns=['P', 'Q'])
    long = staff.stack().rename('employees').reset_index()
    long.columns = ['cod_mun', 'sector', 'employees']
    long = long[long.employees > 0]
    long.to_csv('input/sector_employees_municipality.csv', sep=';', index=False, float_format='%.3f')
    print(len(long), 'rows,', long.cod_mun.nunique(), 'municipalities')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
