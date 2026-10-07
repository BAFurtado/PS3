"""Employee share by sector per ACP in the IBGE nível 12 classification of the input-output matrix, from the Census,
for SECTOR_SHARES 'census' (input/sector_shares_census.csv).

Source: IBGE, Censo Demográfico 2010, microdados da amostra, persons file of each state (local copy, as in
auxiliary/own_account.py). Employees: employed (V6910 = 1) aged 17-69 whose position in the main job is employee
(V6930 = 1, 2, 3: com carteira, militares e estatutários, sem carteira; domestic workers included), by the CNAE
Domiciliar 2.0 division of the main job (V6471), weights V0010; division 00 (atividades mal definidas) left out.
Sectors as in auxiliary/own_account.py, except Government = O and public P/Q as in SECTOR_SHARES 'ibge12': P (85)
and Q (86-88) are split by the share of 'Administração pública' in the state's P and Q salaried staff (Sidra table
6703, auxiliary/sector_shares_ibge12.public_shares), private P/Q staying in OtherServices.

Usage: python auxiliary/sector_shares_census.py [cache_dir]
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
    """Weighted employees by municipality and sector, P and Q apart"""
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
                    cnae = line[203:208]
                    s = sector(cnae)
                    if s == 'Unknown':
                        continue
                    if s == 'OtherServices' and cnae[:2] in ('85', '86', '87', '88'):
                        s = 'P' if cnae[:2] == '85' else 'Q'
                    counts[(int(mun), s)] += float(line[28:44]) / 1e13
        print(uf, len(counts))
    return pd.Series(counts).unstack(fill_value=0.0)


def main(cache_dir):
    staff = employees()
    share = public_shares(cache_dir)
    state = staff.index.map(lambda c: c // 100000)
    public = staff.P * [share(s, 'P') for s in state] + staff.Q * [share(s, 'Q') for s in state]
    staff['Government'] += public
    staff['OtherServices'] += staff.P + staff.Q - public
    staff = staff.drop(columns=['P', 'Q'])
    acps = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';').set_index('cod_mun')
    acp = staff.join(acps).groupby('ACPs').sum()
    acp = acp.div(acp.sum(axis=1), axis=0)
    long = acp.stack().rename('participation').reset_index()
    long.columns = ['concurb_name', 'sector', 'participation']
    long[['sector', 'concurb_name', 'participation']].to_csv('input/sector_shares_census.csv', sep=';', index=False,
                                                            float_format='%.6f')
    ibge12 = pd.read_csv('input/sector_shares_ibge12.csv', sep=';').pivot(index='concurb_name', columns='sector',
                                                                          values='participation')
    print(pd.DataFrame({'ibge12': ibge12.median(), 'census': acp.median()}).round(3).to_string())


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
