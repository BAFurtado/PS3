"""
Census 2010 employees by municipality of work, for STARTUP_JOBS 'census' (world/startup_jobs.py).

Sample microdata (persons, fixed width, latin-1), one file per state: <MICRODATA>/<UF>/Amostra_Pessoas_<code>.txt.
Persons aged 16-69, occupied (V6910 1), employees with and without carteira (V6930 1 and 3) outside public
administration (auxiliary/own_account.sector of the CNAE domiciliar code, Government left out). Military and civil
servants (V6930 2) are left out in every sector: they set the Government headcount (auxiliary/gov_headcount.py).
Work municipality: the residence for V0660 1 (own home) and 2 (this municipality), V6604 for 3 (another
municipality). Kept when the work municipality belongs to the residence's ACP. Persons are summed with the sample weight V0010.

    python auxiliary/census_jobs_workplace_2010.py MICRODATA [ACP ...]   default ACPs: the six ENMU areas

writes input/census_jobs_workplace_2010.csv: acp;work_mun;persons
"""
import glob
import os
import sys
from collections import defaultdict

import pandas as pd

from auxiliary.commute_ap_education_2010 import ACPS, UF
from auxiliary.own_account import sector

OUT = 'input/census_jobs_workplace_2010.csv'


def read(microdata, acps):
    codes = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    codes = codes[codes.ACPs.isin(acps)].drop_duplicates('cod_mun')
    acp_of = dict(zip(codes.cod_mun.astype(str), codes.ACPs))
    states = sorted({UF[m[:2]] for m in acp_of})
    files = [f for st in states for f in glob.glob(os.path.join(microdata, st + '*', 'Amostra_Pessoas_*.txt'))]
    if not files:
        sys.exit('No microdata files for {} under {}'.format(states, microdata))
    persons = defaultdict(float)
    for fn in files:
        print(fn, flush=True)
        with open(fn, encoding='latin-1') as f:
            for line in f:
                mun = line[0:7]
                if mun not in acp_of or not 16 <= int(line[61:64]) <= 69:
                    continue
                if line[391] != '1' or line[393] not in '13' or sector(line[203:208]) == 'Government':
                    continue
                place = line[327]
                work = mun if place in '12' else line[335:342] if place == '3' else None
                if work is None or acp_of.get(work) != acp_of[mun]:
                    continue
                persons[(acp_of[mun], work)] += float(line[28:44]) / 1e13
    return pd.DataFrame([k + (v,) for k, v in sorted(persons.items())], columns=['acp', 'work_mun', 'persons'])


if __name__ == '__main__':
    microdata = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/MyModels/censo2010/data/amostra')
    table = read(microdata, sys.argv[2:] or ACPS)
    table.to_csv(OUT, sep=';', index=False)
    print('wrote', OUT)
