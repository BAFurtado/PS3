"""
Census 2010 commute time of employees by weighting area (AP) and education level, for the validation of the ENMU
runs (analysis/enmu/validacao.py).

Sample microdata (persons, fixed width, latin-1), one file per state: <MICRODATA>/<UF>/Amostra_Pessoas_<code>.txt.
Persons aged 16-69, occupied (V6910 1), employees (V6930 1-3: with and without carteira, military and civil
servants), who work away from home and return daily (V0661 1), by commute class V0662 (1 up to 5 min, 2 6-30 min,
3 31-60 min, 4 61-120 min, 5 over 120 min) and education level V6400 (1-4; 5, undetermined, left out). Persons are
summed with the sample weight V0010. The same employees by municipality of residence and of work (V0660 2: this
municipality, 3: another one, V6604 its code; 5, more than one, left out).

    python auxiliary/commute_ap_education_2010.py MICRODATA [ACP ...]   default ACPs: the six ENMU areas

writes input/commute_ap_education_2010.csv: acp;cod_mun;ap;level;commute_class;persons
and input/commute_workplace_2010.csv: acp;cod_mun;work_mun;persons
"""
import glob
import os
import sys
from collections import defaultdict

import pandas as pd

OUT = 'input/commute_ap_education_2010.csv'
OUT_WORK = 'input/commute_workplace_2010.csv'
ACPS = ['BRASILIA', 'FORTALEZA', 'GOIANIA', 'MANAUS', 'BAIXADA SANTISTA', 'TERESINA']
UF = {'11': 'RO', '12': 'AC', '13': 'AM', '14': 'RR', '15': 'PA', '16': 'AP', '17': 'TO', '21': 'MA', '22': 'PI',
      '23': 'CE', '24': 'RN', '25': 'PB', '26': 'PE', '27': 'AL', '28': 'SE', '29': 'BA', '31': 'MG', '32': 'ES',
      '33': 'RJ', '35': 'SP', '41': 'PR', '42': 'SC', '43': 'RS', '50': 'MS', '51': 'MT', '52': 'GO', '53': 'DF'}


def read(microdata, acps):
    codes = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    codes = codes[codes.ACPs.isin(acps)].drop_duplicates('cod_mun')
    acp_of = dict(zip(codes.cod_mun.astype(str), codes.ACPs))
    states = sorted({UF[m[:2]] for m in acp_of})
    files = [f for st in states for f in glob.glob(os.path.join(microdata, st + '*', 'Amostra_Pessoas_*.txt'))]
    if not files:
        sys.exit('No microdata files for {} under {}'.format(states, microdata))
    persons = defaultdict(float)
    work = defaultdict(float)
    for fn in files:
        with open(fn, encoding='latin-1') as f:
            for line in f:
                mun = line[0:7]
                if mun not in acp_of:
                    continue
                if not 16 <= int(line[61:64]) <= 69:
                    continue
                # V6910 occupied, V6930 employee, V0661 returns daily, V0662 class, V6400 level
                if line[391] != '1' or line[393] not in '123' or line[349] != '1' or line[350] not in '12345':
                    continue
                if line[157] not in '1234':
                    continue
                w = float(line[28:44]) / 1e13
                persons[(acp_of[mun], int(mun), line[7:20], int(line[157]), int(line[350]))] += w
                if line[327] == '2':
                    work[(acp_of[mun], int(mun), int(mun))] += w
                elif line[327] == '3' and line[335:342].strip().isdigit():
                    work[(acp_of[mun], int(mun), int(line[335:342]))] += w
    return (pd.DataFrame([(*k, v) for k, v in persons.items()],
                         columns=['acp', 'cod_mun', 'ap', 'level', 'commute_class', 'persons']),
            pd.DataFrame([(*k, v) for k, v in work.items()], columns=['acp', 'cod_mun', 'work_mun', 'persons']))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    t, w = read(sys.argv[1], sys.argv[2:] or ACPS)
    w.sort_values(['acp', 'cod_mun', 'work_mun']).to_csv(OUT_WORK, sep=';', index=False)
    print('{}: {} rows'.format(OUT_WORK, len(w)))
    t = t.sort_values(['acp', 'cod_mun', 'ap', 'level', 'commute_class'])
    t.to_csv(OUT, sep=';', index=False)
    print('{}: {} rows, {} APs, {:.0f} employees'.format(OUT, len(t), t.ap.nunique(), t.persons.sum()))
    print(t.groupby('acp').persons.sum().round(0).to_string())
