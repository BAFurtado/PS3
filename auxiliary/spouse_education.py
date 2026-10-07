"""Education of spouses by the education of the household head, per ACP, for partner matching
(input/spouse_education_2010.csv).

Source: IBGE, Censo Demográfico 2010, microdados da amostra, persons file of each state (local copy, CENSUS below).
Couples: the responsible person (V0502 = 01) and the spouse (V0502 = 02, 03) of the same household (V0300 within the
municipality), both with a known education level (V6400 1-4; 5 não determinado left out), weighted by the head's
V0010. share = the weighted share of spouses of each level among the couples whose head has head_level. 'BRASIL':
all ACPs pooled, used for an ACP the file lacks.

Rows: acp;head_level;spouse_level;share

Usage: python auxiliary/spouse_education.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from own_account import CENSUS, FILES  # noqa: E402

COLS = [(0, 7, 'mun'), (20, 28, 'dom'), (28, 44, 'w'), (53, 55, 'rel'), (157, 158, 'edu')]


def couples(path, codes):
    d = pd.read_fwf(path, colspecs=[(a, b) for a, b, _ in COLS], names=[n for *_, n in COLS], dtype=str,
                    encoding='latin-1')
    d['mun'] = d.mun.astype(int)
    d = d[d.mun.isin(codes) & d.edu.isin(list('1234'))]
    d['w'] = d.w.astype(float) / 1e13
    heads = d[d.rel == '01'][['mun', 'dom', 'edu', 'w']]
    spouses = d[d.rel.isin(['02', '03'])][['mun', 'dom', 'edu']]
    return heads.merge(spouses, on=['mun', 'dom'], suffixes=('_head', '_spouse'))


def shares(m):
    t = m.groupby(['edu_head', 'edu_spouse']).w.sum()
    t = (t / t.groupby(level=0).transform('sum')).rename('share').reset_index()
    return t.rename(columns={'edu_head': 'head_level', 'edu_spouse': 'spouse_level'})


def main():
    muns = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    acp_of = muns.drop_duplicates('cod_mun').set_index('cod_mun').ACPs
    parts = []
    for uf, files in FILES.items():
        codes = set(acp_of.index[acp_of.index // 100000 == uf])
        if not codes:
            continue
        for f in files:
            parts.append(couples(os.path.join(CENSUS, f), codes))
            print(f, len(parts[-1]))
    m = pd.concat(parts)
    m['acp'] = m.mun.map(acp_of)
    out = [shares(g).assign(acp=acp) for acp, g in m.groupby('acp')] + [shares(m).assign(acp='BRASIL')]
    out = pd.concat(out)[['acp', 'head_level', 'spouse_level', 'share']]
    out.to_csv('input/spouse_education_2010.csv', sep=';', index=False, float_format='%.6g')
    print(out[out.acp == 'BRASIL'].to_string())


if __name__ == '__main__':
    main()
