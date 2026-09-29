"""Public jobs per municipality by level of government, for GOV_WAGE_RULE = 'premium' (input/gov_levels.csv).

Source: Ipea, Atlas do Estado Brasileiro, "Total de pessoas no setor público em cada município, por nível federativo
e Poder - 2021" (RAIS-based published aggregate), <https://www.ipea.gov.br/atlasestado/downloads>. Shares are of the
three levels' sum (the file's 'publico' column differs from it by 0.2 % nationally). Municipalities with no public job
listed get municipal share 1.

Usage: python auxiliary/gov_levels.py [path_to_downloaded_csv]
"""
import sys

import pandas as pd

URL = 'https://www.ipea.gov.br/atlasestado/arquivos/downloads/4875-220mapatotalvinculosmunicipios.csv'


def main(src):
    atlas = pd.read_csv(src, sep=';')
    acps = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    out = acps.merge(atlas, left_on='cod_mun', right_on='codigo', how='left')
    total = out[['federal', 'estadual', 'municipal']].sum(axis=1)
    for level in ('federal', 'estadual', 'municipal'):
        out[level] = (out[level] / total).where(total > 0, 1.0 if level == 'municipal' else 0.0).round(4)
    out[['ACPs', 'cod_mun', 'federal', 'estadual', 'municipal']].to_csv('input/gov_levels.csv', sep=';', index=False)
    print(out[['federal', 'estadual', 'municipal']].describe().round(3).to_string())


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else URL)
