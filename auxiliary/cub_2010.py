"""CUB/m² residential standard projects by state, July 2010 (input/cub_2010.csv).

Source: CBIC, CUB/m² Estadual, report "Tabela do CUB/m² Estadual" (NBR 12.721:2006, CUB 2006 series, without
labour-tax relief), archive of reports up to September 2015 at http://memoria.cub.org.br (one Sinduscon per state; the
state capital's Sinduscon where a state has several). July 2010 is the Census 2010 reference month. Values in R$ / m²,
hard building cost: no land, foundations, elevators, projects, taxes, builder or developer remuneration.
States absent from the archive: AL, AP, RS, SC, SP.

Rows: uf;project;standard;cub   (project R-1, PP-4, R-8, R-16, PIS; standard B, N, A)

Usage: python auxiliary/cub_2010.py
"""
import re
import time

import pandas as pd
import requests

URL = 'http://memoria.cub.org.br/controller.php'
SINDUSCON = {'AC': 4, 'AM': 5, 'BA': 6, 'CE': 7, 'DF': 8, 'ES': 9, 'GO': 10, 'MA': 11, 'MS': 12, 'MT': 13, 'PA': 14,
             'PB': 15, 'PE': 16, 'PI': 17, 'PR': 18, 'MG': 1, 'RJ': 20, 'RN': 21, 'SE': 22, 'TO': 23, 'RO': 35,
             'RR': 30}
YEAR, MONTH = 2010, 7
STANDARDS = {'BAIXO': 'B', 'NORMAL': 'N', 'ALTO': 'A'}


def table(sid):
    data = {'responseType': 'xml', 'url': 'intranet/reports/tabela_cub_uf.uc', 'uniqueId': '1', 'action': 'get_data',
            'yid': YEAR, 'mid': MONTH, 'vid': 0, 'sid': sid, 'rep': 'tabela_cub_uf'}
    text = requests.post(URL, data=data, timeout=60).text
    text = re.sub(r'<[^>]*>|&nbsp;', ' ', text).replace('&Atilde;', 'Ã')
    if 'RESIDENCIAIS' not in text:
        return []
    residential = text.split('RESIDENCIAIS')[1].split('COMERCIAIS')[0]
    rows = []
    for name, block in re.findall(r'PADRÃO\s+(BAIXO|NORMAL|ALTO)(.*?)(?=PADRÃO|$)', residential, re.S):
        standard = STANDARDS[name]
        for project, value in re.findall(r'(R-1|PP-4|R-8|R-16|PIS)\s+([\d.]+,\d+)', block):
            rows.append((project, standard, float(value.replace('.', '').replace(',', '.'))))
    return rows


def main():
    out = []
    for uf, sid in SINDUSCON.items():
        rows = table(sid)
        if not rows:
            print(uf, 'no table for', MONTH, YEAR)
        out += [(uf,) + r for r in rows]
        time.sleep(1)
    df = pd.DataFrame(out, columns=['uf', 'project', 'standard', 'cub'])
    df.to_csv('input/cub_2010.csv', sep=';', index=False)
    print(df.pivot_table(index='uf', columns=['project', 'standard'], values='cub').round(0).to_string())


if __name__ == '__main__':
    main()
