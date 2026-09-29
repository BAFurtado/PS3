"""Public/private wage ratio per municipality, for GOV_WAGE_RATIO_BY_MUN (input/gov_wage_ratio.csv).

Source: IBGE, Cadastro Central de Empresas (CEMPRE), Sidra table 6450, 2010-2019: salaries and other pay (var 662)
and salaried staff (var 708) by municipality, total and CNAE 2.0 section O (public administration, defence and
social security). Public wage = O pay / O staff; private wage = (total - O) pay / (total - O) staff.

The ratio is the median over the years with both figures published. IBGE suppresses cells ('X') in small
municipalities; those, and municipalities with fewer than MIN_YEARS published years, take their ACP's ratio
(pooled over the ACP's published municipalities, median over years), and failing that the national one.

Usage: python auxiliary/gov_wage_ratio.py [cache_dir]
"""
import json
import os
import sys
import tempfile
import urllib.request

import numpy as np
import pandas as pd

YEARS = range(2010, 2020)
TOTAL, SECTION_O = '117897', '117774'
URL = ('https://servicodados.ibge.gov.br/api/v3/agregados/6450/periodos/{year}/variaveis/708|662'
       '?localidades=N6[all]&classificacao=12762[{total},{o}]')
MIN_YEARS = 3


def fetch(year, cache_dir):
    path = os.path.join(cache_dir, f'cempre_{year}.json')
    if not os.path.exists(path):
        with urllib.request.urlopen(URL.format(year=year, total=TOTAL, o=SECTION_O), timeout=600) as r:
            with open(path, 'wb') as f:
                f.write(r.read())
    with open(path) as f:
        return json.load(f)


def load(cache_dir):
    rows = []
    for year in YEARS:
        for var in fetch(year, cache_dir):
            for res in var['resultados']:
                sec = 'T' if TOTAL in res['classificacoes'][0]['categoria'] else 'O'
                for s in res['series']:
                    v = pd.to_numeric(s['serie'][str(year)], errors='coerce')
                    rows.append((int(s['localidade']['id']), year, f"{'pay' if var['id'] == '662' else 'staff'}_{sec}", v))
    d = pd.DataFrame(rows, columns=['cod_mun', 'year', 'col', 'v']).pivot_table(
        index=['cod_mun', 'year'], columns='col', values='v').reset_index()
    ok = (d[['pay_T', 'pay_O', 'staff_T', 'staff_O']].notna().all(axis=1) & (d.staff_O > 0)
          & (d.staff_T > d.staff_O) & (d.pay_O > 0))
    return d[ok]


def ratio(g):
    return (g.pay_O.sum() / g.staff_O.sum()) / ((g.pay_T.sum() - g.pay_O.sum()) / (g.staff_T.sum() - g.staff_O.sum()))


def main(cache_dir):
    d = load(cache_dir)
    acps = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    by_year = d.groupby(['cod_mun', 'year']).apply(ratio, include_groups=False).rename('r').reset_index()
    mun = by_year.groupby('cod_mun').r.agg(['median', 'count'])
    d = d.merge(acps, on='cod_mun', how='left')
    acp = d.dropna(subset=['ACPs']).groupby(['ACPs', 'year']).apply(ratio, include_groups=False).groupby('ACPs').median()
    national = d.groupby('year').apply(ratio, include_groups=False).median()

    out = acps.copy()
    out['years'] = out.cod_mun.map(mun['count']).fillna(0).astype(int)
    own = out.years >= MIN_YEARS
    out['ratio'] = np.where(own, out.cod_mun.map(mun['median']), out.ACPs.map(acp))
    out['source'] = np.where(own, 'municipality', 'acp')
    missing = out.ratio.isna()
    out.loc[missing, 'ratio'] = national
    out.loc[missing, 'source'] = 'national'
    out['ratio'] = out.ratio.round(3)
    out[['ACPs', 'cod_mun', 'ratio', 'source', 'years']].to_csv('input/gov_wage_ratio.csv', index=False, sep=';')
    print(f'national {national:.3f}; sources: {out.source.value_counts().to_dict()}')
    print(out.ratio.describe().round(2).to_string())


if __name__ == '__main__':
    cache = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), 'cempre_cache')
    os.makedirs(cache, exist_ok=True)
    main(cache)
