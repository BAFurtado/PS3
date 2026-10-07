"""Federal and state pay relative to private pay, per municipality, for federal and state public pay
(input/gov_pay.csv).

Public pay: Ipea, Atlas do Estado Brasileiro, "Remuneração média por nível federativo - Brasil, grandes regiões e UFs"
(RAIS-based published aggregate, all powers, mean monthly pay), <https://www.ipea.gov.br/atlasestado/downloads>,
2010, in the municipality's state. The Atlas states pay in prices of its last year (2021) with the INPC; it is brought
to 2010 prices with the ratio of the annual mean INPC index, 2010 to 2021 (BCB SGS 188).
Private pay: IBGE CEMPRE 2010 (Sidra table 6450, as auxiliary/gov_wage_ratio.py), salaries and other pay over
salaried staff, all sections but O, pooled over the ACP's municipalities, divided by 13.33 (12 months, the 13th salary
and the vacation third) to a monthly mean. ACPs with no published figure take the national one.

Output: per municipality, federal and state pay as multiples of its ACP's private pay.

Usage: python auxiliary/gov_pay.py [path_to_downloaded_atlas_csv] [cempre_cache_dir]
(the conda env may need SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt)
"""
import json
import os
import sys
import tempfile
import urllib.request

import pandas as pd

import gov_wage_ratio

URL = 'https://www.ipea.gov.br/atlasestado/arquivos/downloads/2979-118remuneracaomedianivelfederativobrgruf.csv'
INPC = ('https://api.bcb.gov.br/dados/serie/bcdata.sgs.188/dados?formato=json'
        '&dataInicial=01/01/2010&dataFinal=31/12/{year}')
YEAR, BASE_YEAR = 2010, 2021
MONTHS_PAID = 12 + 1 + 1 / 3


def inpc_ratio():
    with urllib.request.urlopen(INPC.format(year=BASE_YEAR)) as r:
        months = json.load(r)
    level, index = 1.0, {}
    for m in months:
        level *= 1 + float(m['valor']) / 100
        index.setdefault(m['data'][-4:], []).append(level)
    mean = {y: sum(v) / len(v) for y, v in index.items()}
    return mean[str(YEAR)] / mean[str(BASE_YEAR)]


def private_pay(cache_dir):
    gov_wage_ratio.YEARS = [YEAR]
    d = gov_wage_ratio.load(cache_dir)
    d = d.assign(pay=1000 * (d.pay_T - d.pay_O), staff=d.staff_T - d.staff_O)
    acps = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    by_acp = d.merge(acps, on='cod_mun').groupby('ACPs')[['pay', 'staff']].sum()
    return (by_acp.pay / by_acp.staff / MONTHS_PAID), d.pay.sum() / d.staff.sum() / MONTHS_PAID


def main(src, cache_dir):
    atlas = pd.read_csv(src, sep=';', decimal=',')
    atlas = atlas[(atlas.ano == YEAR) & ((atlas.codigo_localizacao == 0) | (atlas.codigo_localizacao > 10))]
    public = atlas.pivot(index='codigo_localizacao', columns='poder', values='rem_media') * inpc_ratio()
    by_acp, national = private_pay(cache_dir)
    out = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')[['ACPs', 'cod_mun']]
    private = out.ACPs.map(by_acp).fillna(national)
    for level in ('federal', 'estadual'):
        out[level] = (out.cod_mun // 100000).map(public[level]).fillna(public.loc[0, level])
        out[level] = (out[level] / private).round(3)
    out.to_csv('input/gov_pay.csv', sep=';', index=False)
    print(f'private pay R$ 2010/month: national {national:.0f}; public Brazil:',
          public.loc[0].round(0).to_dict())
    print(out.groupby('ACPs')[['federal', 'estadual']].first().describe().round(2).to_string())


if __name__ == '__main__':
    cache = sys.argv[2] if len(sys.argv) > 2 else os.path.join(tempfile.gettempdir(), 'cempre_cache')
    os.makedirs(cache, exist_ok=True)
    main(sys.argv[1] if len(sys.argv) > 1 else URL, cache)
