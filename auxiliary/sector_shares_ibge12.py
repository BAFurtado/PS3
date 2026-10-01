"""Employment share by sector per ACP in the IBGE nível 12 classification of the input-output matrix, for
SECTOR_SHARES 'ibge12' (input/sector_shares_ibge12.csv).

input/CONCURBs_SECTOR.csv (RAIS 2010) groups CNAE 2.0 sections as Trade = G+I, Business = J+M+N, OtherServices = R+S+T,
Government = O+P+Q+U; nível 12 has Trade = G, Business = J (informação e comunicação), Government = O and public
P/Q, and the rest in OtherServices. The other sectors are the same sections in both and keep their RAIS shares. Each
of the four groups is split by the ACP's staff by section in IBGE's Cadastro Central de Empresas (CEMPRE) 2010, Sidra
table 6450, salaried staff (var 708) by municipality. Public P and Q are the municipality's P and Q times the share of
'Administração pública' in its state's P and Q salaried staff (table 6703; the national share where the state's cell
is suppressed). A group with no CEMPRE staff in the ACP is split by the national staff.

IBGE suppresses small cells ('X'). A municipality's suppressed staff (its total less its published sections) is
spread over its suppressed sections in proportion to their national staff.

Usage: python auxiliary/sector_shares_ibge12.py [cache_dir]
"""
import json
import os
import sys
import tempfile
import urllib.parse

import numpy as np
import pandas as pd
import requests

YEAR = 2010
TOTAL = '117897'
SECTIONS = {'116830': 'A', '116880': 'B', '116910': 'C', '117296': 'D', '117307': 'E', '117329': 'F', '117363': 'G',
            '117484': 'H', '117543': 'I', '117555': 'J', '117608': 'K', '117666': 'L', '117673': 'M', '117714': 'N',
            '117774': 'O', '117788': 'P', '117810': 'Q', '117838': 'R', '117861': 'S', '117888': 'T', '117892': 'U'}
# RAIS group: {nível 12 sector: staff columns}
SPLITS = {'Trade': {'Trade': ['G'], 'OtherServices': ['I']},
          'Business': {'Business': ['J'], 'OtherServices': ['M', 'N']},
          'OtherServices': {'OtherServices': ['R', 'S', 'T']},
          'Government': {'Government': ['O', 'public_PQ'], 'OtherServices': ['private_PQ', 'U']}}
URL = 'https://servicodados.ibge.gov.br/api/v3/agregados/6450/periodos/{year}/variaveis/708?localidades={loc}' \
      '&classificacao=12762{cat}'
# Table 6703: sections P and Q by legal nature (total, public administration), all firm sizes
LEGAL_URL = ('https://servicodados.ibge.gov.br/api/v3/agregados/6703/periodos/{year}/variaveis/708?localidades='
             + urllib.parse.quote('N1[all]|N3[all]') + '&classificacao='
             + urllib.parse.quote('12762[117788,117810]|2703[117933,107315]|319[104029]'))


def fetch(loc, name, cache_dir, url=None):
    path = os.path.join(cache_dir, f'cempre_{YEAR}_{name}.json')
    if not os.path.exists(path):
        cat = urllib.parse.quote('[' + ','.join([TOTAL] + list(SECTIONS)) + ']')
        url = url or URL.format(year=YEAR, loc=urllib.parse.quote(loc), cat=cat)
        response = requests.get(url, timeout=600)
        response.raise_for_status()
        with open(path, 'wb') as f:
            f.write(response.content)
    with open(path) as f:
        return json.load(f)


def parse(data):
    rows = []
    for res in data[0]['resultados']:
        code = list(res['classificacoes'][0]['categoria'])[0]
        section = 'Total' if code == TOTAL else SECTIONS[code]
        for s in res['series']:
            rows.append((int(s['localidade']['id']), section, s['serie'][str(YEAR)]))
    return pd.DataFrame(rows, columns=['cod_mun', 'section', 'v']).pivot(index='cod_mun', columns='section', values='v')


def staff_by_section(cache_dir, codes):
    sections = list(SECTIONS.values())
    national = pd.to_numeric(parse(fetch('N1[all]', 'brasil', cache_dir)).iloc[0], errors='coerce')
    national = national[sections].fillna(0.0)
    raw = pd.concat([parse(fetch('N6[' + ','.join(codes[i:i + 100]) + ']', f'mun_{i}', cache_dir))
                     for i in range(0, len(codes), 100)])
    # '-' is a published zero, 'X' suppressed
    staff = raw[sections].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    total = pd.to_numeric(raw['Total'], errors='coerce')
    residual = (total - staff.sum(axis=1)).clip(lower=0).fillna(0.0)
    weights = (raw[sections] == 'X') * national
    weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    print(f'suppressed staff imputed: {residual.sum() / total.sum():.4f} of the total')
    return staff + weights.mul(residual, axis=0), national


def public_shares(cache_dir):
    """Share of public administration in P and Q salaried staff, by state code (0 = Brazil)"""
    staff = {}
    for res in fetch(None, 'legal_nature', cache_dir, LEGAL_URL.format(year=YEAR))[0]['resultados']:
        cats = {c['id']: list(c['categoria'])[0] for c in res['classificacoes']}
        section = 'P' if cats['12762'] == '117788' else 'Q'
        for s in res['series']:
            place = 0 if s['localidade']['nivel']['id'] == 'N1' else int(s['localidade']['id'])
            staff[(place, section, cats['2703'])] = pd.to_numeric(s['serie'][str(YEAR)], errors='coerce')
    shares = {}
    for (place, section, nature), v in staff.items():
        if nature == '107315':
            share = v / staff[(place, section, '117933')]
            shares[(place, section)] = share if np.isfinite(share) else np.nan
    national = {sec: shares[(0, sec)] for sec in 'PQ'}
    return lambda state, sec: national[sec] if np.isnan(shares.get((state, sec), np.nan)) else shares[(state, sec)]


def public_split(staff, share):
    staff = staff.copy()
    state = staff.index.map(lambda c: c // 100000)
    staff['public_PQ'] = (staff.P * [share(s, 'P') for s in state] + staff.Q * [share(s, 'Q') for s in state])
    staff['private_PQ'] = staff.P + staff.Q - staff.public_PQ
    return staff


def main(cache_dir):
    acps = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    staff, national = staff_by_section(cache_dir, [str(c) for c in acps.cod_mun])
    share = public_shares(cache_dir)
    staff = public_split(staff, share)
    acp_staff = staff.join(acps.set_index('cod_mun')).groupby('ACPs').sum()
    national = national.copy()
    national['public_PQ'] = national.P * share(0, 'P') + national.Q * share(0, 'Q')
    national['private_PQ'] = national.P + national.Q - national.public_PQ

    rais = pd.read_csv('input/CONCURBs_SECTOR.csv', sep=';', decimal=',')
    rais = rais.pivot(index='concurb_name', columns='sector', values='participation').fillna(0.0)
    rais = rais.div(rais.sum(axis=1), axis=0)
    out = rais.copy()
    for group in SPLITS:
        out[group] = 0.0
    for group, parts in SPLITS.items():
        sums = pd.DataFrame({s: acp_staff.reindex(rais.index)[cols].sum(axis=1) for s, cols in parts.items()})
        fallback = pd.Series({s: national[cols].sum() for s, cols in parts.items()})
        empty = sums.sum(axis=1) <= 0
        sums.loc[empty] = fallback.values
        fractions = sums.div(sums.sum(axis=1), axis=0)
        for sector in parts:
            out[sector] += rais[group] * fractions[sector]
    out = out[rais.columns]
    long = out.stack().rename('participation').reset_index()
    long.columns = ['concurb_name', 'sector', 'participation']
    long[['sector', 'concurb_name', 'participation']].to_csv('input/sector_shares_ibge12.csv', sep=';', index=False,
                                                            float_format='%.6f')
    print(pd.DataFrame({'RAIS groups': rais.median(), 'nível 12': out.median()}).round(3).to_string())


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
