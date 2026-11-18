"""R2 §2 gate bounds for the step-test ACPs, from published data.

usage (repo root): python analysis/mcmv2026/validation/gate_bounds.py  -> analysis/mcmv2026/validation/gate_bounds.csv + printout

G2 floor, per ACP: Census 2010 household income / IBGE 2010 value added, both net of imputed rent.
  income = Census income per person aged 10+ by AP (input/income_per_person_AP_2010.csv, Básico V009) x persons
  aged 10+ (input/num_people_age_gender_AP_2010.csv) x 12; value added = Sidra 5938 var 498 (total) minus imputed
  rent at its national share of market value added (input/municipal_va_2010.csv carries it: va_market = market VA x
  (1 - share)).
G2 ceiling, per ACP: the floor x the national under-reporting factor = national accounts household income of the
  kinds the Census asks about (wages D.11 + mixed income B.3 + social benefits D.62 received by households, IBGE
  Contas Econômicas Integradas 2010) / Census 2010 income (Sidra 1381: persons aged 10+ x mean income, x 12), Brazil.
  Same denominator (value added net of imputed rent) on both sides, so the factor is the ratio of the incomes.
G3: Census 2010 unemployment among the active (input/participation_2010.csv, all groups), and PNAD Contínua
  (Sidra 4099, quarterly, persons 14+) annual means 2012-2019: RM Belo Horizonte and RM Goiânia; Brasília = DF;
  Palmas = the capital (no RM in the survey).
Report: M2 and M4 / GDP, Brazil 2010-2019 (BCB SGS 27810, 27815; GDP 12 months SGS 4382).
"""
import io
import zipfile

import pandas as pd
import requests

ACPS = ['BELO HORIZONTE', 'BRASILIA', 'GOIANIA', 'PALMAS']
PNAD = {'BELO HORIZONTE': 'n7/3101', 'GOIANIA': 'n7/5201', 'BRASILIA': 'n3/53', 'PALMAS': 'n6/1721000'}
SIDRA = 'https://servicodados.ibge.gov.br/api/v3/agregados/5938/periodos/2010/variaveis/498?localidades=N6[{}]'
CEI = ('https://ftp.ibge.gov.br/Contas_Nacionais/Sistema_de_Contas_Nacionais/2015/tabelas_xls/'
       'contas_economicas_integradas/contas_economicas_integradas_2000a2015_xls.zip')
SGS = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json&dataInicial=01/01/2010&dataFinal=31/12/2019'


def total_va(codes):
    out = {}
    for i in range(0, len(codes), 100):
        chunk = ','.join(str(c) for c in codes[i:i + 100])
        for s in requests.get(SIDRA.format(chunk), timeout=300).json()[0]['resultados'][0]['series']:
            out[int(s['localidade']['id'])] = float(s['serie']['2010']) * 1000
    return pd.Series(out)


def underreporting():
    """National accounts / Census household income, Brazil 2010, R$ a year"""
    raw = zipfile.ZipFile(io.BytesIO(requests.get(CEI, timeout=300).content)).read('CEI2010.xls')
    d = pd.read_excel(io.BytesIO(raw), sheet_name='CEI', header=None).iloc[:, :18]
    code = d[10].astype(str).str.strip()
    received = 15  # resources column of households
    accounts = sum(float(d[code == c].iloc[0, received]) for c in ('D.11', 'B.3', 'D.62')) * 1e6
    rows = requests.get('https://apisidra.ibge.gov.br/values/t/1381/n1/all/v/140,58/p/2010/c2/6794/c86/95251',
                        timeout=120).json()[1:]
    people, mean = (float(r['V']) for r in rows)
    return accounts / (people * mean * 12)


def pnad(level):
    rows = requests.get(f'https://apisidra.ibge.gov.br/values/t/4099/{level}/v/4099/p/201201-201904',
                        timeout=120).json()[1:]
    s = pd.DataFrame(rows)
    s = s[~s.V.isin(['-', '...', '..', 'X'])]
    return s.assign(year=s.D3C.str[:4], V=s.V.astype(float)).groupby('year').V.mean() / 100


def sgs(code):
    df = pd.DataFrame(requests.get(SGS.format(code), timeout=60).json())
    return df.assign(valor=df.valor.astype(float)).valor


def main():
    muns = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    muns = muns[muns.ACPs.isin(ACPS)]
    income = pd.read_csv('input/income_per_person_AP_2010.csv', dtype={'AREAP': str})
    people = pd.read_csv('input/num_people_age_gender_AP_2010.csv', sep=';', dtype={'AREAP': str})
    people = people[people.age >= 10].groupby(['AREAP', 'mun']).num_people.sum().reset_index()
    people = people.merge(income, on='AREAP')
    people['income'] = people.num_people * people.income_per_person * 12
    market = pd.read_csv('input/municipal_va_2010.csv', sep=';', index_col='cod_mun')
    va = total_va(sorted(muns.cod_mun.unique()))
    part = pd.read_csv('input/participation_2010.csv', sep=';')
    factor = underreporting()
    rows = []
    for acp in ACPS:
        codes = muns[muns.ACPs == acp].cod_mun
        hh_income = people[people.mun.isin(codes)].income.sum()
        market_va = market.va_market.reindex(codes).sum()
        # imputed rent = market VA before the cut x share; va_market = that x (1 - share), share 0.0937 in the file
        share = 0.0937
        imputed = market_va / (1 - share) * share
        value_added = va.reindex(codes).sum() - imputed
        p = part[part.cod_mun.isin(codes)]
        census_u = 1 - p.employed.sum() / p.active.sum()
        u = pnad(PNAD[acp])
        rows.append({'acp': acp, 'g2_floor': hh_income / value_added, 'g2_ceiling': hh_income / value_added * factor,
                     'census_u': census_u, 'pnad_min': u.min(), 'pnad_max': u.max(), 'pnad_mean': u.mean()})
    out = pd.DataFrame(rows).set_index('acp')
    print(f'under-reporting factor (national accounts / Census) {factor:.3f}')
    gdp = sgs(4382)
    out.to_csv('analysis/mcmv2026/validation/gate_bounds.csv', float_format='%.4f')
    print(out.round(3).to_string())
    print('M2 / GDP 2010-19', (sgs(27810) / 1e3 / gdp).agg(['min', 'max']).round(3).to_dict(),
          'M4 / GDP 2010-19', (sgs(27815) / 1e3 / gdp).agg(['min', 'max']).round(3).to_dict())


if __name__ == '__main__':
    main()
