"""Gate bounds (G2, G3) and reference levels for the 25-capital screen, same sources as gate_bounds.py.

usage (repo root): python analysis/mcmv2026/validation/gate_bounds25.py  -> analysis/mcmv2026/validation/gate_bounds25.csv + printout

G2 and G3 as in gate_bounds.py. PNAD Contínua (Sidra 4099) for the metro where the survey has one, else the capital
(Teresina, Porto Velho, Rio Branco, Boa Vista, Palmas, Campo Grande); Brasília = DF.
Also: real GDP growth 2012-19 %/yr (Sidra 5938 var 37 summed over the ACP, deflated by the national GDP deflator
2012-19, 1.522); population growth 2010-22 %/yr (input/census_population_2010_2022.csv); state Gini of
household income per capita, mean 2012-19 (PNAD Contínua, Sidra 7435; a different concept from the model's family
permanent income, reference only).
"""
import os
import sys

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_bounds as GB  # noqa: E402

PNAD = {'MANAUS': 'n7/1301', 'BELEM': 'n7/1501', 'MACAPA': 'n7/1601', 'SAO LUIS': 'n7/2101', 'FORTALEZA': 'n7/2301',
        'NATAL': 'n7/2401', 'JOAO PESSOA': 'n7/2501', 'RECIFE': 'n7/2601', 'MACEIO': 'n7/2701', 'ARACAJU': 'n7/2801',
        'SALVADOR': 'n7/2901', 'BELO HORIZONTE': 'n7/3101', 'RIO DE JANEIRO': 'n7/3301', 'SAO PAULO': 'n7/3501', 'VITORIA': 'n7/3201', 'CURITIBA': 'n7/4101',
        'FLORIANOPOLIS': 'n7/4201', 'PORTO ALEGRE': 'n7/4301', 'CUIABA': 'n7/5101', 'GOIANIA': 'n7/5201',
        'BRASILIA': 'n3/53', 'TERESINA': 'n6/2211001', 'PORTO VELHO': 'n6/1100205', 'RIO BRANCO': 'n6/1200401',
        'BOA VISTA': 'n6/1400100', 'PALMAS': 'n6/1721000', 'CAMPO GRANDE': 'n6/5002704'}
DEFLATOR_12_19 = 1.522
GDP = 'https://servicodados.ibge.gov.br/api/v3/agregados/5938/periodos/2012|2019/variaveis/37?localidades=N6[{}]'


def nominal_gdp(codes):
    out = {}
    for i in range(0, len(codes), 100):
        chunk = ','.join(str(c) for c in codes[i:i + 100])
        for s in requests.get(GDP.format(chunk), timeout=300).json()[0]['resultados'][0]['series']:
            out[int(s['localidade']['id'])] = {y: float(v) for y, v in s['serie'].items()}
    return pd.DataFrame(out).T


def state_gini():
    rows = requests.get('https://apisidra.ibge.gov.br/values/t/7435/n3/all/v/10681/p/2012-2019', timeout=120).json()[1:]
    s = pd.DataFrame(rows)
    s = s[~s.V.isin(['-', '...', '..', 'X'])]
    return s.assign(V=s.V.astype(float), uf=s.D1C.astype(int)).groupby('uf').V.mean()


def main():
    muns = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    muns = muns[muns.ACPs.isin(PNAD)]
    income = pd.read_csv('input/income_per_person_AP_2010.csv', dtype={'AREAP': str})
    people = pd.read_csv('input/num_people_age_gender_AP_2010.csv', sep=';', dtype={'AREAP': str})
    people = people[people.age >= 10].groupby(['AREAP', 'mun']).num_people.sum().reset_index()
    people = people.merge(income, on='AREAP')
    people['income'] = people.num_people * people.income_per_person * 12
    market = pd.read_csv('input/municipal_va_2010.csv', sep=';', index_col='cod_mun')
    codes_all = sorted(muns.cod_mun.unique())
    va = GB.total_va(codes_all)
    gdp = nominal_gdp(codes_all)
    pop = pd.read_csv('input/census_population_2010_2022.csv', sep=';', index_col='cod_mun')
    part = pd.read_csv('input/participation_2010.csv', sep=';')
    gini = state_gini()
    factor = GB.underreporting()
    rows = []
    for acp in sorted(PNAD):
        codes = muns[muns.ACPs == acp].cod_mun.unique()
        hh_income = people[people.mun.isin(codes)].income.sum()
        market_va = market.va_market.reindex(codes).sum()
        imputed = market_va / (1 - 0.0937) * 0.0937
        value_added = va.reindex(codes).sum() - imputed
        p = part[part.cod_mun.isin(codes)]
        u = GB.pnad(PNAD[acp])
        g = gdp.reindex(codes).sum()
        c = pop.reindex(codes).sum()
        rows.append({'acp': acp, 'g2_floor': hh_income / value_added, 'g2_ceiling': hh_income / value_added * factor,
                     'census_u': 1 - p.employed.sum() / p.active.sum(), 'pnad_min': u.min(), 'pnad_max': u.max(),
                     'pnad_mean': u.mean(),
                     'real_gdp_12_19': ((g['2019'] / g['2012'] / DEFLATOR_12_19) ** (1 / 7) - 1) * 100,
                     'pop_10_22': ((c.pop_2022 / c.pop_2010) ** (1 / 12) - 1) * 100,
                     'state_gini_12_19': gini.get(int(str(codes[0])[:2]))})
    out = pd.DataFrame(rows).set_index('acp')
    print(f'under-reporting factor (national accounts / Census) {factor:.3f}')
    out.to_csv('analysis/mcmv2026/validation/gate_bounds25.csv', float_format='%.4f')
    print(out.round(3).to_string())


if __name__ == '__main__':
    main()
