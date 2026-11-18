"""Public administration value added (IBGE PIB dos Municípios, Sidra 5938 var 525, 2010, R$ thousand) and market VA
(input/municipal_va_2010.csv) per ACP, R$ of 2010 a month per resident.
usage (repo root): python analysis/mcmv2026/validation/apu_va25.py -> analysis/mcmv2026/validation/apu_va25.csv (the 25 capitals of the screen)"""
import pandas as pd
import requests

URL = 'https://servicodados.ibge.gov.br/api/v3/agregados/5938/periodos/2010/variaveis/525?localidades=N6[{codes}]'
ACPS = pd.read_csv('analysis/mcmv2026/validation/gate_bounds25.csv').acp.tolist()
muns = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
va = pd.read_csv('input/municipal_va_2010.csv', sep=';').set_index('cod_mun')
rows = []
for acp in ACPS:
    codes = [c for c in muns[muns.ACPs == acp].cod_mun.unique() if c in va.index]
    apu = 0.0
    for i in range(0, len(codes), 100):
        for s in requests.get(URL.format(codes=','.join(map(str, codes[i:i + 100]))), timeout=300).json()[0][
                'resultados'][0]['series']:
            apu += 1000 * float(s['serie']['2010'])
    pop = va.loc[codes, 'pop'].sum()
    rows.append({'acp': acp, 'pop': pop, 'market_pc': va.loc[codes, 'va_market'].sum() / pop / 12,
                 'apu_pc': apu / pop / 12})
out = pd.DataFrame(rows).set_index('acp')
out['gdp_pc'] = out.market_pc + out.apu_pc
out.to_csv('analysis/mcmv2026/validation/apu_va25.csv', float_format='%.2f')
print(out.round(1))
