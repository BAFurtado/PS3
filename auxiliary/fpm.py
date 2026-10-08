"""FPM a year per resident in R$ of 2010 (input/fpm_real_pc.csv), from the FPM paid to each municipality
(input/fpm/{UF}.csv, R$ of the year, net of the FUNDEB retention).

input/fpm/{UF}.csv: Secretaria do Tesouro Nacional, Transferências Constitucionais para Municípios; it equals the
rows with item and transfer 'FPM' of the STN files (Tesouro Transparente, dataset
transferencias-constitucionais-para-municipios), three decêndios summed, wherever those files are complete (yearly files
2010-2015, monthly 2016-2017 checked 2026-10-08). Later monthly files have gaps (October 2019 lists only the FUNDEB
retention; May 2018 has no transfer column), so extending the table needs the STN transfers API.
Deflated by the IPCA annual mean (Sidra 1737, variable 2266), over the municipality's population on its 2010-2022
Census path (input/census_population_2010_2022.csv, geometric, extended at the same rate after 2022).

Usage: python auxiliary/fpm.py
"""
import os

import pandas as pd
import requests

IPCA = 'https://servicodados.ibge.gov.br/api/v3/agregados/1737/periodos/{periods}/variaveis/2266?localidades=N1[all]'
FIRST = 2010


def main():
    fpm = pd.concat(pd.read_csv(f'input/fpm/{f}') for f in sorted(os.listdir('input/fpm')))
    fpm = fpm[fpm.ano >= FIRST]
    years = range(FIRST, int(fpm.ano.max()) + 1)
    periods = '|'.join(f'{y}{m:02d}' for y in years for m in range(1, 13))
    serie = requests.get(IPCA.format(periods=periods), timeout=300).json()[0]['resultados'][0]['series'][0]['serie']
    index = {y: sum(float(v) for k, v in serie.items() if k.startswith(str(y))) / 12 for y in years}
    pop = pd.read_csv('input/census_population_2010_2022.csv', sep=';').set_index('cod_mun')
    growth = (pop.pop_2022 / pop.pop_2010) ** (1 / 12)
    fpm = fpm[fpm.cod.isin(pop.index)]
    residents = pop.pop_2010[fpm.cod].values * growth[fpm.cod].values ** (fpm.ano.values - FIRST)
    deflator = fpm.ano.map(lambda y: index[y] / index[FIRST]).values
    out = pd.DataFrame({'cod': fpm.cod.values, 'ano': fpm.ano.values, 'fpm_pc': fpm.fpm.values / deflator / residents})
    out.sort_values(['cod', 'ano']).to_csv('input/fpm_real_pc.csv', sep=';', index=False, float_format='%.4f')
    print(out.groupby('ano').fpm_pc.median().round(1).to_string())


if __name__ == '__main__':
    main()
