import pandas as pd

from analysis.output import OUTPUT_DATA_SPEC


def read_model_output_regional_gdp(path, cols):
    data = pd.read_csv(path, names=cols, sep=';')
    return data


if __name__ == '__main__':
    # Simulated data
    run = 'run__2024-11-26T10_02_16.935529'
    regional_file = f'../../output/{run}/0/regional.csv'
    cols_spec = OUTPUT_DATA_SPEC['regional']['columns']
    s = read_model_output_regional_gdp(regional_file, cols_spec)
    cols_s = ['mun_id', 'gdp_region', 'gdp_percapita']
    s = s.loc[s.month == '2019-12-01'][cols_s]

    # Real data
    d = pd.read_csv('pib_municipios2021.csv')
    cols_d = ['cod_mun', 'pib_corrente', 'pib_percapita_corrente']
    d = d[cols_d]
    d = d[d['cod_mun'].isin(s['mun_id'])]
