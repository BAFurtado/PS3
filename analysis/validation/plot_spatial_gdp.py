import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm

from analysis.output import OUTPUT_DATA_SPEC


def read_model_output_regional_gdp(path, cols):
    data = pd.read_csv(path, names=cols, sep=';')
    return data


def plot(data):
    """Generate a spatial plot"""
    # Loading the shapefiles
    full_region = gpd.read_file('../../input/shapes/mun_ACPS_ibge_2014_latlong_wgs1984_fixed.shp')
    urban_region = gpd.read_file('../../input/shapes/URBAN_IBGE_ACPs.shp')

    plots = data.columns[1:]
    figs = []

    for p in plots:
        # Starting the plot
        fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'aspect': 'equal'})

        # Plotting each polygon in the selection process (based on mun_codes)
        # Urban areas (ACPs IBGE)
        for mun in data.cod_mun:
            shape_select = urban_region[urban_region['GEOCODI'] == str(mun)].copy()
            shape_select.plot(ax=ax, color='black', linewidth=0.2, alpha=.2, edgecolor='black')

        for mun in data.cod_mun:
            shape_select = full_region[full_region['CD_GEOCMU'] == str(mun)].copy()
            shape_select.CD_GEOCMU = shape_select.CD_GEOCMU.astype(int)
            shape_select.plot(ax=ax, color='grey', linewidth=0.5, alpha=.7, edgecolor='black')

        # Plotting=
        merged = shape_select.merge(data, left_on='CD_GEOCMU', right_on='cod_mun')
        merged.plot(
            ax=ax,
            column=p,
            cmap='viridis',
            legend=False,
            legend_kwds={'shrink': 0.5, 'label': p.capitalize().replace('_', ' ')},
            linewidth=0.5,
            edgecolor='black',
        )
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []
        fig.colorbar(sm, cax=cax)
        # Adding the grid location, title, axes labels
        ax.grid(True, color='grey', linestyle='-')
        # ax.set_title(p.capitalize().replace('_', ' '))
        ax.set_xlabel('Longitude (in degrees)')
        ax.set_ylabel('Latitude (in degrees)')
        figs.append((p, fig))
        plt.show()

    return figs


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

    # Plot
    fs = plot(d)

