import geopandas as gpd

import matplotlib.pyplot as plt
import pandas as pd

from analysis.output import OUTPUT_DATA_SPEC


def read_model_output_regional_gdp(path, cols):
    data = pd.read_csv(path, names=cols, sep=';')
    return data


def plot(data, text, full_region, urban_region):
    """Generate a spatial plot"""
    # Loading the shapefiles

    if len(data.columns) == 4:

        plots = data.columns[2:]
    else:
        plots = data.columns[1:]

    for p in plots:
        # Starting the plot
        fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'aspect': 'equal'})
        vmin, vmax = data[p].min(), data[p].max()

        # Plotting each polygon in the selection process (based on mun_codes)
        # Urban areas (ACPs IBGE)
        for mun in data.cod_mun:
            shape_select = urban_region[urban_region['GEOCODI'] == str(mun)].copy()
            shape_select.plot(ax=ax, color='black', linewidth=0.2, alpha=.2, edgecolor='black')

        for mun in data.cod_mun:
            shape_select = full_region[full_region['CD_GEOCMU'] == str(mun)].copy()
            shape_select.CD_GEOCMU = shape_select.CD_GEOCMU.astype(int)
            shape_select.plot(ax=ax, color='grey', linewidth=0.5, alpha=.7, edgecolor='black')

            # Plotting
            merged = shape_select.merge(data, left_on='CD_GEOCMU', right_on='cod_mun')
            merged.plot(
                ax=ax,
                column=p,
                cmap='viridis',
                legend=False,
                legend_kwds={'shrink': 0.5, 'label': p.capitalize().replace('_', ' ')},
                linewidth=0.5,
                edgecolor='black',
                vmin=vmin,
                vmax=vmax
            )
            # Add labels for each polygon (municipality names)
            for i, row in shape_select.iterrows():
                x, y = row.geometry.centroid.x, row.geometry.centroid.y  # Get the centroid for label placement
                ax.text(
                    x, y, row['NM_MUNICIP'],  # The 'nome_mun' column contains the municipality names
                    fontsize=14,  # Adjust font size as needed
                    ha='center',  # Align the text horizontally
                    color='white',  # Adjust text color as needed
                    weight='bold'  # Optional: makes the text bold
                )

        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        fig.colorbar(sm, cax=cax)
        # Adding the grid location, title, axes labels
        ax.grid(True, color='grey', linestyle='-')
        title = f'{p.capitalize().replace('_', ' ')} ({text} data)'
        ax.set_title(title)
        ax.set_xlabel('Longitude (in degrees)')
        ax.set_ylabel('Latitude (in degrees)')
        plt.savefig(f'results/{title}.png')
        plt.show()


if __name__ == '__main__':
    # Simulated data
    run = 'run__2024-11-27T14_51_00.553290'
    regional_file = f'../../output/{run}/0/regional.csv'
    cols_spec = OUTPUT_DATA_SPEC['regional']['columns']
    s = read_model_output_regional_gdp(regional_file, cols_spec)
    cols_s = ['mun_id', 'gdp_region', 'gdp_percapita']
    s = s.loc[s.month == '2019-12-01'][cols_s]
    s.rename(columns={'mun_id': 'cod_mun'}, inplace=True)

    # Real data
    d = pd.read_csv('pib_municipios2021.csv')
    cols_d = ['cod_mun', 'nome_mun', 'pib_corrente', 'pib_percapita_corrente']
    d = d[cols_d]
    d = d[d['cod_mun'].isin(s['cod_mun'])]

    full_r = gpd.read_file('../../input/shapes/mun_ACPS_ibge_2014_latlong_wgs1984_fixed.shp')
    urban_r = gpd.read_file('../../input/shapes/URBAN_IBGE_ACPs.shp')

    # Plot
    for each in zip([d, s], ['real', 'simulated']):
        plot(each[0], each[1], full_r, urban_r)

