import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys,os


project_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
print(project_folder)
sys.path.insert(0, project_folder)

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
        title = f'{p.capitalize().replace("_", " ")} ({text} data)'
        ax.set_title(title)
        ax.set_xlabel('Longitude (in degrees)')
        ax.set_ylabel('Latitude (in degrees)')
        plt.savefig(project_folder +'/analysis/validation/'+f'results/{title}.png')
        plt.show()

def plot_mun_hist(real,sim, text, full_region, urban_region):
    real = real.sort_values(by='massa_salarial_sum', ascending=False)
    sim = sim.sort_values(by='gdp_percapita', ascending=False)
    real['wages'] = real['massa_salarial_sum']/np.mean(real['massa_salarial_sum'])
    sim['wages'] = sim['gdp_percapita']/np.mean(sim['gdp_percapita'])
    sim['rank'] = range(1, len(sim) + 1)
    real['rank'] = range(1, len(real) + 1)
    #plt.figure(figsize=(10, 6))
    
    bar_width = 0.4
    x = np.arange(len(sim))
    
    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'aspect': 'equal'})
    ax.bar(x - bar_width / 2, sim['wages'], width=bar_width, label='Simulated Data', color='blue')
    ax.bar(x + bar_width / 2, real['wages'], width=bar_width, label='Real Data', color='green')
    

    ax.grid(True, color='grey', linestyle='-')
    title = f'Normalized mean wage distribution'
    ax.set_title(title)


    # Customize plot
   
    
    plt.xlabel('Rank')
    plt.ylabel('Normalized GDP')
    #plt.xticks(x, [f"Rank {i+1}" for i in range(len(df1))])  # Label ranks
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(project_folder +'/analysis/validation/'+f'results/{title}.png')
    plt.show()

if __name__ == '__main__':
    # Simulated data
    run = 'run__2024-12-19T15_57_43.169698' #PS3\output\PS3\output\
    regional_file = project_folder + f'/output/{run}/0/regional.csv'
    cols_spec = OUTPUT_DATA_SPEC['regional']['columns']
    s = read_model_output_regional_gdp(regional_file, cols_spec)
    cols_s = ['mun_id', 'gdp_percapita', 'pop']
    s = s.loc[s.month == '2019-12-01'][cols_s]
    #s['employment'] = s['pop'] *(1- s['regional_unemployment'])
    s['6digit'] = s['mun_id'].apply(lambda x: int(str(x)[:-1]))
    s.rename(columns={'mun_id': 'cod_mun'}, inplace=True)

    # Real data
    d = pd.read_csv(project_folder +'/analysis/validation/real_world_data/mun_isic12_2010.csv')
    cols_d = ['cod_mun', 'qtde_vinc_ativos_sum', 'massa_salarial_sum']
    #d.rename(columns={'codemun': 'cod_mun'}, inplace=True)
    d = d.groupby(['codemun'], as_index=False).sum()
    
    d = d[d['codemun'].isin(s['6digit'])]
    d = pd.merge(d, s[['6digit', 'cod_mun']], how='left', left_on='codemun', right_on='6digit')
    d = d[cols_d]
    s.drop('6digit', axis=1, inplace=True)

    full_r = gpd.read_file(project_folder +'/input/shapes/mun_ACPS_ibge_2014_latlong_wgs1984_fixed.shp')
    urban_r = gpd.read_file(project_folder +'/input/shapes/URBAN_IBGE_ACPs.shp')
    plot_mun_hist(d, s, 'real', full_r, urban_r)

    # Plot
    for each in zip([d, s], ['real', 'simulated']):
        pass
        #plot(each[0], each[1], full_r, urban_r)
        #project_folder

    pass