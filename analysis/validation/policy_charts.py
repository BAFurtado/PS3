import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import os
from data_utils import *
from validation_charts import *
def create_grouped_boxplot(data, x_col, y_col, hue_col, title, xlabel, ylabel, figsize=(12, 6), palette="Set2", rotation=45):
    """
    Generates grouped boxplots to compare policy effects across urban agglomerations.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing urban agglomeration, policy type, and emission data.
    - x_col (str): Column name for the x-axis (urban agglomeration).
    - y_col (str): Column name for the y-axis (e.g., Emission per GDP).
    - hue_col (str): Column for policy grouping.
    - title (str): Plot title.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - figsize (tuple): Size of the plot.
    - palette (str): Color palette for policies.
    - rotation (int): Rotation angle for x-axis labels.
    """
    plt.figure(figsize=figsize)
    sns.boxplot(data=data, x=x_col, y=y_col, hue=hue_col, palette=palette, width=0.7)
    
    plt.xticks(rotation=rotation, ha='right', fontsize=10)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title=hue_col, loc='upper right')
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()

def create_faceted_boxplot(data, x_col, y_col, col_col, title, xlabel, ylabel, ncols=2, figsize=(14, 8), palette="muted", rotation=90):
    """
    Generates faceted boxplots, splitting policies into separate subplots.
    """
    g = sns.catplot(data=data, x=x_col, y=y_col, col=col_col, kind="box", palette=palette, col_wrap=ncols, height=4, aspect=1.5)
    
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=7)
    
    g.fig.suptitle(title, fontsize=16)
    g.set_axis_labels(xlabel, ylabel)
    g.tight_layout()
    plt.show()

def create_heatmap(data, index_col, columns_col, values_col,title,filter_name="emissions", figsize=(12, 8), cmap="coolwarm",agg='mean'):
    """
    Generates a heatmap showing emissions per GDP across urban agglomerations and policies.
    Aggregates data using mean values before pivoting.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing urban agglomeration, policy type, and emission data.
    - index_col (str): Column name for the index (urban agglomeration).
    - columns_col (str): Column name for the columns (policy type).
    - values_col (str): Column name for the heatmap values (e.g., Emission per GDP).
    - title (str): Plot title.
    - figsize (tuple): Figure size.
    - cmap (str): Color map for visualization.
    """
    # Aggregate by mean to ensure unique values for pivot
    data = data[data["description"].isin([filter_name])]
    #data = data[data["Date"]>='2019-01-01']
    if agg == 'mean':
        aggregated_data = data.groupby([index_col, columns_col])[values_col].mean().reset_index()
    else:
        aggregated_data = data.groupby([index_col, columns_col])[values_col].sum().reset_index()
    

    # Pivot the data
    pivot_table = aggregated_data.pivot(index=index_col, columns=columns_col, values=values_col)
    pivot_table.columns = ['Both', 'Carbon Tax','No Policy','Subsidies']
    hue_order = ['No Policy', 'Carbon Tax','Subsidies','Both']
    print(pivot_table.columns)
    pivot_table = pivot_table[hue_order]
    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(pivot_table, 
                annot=True, 
                fmt=".2f",
                cmap=cmap, linewidths=0.5, cbar_kws={'label': values_col})
    
    plt.title(title, fontsize=14)
    plt.xlabel(columns_col, fontsize=12)
    plt.ylabel(index_col, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

def create_emissions_violinplot_by_sector(data, sector_col, policy_col, value_col, xlabel, ylabel,
                                          title,filter_name="emissions", figsize=(16, 12), 
                                          palette="Set2", rotation=45, col_wrap=4,normalize=False):
    """
    Generates faceted violin plots (one panel per sector) to visualize the distribution 
    of firms' emissions reduction values for each policy, with a shared y-axis.
    Also adds a legend to show policy colors.
    """
    # Create the color mapping
    import seaborn as sns
    data = data[data["description"].isin([filter_name])]
    if normalize:
        data[value_col] = (data[value_col]-data[value_col].min()) / (data[value_col].max()-data[value_col].min()) 
    unique_policies = data[policy_col].unique()
    color_palette = sns.color_palette(palette, len(unique_policies))
    policy_colors = dict(zip(unique_policies, color_palette))

    # Create the facet grid with violin plots
    g = sns.catplot(
        data=data,
        kind="violin",
        x=policy_col,
        y=value_col,
        col=sector_col,
        col_wrap=col_wrap,
        palette=policy_colors,  # Ensure consistent colors
        inner="quartile",
        height=4,
        aspect=1.2,
        sharey=True
    )
    
    # Remove the "Sector=" prefix from facet titles
    g.set_titles("{col_name}")

    # Manually set sector names as x-axis labels
    for ax, sector_name in zip(g.axes.flatten(), data[sector_col].unique()):
        ax.set_xlabel(sector_name, fontsize=12) 
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)        
    
    # Set overall title and axis labels
    g.figure.suptitle(title, fontsize=18)
    g.set_axis_labels(xlabel, ylabel)
    
    # Create a legend
    legend_patches = [mpatches.Patch(color=color, label=policy) for policy, color in policy_colors.items()]
    g.figure.legend(handles=legend_patches, title="Policy", loc='upper right', bbox_to_anchor=(.99, 1))
    
    # Adjust layout
    #g.figure.subplots_adjust(top=0.95, left=0.05, right=0.85, bottom=0.08)  # Make space for legend
    #plt.tight_layout()
    plt.show()

def create_grouped_boxplot_by_sector(data, sector_col, policy_col, value_col,
                                     title, xlabel, ylabel,filter_name="emissions",figsize=(16, 8), 
                                     palette="Set2", rotation=45,normalize=False):
    """
    Creates a grouped boxplot of emission reductions per sector, ranking them 
    by the largest average reduction (across policies).
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data.
    - sector_col (str): Column name for the sector.
    - policy_col (str): Column name for the policy.
    - value_col (str): Column name for the emissions reduction values.
    - title (str): Title of the chart.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - figsize (tuple): Figure size.
    - palette (str): Color palette for policies.
    - rotation (int): Rotation angle for x-axis tick labels.
    """
    # Compute mean emission reduction per sector (ignoring policy)
    data = data.loc[data["description"].isin([filter_name])]
    if normalize:
        #data[value_col] = (data[value_col]-data[value_col].min()) / (data[value_col].max()-data[value_col].min()) 
        data[value_col] = (data[value_col]-data[value_col].mean()) / (data[value_col].std()) 
    sector_means = data.groupby(sector_col)[value_col].mean().sort_values(ascending=False)

    # Reorder the DataFrame based on the computed ranking
    data[sector_col] = pd.Categorical(data[sector_col], categories=sector_means.index, ordered=True)

    # Create the boxplot
    plt.figure(figsize=figsize)
    ax = sns.boxplot(
        data=data,
        x=sector_col,
        y=value_col,
        hue=policy_col,  # Different colors per policy
        palette=palette
    )
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=rotation, ha='right', fontsize=12)

    # Set labels and title
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=18)

    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Move the legend outside the plot
    plt.legend(title="Policy", loc="upper right", bbox_to_anchor=(1.05, 1))

    # Adjust layout to fit everything nicely
    plt.tight_layout()
    plt.show()

def make_plot_sectors_emission(data, name='figures/emission_policy_effects',
                               policy_col='policy',filter_name="emissions",
                               exclude_sectors=[]):
    """
    Creates a scatter plot with vertical error bars (±1.96*std_error) for each sector,
    grouped by emission policy. The x-axis shows sectors (as numeric positions with labels)
    and each policy is given its own color. Marker transparency is based on p-value.

    Parameters:
    - data (pd.DataFrame): Must contain columns 'sectors', 'parameter', 'policy', 'pvalue', 'std_error'.
    - name (str): File name prefix for saving figures (EPS, PDF, etc.).
    - exclude_sectors (list): List of sectors to exclude from the plot.
    """

    # Define colors for each emission policy.
    # Here, we assume three emission policies:
    # For example, "Taxation", "Subsidies", "Combined Policy".
    colors = ['tab:red', 'tab:blue', 'tab:green']
    policy_list = data[policy_col].unique()

    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(8, 6))

    #adjust data
    data = data.loc[data["description"].isin([filter_name])]
    # Sort the data by 'parameter' (descending) and set 'sectors' as index.
    data = data.sort_values(by='parameter', ascending=False).set_index('sectors')

    # Create a list of sector labels (exclude any sectors as specified).
    labels = [sector for sector in data.index.unique().to_list() if sector not in exclude_sectors]

    # To plot on a numeric x-axis, get positions corresponding to each sector.
    positions = {sector: idx for idx, sector in enumerate(labels)}

    # Loop over each emission policy.
    for i, policy in enumerate(policy_list):
        sub = data[data['policy'] == policy]
        for sector in labels:
            try:
                # Extract the parameter value, p-value, and standard error.
                value = sub.loc[sector, 'parameter']
                pval = sub.loc[sector, 'pvalue']
                std_err = sub.loc[sector, 'std_error']
            except KeyError:
                continue  # Skip sectors with no data for this policy.

            # Use the numeric position for the x-axis.
            pos = positions[sector]

            # Plot the scatter point.
            ax.scatter(pos, value,
                       color=colors[i],
                       alpha=0.9 if pval < 0.1 else 0.4,
                       marker='o')

            # Draw vertical lines representing the 95% confidence interval.
            ax.vlines(x=pos,
                      ymin=value - 1.96 * std_err,
                      ymax=value + 1.96 * std_err,
                      color=colors[i],
                      alpha=0.7,
                      lw=0.7)

    # Draw a horizontal reference line at y = 0.
    ax.hlines(y=0, xmin=-0.5, xmax=len(labels)-0.5, colors='black', lw=1, alpha=0.7)

    # Set x-ticks using the numeric positions and assign the sector labels.
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation='vertical', fontsize=10)

    # Create legend elements for each emission policy.
    legend_elements = [
        Line2D([0], [0], marker='o', label='Taxation', color=colors[0], lw=0.5),
        Line2D([0], [0], marker='o', label='Subsidies', color=colors[1], lw=0.5),
        Line2D([0], [0], marker='o', label='Combined Policy', color=colors[2], lw=0.5)
    ]
    ax.set_ylabel("Effect of Emission Policy relative to Baseline")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(handles=legend_elements, edgecolor='white', loc='upper right', facecolor='white', framealpha=1)

    plt.tight_layout()
    # Save the figure in EPS and PDF formats.
    #plt.savefig(f'{name}.eps', dpi=1200, format='eps')
    #plt.savefig(f'{name}.pdf', dpi=1200)
    plt.show()

def plot_variable_distribution(df, variable, bins=30,initial_month="2020-01-01",save_path='dist_emission.png'):
    """
    Plots a histogram of the selected variable's distribution across different policies.

    Parameters:
    - df (pd.DataFrame): The data frame containing the data.
    - variable (str): The variable (description) to filter and plot.
    - bins (int, optional): Number of bins in the histogram. Default is 30.

    Returns:
    - Displays the histogram.
    """
    # Filter the dataframe for the selected variable
    df_filtered = df[df["description"] == variable]
    df_filtered = df_filtered.loc[df_filtered['value']!=0]
    df_filtered = df_filtered.loc[df_filtered['month']>=initial_month]

    # Check if there's data to plot
    if df_filtered.empty:
        print(f"No data found for variable: {variable}")
        return

    # Plot histogram
    r = 1.5
    plt.figure(figsize=(10/r, 8/r))
    
    sns.histplot(data=df_filtered, 
                 x="value",
                 hue="Policy", 
                 stat="density",
                 bins=bins, kde=True, element="step", common_norm=False,legend=True,alpha=0.1)
    sns.set_style({'axes.facecolor':'white', 'grid.color': '.8'})
    sns.despine(offset=10, trim=True);
    for n,policy in enumerate(df_filtered["Policy"].unique()):
        plt.axvline(x=df_filtered.loc[df_filtered["Policy"] == policy,"value"].median(),
                    linestyle='--',color=sns.color_palette("tab10")[n],label=policy)
    # Labels and title
    plt.xlabel("Value")#f"{variable.capitalize()} 
    plt.ylabel("Frequency")
    #plt.title(f"Distribution of {variable} Across Policies")
    save_path = save_path+'.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Figure saved to {save_path}")


    
    # Show legend and plot
    #plt.legend(title="Policy", loc="best",)
    plt.show()

def plot_dual_variable_distribution(df, x_variable, y_variable, initial_month="2020-01-01", bins=30):
    """
    Plots a 2D histogram of two selected variables' distribution across different policies,
    with an optional date range filter.

    Parameters:
    - df (pd.DataFrame): The data frame containing the data.
    - x_variable (str): The first variable (description) to filter and plot on the x-axis.
    - y_variable (str): The second variable (description) to filter and plot on the y-axis.
    - start_month (str, optional): The start month in 'YYYY-MM' format.
    - end_month (str, optional): The end month in 'YYYY-MM' format.
    - bins (int, optional): Number of bins in the histogram. Default is 30.

    Returns:
    - Displays the 2D histogram.
    """
    # Convert month column to datetime if not already
    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    # Filter the dataframe for the selected variables
    df_filtered = df[df["description"].isin([x_variable, y_variable])]
    df_filtered = df_filtered.loc[df_filtered['month']>=initial_month]
    df_grouped = df_filtered.groupby(["month", "Policy", "description"])["value"].mean().reset_index()

    # Pivot so that x and y variables are in separate columns
    df_pivot = df_grouped.pivot(index=["month", "Policy"], columns="description", values="value").reset_index()

    # App

    # Check if there's data to plot
    if df_pivot.empty:
        print(f"No data found for variables: {x_variable} and {y_variable} in the given date range.")
        return

    # Set up the figure
    plt.figure(figsize=(10, 6))
    policies = df_filtered["Policy"].unique()
    palette = sns.color_palette("tab10", n_colors=len(policies))

    # Create 2D histogram
    ax = sns.kdeplot(
        data=df_pivot, 
        x=x_variable, 
        y=y_variable, 
        hue="Policy", 
        hue_order=policies,
        bins=bins, 
        cbar=False,
        fill=True,
        alpha=.5,
        palette=palette
    )

    # Labels and title
    plt.xlabel(f"{x_variable} Value")
    plt.ylabel(f"{y_variable} Value")

    # Add mean lines for each policy and variable
        # Calculate the means for both x_variable and y_variable
    mean_x = df_filtered[df_filtered["description"] == x_variable].groupby("Policy")["value"].mean()
    mean_y = df_filtered[df_filtered["description"] == y_variable].groupby("Policy")["value"].mean()

    # Add mean dots for each policy (mean of x_variable and y_variable together)
    for i, policy in enumerate(policies):
        x_mean = mean_x.loc[policy]
        y_mean = mean_y.loc[policy]
        print(policy,x_mean,y_mean)

    # Plot the mean dot for each policy (x_mean, y_mean)
        plt.scatter(x_mean, y_mean, color=palette[i], s=100, zorder=5, 
                    label=f"Mean {policy}", marker='o')
    title = f"Distribution of {x_variable} vs {y_variable} Across Policies"
    

    plt.title(title)
    # Show plot
    plt.show()


def plot_firms_on_income_map(subdivisions_gdf,
    firms_df,
    lat_col='lat',
    lon_col='long',
    description_col='description',
    value_col='value',
    region_col='region_id',  # or whatever uniquely identifies regions
    efficiency_var='eco_eff',
    income_var='wages_paid',
    policy=None
):
    # Pivot to wide format
    filtered_df = firms_df[firms_df[description_col].isin([efficiency_var, income_var])].copy()
    #Correct the region id 
    filtered_df[region_col] = filtered_df[region_col][:7]
    if policy is not None:
        filtered_df = filtered_df[filtered_df["Policy"] == policy]
    subdivisions_gdf[region_col] = subdivisions_gdf['CD_GEOCMU']
    pivot_df = filtered_df.pivot_table(
        index=['lat', 'long'],
        columns=description_col,
        values=value_col
    ).reset_index()

    # Create GeoDataFrame for firms
    firms_gdf = gpd.GeoDataFrame(
        pivot_df,
        geometry=gpd.points_from_xy(pivot_df[lon_col], pivot_df[lat_col]),
        crs='EPSG:4326'
    )
    firms_gdf = firms_gdf.to_crs(subdivisions_gdf.crs)

    # Spatial join to attach region info to each firm
    joined = gpd.sjoin(firms_gdf, subdivisions_gdf[[region_col, 'geometry']], how='inner', predicate='within')

    # Compute mean income per region for the background map
    income_by_region = joined.groupby(region_col)[income_var].mean().reset_index()
    subdivisions_gdf = subdivisions_gdf.merge(income_by_region, on=region_col, how='left')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    subdivisions_gdf.plot(column=income_var, ax=ax, legend=True, cmap='Blues', alpha=0.6, edgecolor='gray')

    joined.plot(ax=ax, column=efficiency_var, cmap='RdYlGn', markersize=5, legend=True, alpha=0.8)

    ax.set_title('Firm Eco-efficiency over Income Map: {}'.format(policy))
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_kde_contours(
    firms_df,
    subdivisions_gdf,
    lat_col='lat',
    lon_col='long',
    description_col='description',
    value_col='value',
    efficiency_var='eco_eff',
    region_col='region_id',
    projection='EPSG:3857',
    policy=None,
    save_path='kde_eco_efficiency'
):
    
    # Filter and pivot to get eco-efficiency   
    filtered_df = firms_df[firms_df[description_col] == efficiency_var].copy()
    #Correct the region id 
    filtered_df[region_col] = filtered_df[region_col].apply(lambda x: str(x)[:7])
    if policy is not None:
        filtered_df = filtered_df[filtered_df["Policy"] == policy]
    pivot_df = filtered_df.pivot_table(
        index=['lat', 'long'],
        values=value_col,
        aggfunc='mean'
    ).reset_index().rename(columns={value_col: efficiency_var})

    # Convert firm points to GeoDataFrame
    firm_gdf = gpd.GeoDataFrame(
        pivot_df,
        geometry=gpd.points_from_xy(pivot_df[lon_col], pivot_df[lat_col]),
        crs='EPSG:4326'
    ).to_crs(projection)

    # Reproject subdivisions
    subdivisions_gdf = subdivisions_gdf.to_crs(projection)
    subdivisions_gdf[region_col] = subdivisions_gdf['CD_GEOCMU']

    # Spatial join to identify intersecting regions
    joined = gpd.sjoin(firm_gdf, subdivisions_gdf[[region_col, 'geometry']], how='inner', predicate='within')

    # Filter subdivisions to only those seen in firms
    relevant_regions = joined[region_col].unique()
    filtered_subdivisions = subdivisions_gdf[subdivisions_gdf[region_col].isin(relevant_regions)]

    # Plot setup (optimized for A4 one-column width: ~3.5 inches)
    fig_width = 3.5  # inches
    dpi = 300
    fig, ax = plt.subplots(figsize=(fig_width, fig_width), dpi=dpi)

    # Plot the filtered subdivisions
    filtered_subdivisions.plot(ax=ax, color='white', edgecolor='gray', linewidth=0.5, alpha=0.3)

    # KDE plot with weights
    sns.kdeplot(
        x=joined.geometry.x,
        y=joined.geometry.y,
        weights=joined[efficiency_var],
        cmap="RdYlGn",
        fill=True,
        levels=15,
        alpha=0.6,
        ax=ax
    )

    ax.set_title('Eco-Efficiency KDE: {}'.format(policy), fontsize=14)  # Set the title with the policy, fontsize=10)
    ax.axis('off')
    plt.tight_layout()

    # Save to SVG
    save_path = save_path+'_{}.svg'.format(policy)
    fig.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close(fig)

    print(f"Figure saved to: {save_path}")

def plot_bivariate_choropleth(
    subdivisions_gdf,
    firms_df,
    lat_col='lat',
    lon_col='long',
    description_col='description',
    value_col='value',
    efficiency_var='eco_eff',
    income_var='wages_paid',
    region_col='region_id',
    n_quantiles=3,
    policy=None
            ):
   

    # Pivot firm data
    filtered_df = firms_df[firms_df[description_col].isin([efficiency_var, income_var])].copy()
    #Correct the region id 
    filtered_df[region_col] = filtered_df[region_col][:7]
    if policy is not None:
        filtered_df = filtered_df[filtered_df["Policy"] == policy]
    subdivisions_gdf[region_col] = subdivisions_gdf['CD_GEOCMU']
    pivot_df = filtered_df.pivot_table(
        index=['lat', 'long'],
        columns=description_col,
        values=value_col
    ).reset_index()

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        pivot_df,
        geometry=gpd.points_from_xy(pivot_df[lon_col], pivot_df[lat_col]),
        crs='EPSG:4326'
    ).to_crs(subdivisions_gdf.crs)

    # Spatial join
    joined = gpd.sjoin(gdf, subdivisions_gdf[[region_col, 'geometry']], how='inner', predicate='within')

    # Aggregate
    summary = joined.groupby(region_col).agg({
        efficiency_var: 'mean',
        income_var: 'mean'
    }).reset_index()

    # Merge into subdivisions
    merged = subdivisions_gdf.merge(summary, on=region_col, how='inner')

    # Quantile binning
    merged['eff_bin'] = pd.qcut(merged[efficiency_var], n_quantiles, labels=False)
    merged['inc_bin'] = pd.qcut(merged[income_var], n_quantiles, labels=False)

    # Create bivariate class
    merged['bi_class'] = merged['inc_bin'].astype(str) + '-' + merged['eff_bin'].astype(str)

    # Define a simple 3x3 bivariate color palette (from ColorBrewer-like schemes)
    bivariate_colors = {
        '0-0': '#e8e8e8', '0-1': '#ace4e4', '0-2': '#5ac8c8',
        '1-0': '#dfb0d6', '1-1': '#a5add3', '1-2': '#5698b9',
        '2-0': '#be64ac', '2-1': '#8c62aa', '2-2': '#3b4994',
    }
    merged['color'] = merged['bi_class'].map(bivariate_colors)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    merged.plot(color=merged['color'], ax=ax, edgecolor='white', linewidth=0.5)

    ax.set_title('Bivariate Choropleth: Income vs Eco-efficiency')
    ax.axis('off')

    # Optional: draw legend manually
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgba

    legend_elements = [
        mpatches.Patch(color=v, label=k) for k, v in bivariate_colors.items()
    ]
    ax.legend(handles=legend_elements, title='[Income]-[Efficiency]', loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
# Generate the plot.
    project_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    gdf = gpd.read_file(project_folder +'/input/shapes/mun_ACPS_ibge_2014_latlong_wgs1984_fixed.shp')
    gdf.crs 
    base_fp = "C:\\Users\\gusta\\Desktop\\Pulmonar\\Projeto_IPEA\\PS3\\analysis\\validation\\simulated_data"
    
    lista_no_policy = find_firms_csv(base_fp+"\\baseline")[0:3]
    lista_tax = find_firms_csv(base_fp+"\\tax")[0:3]
    lista_subsidies = find_firms_csv(base_fp+"\\subsidies")[0:3]
    lista_both = find_firms_csv(base_fp+"\\both")[0:3]

    no_policy = read_many_firm_files(lista_no_policy, policy="No policy",consolidate=False)
    tax = read_many_firm_files(lista_tax, policy="Carbon Tax",consolidate=False)
    subsidies = read_many_firm_files(lista_subsidies, policy="Subsidies",consolidate=False)
    both = read_many_firm_files(lista_both, policy="Both",consolidate=False)

    data_firm = pd.concat([no_policy, tax, subsidies, both], ignore_index=True)

    #plot_firms_on_income_map(gdf, data_firm, policy="No policy")
    #plot_firms_on_income_map(gdf, data_firm, policy="Subsidies")
    #plot_firms_on_income_map(gdf, data_firm, policy="Carbon Tax")
    #plot_firms_on_income_map(gdf, data_firm, policy="Both")
    #plot_bivariate_choropleth(gdf, data_firm)
    plot_kde_contours(data_firm, gdf, policy="No policy")
    plot_kde_contours(data_firm, gdf, policy="Subsidies")
    plot_kde_contours(data_firm, gdf, policy="Carbon Tax")
    plot_kde_contours(data_firm, gdf, policy="Both")

    pass