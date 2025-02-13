import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# Example Usage: Simulated Data
np.random.seed(42)
urban_areas = [f"Urban Area {i+1}" for i in range(27)]
policies = ["Taxation", "Subsidy", "Cap & Trade"]

# Generate synthetic emissions per GDP for each region-policy combination
data = {
    "Urban Agglomeration": np.tile(np.repeat(urban_areas, 50), len(policies)),
    "Policy": np.repeat(policies, 50 * len(urban_areas)),
    "Emission per GDP": np.concatenate([
        20-np.random.normal(loc, 2, 50) for loc in np.linspace(1.2, 2.0, len(urban_areas) * len(policies))
    ])
}

df_policies = pd.DataFrame(data)

# Generate the grouped boxplot
create_grouped_boxplot(
    data=df_policies,
    x_col="Urban Agglomeration",
    y_col="Emission per GDP",
    hue_col="Policy",
    title="Grouped Boxplot: Policy Effects on Emissions per GDP",
    xlabel="Urban Agglomeration",
    ylabel="Emission per GDP"
)


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


def create_heatmap(data, index_col, columns_col, values_col, title, figsize=(12, 8), cmap="coolwarm"):
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
    aggregated_data = data.groupby([index_col, columns_col])[values_col].mean().reset_index()

    # Pivot the data
    pivot_table = aggregated_data.pivot(index=index_col, columns=columns_col, values=values_col)

    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5, cbar_kws={'label': values_col})
    
    plt.title(title, fontsize=14)
    plt.xlabel(columns_col, fontsize=12)
    plt.ylabel(index_col, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

# Call the function with df_policies
create_heatmap(
    data=df_policies,
    index_col="Urban Agglomeration",
    columns_col="Policy",
    values_col="Emission per GDP",
    title="Heatmap: Emission per GDP by Policy and Urban Agglomeration"
)

def create_emissions_violinplot_by_sector(data, sector_col, policy_col, value_col, 
                                          title, xlabel, ylabel, figsize=(16, 12), 
                                          palette="Set2", rotation=45, col_wrap=4):
    """
    Generates faceted violin plots (one panel per sector) to visualize the distribution 
    of firms' emissions reduction values for each policy, with a shared y-axis.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the emissions reduction data.
    - sector_col (str): Column name for the sector (used for faceting).
    - policy_col (str): Column name for the policy.
    - value_col (str): Column name for the emissions reduction values.
    - title (str): Overall title of the chart.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - figsize (tuple): Overall figure size.
    - palette (str): Color palette for the policies.
    - rotation (int): Rotation angle for x-axis tick labels.
    - col_wrap (int): Number of facets per row.
    """
    # Create the facet grid with a violin plot for each sector, sharing the y-axis scale.
    g = sns.catplot(
        data=data,
        kind="violin",
        x=policy_col,
        y=value_col,
        col=sector_col,
        col_wrap=col_wrap,
        palette=palette,
        inner="quartile",
        height=4,
        aspect=1.2,
        sharey=True  # All subplots share the same y-axis scale.
    )
    
    # Remove the "Sector=" prefix from facet titles by customizing the title format.
    g.set_titles("{col_name}")
    
    # Adjust x-axis tick labels and add gridlines in each subplot.
    for ax in g.axes.flatten():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set overall title and axis labels.
    g.fig.suptitle(title, fontsize=18)
    g.set_axis_labels(xlabel, ylabel)
    
    # Adjust layout to prevent the title and labels from overlapping the subplots.
    g.fig.subplots_adjust(top=0.9, left=0.05, right=0.95, bottom=0.08)
    
    plt.tight_layout()
    plt.show()

# --- Example Usage ---

# Create example synthetic data for 12 sectors.
np.random.seed(42)
sectors = [f"Sector {i}" for i in range(1, 13)]
policies = ["Policy A", "Policy B", "Policy C"]
data_list = []

for sector in sectors:
    for policy in policies:
        # Generate 100 sample values for each (sector, policy) combination.
        if policy == "Policy A":
            values = np.random.normal(loc=0.3, scale=0.1, size=100)
        elif policy == "Policy B":
            values = np.random.normal(loc=0.5, scale=0.15, size=100)
        else:  # Policy C
            values = np.random.normal(loc=0.4, scale=0.2, size=100)
        for val in values:
            data_list.append({
                "Sector": sector,
                "Policy": policy,
                "EmissionsReduction": val
            })

df_emissions = pd.DataFrame(data_list)

# Generate the faceted violin plot with shared y-axis and clean facet titles.
create_emissions_violinplot_by_sector(
    data=df_emissions,
    sector_col="Sector",
    policy_col="Policy",
    value_col="EmissionsReduction",
    title="Firms' Emissions Reduction Distributions by Policy and Sector (12 Sectors)",
    xlabel="Policy",
    ylabel="Emissions Reduction (%)",
    figsize=(16, 12),
    palette="Set2",
    rotation=45,
    col_wrap=4
)

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

def make_plot_sectors_emission(data, name='figures/emission_policy_effects',
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
    policy_list = ["Taxation", "Subsidies", "Combined Policy"]

    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(8, 6))

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

# --- Example Usage ---

# Suppose you have a DataFrame 'df_emission' with the following columns:
# 'sectors', 'parameter', 'policy', 'pvalue', 'std_error'.
# For demonstration, here is a synthetic example:

import numpy as np
np.random.seed(42)
sectors = [f"Sector {i}" for i in range(1, 13)]
policies = ["Taxation", "Subsidies", "Combined Policy"]
data_list = []

for sector in sectors:
    for policy in policies:
        # Generate one sample value per (sector, policy) combination.
        # (In practice, you might have averages from multiple firms.)
        parameter = np.random.normal(loc=0.5, scale=0.1)
        pvalue = np.random.uniform(0, 0.2)
        std_error = np.random.uniform(0.05, 0.15)
        data_list.append({
            "sectors": sector,
            "parameter": parameter,
            "policy": policy,
            "pvalue": pvalue,
            "std_error": std_error
        })

df_emission = pd.DataFrame(data_list)

# Generate the plot.
make_plot_sectors_emission(df_emission,
                           name='figures/emission_policy_effects',
                           exclude_sectors=[])  # You can list sectors to exclude if needed.
