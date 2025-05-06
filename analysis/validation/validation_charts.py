import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_boxplot_with_star(data, x_col, y_col, real_data_means, title, xlabel, ylabel, figsize=(12, 6), palette="coolwarm", rotation=45):
    """
    Generates a boxplot with a star marker representing the real data mean inside each box.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the simulated data.
    - x_col (str): Column name for the x-axis (categorical variable, e.g., Urban Agglomeration).
    - y_col (str): Column name for the y-axis (numerical variable, e.g., Emission per GDP).
    - real_data_means (dict): Dictionary mapping each category in x_col to its real data mean.
    - title (str): Plot title.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - figsize (tuple): Size of the plot.
    - palette (str): Color palette.
    - rotation (int): Rotation angle for x-axis labels.
    """
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=data, x=x_col, y=y_col, palette=palette, width=0.7,zorder=1)

    # Add star markers at real data means
    for i, category in enumerate(data[x_col].unique()):
        if category in real_data_means:
            plt.scatter(i, real_data_means[category], color='black', marker='*', s=150,zorder=1, label="Real Data" if i == 0 else "")

    plt.xticks(rotation=rotation, ha='right', fontsize=10)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()

def create_histogram_comparison(data, value_col, group_col, title, xlabel, bins=30, alpha=0.5, figsize=(10, 6), palette=("blue", "red")):
    """
    Generates an overlaid histogram comparing real and simulated income distributions.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing real and simulated income data.
    - value_col (str): Column name representing income values.
    - group_col (str): Column name distinguishing "Real" vs. "Simulated" data.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - bins (int): Number of bins in the histogram.
    - alpha (float): Transparency level for better overlay visibility.
    - figsize (tuple): Size of the figure.
    - palette (tuple): Colors for real and simulated distributions.
    """
    plt.figure(figsize=figsize)

    # Plot histogram for real data
    sns.histplot(data[data[group_col] == "Real"], x=value_col, bins=bins, kde=True, color=palette[0], alpha=alpha, label="Real Data")

    # Plot histogram for simulated data
    sns.histplot(data[data[group_col] == "Simulated"], x=value_col, bins=bins, kde=True, color=palette[1], alpha=alpha, label="Simulated Data")

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()



def create_bubble_chart(data, x_col, y_col, size_col, hue_col, title, xlabel, ylabel, figsize=(10, 6), palette="coolwarm", alpha=0.6):
    """
    Generates a bubble chart comparing real vs. simulated emissions, with bubble size representing sector participation.
    Includes a 1:1 reference line.

    Parameters:
    - data (pd.DataFrame): DataFrame containing emissions and sector participation data.
    - x_col (str): Column name for real emissions (X-axis).
    - y_col (str): Column name for simulated emissions (Y-axis).
    - size_col (str): Column representing sector participation (bubble size).
    - hue_col (str): Column for coloring bubbles (e.g., sector type).
    - title (str): Title of the chart.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - figsize (tuple): Figure size.
    - palette (str): Color palette for different sectors.
    - alpha (float): Transparency of bubbles.
    """
    plt.figure(figsize=figsize)

    # Normalize bubble sizes for better scaling
    max_size = data[size_col].max()
    data["BubbleSize"] = (data[size_col] / max_size) * 1  # Scale to 1000 max

    # Scatter plot with bubble sizes
    sns.scatterplot(
        data=data, x=x_col, y=y_col, size="BubbleSize", hue=hue_col, sizes=(20, 1000),
        alpha=alpha, palette=palette, edgecolor="black", linewidth=0.5
    )

    # 1:1 Reference Line
    min_val = min(data[x_col].min(), data[y_col].min()) * 0.95  # Extend slightly for padding
    max_val = max(data[x_col].max(), data[y_col].max()) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1.5, label="1:1 Line")

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title=hue_col, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()


def create_grouped_boxplot(data, x_col, y_col, hue_col, title, xlabel, ylabel, figsize=(10, 6), palette="Set2", rotation=45):
    """
    Generates grouped boxplots comparing simulated and real economic indicators.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the economic indicators and real/simulated labels.
    - x_col (str): Column name for the x-axis (Indicator type).
    - y_col (str): Column name for the y-axis (values).
    - hue_col (str): Column indicating whether data is "Real" or "Simulated".
    - title (str): Title of the chart.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - figsize (tuple): Figure size.
    - palette (str): Color palette for real vs. simulated data.
    - rotation (int): Rotation angle for x-axis labels.
    """
    plt.figure(figsize=figsize)
    sns.boxplot(data=data, x=x_col, y=y_col, hue=hue_col, palette=palette, width=0.6)

    plt.xticks(rotation=rotation, ha='right', fontsize=10)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title=hue_col, loc='upper right')
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()



def create_grouped_boxplot(data, x_col, y_col, hue_col, title, xlabel, ylabel, figsize=(10, 6), palette="Set2", 
                           rotation=45, y_padding=0.2,normalize=True):
    """
    Generates grouped boxplots comparing simulated and real economic indicators.
    Includes extra padding on the y-axis for better readability.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the economic indicators and real/simulated labels.
    - x_col (str): Column name for the x-axis (Indicator type).
    - y_col (str): Column name for the y-axis (values).
    - hue_col (str): Column indicating whether data is "Real" or "Simulated".
    - title (str): Title of the chart.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - figsize (tuple): Figure size.
    - palette (str): Color palette for real vs. simulated data.
    - rotation (int): Rotation angle for x-axis labels.
    - y_padding (float): Extra percentage of padding on the y-axis.
    """
    plt.figure(figsize=figsize)
    if normalize:
        data[y_col] = data.groupby(x_col)[y_col].transform(lambda x: (x - x.mean()) / x.std())
    ax = sns.boxplot(data=data, x=x_col, y=y_col, hue=hue_col, palette=palette, width=0.6)

    # Get current y-axis limits
    ymin, ymax = ax.get_ylim()
    
    # Add padding to the y-axis
    plt.ylim(ymin - y_padding * (ymax - ymin), ymax + y_padding * (ymax - ymin))

    # Adjust x-axis margins for spacing
    plt.margins(x=0.05)

    plt.xticks(rotation=rotation, ha='right', fontsize=10)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title=hue_col, loc='upper right')

    # Fine-tune spacing to prevent label cutoff
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
    
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()

def create_grouped_boxplot_facet(data, indicator_col, x_col, y_col, hue_col, 
                                 title, xlabel, ylabel, figsize=(14, 8), 
                                 palette="Set2", rotation=45, y_padding=0, col_wrap=3):
    """
    Generates grouped boxplots for each indicator (facet by indicator), comparing simulated 
    and real economic indicators. Each facet gets its own y-axis with extra padding.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the economic indicators, the source (Real/Simulated), 
                           and an indicator label.
    - indicator_col (str): Column name for the indicator (used for faceting).
    - x_col (str): Column name for the x-axis (e.g., typically a grouping variable like "Source" or similar).
    - y_col (str): Column name for the y-axis (the value).
    - hue_col (str): Column indicating the subgroup (e.g., "Real" vs. "Simulated").
    - title (str): Overall title of the chart.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - figsize (tuple): Figure size for the overall plot.
    - palette (str): Color palette.
    - rotation (int): Rotation angle for x-axis tick labels.
    - y_padding (float): Extra fraction of the y-axis range to add as padding on each subplot.
    - col_wrap (int): How many subplots per row.
    """
    # Create the facet grid with a boxplot for each indicator.
    g = sns.catplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        col=indicator_col,
        kind="box",
        palette=palette,
        col_wrap=col_wrap,
        height=4,   # height per facet
        aspect=1.2, # aspect ratio per facet
        sharey=False  # allow each subplot its own y-axis scale
    )
    
    # Set the overall title and x-axis label for the FacetGrid
    g.figure.suptitle(title, fontsize=16)
    g.set_axis_labels(xlabel, ylabel)
    
    # Adjust rotation for each subplot's x-axis labels and add y-axis padding.
    for ax in g.axes.flatten():
        # Rotate x-axis tick labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=10)
        
        # Get current y-axis limits and add padding
        ymin, ymax = ax.get_ylim()
        #pad = y_padding * (ymax - ymin)
        #ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    
    # Adjust subplot spacing so labels and titles are not cut off
    g.figure.subplots_adjust(top=0.88, left=0.07, right=0.95, bottom=0.1)
    plt.show()




if __name__ == "__main__":
    # Example Usage: Simulated Data
   pass