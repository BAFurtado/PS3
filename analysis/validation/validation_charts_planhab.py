import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
from matplotlib.ticker import MaxNLocator
import numpy as np

# --- Configuration ---

DATA_ROOT_DIRECTORIES = [
    'analysis/validation/simulated_data/baseline',
    'analysis/validation/simulated_data/ECO_INVESTMENT_SUBSIDIES__2025-08-31T12_30_31.123224', # Example path 2
    'analysis/validation/simulated_data/PRODUCTIVITY_MAGNITUDE_DIVISOR__2025-08-30T16_07_18.167714'     # Example path 3
]

# --- NEW: Configuration for Annualization ---
# Add the names of any columns you want to convert to a 12-month moving average
VARIABLES_TO_ANNUALIZE = [
    'gdp_growth',
    'inflation',
    'unemployment',
    'emissions',
    'firms_median_innovation_investment'
]

# --- Updated Plotting Configuration ---
# We now use the '_annual' suffix to plot the smoothed data.
# To plot raw monthly data again, just remove the '_annual' suffix from the names.

OUTCOME_VARIABLES_TO_PLOT = [
    'emissions_annual', 
    'firms_median_innovation_investment_annual',
    'unemployment_annual', 
    'gdp_index', # gdp_index is cumulative, so we don't annualize it
    'price_index' # price_index is also cumulative
]

# The keys in PROJECTED_DATA and the names in PROJECTION_VARIABLES_TO_PLOT must match.
PROJECTED_DATA = {
    'gdp_growth_annual': 0.25,
    'inflation_annual': 0.035/12,
    'unemployment_annual': 0.065
}

PROJECTION_VARIABLES_TO_PLOT = [
    'gdp_growth_annual', 
    'inflation_annual', 
    'unemployment_annual'
]


# --- Data Specification ---

OUTPUT_DATA_SPEC = {
    'stats': {
        'columns': ['month', 'pop', 'price_index', 'gdp_index', 'gdp_growth',
                    'unemployment', 'median_workers', 'families_median_wealth',
                    'families_wages_received', 'families_commuting', 'families_savings',
                    'families_helped', 'amount_subsidised', 'firms_profit',
                    'firms_median_stock', 'firms_avg_eco_eff', 'firms_median_wage_paid',
                    'firms_median_innovation_investment', 'emissions', 'gini_index',
                    'average_utility', 'pct_zero_consumption', 'rent_default',
                    'inflation', 'average_qli', 'house_vacancy', 'house_price',
                    'house_rent', 'affordable', 'p_delinquent', 'equally',
                    'locally', 'fpm', 'bank', 'emissions_fund',
                    'ext_amount_sold', 'affordability_median']
    }
}


# --- Helper Functions ---

def parse_folder_name(folder_name: str) -> Tuple[str, float]:
    """
    Parses a folder name to extract the parameter name and its value.
    """
    match = re.match(r'([^=]+)=([0-9\.\-]+)', folder_name)
    if not match:
        raise ValueError(f"Folder name '{folder_name}' does not match the expected 'PARAM=VALUE' format.")
    param_name, param_value_str = match.groups()
    return param_name, float(param_value_str)

def process_simulation_data(root_directory: str) -> pd.DataFrame:
    """
    Scans a directory, aggregates stats.csv files, and computes annual moving averages.
    """
    root_path = Path(root_directory)
    if not root_path.is_dir():
        print(f"Error: Directory '{root_directory}' not found.")
        return pd.DataFrame()

    all_results = []
    print(f"\n--- Processing Directory: {root_directory} ---")
    for param_path in root_path.iterdir():
        if param_path.is_dir():
            try:
                param_name, param_value = parse_folder_name(param_path.name)
                csv_files = list(param_path.rglob('stats.csv'))
                if not csv_files:
                    continue
                
                # MODIFIED: Process each CSV file and assign a unique run_id
                for run_idx, f in enumerate(csv_files):
                    df = pd.read_csv(f, sep=';', decimal=',', header=None, names=OUTPUT_DATA_SPEC['stats']['columns'])
                    df['parameter_name'] = param_name
                    df['parameter_value'] = param_value
                    # Unique identifier for each simulation run
                    df['run_id'] = f"{param_name}_{param_value}_{run_idx}"
                    all_results.append(df)

            except ValueError:
                continue 

    if not all_results:
        return pd.DataFrame()

    master_df = pd.concat(all_results, ignore_index=True)
    master_df['month'] = pd.to_datetime(master_df['month'])
    
    # Filter for the desired date range before calculations
    master_df = master_df[master_df['month'] >= '2016-01-01'].copy()
    
    # --- NEW: Annualize Data using 12-month moving average ---
    if VARIABLES_TO_ANNUALIZE:
        print("Calculating annual moving averages...")
        # Ensure data is sorted correctly for rolling calculations
        master_df.sort_values(by=['run_id', 'month'], inplace=True)

        for col in VARIABLES_TO_ANNUALIZE:
            if col in master_df.columns:
                annual_col_name = f"{col}_annual"
                # The groupby ensures the rolling average restarts for each simulation run
                # transform applies the calculation and returns a DataFrame of the same shape
                master_df[annual_col_name] = master_df.groupby('run_id')[col].transform(
                    lambda x: x.rolling(window=12, min_periods=12).mean()
                )
            else:
                print(f"Warning: Column '{col}' not found for annualization.")
        
        # Drop rows where the rolling average is NaN (i.e., the first 11 months of each run)
        original_rows = len(master_df)
        master_df.dropna(subset=[f"{col}_annual" for col in VARIABLES_TO_ANNUALIZE if f"{col}_annual" in master_df.columns], inplace=True)
        print(f"Dropped {original_rows - len(master_df)} rows with incomplete annual data.")

    return master_df[master_df['month'] >= '2018-01-01']

def plot_single_sensitivity_on_ax(ax: plt.Axes, df: pd.DataFrame, outcome_var: str):
    # This function remains unchanged and works with both raw and annual data
    # (as long as the column name exists in the dataframe)
    if outcome_var not in df.columns:
        ax.text(0.5, 0.5, f"'{outcome_var}'\nnot found", ha='center', va='center')
        return
    # ... (rest of the function is the same as before)
    df[outcome_var] = pd.to_numeric(df[outcome_var], errors='coerce')
    if df[outcome_var].isnull().all():
        ax.text(0.5, 0.5, f"'{outcome_var}'\nhas no data", ha='center', va='center')
        return
    sns.lineplot(data=df, x='parameter_value', y=outcome_var, marker='o', err_style='band', errorbar='sd', ax=ax)
    ax.set_title(outcome_var.replace('_', ' ').title(), fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.tick_params(axis='x', labelrotation=30)


def plot_boxplot_with_projection(ax: plt.Axes, df: pd.DataFrame, outcome_var: str, projected_value: float):
    # This function also remains unchanged
    if outcome_var not in df.columns:
        ax.text(0.5, 0.5, f"'{outcome_var}'\nnot found in data", ha='center', va='center', fontsize=9)
        return
    # ... (rest of the function is the same as before)
    df[outcome_var] = pd.to_numeric(df[outcome_var], errors='coerce')
    df.dropna(subset=[outcome_var], inplace=True)
    if df.empty:
        ax.text(0.5, 0.5, f"No valid data for\n'{outcome_var}'", ha='center', va='center', fontsize=9)
        return
    df_sorted = df.sort_values('parameter_value')
    sns.boxplot(data=df_sorted, x='parameter_value', y=outcome_var, ax=ax, showfliers=False)
    ax.axhline(y=projected_value, color='red', linestyle='--', linewidth=2, label='Projected Value')
    ax.set_title(outcome_var.replace('_', ' ').title(), fontsize=12)
    ax.tick_params(axis='x', labelrotation=30, labelsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.set_xlabel('') 
    ax.set_ylabel('Value', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)


# --- Main Execution ---
if __name__ == '__main__':
    save_fig = False 

    for data_dir in DATA_ROOT_DIRECTORIES:
        master_dataframe = process_simulation_data(data_dir)
        
        if master_dataframe.empty:
            print(f"Warning: No data found in '{data_dir}'. Skipping plots for this directory.")
            continue

        param_name = master_dataframe['parameter_name'].iloc[0]
        
        # --- 1. Sensitivity Analysis Plots (now using annual data where specified) ---
        print(f"\n--- Generating Sensitivity Line Plots for: {param_name} ---")
        num_cols_sens = len(OUTCOME_VARIABLES_TO_PLOT)
        fig_sens, axes_sens = plt.subplots(
            nrows=1, 
            ncols=num_cols_sens, 
            figsize=(11.69, 4),
            constrained_layout=True
        )
        fig_sens.suptitle(f'Sensitivity Analysis for: {param_name}', fontsize=16, fontweight='bold')

        for col_idx, var in enumerate(OUTCOME_VARIABLES_TO_PLOT):
            ax = axes_sens[col_idx] if num_cols_sens > 1 else axes_sens
            plot_single_sensitivity_on_ax(ax, master_dataframe, var)
            ax.set_xlabel('Parameter Value', fontsize=8)
            if col_idx > 0: ax.set_ylabel('')
            else: ax.set_ylabel('Outcome', fontsize=8)
        
        if save_fig:
            pdf_filename = f"sensitivity_dashboard_{param_name}.pdf"
            fig_sens.savefig(pdf_filename, dpi=300)
        plt.show()
        plt.close(fig_sens)

        # --- 2. Boxplots with Projections (now using annual data where specified) ---
        print(f"\n--- Generating Boxplots with Projections for: {param_name} ---")
        num_cols_proj = len(PROJECTION_VARIABLES_TO_PLOT)
        fig_proj, axes_proj = plt.subplots(
            nrows=1, 
            ncols=num_cols_proj, 
            figsize=(12, 5), 
            constrained_layout=True,
            sharey=False
        )
        
        fig_proj.suptitle(f'Simulated Distribution vs. Projection\nParameter: {param_name}', fontsize=16, fontweight='bold')

        # Handle case of a single plot
        if num_cols_proj == 1:
            axes_proj = [axes_proj]

        for i, var in enumerate(PROJECTION_VARIABLES_TO_PLOT):
            projected_val = PROJECTED_DATA.get(var)
            
            if projected_val is None:
                print(f"Warning: No projected value found for '{var}'. Skipping plot.")
                axes_proj[i].text(0.5, 0.5, f"No projection data\nfor '{var}'", ha='center', va='center')
                continue
                
            plot_boxplot_with_projection(axes_proj[i], master_dataframe.copy(), var, projected_val)

        handles, labels = axes_proj[0].get_legend_handles_labels()
        if handles:
             fig_proj.legend(handles, labels, loc='outside upper right', bbox_to_anchor=(0.99, 0.98))
        
        fig_proj.supxlabel('Parameter Value', fontsize=12)

        if save_fig:
            pdf_filename = f"projection_boxplot_{param_name}.pdf"
            fig_proj.savefig(pdf_filename, dpi=300)
        plt.show()
        plt.close(fig_proj)