import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
from matplotlib.ticker import MaxNLocator

# --- Configuration ---

DATA_ROOT_DIRECTORIES = [
    'analysis/validation/simulated_data/baseline',
    'analysis/validation/simulated_data/ECO_INVESTMENT_SUBSIDIES__2025-08-31T12_30_31.123224', # Example path 2
    'analysis/validation/simulated_data/PRODUCTIVITY_MAGNITUDE_DIVISOR__2025-08-30T16_07_18.167714'    # Example path 3
]

OUTCOME_VARIABLES_TO_PLOT = [
    'emissions', 
    'firms_median_innovation_investment',
    'unemployment', 
    'gdp_index',
    'price_index'
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
    Example: 'ECO_INVESTMENT_SUBSIDIES=0.5' -> ('ECO_INVESTMENT_SUBSIDIES', 0.5)
    """
    match = re.match(r'([^=]+)=([0-9\.\-]+)', folder_name)
    if not match:
        raise ValueError(f"Folder name '{folder_name}' does not match the expected 'PARAM=VALUE' format.")
    
    param_name, param_value_str = match.groups()
    return param_name, float(param_value_str)

def process_simulation_data(root_directory: str) -> pd.DataFrame:
    """
    Scans a directory, parses folder names, and aggregates all stats.csv files.
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
                # Find all 'stats.csv' files recursively
                csv_files = list(param_path.rglob('stats.csv'))
                if not csv_files:
                    continue
                
                df_list = [pd.read_csv(f, sep=';', decimal=',', header=None, names=OUTPUT_DATA_SPEC['stats']['columns']) for f in csv_files]
                combined_df = pd.concat(df_list, ignore_index=True)
                combined_df['parameter_name'] = param_name
                combined_df['parameter_value'] = param_value
                all_results.append(combined_df)
            except ValueError:
                continue # Skip folders that don't match the format

    if not all_results:
        return pd.DataFrame()

    master_df = pd.concat(all_results, ignore_index=True)
    return master_df[master_df['month'] >= '2016-01-01']

def plot_single_sensitivity_on_ax(ax: plt.Axes, df: pd.DataFrame, outcome_var: str):
    """
    Draws a single sensitivity plot onto a provided Matplotlib Axes object.
    """
    if outcome_var not in df.columns:
        ax.text(0.5, 0.5, f"'{outcome_var}'\nnot found", ha='center', va='center')
        return

    df[outcome_var] = pd.to_numeric(df[outcome_var], errors='coerce')
    if df[outcome_var].isnull().all():
        ax.text(0.5, 0.5, f"'{outcome_var}'\nhas no data", ha='center', va='center')
        return

    sns.lineplot(
        data=df,
        x='parameter_value',
        y=outcome_var,
        marker='o',
        err_style='band',
        errorbar='sd',
        ax=ax
    )
    
    # Set a concise title for each subplot (the variable name)
    ax.set_title(outcome_var.replace('_', ' ').title(), fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.tick_params(axis='x', labelrotation=30)


# --- Main Execution ---
if __name__ == '__main__':
    num_cols = len(OUTCOME_VARIABLES_TO_PLOT)
    save_fig = False
    for data_dir in DATA_ROOT_DIRECTORIES:
        master_dataframe = process_simulation_data(data_dir)
        
        if master_dataframe.empty:
            print(f"Warning: No data found in '{data_dir}'. Skipping dashboard creation for this directory.")
            continue

        param_name = master_dataframe['parameter_name'].iloc[0]
        
        fig, axes = plt.subplots(
            nrows=1, 
            ncols=num_cols, 
            figsize=(11.69, 4), # Landscape A4 width
            constrained_layout=True
        )
        
        fig.suptitle(f'Sensitivity Analysis for: {param_name}', fontsize=16, fontweight='bold')

        for col_idx, var in enumerate(OUTCOME_VARIABLES_TO_PLOT):
            ax = axes[col_idx]
            plot_single_sensitivity_on_ax(ax, master_dataframe, var)
            
            # --- Clean up axis labels for a tidier look ---
            ax.set_xlabel('Parameter Value', fontsize=8)
            if col_idx > 0:
                ax.set_ylabel('')
            else:
                ax.set_ylabel('Outcome', fontsize=8)

        # --- Save the complete dashboard row as a PDF ---
        if save_fig:
            pdf_filename = f"sensitivity_dashboard_{param_name}.pdf"
            print(f"\n--- Saving Dashboard to {pdf_filename} ---")
            plt.savefig(pdf_filename, dpi=300)
        plt.close(fig)



