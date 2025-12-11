import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional


OUTPUT_DATA_SPEC = {
    'stats': {
        'avg': {
            'groupings': ['month'],
            'columns': 'ALL'
        },
        'columns': ['month',
                    'pop',
                    'price_index',
                    'gdp_index',
                    'gdp_growth',
                    'unemployment',
                    'median_workers',
                    'families_median_wealth',
                    'families_wages_received',
                    'families_commuting',
                    'families_savings',
                    'families_helped',
                    'amount_subsidised',
                    'firms_profit',
                    'firms_median_stock',
                    'firms_avg_eco_eff',
                    'firms_median_wage_paid',
                    'firms_median_innovation_investment',
                    'total_subsidies',
                    'total_emission_tax',
                    'emissions',
                    'gini_index',
                    'average_utility',
                    'pct_zero_consumption',
                    'rent_default',
                    'inflation',
                    'average_qli',
                    'house_vacancy',
                    'house_price',
                    'house_rent',
                    'affordable',
                    'p_delinquent',
                    'equally',
                    'locally',
                    'fpm',
                    'bank',
                    'emissions_fund',
                    'ext_amount_sold',
                    'affordability_median'
                    ]
    },
    'families': {
        'avg': {
            'groupings': ['month', 'mun_id'],
            'columns': ['house_price', 'house_rent', 'total_wage', 'savings', 'num_members']
        },
        'columns': ['month', 'id', 'mun_id', 'house_price', 'house_rent',
                    'total_wage', 'savings', 'num_members']
    },
    'banks': {
        'avg': {
            'groupings': ['month'],
            'columns': 'ALL'
        },
        'columns': ['month', 'balance', 'active_loans', 'mortgage_rate', 'p_delinquent_loans',
                    'mean_loan_age', 'min_loan', 'max_loan', 'mean_loan']
    },
    'houses': {
        'avg': {
            'groupings': ['month', 'mun_id'],
            'columns': ['price', 'on_market']
        },
        'columns': ['month', 'id', 'x', 'y', 'size', 'price', 'rent', 'quality', 'qli',
                    'on_market', 'family_id', 'region_id', 'mun_id']
    },
    'firms': {
        'avg': {
            'groupings': ['month', 'firm_id'],
            'columns': ['total_balance$', 'number_employees',
                        'stocks', 'amount_produced', 'price', 'amount_sold',
                        'revenue', 'profit', 'wages_paid']
        },
        'columns': ['month', 'firm_id', 'region_id', 'mun_id',
                    'long', 'lat', 'total_balance$', 'number_employees',
                    'stocks', 'amount_produced', 'price', 'amount_sold',
                    'revenue', 'profit', 'wages_paid', 'input_cost', 
                    'emissions', 'eco_eff', 'innov_investment', 'sector']
    },
    'construction': {
        'avg': {
            'groupings': ['month', 'firm_id'],
            'columns': ['total_balance$', 'number_employees',
                        'stocks', 'amount_produced', 'price', 'amount_sold',
                        'revenue', 'profit', 'wages_paid']
        },
        'columns': ['month', 'firm_id', 'region_id', 'mun_id',
                    'long', 'lat', 'total_balance$', 'number_employees',
                    'stocks', 'amount_produced', 'price', 'amount_sold',
                    'revenue', 'profit', 'wages_paid']
    },
    'regional': {
        'avg': {
            'groupings': ['month', 'mun_id'],
            'columns': 'ALL'
        },
        'columns': ['month', 'mun_id', 'commuting', 'pop', 'gdp_region',
                    'regional_gini', 'regional_house_values', 'regional_unemployment',
                    'qli_index', 'gdp_percapita', 'treasure', 'equally', 'locally', 'fpm',
                    'licenses']
    }
}

def label_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    df['TAX_EMISSION'] = pd.to_numeric(df['TAX_EMISSION'])
    df['ECO_INVESTMENT_SUBSIDIES'] = pd.to_numeric(df['ECO_INVESTMENT_SUBSIDIES'])
    
    for col in ['CARBON_TAX_RECYCLING', 'TARGETED_TAX_SUBSIDIES']:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip() == 'True'

    conditions = [
        (df['TAX_EMISSION'] == 0) & (df['ECO_INVESTMENT_SUBSIDIES'] == 0) & (~df['CARBON_TAX_RECYCLING']) & (~df['TARGETED_TAX_SUBSIDIES']),
        (df['TAX_EMISSION'] > 0) & (df['ECO_INVESTMENT_SUBSIDIES'] == 0) & (~df['CARBON_TAX_RECYCLING'])& (~df['TARGETED_TAX_SUBSIDIES']),
        (df['TAX_EMISSION'] == 0) & (df['ECO_INVESTMENT_SUBSIDIES'] > 0) & (~df['TARGETED_TAX_SUBSIDIES'])& (~df['CARBON_TAX_RECYCLING']),
        (df['TAX_EMISSION'] > 0) & (df['ECO_INVESTMENT_SUBSIDIES'] > 0) & (~df['CARBON_TAX_RECYCLING']) & (~df['TARGETED_TAX_SUBSIDIES']),
        (df['TAX_EMISSION'] > 0) & (df['ECO_INVESTMENT_SUBSIDIES'] > 0) & (df['CARBON_TAX_RECYCLING'])& (~df['TARGETED_TAX_SUBSIDIES']),
        (df['TAX_EMISSION'] > 0) & (df['ECO_INVESTMENT_SUBSIDIES'] > 0) & (~df['CARBON_TAX_RECYCLING'])& (df['TARGETED_TAX_SUBSIDIES'])
    ]

    choices = [
        "Baseline", 
        "Carbon Tax", 
        "Subsidies", 
        "Combined", 
        "Tax Recycling", 
        "Directed Subsidies"
    ]

    df['scenario'] = np.select(conditions, choices, default="Unclassified")
    df['scenario'] = pd.Categorical(df['scenario'], categories=choices, ordered=True)

    return df

def agg_simulation_data(root_folder: str, param_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Recursively reads 'firms.csv' files, extracts parameters from folder names,
    and aggregates them into a single DataFrame.

    Args:
        root_folder (str): The path to the main directory.
        param_names (list, optional): A list of parameter names if the folder strings 
                                      only contain values (e.g. "0.1;100"). 
                                      If None, the function assumes folders are named 
                                      "key=value;key=value".

    Returns:
        pd.DataFrame: Concatenated data with 'sim_id' and parameter columns.
    """
    root_path = Path(root_folder)
    
    # 1. Find all firms.csv files recursively
    # rglob('*') searches subdirectories. You can specify pattern.
    target_files = list(root_path.rglob('firms.csv'))
    
    if not target_files:
        print("No firms.csv files found.")
        return pd.DataFrame()

    df_list = []

    print(f"Found {len(target_files)} files. Processing...")

    for file_path in target_files:
        try:
            # 2. Extract Metadata from Path
            # Structure: root / params_folder / sim_number_folder / firms.csv
            
            sim_num = file_path.parent.name # The immediate parent (e.g., "0", "1")
            param_str = file_path.parent.parent.name # The grandparent (e.g., "alpha=0.1;beta=0.2")

            # 3. Read the CSV
            # Use appropriate separators (default is comma)
            current_df = pd.read_csv(file_path,sep=';',header=None)
            current_df.columns = OUTPUT_DATA_SPEC['firms']['columns']
            
            # 4. Add Simulation Number
            current_df['sim_id'] = sim_num

            # 5. Parse Parameters
            params = {}
            raw_parts = param_str.split(';')
            
            if param_names:
                # Scenario A: Folder is "0.1;100" and user provided ["alpha", "beta"]
                if len(raw_parts) != len(param_names):
                    print(f"Warning: Param count mismatch in {param_str}")
                else:
                    for name, val in zip(param_names, raw_parts):
                        params[name] = val
            else:
                # Scenario B: Folder is "alpha=0.1;beta=100" (Auto-detect)
                for part in raw_parts:
                    if '=' in part:
                        key, val = part.split('=')
                        params[key] = val
                    else:
                        # Fallback if no '=' found and no names provided
                        params[f"param_{raw_parts.index(part)}"] = part

            # Assign parameter values to new columns in the DataFrame
            for key, value in params.items():
                current_df[key] = value
            current_df = label_scenarios(current_df)

            df_list.append(current_df)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # 6. Concatenate all dataframes
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)    
        return final_df
    else:
        return pd.DataFrame()
    
def agg_stats_data(root_folder: str, param_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Recursively reads 'stats.csv' files, extracts parameters from folder names,
    and aggregates them into a single DataFrame.
    """
    root_path = Path(root_folder)
    target_files = list(root_path.rglob('stats.csv'))
    
    if not target_files:
        print("No stats.csv files found.")
        return pd.DataFrame()

    df_list = []
    print(f"Found {len(target_files)} stats files. Processing...")

    for file_path in target_files:
        try:
            sim_num = file_path.parent.name
            param_str = file_path.parent.parent.name

            # Read CSV
            current_df = pd.read_csv(file_path, sep=';', header=None)
            
            # Map columns if spec exists
            if 'stats' in OUTPUT_DATA_SPEC:
                current_df.columns = OUTPUT_DATA_SPEC['stats']['columns']
            
            current_df['sim_id'] = sim_num

            # Parse Parameters
            params = {}
            raw_parts = param_str.split(';')
            
            if param_names:
                if len(raw_parts) != len(param_names):
                    print(f"Warning: Param count mismatch in {param_str}")
                else:
                    params = dict(zip(param_names, raw_parts))
            else:
                for part in raw_parts:
                    if '=' in part:
                        key, val = part.split('=')
                        params[key] = val
                    else:
                        params[f"param_{raw_parts.index(part)}"] = part

            # Assign parameters to columns
            for key, value in params.items():
                current_df[key] = value
            
            # Label scenarios (Relies on params being present)
            current_df = label_scenarios(current_df)

            df_list.append(current_df)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

import matplotlib.pyplot as plt
import pandas as pd

def plot_spaghetti(df: pd.DataFrame, variable: str, scenario: str, n_lines: int = 50, sigma: float = 1.0):
    subset = df[df['scenario'] == scenario].copy()
    if subset.empty:
        print(f"No data for: {scenario}"); return

    subset['month'] = pd.to_datetime(subset['month'])
    pivoted = subset.pivot(index='month', columns='sim_id', values=variable)

    mean_series = pivoted.mean(axis=1)
    std_series = pivoted.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    if n_lines and n_lines < pivoted.shape[1]:
        plot_data = pivoted.sample(n=n_lines, axis=1, random_state=42)
    else:
        plot_data = pivoted

    ax.plot(plot_data.index, plot_data.values, color='lightgray', alpha=0.3, linewidth=1, zorder=1)

    ax.fill_between(
        pivoted.index, 
        mean_series - (sigma * std_series), 
        mean_series + (sigma * std_series), 
        color='tab:blue', alpha=0.2, label=f'±{sigma} SD', zorder=2
    )

    ax.plot(pivoted.index, mean_series, color='tab:blue', linewidth=2, label='Mean', zorder=3)

    ax.set_title(f"{variable}: {scenario}")
    ax.set_ylabel(variable)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    path = '.\output\TAX_EMISSION_TARGETED_TAX_SUBSIDIES_CARBON_TAX_RECYCLING_ECO_INVESTMENT_SUBSIDIES__2025-12-10T22_00_18.323851'
    #df = agg_simulation_data(path)
    stats_df = agg_stats_data(path)
    plot_spaghetti(stats_df, variable='total_subsidies', scenario='Subsidies', n_lines=10, sigma=1.0)
    pass
