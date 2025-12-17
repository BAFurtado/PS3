import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import math, os
import matplotlib.pyplot as plt
import seaborn as sns


# Apply the style from your snippet
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

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
                    'total_subsidies',#
                    'total_emission_tax',#
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

def load_and_process_real_data(filepath: str='analysis\\validation\\emission_plots\\ipeadata[16-12-2025-05-25].csv', start_year: int = 2010) -> pd.DataFrame:
    """
    Parses the Ipeadata CSV file, fixes dates, and prepares it for plotting.
    
    Args:
        filepath: Path to the CSV file.
        start_year: Filter data to keep only years >= start_year.
        
    Returns:
        pd.DataFrame: Cleaned dataframe indexed by 'month'.
    """
    # 1. Load Data
    # 'usecols=[0,1,2,3]' drops the empty "Unnamed" column from the trailing ';'
    df = pd.read_csv(
        filepath, 
        sep=';', 
        decimal=',', 
        usecols=[0, 1, 2, 3]
    )

    # 2. Rename Columns
    df.columns = ['month', 'inflation', 'gdp_index', 'unemployment']

    # 3. Parse Dates (YYYY.MM -> Datetime)
    def parse_ipea_date(val):
        try:
            s = str(val)
            if '.' in s:
                year, month = s.split('.')
                return pd.Timestamp(f"{year}-{month}-01")
            return pd.NaT
        except:
            return pd.NaT

    df['month'] = df['month'].apply(parse_ipea_date)

    # 4. Filter and Sort
    df = df.dropna(subset=['month'])
    df = df[df['month'].dt.year >= start_year]
    df = df.sort_values('month').set_index('month')
    
    # Ensure numeric columns are floats
    cols = ['inflation', 'gdp_index', 'unemployment']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['gdp_growth'] = df['gdp_index'].pct_change()

    print(f"Data loaded successfully: {len(df)} rows from {start_year} onwards.")
    return df

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
    #conditions = [
    #    (df['TAX_EMISSION'] == 0) & (df['ECO_INVESTMENT_SUBSIDIES'] == 0), #& (~df['CARBON_TAX_RECYCLING']) & (~df['TARGETED_TAX_SUBSIDIES']),
    #    (df['TAX_EMISSION'] > 0) & (df['ECO_INVESTMENT_SUBSIDIES'] == 0),# & (~df['CARBON_TAX_RECYCLING'])& (~df['TARGETED_TAX_SUBSIDIES']),
    #    (df['TAX_EMISSION'] == 0) & (df['ECO_INVESTMENT_SUBSIDIES'] > 0),# & (~df['TARGETED_TAX_SUBSIDIES'])& (~df['CARBON_TAX_RECYCLING']),
    #    (df['TAX_EMISSION'] > 0) & (df['ECO_INVESTMENT_SUBSIDIES'] > 0),# & (~df['CARBON_TAX_RECYCLING']) & (~df['TARGETED_TAX_SUBSIDIES']),
        #(df['TAX_EMISSION'] > 0) & (df['ECO_INVESTMENT_SUBSIDIES'] > 0),# & (df['CARBON_TAX_RECYCLING'])& (~df['TARGETED_TAX_SUBSIDIES']),
        #(df['TAX_EMISSION'] > 0) & (df['ECO_INVESTMENT_SUBSIDIES'] > 0),# & (~df['CARBON_TAX_RECYCLING'])& (df['TARGETED_TAX_SUBSIDIES'])
    #]

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

def plot_baseline_spaghetti(df: pd.DataFrame, 
                            variable: str, 
                            baseline_name: str = 'Baseline',
                            actual_data: pd.Series = None,
                            n_lines: int = 50, 
                            sigma: float = 1.0,
                            save_path: str = 'analysis\\validation\\emission_plots\\figures\\',
                            normalize: bool = True):
    """
    Plots a single spaghetti chart for the Baseline scenario.
    """
    # 1. Setup
    subset = df[df['scenario'] == baseline_name].copy()
    max_date = subset['month'].max()
    min_date = subset['month'].min()
    
    if subset.empty:
        print(f"No data found for scenario: {baseline_name}")
        return

    if not pd.api.types.is_datetime64_any_dtype(subset['month']):
        subset['month'] = pd.to_datetime(subset['month'])

    # Pivot: Index=Time, Columns=Sims
    pivoted = subset.pivot(index='month', columns='sim_id', values=variable)
    if normalize:
        # Scale each simulation run independently to [0, 1]
        # Formula: (x - min) / (max - min)
        pivoted = (pivoted - pivoted.min()) / (pivoted.max() - pivoted.min())
        
        if actual_data is not None:
            # Scale actual data to [0, 1] based on its own range
            actual_data = (actual_data - actual_data.nsmallest(3).iloc[-1]) / (actual_data.max() - actual_data.nsmallest(3).iloc[-1])
    mean_series = pivoted.mean(axis=1)
    std_series = pivoted.std(axis=1)

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors
    c_line = '#b0b0b0'    # Light Gray
    c_mean = '#4c72b0'    # Muted Blue
    c_actual = '#c44e52'  # Muted Red

    # A. Individual Runs (Sampled)
    # Plot first to be in background
    plot_data = pivoted.sample(n=min(n_lines, pivoted.shape[1]), axis=1, random_state=42)
    ax.plot(plot_data.index, plot_data.values, color=c_line, alpha=0.2, linewidth=0.8, label='Individual Runs')

    # B. Confidence Band
    ax.fill_between(mean_series.index, 
                    mean_series - sigma*std_series, 
                    mean_series + sigma*std_series, 
                    color=c_mean, alpha=0.15, linewidth=0)

    # C. Mean Trend
    ax.plot(mean_series.index, mean_series, color=c_mean, linewidth=2.5, label='Mean Trend')

    # D. Actual Data (Optional)
    if actual_data is not None:
        if not pd.api.types.is_datetime64_any_dtype(actual_data.index):
            actual_data.index = pd.to_datetime(actual_data.index)
        actual_data = actual_data.loc[actual_data.index <= max_date]
        actual_data = actual_data.loc[actual_data.index >= min_date]
        ax.plot(actual_data.index, actual_data, color=c_actual, linestyle='--', linewidth=2, label='Actual Data')

    # 3. Styling (Clean for LaTeX)
    ax.set_ylabel(variable.replace('_', ' ').title(), fontsize=12)
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Grid
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    
    # Date Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Legend (Deduplicate handles)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', frameon=False)

    plt.tight_layout()

    # 4. Save
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path+variable+"_spaghetti.pdf", bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.show()

def plot_combined_trends(df: pd.DataFrame, 
                         variable: str, 
                         scenarios: Optional[List[str]] = None,
                         sigma: float = 1.0):
    """
    Plots the mean trends and confidence bands of multiple scenarios 
    on a single plot for direct comparison.
    """
    if scenarios is None:
        scenarios = df['scenario'].unique().tolist()
    
    # Define a distinct color palette
    colors = plt.cm.tab10.colors 

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, scen in enumerate(scenarios):
        # 1. Prepare Data
        subset = df[df['scenario'] == scen].copy()
        if subset.empty: continue
        
        if not pd.api.types.is_datetime64_any_dtype(subset['month']):
            subset['month'] = pd.to_datetime(subset['month'])
            
        # Pivot to get stats across simulations
        pivoted = subset.pivot(index='month', columns='sim_id', values=variable)
        mean_series = pivoted.mean(axis=1)
        std_series = pivoted.std(axis=1)
        
        # 2. Plot Mean Line
        color = colors[i % len(colors)]
        ax.plot(mean_series.index, mean_series, color=color, linewidth=2.5, label=scen)
        
        # 3. Plot Confidence Band (Transparent)
        ax.fill_between(
            mean_series.index, 
            mean_series - (sigma * std_series), 
            mean_series + (sigma * std_series), 
            color=color, alpha=0.15, linewidth=0
        )

    # Styling
    ax.set_title(f"Comparative Trends: {variable}", fontsize=14)
    ax.set_ylabel(variable)
    ax.set_xlabel("Time")
    ax.legend(title="Scenario", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def decompose_emissions(df: pd.DataFrame, 
                        gdp_col: str = 'gdp_index', 
                        em_col: str = 'emissions', 
                        baseline_name: str = 'Baseline') -> pd.DataFrame:
    # 1. Aggregate to Ensemble Means (Scenario, Month)
    means = df.groupby(['scenario', 'month'], observed=True)[[gdp_col, em_col]].mean().reset_index()
    
    # 2. Calculate Intensity (E / Q)
    means['intensity'] = means[em_col] / means[gdp_col]

    # 3. Isolate Baseline for Merging
    base_df = means[means['scenario'] == baseline_name].set_index('month')
    
    # 4. Merge Baseline columns into Main DataFrame
    # Suffixes: '' for Scenario, '_base' for Baseline
    merged = means.merge(base_df, on='month', suffixes=('', '_base'), how='left')

    # 5. Calculate Decomposition Components
    # Scale Effect: (Q_scen - Q_base) * I_base
    merged['scale_effect'] = (merged[gdp_col] - merged[f"{gdp_col}_base"]) * merged['intensity_base']

    # Intensity Effect: Q_scen * (I_scen - I_base)
    merged['intensity_effect'] = merged[gdp_col] * (merged['intensity'] - merged['intensity_base'])

    # Total Variation Check: E_scen - E_base
    merged['total_diff'] = merged[em_col] - merged[f"{em_col}_base"]
    final_df = merged[merged['scenario'] != baseline_name].copy()
    
    return final_df[['month', 'scenario', 'total_diff', 'scale_effect', 'intensity_effect']]

def decompose_emissions_avg(df: pd.DataFrame, 
                            gdp_col: str = 'families_wages_received', 
                            em_col: str = 'emissions', 
                            baseline_name: str = 'Baseline') -> pd.DataFrame:
    
    # 1. Calculate Grand Means (Average over all time and sims)
    means = df.groupby('scenario', observed=True)[[gdp_col, em_col]].mean()
    
    # 2. Derived Intensity of the Average (I = Avg_E / Avg_Q)
    means['intensity'] = means[em_col] / means[gdp_col]

    if baseline_name not in means.index:
        print(f"Error: '{baseline_name}' not found in data.")
        return pd.DataFrame()

    # 3. Isolate Baseline Scalars
    base_Q = means.loc[baseline_name, gdp_col]
    base_I = means.loc[baseline_name, 'intensity']
    base_E = means.loc[baseline_name, em_col]

    # 4. Vectorized Decomposition
    # Scale Effect: Change in Output * Baseline Intensity
    means['scale_effect'] = (means[gdp_col] - base_Q) * base_I

    # Intensity Effect: Scenario Output * Change in Intensity
    means['intensity_effect'] = means[gdp_col] * (means['intensity'] - base_I)

    # Total Difference Check
    means['total_diff'] = means[em_col] - base_E

    # 5. Cleanup
    means = means.drop(index=baseline_name).reset_index()
    
    return means[['scenario', 'scale_effect', 'intensity_effect', 'total_diff']]

def plot_decomposition_waterfall(decomp_df: pd.DataFrame, save_path = 'analysis\\validation\\emission_plots\\figures\\decomposed_emissions.pdf'):
    """
    Generates a grid of waterfall charts sharing Y-axis, clean for LaTeX.
    
    Args:
        decomp_df: The DataFrame containing decomposition results.
        save_path: Optional string path (e.g. '../figures/decomp.pdf'). 
                   If provided, saves the figure.
    """
    # 1. Filter and Setup
    if 'month' in decomp_df.columns:
        snapshot = decomp_df[decomp_df['month'] == decomp_df['month'].max()].copy()
    else:
        snapshot = decomp_df.copy()
        
    snapshot = snapshot[snapshot['total_diff'] != 0]
    scenarios = snapshot['scenario'].unique()
    n_scens = len(scenarios)
    
    # 2. Grid Setup
    cols = 3
    rows = math.ceil(n_scens / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), sharey=True)
    axes = axes.flatten() if n_scens > 1 else [axes]

    # Colors
    c_decrease = '#55a868' 
    c_increase = '#c44e52' 
    c_total = '#4c72b0'    

    for i, scen in enumerate(scenarios):
        ax = axes[i]
        row = snapshot[snapshot['scenario'] == scen].iloc[0]
        
        vals = [row['scale_effect'], row['intensity_effect'], row['total_diff']]
        labels = ['Scale', 'Intensity', 'Net Change']
        bottoms = [0, vals[0], 0]
        
        colors = []
        for val, is_total in zip(vals, [False, False, True]):
            if is_total: colors.append(c_total)
            else: colors.append(c_decrease if val < 0 else c_increase)

        # Plot
        ax.bar(labels, vals, bottom=bottoms, color=colors, 
               edgecolor='white', linewidth=1, alpha=0.9, width=0.6)

        # Connectors
        scale_top = vals[0]
        ax.plot([0, 1], [scale_top, scale_top], color='gray', linestyle='--', linewidth=1)
        net_top = vals[0] + vals[1]
        ax.plot([1, 2], [net_top, net_top], color='gray', linestyle='--', linewidth=1)

        # --- Updated Annotations ---
        # Calculate dynamic padding based on the data range of this subplot
        data_range = max(map(abs, vals)) if any(vals) else 1.0
        pad = 0.03 * data_range

        for j, (val, bottom) in enumerate(zip(vals, bottoms)):
            # Determine position and vertical alignment based on sign
            if val >= 0:
                y_pos = bottom + val + pad
                va = 'bottom' # Text sits on top of the position point
            else:
                y_pos = bottom + val - pad
                va = 'top'    # Text hangs below the position point

            ax.text(j, y_pos, f"{val:.2f}", ha='center', va=va, 
                    fontsize=9, fontweight='bold', color='#333333')
        # ---------------------------

        # Formatting
        ax.set_xlabel(scen, fontweight='bold', fontsize=12, labelpad=10)
        ax.axhline(0, color='black', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', visible=False)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Cleanup
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    # 3. Save Logic
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure successfully saved to: {save_path}")

    plt.show()
    
    # 3. Save Logic
    if save_path:
        # Create directory if it doesn't exist
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
            
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure successfully saved to: {save_path}")

    plt.show()

def plot_decomposition_waterfall(decomp_df: pd.DataFrame, save_path='analysis\\validation\\emission_plots\\figures\\decomposed_emissions.pdf'):
    """
    Generates a grid of waterfall charts sharing Y-axis, clean for LaTeX.
    
    Args:
        decomp_df: The DataFrame containing decomposition results.
        save_path: Optional string path (e.g. '../figures/decomp.pdf'). 
                   If provided, saves the figure.
    """
    # 1. Filter and Setup
    if 'month' in decomp_df.columns:
        snapshot = decomp_df[decomp_df['month'] == decomp_df['month'].max()].copy()
    else:
        snapshot = decomp_df.copy()
        
    snapshot = snapshot[snapshot['total_diff'] != 0]
    scenarios = snapshot['scenario'].unique()
    n_scens = len(scenarios)
    
    # 2. Grid Setup
    cols = 3
    rows = math.ceil(n_scens / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), sharey=True)
    
    # Ensure axes is always iterable
    if n_scens > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # --- Colors 
    c_decrease = '#55a868'
    c_increase = '#c44e52'
    c_total = '#4c72b0'   

    for i, scen in enumerate(scenarios):
        ax = axes[i]
        row = snapshot[snapshot['scenario'] == scen].iloc[0]
        
        vals = [row['scale_effect'], row['intensity_effect'], row['total_diff']]
        labels = ['Scale', 'Intensity', 'Net Change']
        bottoms = [0, vals[0], 0]
        
        colors = []
        for val, is_total in zip(vals, [False, False, True]):
            if is_total: 
                colors.append(c_total)
            else: 
                colors.append(c_decrease if val < 0 else c_increase)

        # Plot with transparency (alpha=0.7)
        ax.bar(labels, vals, bottom=bottoms, color=colors, 
               edgecolor='white', linewidth=1, alpha=0.9, width=0.6)

        # Connectors
        scale_top = vals[0]
        ax.plot([0, 1], [scale_top, scale_top], color='gray', linestyle='--', linewidth=1)
        net_top = vals[0] + vals[1]
        ax.plot([1, 2], [net_top, net_top], color='gray', linestyle='--', linewidth=1)

        # --- Updated Annotations (No Bold) ---
        data_range = max(map(abs, vals)) if any(vals) else 1.0
        pad = 0.03 * data_range

        for j, (val, bottom) in enumerate(zip(vals, bottoms)):
            if val >= 0:
                y_pos = bottom + val + pad
                va = 'bottom'
            else:
                y_pos = bottom + val - pad
                va = 'top'

            ax.text(j, y_pos, f"{val:.2f}", ha='center', va=va, 
                    fontsize=9, fontweight='normal', color='#333333') # Changed to normal
        # ---------------------------

        # Formatting (No Bold)
        ax.set_xlabel(scen, fontweight='normal', fontsize=12, labelpad=10)
        ax.axhline(0, color='black', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', visible=False)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Add Global Y-Axis Label
    fig.supylabel('Change in Emissions', fontweight='normal', fontsize=12)

    # Cleanup empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    # 3. Save Logic
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
            
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure successfully saved to: {save_path}")

    plt.show()

def plot_decomposition_waterfall(decomp_df: pd.DataFrame, save_path='analysis\\validation\\emission_plots\\figures\\decomposed_emissions.pdf'):
    """
    Generates a grid of waterfall charts with updated aesthetics (Seaborn style, Viridis colors).
    """
    # Set Seaborn style (White background, no grid by default if we turn it off later)
    sns.set_context("notebook", font_scale=1.2) # Enlarges fonts globally
    sns.set_style("white") 

    # 1. Filter and Setup
    if 'month' in decomp_df.columns:
        snapshot = decomp_df[decomp_df['month'] == decomp_df['month'].max()].copy()
    else:
        snapshot = decomp_df.copy()
        
    snapshot = snapshot[snapshot['total_diff'] != 0]
    scenarios = snapshot['scenario'].unique()
    n_scens = len(scenarios)
    
    # 2. Grid Setup
    cols = 3
    rows = math.ceil(n_scens / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows), sharey=True)
    
    if n_scens > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # --- Colors 
    c_decrease = '#55a868'
    c_increase = '#c44e52'
    c_total = '#4c72b0'   


    for i, scen in enumerate(scenarios):
        ax = axes[i]
        row = snapshot[snapshot['scenario'] == scen].iloc[0]
        
        vals = [row['scale_effect'], row['intensity_effect'], row['total_diff']]
        labels = ['Scale', 'Intensity', 'Net Change']
        bottoms = [0, vals[0], 0]
        
        colors = []
        for val, is_total in zip(vals, [False, False, True]):
            if is_total: 
                colors.append(c_total)
            else: 
                colors.append(c_decrease if val < 0 else c_increase)

        # Plot with transparency (alpha=0.7)
        ax.bar(labels, vals, bottom=bottoms, color=colors, 
               edgecolor='black', linewidth=1, alpha=0.7, width=0.6)

        # Connectors
        scale_top = vals[0]
        ax.plot([0, 1], [scale_top, scale_top], color='gray', linestyle='--', linewidth=1, alpha=0.5)
        net_top = vals[0] + vals[1]
        ax.plot([1, 2], [net_top, net_top], color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # --- Annotations ---
        data_range = max(map(abs, vals)) if any(vals) else 1.0
        pad = 0.05 * data_range # Slightly more padding for larger fonts

        for j, (val, bottom) in enumerate(zip(vals, bottoms)):
            if val >= 0:
                y_pos = bottom + val + pad
                va = 'bottom'
            else:
                y_pos = bottom + val - pad
                va = 'top'

            ax.text(j, y_pos, f"{val:.2f}", ha='center', va=va, 
                    fontsize=10, fontweight='normal', color='#333333') 
        # ---------------------------

        # Formatting
        ax.set_xlabel(scen, fontweight='normal', fontsize=16, labelpad=12)
        ax.axhline(0, color='black', linewidth=1)
        
        # Remove spines
        sns.despine(ax=ax, left=True, bottom=False)
        
        # Remove Grid completely as requested
        ax.grid(False)
        
        # Remove y-ticks for inner plots to clean up view (since they share Y)
        if i % cols != 0:
            ax.set_yticks([])
            ax.set_ylabel("")

    # Global Y-Axis Label
    fig.supylabel('Change in Emissions', fontweight='normal', fontsize=16)

    # Cleanup empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    # 3. Save Logic
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure successfully saved to: {save_path}")

    plt.show()

def plot_distribution_comparison(df: pd.DataFrame, 
                                 variable: str, 
                                 method: str = 'mean'):
    """
    Plots the distribution of a variable (e.g., 'gini') across scenarios 
    using boxplots overlaid with individual simulation points.

    Args:
        df: The stats DataFrame.
        variable: The column name to analyze (e.g., 'gini_index', 'emissions').
        method: 'mean' (time-average) or 'final' (last period value) per simulation.
    """
    # 1. Aggregate per Simulation
    if method == 'final':
        # Get last date's value for each sim
        last_date = df['month'].max()
        data = df[df['month'] == last_date].copy()
    else:
        # Get time-average for each sim
        data = df.groupby(['scenario', 'sim_id'], observed=True)[variable].mean().reset_index()

    # 2. Prepare Data for Plotting
    scenarios = data['scenario'].unique()
    plot_data = [data[data['scenario'] == s][variable].values for s in scenarios]

    # 3. Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Boxplot (Distribution summary)
    ax.boxplot(plot_data, labels=scenarios, patch_artist=True, 
               boxprops=dict(facecolor='lightgray', alpha=0.5),
               medianprops=dict(color='red', linewidth=1.5))

    # Jitter Plot (Show individual runs to detect clustering/outliers)
    for i, vals in enumerate(plot_data):
        # Add random x-jitter
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        ax.plot(x, vals, 'o', color='tab:blue', alpha=0.4, markersize=4)

    ax.set_title(f"Distribution of {variable} by Scenario ({method.title()})")
    ax.set_ylabel(variable)
    ax.set_xlabel("Scenario")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
def plot_concentration_violin(firms_df: pd.DataFrame, 
                              value_col: str = 'revenue',
                              scenarios: list = None,
                              baseline_name: str = 'Baseline',
                              diff_from_baseline: bool = True,
                              save_path: str = 'analysis\\validation\\emission_plots\\figures\\income_concentration.pdf'):
    """
    Plots the distribution of market concentration (Top 10%, Middle 40%, Bottom 50%)
    using Violin Plots.
    
    Args:
        diff_from_baseline: If True, plots the CHANGE relative to the Baseline 
                            (Scenario - Baseline) instead of absolute values.
    """
    if scenarios is None:
        scenarios = sorted(firms_df['scenario'].unique())

    # 1. Calculation (Per Simulation)
    print("Calculating concentration distributions...")
    
    cols = ['scenario', 'sim_id', 'month', value_col]
    data = firms_df[cols].copy()
    
    def get_shares(g):
        total = g[value_col].sum()
        if total == 0: return pd.Series([0, 0, 0], index=['Top 10%', 'Middle 40%', 'Bottom 50%'])
        
        vals = np.sort(g[value_col].values)
        n = len(vals)
        n_top = max(1, int(n * 0.10))
        n_bot = int(n * 0.50)
        # Middle is whatever is left between Top and Bottom indices
        # Indices: 0 to n_bot-1 (Bottom), n_bot to n-n_top-1 (Middle), n-n_top to n-1 (Top)
        
        top_sum = vals[-n_top:].sum()
        bot_sum = vals[:n_bot].sum()
        mid_sum = total - top_sum - bot_sum
        
        return pd.Series({
            'Top 10%': (top_sum / total) * 100,
            'Middle 40%': (mid_sum / total) * 100,
            'Bottom 50%': (bot_sum / total) * 100
        })

    # Group by Sim+Month to get snapshot
    monthly_shares = data.groupby(['scenario', 'sim_id', 'month'], observed=True).apply(get_shares).reset_index()

    # Time-Average per Simulation
    sim_averages = monthly_shares.groupby(['scenario', 'sim_id'])[['Top 10%', 'Middle 40%', 'Bottom 50%']].mean().reset_index()

    # --- Diff from Baseline Logic ---
    if diff_from_baseline:
        if baseline_name not in sim_averages['scenario'].values:
            print(f"Error: Baseline '{baseline_name}' not found for diff calculation.")
            return

        # Pivot to match Sim IDs: Index=sim_id, Columns=Scenario, Values=[Groups]
        pivoted = sim_averages.pivot(index='sim_id', columns='scenario', values=['Top 10%', 'Middle 40%', 'Bottom 50%'])
        
        # Calculate Diffs for each scenario against Baseline
        diff_data = []
        for scen in scenarios:
            if scen == baseline_name: continue # Skip baseline (it would be 0)
            
            for group in ['Top 10%', 'Middle 40%', 'Bottom 50%']:
                # Series of differences
                diffs = pivoted[(group, scen)] - pivoted[(group, baseline_name)]
                
                # Create temp df
                temp = pd.DataFrame({
                    'scenario': scen,
                    'sim_id': diffs.index,
                    'Share Group': group,
                    'Value': diffs.values
                })
                diff_data.append(temp)
        
        plot_data = pd.concat(diff_data, ignore_index=True)
        y_label = "Change in Share (percentage points)"
        title_suffix = "(Difference from Baseline)"
        
    else:
        # Standard Absolute Values
        plot_data = sim_averages.melt(id_vars=['scenario', 'sim_id'], 
                                      value_vars=['Top 10%', 'Middle 40%', 'Bottom 50%'],
                                      var_name='Share Group', value_name='Value')
        plot_data = plot_data[plot_data['scenario'].isin(scenarios)]
        y_label = "Share of Total Value (%)"
        title_suffix = "(Absolute Share)"

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(12, 7))

    # Palette: Blue (Rich), Purple (Middle), Green (Poor)
    palette = {'Top 10%': '#4c72b0', 'Middle 40%': '#8172b3', 'Bottom 50%': '#55a868'}

    sns.violinplot(data=plot_data, x='scenario', y='Value', hue='Share Group',
                   split=False,
                   inner='box',
                   cut=0, 
                   palette=palette, 
                   linewidth=1.2,
                   ax=ax)

    # 3. Styling
    #ax.set_title(f"Distribution of Market Concentration Impacts {title_suffix}", 
    #             fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xlabel("")
    ax.set_ylabel(y_label, fontsize=12)
    
    # Add a zero line if plotting differences
    if diff_from_baseline:
        ax.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(False)

    ax.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)

    plt.tight_layout()

    # 4. Save
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved: {save_path}")

def plot_concentration_violin(firms_df: pd.DataFrame, 
                              value_col: str = 'revenue',
                              scenarios: list = None,
                              baseline_name: str = 'Baseline',
                              diff_from_baseline: bool = True,
                              save_path: str = 'analysis\\validation\\emission_plots\\figures\\income_concentration.pdf'):
    """
    Plots the distribution of market concentration (Top 10%, Middle 40%, Bottom 50%)
    using Violin Plots with Viridis coloring and clean formatting.
    """
    if scenarios is None:
        scenarios = sorted(firms_df['scenario'].unique())

    # 1. Calculation (Per Simulation)
    print("Calculating concentration distributions...")
    
    cols = ['scenario', 'sim_id', 'month', value_col]
    data = firms_df[cols].copy()
    
    def get_shares(g):
        total = g[value_col].sum()
        if total == 0: return pd.Series([0, 0, 0], index=['Top 10%', 'Middle 40%', 'Bottom 50%'])
        
        vals = np.sort(g[value_col].values)
        n = len(vals)
        n_top = max(1, int(n * 0.10))
        n_bot = int(n * 0.50)
        
        top_sum = vals[-n_top:].sum()
        bot_sum = vals[:n_bot].sum()
        mid_sum = total - top_sum - bot_sum
        
        return pd.Series({
            'Top 10%': (top_sum / total) * 100,
            'Middle 40%': (mid_sum / total) * 100,
            'Bottom 50%': (bot_sum / total) * 100
        })

    # Group by Sim+Month to get snapshot
    monthly_shares = data.groupby(['scenario', 'sim_id', 'month'], observed=True).apply(get_shares).reset_index()

    # Time-Average per Simulation
    sim_averages = monthly_shares.groupby(['scenario', 'sim_id'])[['Top 10%', 'Middle 40%', 'Bottom 50%']].mean().reset_index()

    # --- Diff from Baseline Logic ---
    if diff_from_baseline:
        if baseline_name not in sim_averages['scenario'].values:
            print(f"Error: Baseline '{baseline_name}' not found for diff calculation.")
            return

        # Pivot to match Sim IDs
        pivoted = sim_averages.pivot(index='sim_id', columns='scenario', values=['Top 10%', 'Middle 40%', 'Bottom 50%'])
        
        diff_data = []
        for scen in scenarios:
            if scen == baseline_name: continue 
            
            for group in ['Top 10%', 'Middle 40%', 'Bottom 50%']:
                diffs = pivoted[(group, scen)] - pivoted[(group, baseline_name)]
                temp = pd.DataFrame({
                    'scenario': scen,
                    'sim_id': diffs.index,
                    'Share Group': group,
                    'Value': diffs.values
                })
                diff_data.append(temp)
        
        plot_data = pd.concat(diff_data, ignore_index=True)
        y_label = "Change in Share (percentage points)"
        
    else:
        # Standard Absolute Values
        plot_data = sim_averages.melt(id_vars=['scenario', 'sim_id'], 
                                      value_vars=['Top 10%', 'Middle 40%', 'Bottom 50%'],
                                      var_name='Share Group', value_name='Value')
        plot_data = plot_data[plot_data['scenario'].isin(scenarios)]
        y_label = "Share of Total Value (%)"

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Colors (Viridis with Transparency) ---
    cmap = plt.get_cmap('viridis')
    # Use RGBA tuples where the 4th value is alpha (0.7)
    c_top = cmap(0.9, alpha=0.5)    # Yellowish
    c_mid = cmap(0.5, alpha=0.5)    # Teal
    c_bot = cmap(0.2, alpha=0.5)    # Purple
    
    palette = {'Top 10%': c_top, 'Middle 40%': c_mid, 'Bottom 50%': c_bot}

    sns.violinplot(data=plot_data, x='scenario', y='Value', hue='Share Group',
                   split=False,
                   inner='box',
                   cut=0, 
                   palette=palette, 
                   linewidth=1.0, # Slightly thinner line
                   ax=ax)
    sns.despine()
    # 3. Styling
    ax.set_xlabel("", fontweight='normal')
    ax.set_ylabel(y_label, fontsize=12, fontweight='normal')
    
    if diff_from_baseline:
        ax.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(False)

    # Clean Legend
    ax.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, -0.08), 
              ncol=3, frameon=False, fontsize=10)

    plt.tight_layout()

    # 4. Save
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved: {save_path}")
    
    plt.show()

def plot_firm_size_distribution(firms_df: pd.DataFrame, 
                                variable: str = 'revenue', 
                                scenarios: list = None,
                                log_scale: bool = True,
                                save_path: str = None):
    """
    Generates a Firm Size Distribution chart with overlaid Histogram and KDE.
    
    Args:
        variable: The column to plot (e.g., 'revenue', 'emissions').
        log_scale: If True, plots Log10(variable).
    """
    # 1. Setup Data
    last_date = firms_df['month'].max()
    #print(f"Generating distribution for snapshot: {last_date.date()}")
    
    subset = firms_df[firms_df['month'] == last_date].copy()
    
    if scenarios is None:
        scenarios = sorted(subset['scenario'].unique())
    
    subset = subset[subset['scenario'].isin(scenarios)]

    # 2. Log Transformation
    plot_var = variable
    xlabel = variable.replace('_', ' ').title()
    
    if log_scale:
        subset = subset[subset[variable] > 0] # Filter zeros for log
        subset['log_value'] = np.log10(subset[variable])
        plot_var = variable
        xlabel = f"{xlabel}"

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # A. Histogram (Transparent Background)
    # stat='density' ensures it aligns with the KDE scale
    # common_norm=False means each scenario is normalized independently (comparing shapes)
    sns.histplot(data=subset, x=plot_var, hue='scenario', 
                 stat='density', common_norm=False,
                 element="step",  # 'step' is cleaner than 'bars' for overlaps
                 alpha=0.15,      # Very transparent
                 linewidth=0,     # No border on bars
                 palette='tab10', legend=False, ax=ax)

    # B. KDE (Solid Lines on Top)
    sns.kdeplot(data=subset, x=plot_var, hue='scenario', 
                common_norm=False, 
                fill=False,       # No fill, just lines
                linewidth=2.5, 
                palette='tab10', ax=ax)

    # 4. Styling
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("") 

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Grid
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.grid(axis='x', visible=False)
    
    # Legend
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), frameon=False, title=None)

    plt.tight_layout()

    # 5. Save
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved: {save_path}")

    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_fiscal_cost_benefit_horizontal(df, 
                                        fiscal_col='emissions_fund', 
                                        gdp_col='gdp_index', 
                                        emissions_col='emissions', 
                                        gini_col='gini_index'):
    """
    Creates 3 horizontal charts sharing the Fiscal Y-Axis.
    Y-Axis: Fiscal Balance (Surplus/Deficit).
    X-Axes: GDP, Emissions, Gini.
    """
    
    # 1. Aggregation (One point per simulation ID)
    df_agg = df.groupby(['scenario', 'sim_id']).agg({
        fiscal_col: 'last',       # Stock variable (end of run)
        gdp_col: 'mean',          # Flow variable (average)
        emissions_col: 'sum',     # Flow variable (total)
        gini_col: 'mean'          # Flow variable (average)
    }).reset_index()

    # 2. Setup Plot (1 Row, 3 Columns, Share Y)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # Define Palette
    scenarios = df_agg['scenario'].unique()
    palette = sns.color_palette("husl", len(scenarios))
    
    # Common settings for clarity
    alpha_val = 0.7
    size_val = 80

    # --- Chart 1: Fiscal vs GDP ---
    sns.scatterplot(
        data=df_agg, y=fiscal_col, x=gdp_col, 
        hue='scenario', palette=palette, ax=axes[0], s=size_val, alpha=alpha_val
    )
    axes[0].set_title("Economic Output vs. Budget", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("GDP Index (Further Right is Better)")
    axes[0].set_ylabel("Fiscal Balance\n(↓ Deficit | Surplus ↑)")
    axes[0].legend_.remove() # Clean up legends, keep only one

    # --- Chart 2: Fiscal vs Emissions ---
    sns.scatterplot(
        data=df_agg, y=fiscal_col, x=emissions_col, 
        hue='scenario', palette=palette, ax=axes[1], s=size_val, alpha=alpha_val
    )
    axes[1].set_title("Emissions vs. Budget", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Total Emissions (Further Left is Better)")
    axes[1].set_ylabel("") # Hide Y label for inner plots
    axes[1].legend_.remove()

    # --- Chart 3: Fiscal vs Gini ---
    sns.scatterplot(
        data=df_agg, y=fiscal_col, x=gini_col, 
        hue='scenario', palette=palette, ax=axes[2], s=size_val, alpha=alpha_val
    )
    axes[2].set_title("Inequality vs. Budget", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Gini Index (Further Left is Better)")
    axes[2].set_ylabel("")
    
    # 3. Reference Lines & Polish
    # Calculate Baseline Averages for vertical reference lines on X-axes
    base_stats = df_agg[df_agg['scenario'] == 'Baseline'].mean(numeric_only=True)
    
    # Draw Lines
    axes[0].axvline(base_stats[gdp_col], color='grey', linestyle='--', alpha=0.5, label='Baseline GDP')
    axes[1].axvline(base_stats[emissions_col], color='red', linestyle=':', alpha=0.5, label='Baseline Emissions')
    axes[2].axvline(base_stats[gini_col], color='grey', linestyle='--', alpha=0.5, label='Baseline Gini')

    # Draw the Critical "Budget Neutrality" Line (Horizontal)
    for ax in axes:
        ax.axhline(0, color='black', linewidth=1.5, linestyle='-') # The Zero Deficit Line
        ax.grid(True, alpha=0.3)

    # Single Legend on the far right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.08, 0.5), title="Policy Scenarios")
    
    plt.tight_layout()
    return fig

if __name__=="__main__":
    path = '.\output\TAX_EMISSION_TARGETED_TAX_SUBSIDIES_CARBON_TAX_RECYCLING_ECO_INVESTMENT_SUBSIDIES__2025-12-10T22_00_18.323851'
    path = '.\output\TAX_EMISSION_ECO_INVESTMENT_SUBSIDIES__2025-09-03T22_21_39.971705'#_2025-12-15T21_34_51.502117'EMISSIONS_2025-12-15T21_34_51.502117
    path = '.\output\EMISSIONS_2025-12-15T21_34_51.502117'
    stats_df = agg_stats_data(path)
    
    stats_df = stats_df[stats_df['month']>='2010-12-01']
    real = load_and_process_real_data()

    #plot_baseline_spaghetti(stats_df, variable='inflation', baseline_name='Baseline', n_lines=10, sigma=1.0, actual_data=real['inflation'])
    #plot_baseline_spaghetti(stats_df, variable='unemployment', baseline_name='Baseline', n_lines=10, sigma=1.0, actual_data=real['unemployment'])
    #plot_baseline_spaghetti(stats_df, variable='gdp_growth', baseline_name='Baseline', n_lines=10, sigma=1.0, actual_data=real['gdp_growth'])
    
    #firms_df = agg_simulation_data(path)
    #stats_df = stats_df.merge(
    #    firms_df.groupby(['scenario', 'sim_id', 'month'])['wages_paid'].sum().rename('total_wages'), 
    #    on=['scenario', 'sim_id', 'month'], 
    #    how='left'
    #)
    stats_df['fiscal_balance'] = stats_df['total_emission_tax'] - stats_df['total_subsidies']
    #decomp_df = decompose_emissions_avg(
    ##        stats_df[stats_df['month']>='2018-12-01'], #
     #       gdp_col='total_wages', 
     #       em_col='emissions', 
     #      baseline_name='Baseline'
     #   )
    plot_fiscal_cost_benefit_horizontal(stats_df,fiscal_col='fiscal_balance')
    #plot_decomposition_waterfall(decomp_df)
    #plot_concentration_violin(firms_df, value_col='wages_paid')
    #plot_firm_size_distribution(firms_df, variable='revenue')

    #plot_combined_trends(stats_df, variable='emissions', scenarios=['Baseline','Subsidies', 'Carbon Tax','Combined'], sigma=1.0)
    #plot_combined_trends(stats_df, variable='firms_avg_eco_eff', scenarios=['Baseline','Subsidies', 'Carbon Tax','Combined'], sigma=1.0)
    #plot_combined_trends(stats_df, variable='total_subsidies', scenarios=['Baseline','Subsidies', 'Carbon Tax','Combined'], sigma=1.0)
    #plot_combined_trends(stats_df, variable='firms_median_innovation_investment', scenarios=['Baseline','Subsidies', 'Carbon Tax','Combined'], sigma=1.0)
    #plot_combined_trends(stats_df, variable='families_wages_received', scenarios=['Baseline','Subsidies', 'Carbon Tax','Combined'], sigma=1.0)


    pass
    