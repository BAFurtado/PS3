import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import math


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

def plot_spaghetti(df: pd.DataFrame, variable: str, scenario: str, 
                   actual_data = None, 
                   n_lines: int = 50, sigma: float = 1.0):
    
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

    if actual_data is not None:
        # Ensure index is datetime if it isn't already
        if not pd.api.types.is_datetime64_any_dtype(actual_data.index):
            if 'month' in actual_data.columns:
                actual_data = actual_data.set_index('month')
                actual_data.index = pd.to_datetime(actual_data.index)
        
        # Select column if DataFrame, otherwise use Series
        y_vals = actual_data[variable] if isinstance(actual_data, pd.DataFrame) else actual_data
        
        ax.plot(y_vals.index, y_vals, color='red', linestyle='--', linewidth=2, label='Actual', zorder=4)

    ax.set_title(f"{variable}: {scenario}")
    ax.set_ylabel(variable)
    ax.legend(loc='best')
    plt.tight_layout()
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

def plot_decomposition_waterfall(decomp_df: pd.DataFrame):
    """
    Generates a grid of waterfall charts (one per scenario) decomposing 
    the total emission change into Scale and Intensity effects.
    Uses data from the final month available.
    """
    # 1. Select Snapshot (Last Month)
    #last_month = '2011-01-01'# decomp_df['month'].min()
    snapshot = decomp_df.copy() #[decomp_df['month'] == last_month]
    
    # Filter out scenarios with no change (like Baseline itself if present)
    snapshot = snapshot[snapshot['total_diff'] != 0]
    
    scenarios = snapshot['scenario'].unique()
    n_scens = len(scenarios)
    
    # 2. Setup Grid
    cols = 3
    rows = math.ceil(n_scens / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), sharey=True)
    axes = axes.flatten() # Flatten for easy indexing

    # 3. Plot Loop
    for i, scen in enumerate(scenarios):
        ax = axes[i]
        row = snapshot[snapshot['scenario'] == scen].iloc[0]
        
        # Data for Waterfall
        # [Scale Effect, Intensity Effect, Total Change]
        values = [row['scale_effect'], row['intensity_effect'], row['total_diff']]
        labels = ['Scale\nEffect', 'Intensity\nEffect', 'Total\nChange']
        
        # Calculate bottoms for the "stacking" effect of a waterfall
        # Bar 1 starts at 0. Bar 2 starts where Bar 1 ended.
        bottoms = [0, values[0], 0] 
        
        # Determine Colors (Green for reduction/negative, Red for increase/positive, Blue for Total)
        colors = []
        for val, is_total in zip(values, [False, False, True]):
            if is_total:
                colors.append('tab:blue') # Total is neutral/summary
            else:
                colors.append('tab:green' if val < 0 else 'tab:red')

        # Draw Bars
        # We assume the user wants to see the contribution of each part
        ax.bar(labels, values, bottom=bottoms, color=colors, edgecolor='black', alpha=0.7)

        # Connector line (optional visual aid between Scale and Intensity)
        ax.plot([0, 1], [values[0], values[0]], color='black', linestyle='--', linewidth=1)

        # Styling
        ax.set_title(scen, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Add value labels
        for j, val in enumerate(values):
            y_pos = bottoms[j] + val if val < 0 else bottoms[j] + val
            # Adjust label position slightly based on bar direction
            offset = 0.05 * max(abs(x) for x in values) 
            y_pos = bottoms[j] + val + offset if val >= 0 else bottoms[j] + val - offset*2
            
            ax.text(j, y_pos, f"{val:.2f}", ha='center', va='bottom' if val>0 else 'top', fontsize=9)

    # 4. Cleanup Empty Axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Emissions Decomposition by Scenario ", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_subsidy_bias_binscatter(firms_df: pd.DataFrame, 
                                 revenue_col: str = 'revenue', 
                                 subsidy_col: str = 'subsidy_amount', 
                                 n_bins: int = 20):
    """
    Plots a binned scatterplot to analyze if larger firms (by revenue) 
    receive disproportionately higher subsidies.
    """
    # 1. Filter for relevant firms (e.g., only those in a Subsidy scenario)
    # We remove rows with 0 revenue to avoid log errors or division by zero
    data = firms_df[(firms_df[revenue_col] > 0) & (firms_df[subsidy_col] > 0) & (firms_df['month']>='2010-01-01')].copy()
    data[revenue_col] = np.log(data[revenue_col])
    if data.empty:
        print("No firms with positive revenue and subsidies found.")
        return

    # 2. Create Bins based on Revenue (e.g., 20 quantiles)
    # qcut creates bins with equal number of firms in each
    data['bin'] = pd.qcut(data[revenue_col], q=n_bins, labels=False, duplicates='drop')

    # 3. Aggregate Means per Bin
    bin_stats = data.groupby('bin')[[revenue_col, subsidy_col]].mean()

    # 4. Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter of the bins
    ax.scatter(bin_stats[revenue_col], bin_stats[subsidy_col], color='tab:blue', s=100, alpha=0.8, label='Revenue Deciles')
    
    # Fit a regression line through the bins for visual trend
    m, b = np.polyfit(bin_stats[revenue_col], bin_stats[subsidy_col], 1)
    ax.plot(bin_stats[revenue_col], m*bin_stats[revenue_col] + b, color='red', linestyle='--', linewidth=2, label='Trend')

    # Formatting
    ax.set_title(f"Subsidy Allocation Bias (Binned Scatter, n={n_bins})")
    ax.set_xlabel("Mean Firm Revenue (Log Scale recommended if skewed)")
    ax.set_ylabel("Mean Subsidy Received")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_growth_incidence_curve(firms_df: pd.DataFrame, 
                                value_col: str = 'revenue', 
                                scenarios: Optional[List[str]] = None,
                                baseline_name: str = 'Baseline',
                                n_bins: int = 10,
                                show_error: bool = True):
    """
    Plots the Growth Incidence Curve (percentage change relative to Baseline
    conditional on initial firm size percentile).
    """
    if scenarios is None:
        scenarios = [s for s in firms_df['scenario'].unique() if s != baseline_name]
    
    wide = firms_df[firms_df['scenario'].isin(scenarios + [baseline_name])].pivot_table(
        index=['sim_id', 'firm_id'], columns='scenario', values=value_col
    ).dropna(subset=[baseline_name])

    def calc_bin_growth(g):
        try:
            bins = pd.qcut(g[baseline_name], n_bins, labels=False, duplicates='drop')
            growth = g[scenarios].sub(g[baseline_name], axis=0)#.div(g[baseline_name], axis=0)
            return growth.groupby(bins).mean()
        except ValueError: return None

    # Calculate mean growth per bin for every simulation
    sim_stats = wide.groupby('sim_id').apply(calc_bin_growth).stack().rename('growth').reset_index()
    sim_stats.columns = ['sim_id', 'bin', 'scenario', 'growth']

    # Aggregate across simulations
    stats = sim_stats.groupby(['scenario', 'bin'])['growth'].agg(['mean', 'std', 'count'])
    stats['ci'] = 1.96 * (stats['std'] / np.sqrt(stats['count']))

    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals = np.arange(1, n_bins + 1)

    for scen in scenarios:
        data = stats.loc[scen].reindex(range(n_bins))
        y = data['mean'] * 100
        y_err = (data['ci'] * 100) if show_error else None
        
        ax.errorbar(x_vals, y, yerr=y_err, label=scen, capsize=4, marker='o', alpha=0.8)

    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_title(f"Growth Incidence Curve: {value_col.title()}")
    ax.set_xlabel(f"Baseline Size Deciles (1=Smallest, {n_bins}=Largest)")
    ax.set_ylabel("Average Change relative to Baseline (%)")
    ax.set_xticks(x_vals)
    ax.legend() 
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_growth_incidence_curve(firms_df: pd.DataFrame, 
                                value_col: str = 'revenue', 
                                scenarios: Optional[List[str]] = None,
                                baseline_name: str = 'Baseline',
                                n_bins: int = 10,
                                show_error: bool = True):
    """
    Plots the Growth Incidence Curve using the time-averaged value for each firm.
    Compares the average performance in a scenario vs. the average in Baseline,
    conditional on the firm's initial average size.
    """
    if scenarios is None:
        scenarios = [s for s in firms_df['scenario'].unique() if s != baseline_name]
    
    # Pre-process: Average across time periods for each firm/sim/scenario
    # This reduces volatility and captures "permanent" size/impact
    avg_df = firms_df[firms_df['scenario'].isin(scenarios + [baseline_name])].groupby(
        ['scenario', 'sim_id', 'firm_id'], observed=True
    )[value_col].mean().reset_index()

    wide = avg_df.pivot_table(
        index=['sim_id', 'firm_id'], columns='scenario', values=value_col
    ).dropna(subset=[baseline_name])

    def calc_bin_growth(g):
        try:
            # Bin based on the Baseline Average
            bins = pd.qcut(g[baseline_name], n_bins, labels=False, duplicates='drop')
            growth = g[scenarios].sub(g[baseline_name], axis=0).div(g[baseline_name], axis=0)
            return growth.groupby(bins).mean()
        except ValueError: return None

    sim_stats = wide.groupby('sim_id').apply(calc_bin_growth).stack().rename('growth').reset_index()
    sim_stats.columns = ['sim_id', 'bin', 'scenario', 'growth']

    stats = sim_stats.groupby(['scenario', 'bin'])['growth'].agg(['mean', 'std', 'count'])
    stats['ci'] = 1.96 * (stats['std'] / np.sqrt(stats['count']))

    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals = np.arange(1, n_bins + 1)

    for scen in scenarios:
        data = stats.loc[scen].reindex(range(n_bins))
        y = data['mean'] * 100
        y_err = (data['ci'] * 100) if show_error else None
        
        ax.errorbar(x_vals, y, yerr=y_err, label=scen, capsize=4, marker='o', alpha=0.8)

    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_title(f"Growth Incidence Curve: {value_col.title()} (Time-Averaged)")
    ax.set_xlabel(f"Baseline Size Deciles (1=Smallest, {n_bins}=Largest)")
    ax.set_ylabel("Average Change relative to Baseline (%)")
    ax.set_xticks(x_vals)
    ax.legend() 
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
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

def plot_lorenz_curve(firms_df: pd.DataFrame, 
                      value_col: str = 'revenue', 
                      scenarios: Optional[List[str]] = None):
    """
    Plots Lorenz Curves for firm revenues (time-averaged) under different scenarios.
    Includes Gini coefficients in the legend.
    """
    if scenarios is None:
        scenarios = firms_df['scenario'].unique()
    
    # Pre-process: Time-average per firm to capture permanent inequality
    avg_df = firms_df[firms_df['scenario'].isin(scenarios)].groupby(
        ['scenario', 'sim_id', 'firm_id'], observed=True
    )[value_col].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot Line of Equality
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1, label='Equality')

    for scen in scenarios:
        # Pool all firm averages for this scenario
        values = avg_df.loc[avg_df['scenario'] == scen, value_col].values
        values = np.sort(values)
        
        n = len(values)
        if n == 0: continue
        
        # Calculate Lorenz Coordinates
        lorenz_x = np.arange(1, n + 1) / n
        lorenz_y = np.cumsum(values) / values.sum()
        
        # Add (0,0) for correct plotting
        lorenz_x = np.insert(lorenz_x, 0, 0)
        lorenz_y = np.insert(lorenz_y, 0, 0)
        
        # Calculate Gini (1 - 2*AUC)
        gini = 1 - 2 * np.trapz(lorenz_y, lorenz_x)
        
        ax.plot(lorenz_x, lorenz_y, linewidth=2, label=f"{scen} (Gini: {gini:.3f})")

    ax.set_title(f"Lorenz Curve: {value_col.title()} Inequality")
    ax.set_xlabel("Cumulative Share of Firms")
    ax.set_ylabel(f"Cumulative Share of {value_col.title()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    path = '.\output\TAX_EMISSION_TARGETED_TAX_SUBSIDIES_CARBON_TAX_RECYCLING_ECO_INVESTMENT_SUBSIDIES__2025-12-10T22_00_18.323851'
    path = '.\output\EMISSIONS'
    #firms_df = agg_simulation_data(path)
    #firms_df['avg_wage'] = firms_df['wages_paid'] / firms_df['number_employees']

    #plot_growth_incidence_curve(firms_df, value_col='wages_paid', 
    #                            baseline_name='Baseline',
    ##                            scenarios=['Subsidies', 'Carbon Tax','Combined'],
     #                            n_bins=10)
    #plot_lorenz_curve(firms_df, value_col='revenue', 
    #                            scenarios=['Baseline', 'Subsidies','Carbon Tax','Combined'])
    #plot_subsidy_bias_binscatter(firms_df, revenue_col='revenue', subsidy_col='innov_investment', n_bins=30)
    stats_df = agg_stats_data(path)
    stats_df = stats_df[stats_df['month']<='2020-12-01']
    #plot_distribution_comparison(firms_df, variable='wages_paid', method='mean')
    plot_spaghetti(stats_df, variable='emissions', scenario='Baseline', n_lines=10, sigma=1.0)
    plot_combined_trends(stats_df, variable='emissions', scenarios=['Baseline','Subsidies', 'Carbon Tax','Combined'], sigma=1.0)
    decomp_df = decompose_emissions_avg(
            stats_df, 
            gdp_col='families_wages_received', 
            em_col='emissions', 
            baseline_name='Baseline'
        )
        
        # 4. Generate Waterfall Chart
    
    plot_decomposition_waterfall(decomp_df)
    plot_distribution_comparison(stats_df, variable='unemployment', method='mean')
    plot_distribution_comparison(stats_df, variable='emissions', method='mean')
    plot_distribution_comparison(stats_df, variable='gdp_index', method='mean')

    print("Rendering decomposition waterfall...")
