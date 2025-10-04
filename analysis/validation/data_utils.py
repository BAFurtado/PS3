import sys, os

# setting path
sys.path.append('PS3/analysis/validation')
sys.path.append('PS3/analysis')
sys.path.append('PS3')

import pandas as pd
from validation_charts import *
from policy_charts import *
# from output import OUTPUT_DATA_SPEC
import seaborn as sns


def transform_annual_to_monthly(annual_df, avg_cols, rate_cols):
    """ Transforms an annual DataFrame into a monthly DataFrame using two rules: 1. Average: Divides the annual value by 12. 2. Rate Proportion: Calculates the monthly rate by taking the 12th root.

    Args:
        annual_df (pd.DataFrame): DataFrame with an annual index (e.g., year).
        avg_cols (list): A list of column names to transform by averaging.
        rate_cols (list): A list of column names to transform by rate proportion.

    Returns:
        pd.DataFrame: A new DataFrame with monthly data, a 'month' column in 'YYYY-MM-DD' format, and a standard integer index.
    """

    monthly_df = annual_df.loc[annual_df.index.repeat(12)].copy()

    num_years = len(annual_df)
    months_list = list(range(1, 13)) * num_years
    monthly_df['month'] = months_list

    monthly_df = monthly_df.reset_index()

    monthly_df['month'] = pd.to_datetime(
        monthly_df['ano'].astype(str) + '-' + monthly_df['month'].astype(str) + '-01').dt.strftime('%Y-%m-%d')

    monthly_df = monthly_df.drop(columns=['ano','index'])

    if avg_cols:
        monthly_df[avg_cols] = monthly_df[avg_cols] / 12

    if rate_cols:
        monthly_df[rate_cols] = (1+monthly_df[rate_cols]) ** (1 / 12) -1
    cols = list(monthly_df.columns)
    cols.insert(0, cols.pop(cols.index('month')))
    return monthly_df[cols]
def find_stats_csv(root_dir):
    stats_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        if "stats.csv" in filenames:
            stats_files.append(os.path.join(dirpath, "stats.csv"))

    return stats_files


def find_firms_csv(root_dir):
    stats_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        if "firms.csv" in filenames:
            stats_files.append(os.path.join(dirpath, "firms.csv"))

    return stats_files


def read_inflation_data(file_path):
    if file_path:
        fp = file_path
    else:
        fp = "PS3/analysis/validation/real_world_data/inflation_data.csv"

    # Load CSV and melt the DataFrame
    df = pd.read_csv(fp, encoding='utf-8', delimiter=';', )  # skiprows=2)
    df.rename(columns={df.columns[0]: "Month"}, inplace=True)
    df_melted = df.melt(id_vars=["Month"], var_name="Municipality", value_name="Value")

    # Convert "Value" to numeric, forcing errors to NaN
    df_melted["Value"] = pd.to_numeric(df_melted["Value"].str.replace(",", "."), errors='coerce') / 100
    df_melted['Description'] = "Inflation_regional"
    df_melted['Type'] = "Real"

    # Display the cleaned DataFrame
    print(df_melted.head())
    # df_melted.to_csv("PS3/analysis/validation/real_world_data/inflation_data_cleaned.csv", index=False)
    return df_melted


def read_macroeconomic_data(file_path=None):
    if file_path:
        fp = file_path
    else:
        fp = "PS3/analysis/validation/real_world_data/real_data_macroeconomic.csv"

    # Load CSV and melt the DataFrame
    df = pd.read_csv(fp, encoding='utf-8', delimiter=';')
    df.rename(columns={df.columns[0]: "month"}, inplace=True)
    df.columns = ["month", "inflation", 'GDP', "income_growth", "unemployment"]
    df_melted = df.melt(id_vars=["month"], var_name="description", value_name="value")
    df_melted["value"] = pd.to_numeric(df_melted["value"].str.replace(",", "."), errors='coerce')
    df_melted.loc[df_melted['description'] == 'unemployment', 'value'] = df_melted.loc[df_melted[
                                                                                           'description'] == 'unemployment', 'value'] / 100
    df_melted.loc[df_melted['description'] == 'inflation', 'value'] = df_melted.loc[df_melted[
                                                                                        'description'] == 'inflation', 'value'] / 100
    df_melted['source'] = "Real"
    # for var in ["Inflation",'GDP',"Income","Unemployment"]:
    #    values = df_melted.loc[df_melted['Description'] == var,'Value']
    #    values = (values-np.min(values))/(np.max(values)-np.min(values))
    #    df_melted.loc[df_melted['Description'] == var,'Value'] = values
    return df_melted


def read_simulation_data(file_path, policy=None):
    """
    This one stats.csv file and adjusts column names. May have to be called multiple times
    """
    columns = OUTPUT_DATA_SPEC['stats']['columns']
    if file_path:
        fp = file_path
    else:
        print("No file path given")
        return None
    data = pd.read_csv(fp, encoding='utf-8', delimiter=';', header=None)
    data.columns = columns
    data['income_growth'] = data['families_wages_received'] / data['families_wages_received'].shift(1) - 1
    data = data.loc[data['income_growth'] <= 0.03]
    df_melted = data.melt(id_vars=["month"], var_name="description", value_name="value")
    df_melted['source'] = "Simulated"
    if policy:
        df_melted['Policy'] = policy
    return df_melted


def read_many_sim_files(file_path_list, policy=None):
    data_frame_list = [read_simulation_data(file_path, policy) for file_path in file_path_list]
    return pd.concat(data_frame_list)


def read_firms_simulation_data(file_path, policy=None, last_month=True, consolidate=False, consolidate_regions=True):
    """
    Takes one firms.csv file and adjusts column names and format.
    """
    columns = OUTPUT_DATA_SPEC['firms']['columns']
    if file_path:
        fp = file_path
    else:
        print("No file path given")
        return None
    data = pd.read_csv(fp, encoding='utf-8', delimiter=';', header=None)
    data.columns = columns
    sim_month_last = data.iloc[-24, 0]

    if consolidate:
        columns_to_average = ['eco_eff', 'price', "innov_investment"]  # Replace with actual column names
        columns_to_sum = ['stocks', 'amount_produced', 'amount_sold', 'revenue',
                          "profit", "wages_paid", "input_cost", "emissions"]

        agg_dict = {col: 'mean' for col in columns_to_average}  # Average these columns
        agg_dict.update({col: 'sum' for col in columns_to_sum})  # Sum these columns

        if consolidate_regions:
            # data['sector'] = data[' sector']
            data.drop(['mun_id', 'firm_id', 'long', 'lat', 'region_id'], axis=1, inplace=True)

            data = data.groupby(['month', 'sector', ], as_index=False).agg(agg_dict)
            data['emission_per_gdp'] = 1000 * data['emissions'] / data['revenue']
            data['innov_per_gdp'] = 100 * data['innov_investment'] * data['wages_paid'] / data['revenue']
            data['gdp_share'] = 100 * data['revenue'] / sum(data['revenue'])
            data['wage_share'] = 100 * data['wages_paid'] / sum(data['wages_paid'])
            df_melted = data.melt(id_vars=['month', 'sector',
                                           ],
                                  var_name="description",
                                  value_name="value")

        else:
            data.drop(['mun_id', 'firm_id', 'long', 'lat'], axis=1, inplace=True)
            data = data.groupby(['month', 'sector', 'region_id'], as_index=False).agg(agg_dict)
            data['emission_per_gdp'] = data['emissions'] / data['revenue']
            df_melted = data.melt(id_vars=['month', 'region_id', 'sector',
                                           ],
                                  var_name="description",
                                  value_name="value")

    else:
        # data.drop(['mun_id', 'long', 'lat'],axis=1,inplace=True)
        data['emission_per_gdp'] = data['emissions'] / data['revenue']
        df_melted = data.melt(id_vars=['month', 'firm_id', 'region_id', 'sector', 'mun_id', 'long', 'lat'
                                       ],
                              var_name="description",
                              value_name="value")
    if policy:
        df_melted['Policy'] = policy
    if last_month:
        # sim_month_last = '2019-01-01'#data.iloc[-24, 0]
        df_melted = df_melted.loc[df_melted['month'] >= sim_month_last]
    # df_melted['source'] = "Simulated"
    return df_melted


def read_many_firm_files(file_path_list, policy=None, consolidate=False):
    data_frame_list = [read_firms_simulation_data(file_path, policy, consolidate=consolidate) for file_path in
                       file_path_list]
    return pd.concat(data_frame_list)


if __name__ == "__main__":
    os.chdir('C:\\Users\\B04903452123\\Documents\\Projects\\PS3_planhab_pvt')
    paths = ['input/interest_baixa.csv',
             'input/interest_media.csv',
             'input/interest_alta.csv']
    for path in paths:
        df = pd.read_csv(path)
        monthly = transform_annual_to_monthly(df,
                                    ['pib_precos_correntes', 'rec_liq_pib', 'desp_primaria_pib',
                                     'pessoal_pib', 'prev_pib', 'discricionaria_pib'],
                                    ['tx_divida', 'interest', 'divida', 'mortgage'])
        monthly.to_csv(path+'_new',index=False)
