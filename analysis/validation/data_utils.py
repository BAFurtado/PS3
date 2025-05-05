import sys,os
# setting path
sys.path.append('PS3/analysis/validation')
sys.path.append('PS3/analysis')
sys.path.append('PS3')



import pandas as pd
from validation_charts import *
from policy_charts import *
from output import OUTPUT_DATA_SPEC
import seaborn as sns

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
    df = pd.read_csv(fp, encoding='utf-8', delimiter=';',)# skiprows=2)
    df.rename(columns={df.columns[0]: "Month"}, inplace=True)
    df_melted = df.melt(id_vars=["Month"], var_name="Municipality", value_name="Value")

    # Convert "Value" to numeric, forcing errors to NaN
    df_melted["Value"] = pd.to_numeric(df_melted["Value"].str.replace(",", "."), errors='coerce')/100
    df_melted['Description'] = "Inflation_regional"
    df_melted['Type'] = "Real"

    # Display the cleaned DataFrame
    print(df_melted.head())
    #df_melted.to_csv("PS3/analysis/validation/real_world_data/inflation_data_cleaned.csv", index=False)
    return df_melted

def read_macroeconomic_data(file_path=None):
    if file_path:
        fp = file_path
    else:
        fp = "PS3/analysis/validation/real_world_data/real_data_macroeconomic.csv"

    # Load CSV and melt the DataFrame
    df = pd.read_csv(fp, encoding='utf-8', delimiter=';')
    df.rename(columns={df.columns[0]: "month"}, inplace=True)
    df.columns = ["month","inflation",'GDP',"income_growth","unemployment"]
    df_melted = df.melt(id_vars=["month"], var_name="description", value_name="value")
    df_melted["value"] = pd.to_numeric(df_melted["value"].str.replace(",", "."), errors='coerce')
    df_melted.loc[df_melted['description'] == 'unemployment','value'] = df_melted.loc[df_melted['description'] == 'unemployment','value']/100
    df_melted.loc[df_melted['description'] == 'inflation','value'] = df_melted.loc[df_melted['description'] == 'inflation','value']/100
    df_melted['source'] = "Real"
    #for var in ["Inflation",'GDP',"Income","Unemployment"]:
    #    values = df_melted.loc[df_melted['Description'] == var,'Value']
    #    values = (values-np.min(values))/(np.max(values)-np.min(values))
    #    df_melted.loc[df_melted['Description'] == var,'Value'] = values
    return df_melted

def read_simulation_data(file_path,policy=None):
    """
    This one stats.csv file and adjusts column names. May have to be called multiple times
    """
    columns = OUTPUT_DATA_SPEC['stats']['columns']
    if file_path:
        fp = file_path
    else:
        print("No file path given")
        return None
    data = pd.read_csv(fp, encoding='utf-8', delimiter=';',header=None)
    data.columns = columns
    data['income_growth'] = data['families_wages_received']/data['families_wages_received'].shift(1)-1
    data = data.loc[data['income_growth']<=0.03]
    df_melted = data.melt(id_vars=["month"], var_name="description", value_name="value")
    df_melted['source'] = "Simulated"
    if policy:
        df_melted['Policy'] = policy
    return df_melted

def read_many_sim_files(file_path_list,policy=None):
    data_frame_list = [read_simulation_data(file_path,policy) for file_path in file_path_list]
    return pd.concat(data_frame_list)

def read_firms_simulation_data(file_path,policy=None,last_month=True,consolidate=False,consolidate_regions=True):
    """
    Takes one firms.csv file and adjusts column names and format.
    """
    columns = OUTPUT_DATA_SPEC['firms']['columns']
    if file_path:
        fp = file_path
    else:
        print("No file path given")
        return None
    data = pd.read_csv(fp, encoding='utf-8', delimiter=';',header=None)
    data.columns = columns
    sim_month_last = data.iloc[-24, 0]
    
    if consolidate:
        columns_to_average = ['eco_eff','price',"innov_investment"]  # Replace with actual column names
        columns_to_sum = ['stocks', 'amount_produced','amount_sold','revenue',
                          "profit","wages_paid","input_cost","emissions"] 

        agg_dict = {col: 'mean' for col in columns_to_average}  # Average these columns
        agg_dict.update({col: 'sum' for col in columns_to_sum}) # Sum these columns
        
        if consolidate_regions:
            #data['sector'] = data[' sector']
            data.drop(['mun_id','firm_id', 'long', 'lat','region_id'],axis=1,inplace=True)
           
            data = data.groupby(['month','sector',], as_index=False).agg(agg_dict)
            data['emission_per_gdp'] = 1000*data['emissions']/data['revenue']
            data['innov_per_gdp'] = 100*data['innov_investment']*data['wages_paid']/data['revenue']
            data['gdp_share'] = 100*data['revenue']/sum(data['revenue'])
            data['wage_share'] = 100*data['wages_paid']/sum(data['wages_paid'])
            df_melted = data.melt(id_vars=['month','sector',
                                        ], 
                                var_name="description", 
                                value_name="value")
            
        else:
            data.drop(['mun_id','firm_id', 'long', 'lat'],axis=1,inplace=True)
            data = data.groupby(['month','sector', 'region_id'], as_index=False).agg(agg_dict)
            data['emission_per_gdp'] = data['emissions']/data['revenue']
            df_melted = data.melt(id_vars=['month','region_id','sector',
                                        ], 
                                var_name="description", 
                                value_name="value")
            
    else:
        #data.drop(['mun_id', 'long', 'lat'],axis=1,inplace=True)
        data['emission_per_gdp'] = data['emissions']/data['revenue']
        df_melted = data.melt(id_vars=['month', 'firm_id','region_id','sector','mun_id', 'long', 'lat'
                                    ], 
                            var_name="description", 
                            value_name="value")
    if policy:
        df_melted['Policy'] = policy
    if last_month:
        #sim_month_last = '2019-01-01'#data.iloc[-24, 0]
        df_melted = df_melted.loc[df_melted['month'] >= sim_month_last]
    #df_melted['source'] = "Simulated"
    return df_melted



def read_many_firm_files(file_path_list,policy=None,consolidate=False):
    data_frame_list = [read_firms_simulation_data(file_path,policy,consolidate=consolidate) for file_path in file_path_list]
    return pd.concat(data_frame_list)

if __name__ == "__main__":
   
    base_fp = "C:\\Users\\gusta\\Desktop\\Pulmonar\\Projeto_IPEA\\PS3"
    agg_fp = [base_fp + _ for _ in ["\\output\\baseline",
                "\\output\\both",
                "\\output\\subsidies",
                "\\output\\tax"]]
    #data_agg_pol = read_many_sim_files(agg_fp)
    
    
    

    
    baseline = read_firms_simulation_data(base_fp+"\\output\\baseline\\0\\firms.csv",policy="No Policy",consolidate=True)
    mean_base = baseline.drop(['Policy','month','firm_id','region_id'],axis=1).groupby(['sector','description'],as_index=False).mean()
    tax = read_firms_simulation_data(base_fp+"\\output\\tax\\0\\firms.csv",policy="Carbon Tax")
    subsidies = read_firms_simulation_data(base_fp+"\\output\\subsidies\\0\\firms.csv",policy="Subsidies")
    #data_policies_firms_2 = read_firms_simulation_data("C:\\Users\\gusta\\Downloads\\firms.csv",policy="Carbon Tax")
    dat_pol = pd.concat([tax, subsidies,baseline],ignore_index=True)
    #dat_pol = dat_pol.loc[dat_pol['description']#.isin(['emissions','pop'])]
    dat_pol = dat_pol.loc[dat_pol['value']>0]
    dat_pol = dat_pol.merge(mean_base, how='left', left_on=['sector','description'], right_on=['sector','description'],
                            suffixes=('', '_mean'))
    dat_pol = dat_pol.loc[dat_pol['month'] >= '2018-01-01']
    dat_pol = dat_pol.loc[dat_pol['sector'] != ' OtherServices ']
    print(dat_pol['description'].unique())
    

    create_heatmap(
        data=dat_pol.loc[dat_pol['description'].isin(['emissions'])],
        index_col="sector",
        columns_col="Policy",
        values_col="value",
        title="Heatmap: Emission by Policy and Sector"
    )

    create_emissions_violinplot_by_sector(data=dat_pol,
        sector_col="sector",
        policy_col="Policy",
        value_col="value",
        filter_name="emissions",
        title="Firms' Emissions Reduction Distributions by Policy and Sector (12 Sectors)",
        xlabel="",
        ylabel="Emissions Reduction (%)",
        figsize=(16, 12),
        palette="Set2",
        rotation=45,
        col_wrap=4,
        normalize=False)
    
    create_grouped_boxplot_by_sector(data=dat_pol,
        sector_col="sector",
        policy_col="Policy",
        value_col="value",
        filter_name="emissions",
        title="Firms' Emissions Distributions by Policy and Sector (12 Sectors)",
        xlabel="",
        ylabel="Emissions (Co2 tons)",
        figsize=(16, 12),
        palette="Set2",
        rotation=45,
        normalize=False
       )

    #make_plot_sectors_emission(data=dat_pol, name='figures/emission_policy_effects', exclude_sectors=[])
pass