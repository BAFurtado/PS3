from data_utils import *
from policy_charts import *
from validation_charts import *
import pandas as pd


if __name__ == '__main__':
    base_fp = "C:\\Users\\gusta\\Desktop\\Pulmonar\\Projeto_IPEA\\PS3\\analysis\\validation\\simulated_data"
    
    # Read agg data
    lista_no_policy = find_stats_csv(base_fp+"\\baseline")
    lista_tax = find_stats_csv(base_fp+"\\tax")
    lista_subsidies = find_stats_csv(base_fp+"\\subsidies")
    lista_both = find_stats_csv(base_fp+"\\both")

    no_policy = read_many_sim_files(lista_no_policy, policy="No policy")
    tax = read_many_sim_files(lista_tax, policy="Carbon Tax")
    subsidies = read_many_sim_files(lista_subsidies, policy="Subsidies")
    both = read_many_sim_files(lista_both, policy="Both")

    data = pd.concat([no_policy, tax, subsidies, both], ignore_index=True)
    # Read firm-level data
    lista_no_policy = find_firms_csv(base_fp+"\\baseline")
    lista_tax = find_firms_csv(base_fp+"\\tax")
    lista_subsidies = find_firms_csv(base_fp+"\\subsidies")
    lista_both = find_firms_csv(base_fp+"\\both")

    no_policy = read_many_firm_files(lista_no_policy, policy="No policy",consolidate=True)
    tax = read_many_firm_files(lista_tax, policy="Carbon Tax",consolidate=True)
    subsidies = read_many_firm_files(lista_subsidies, policy="Subsidies",consolidate=True)
    both = read_many_firm_files(lista_both, policy="Both",consolidate=True)

    data_firm = pd.concat([no_policy, tax, subsidies, both], ignore_index=True)

    # Read real world data
    data_real = read_macroeconomic_data()
    data['source'] = "Simulated"
    val_data = pd.concat([data[data['Policy'].isin(['No policy'])], data_real], ignore_index=True)
    val_data = val_data[val_data['description'].isin(['inflation','income_growth','unemployment'])]
    #val_data.loc[val_data['description'] == 'unemployment']['value'] = val_data[val_data['description'] == 'unemployment']['value']/100
    val_data.loc[val_data['description'] == 'inflation','value'] = 12*val_data.loc[val_data['description'] == 'inflation',:]['value']
    val_data.loc[val_data['description'] == 'unemployment','value'] = val_data.loc[val_data['description'] == 'unemployment',:]['value']
    # Generate plots
    # Generate plots
    create_grouped_boxplot_by_sector(data_firm,xlabel="Policy",ylabel='Emissions per GDP unit',filter_name='emission_per_gdp',
                            sector_col="sector", value_col="value", policy_col="Policy", 
                            title="Firms' Emissions Distributions by Policy and Sector",)
    plot_variable_distribution(data,'emissions')
    plot_variable_distribution(data,'families_wages_received')
    #plot_variable_distribution(data,'gdp_growth')#unemployment
    plot_variable_distribution(data,'gini_index')
    plot_variable_distribution(data,'inflation')

    plot_variable_distribution(data,'unemployment')

    create_grouped_boxplot(val_data,
                            x_col="description", y_col="value", hue_col="source", 
                            title="Real vs. Simulated Economic Indicators", 
                            xlabel="Economic Indicator",ylabel="Value")
    #plot_dual_variable_distribution(data,'emissions','families_wages_received',bins=10)
    plot_dual_variable_distribution(data,'emissions','gdp_index',bins=10)

    plot_dual_variable_distribution(data,'emissions','unemployment',bins=10)

    create_heatmap(data_firm, 
                   index_col='sector', columns_col='Policy', values_col='value', 
                   title='Firms\' Emissions per GDP Distributions by Policy and Sector',
                    filter_name='emission_per_gdp', figsize=(12, 8), cmap="coolwarm")
    create_heatmap(data_firm, 
                   index_col='sector', columns_col='Policy', values_col='value', 
                   title='Firms\' Emissions per GDP Distributions by Policy and Sector',
                    filter_name='innov_investment', figsize=(12, 8), cmap="coolwarm")
    """index_col (str): Column name for the index (urban agglomeration).
    - columns_col (str): Column name for the columns (policy type).
    - values_col (str): Column name for the heatmap values (e.g., Emission per GDP)."""
    pass