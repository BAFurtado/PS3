from data_utils import *
from policy_charts import *
from validation_charts import *
import pandas as pd


if __name__ == '__main__':
    base_fp = "C:\\Users\\gusta\\Desktop\\Pulmonar\\Projeto_IPEA\\PS3\\analysis\\validation\\simulated_data"
    
    lista_no_policy = find_stats_csv(base_fp+"\\baseline")
    lista_tax = find_stats_csv(base_fp+"\\tax")
    lista_subsidies = find_stats_csv(base_fp+"\\subsidies")
    lista_both = find_stats_csv(base_fp+"\\both")

    no_policy = read_many_sim_files(lista_no_policy, policy="No policy")
    tax = read_many_sim_files(lista_tax, policy="Carbon Tax")
    subsidies = read_many_sim_files(lista_subsidies, policy="Subsidies")
    both = read_many_sim_files(lista_both, policy="Both")

    data = pd.concat([no_policy, tax, subsidies, both], ignore_index=True)
    
    #plot_variable_distribution(data,'emissions')
    #plot_variable_distribution(data,'families_wages_received')
    #plot_variable_distribution(data,'inflation')#unemployment
    #plot_variable_distribution(data,'gini_index')
    #plot_variable_distribution(data,'unemployment')

    plot_dual_variable_distribution(data,'emissions','families_wages_received',bins=10)
    plot_dual_variable_distribution(data,'emissions','unemployment',bins=10)


    pass