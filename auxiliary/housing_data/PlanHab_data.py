import pandas as pd
import numpy as np
import os

os.chdir('auxiliary/housing_data')
print(os.getcwd())

# 1. Tratamento dos Totais Nacionais (Histórico ABECIP)
df_totals = pd.read_excel("inputs/FINANCIAMENTO_UNID_E_VALORES_ABECIP_20.xlsx", sheet_name='resumo').dropna()
mask = df_totals["Período"].str.contains("total", case=False, na=False)
df_totals_filtered = df_totals[mask].copy()
df_totals_filtered["ano"] = df_totals_filtered["Período"].str.extract(r"(\d{4})").astype(int)

df_totals_cleaned = df_totals_filtered[["ano", "Total"]].rename(columns={"Total": "total_nacional"})
df_totals_cleaned["total_nacional"] *= 1_000_000

# 2. Carregamento das Projeções (Dados do Bernardo)
df_fgts_sbpe = pd.read_excel("inputs/Numeros_Bernardo.xlsx", sheet_name="consolidado")

# 3. Construção da Matriz de Pesos Municipais (Grid Espaço-Temporal)
df_pesos_raw = pd.read_csv("outputs/pesos_fgts.csv")
df_pesos_raw.loc[df_pesos_raw['ano'] == 2025, 'peso_fgts'] = np.nan

pesos_medios = df_pesos_raw.groupby("cod_ibge")["peso_fgts"].mean().reset_index().rename(columns={"peso_fgts": "peso_medio"})
municipios = df_pesos_raw[["cod_ibge"]].drop_duplicates()
todos_anos = sorted(list(set(df_totals_cleaned["ano"]).union(set(df_fgts_sbpe["Cenario"]))))

# Criação do produto cartesiano (Municípios x Anos) para evitar gaps
grid = municipios.assign(key=1).merge(pd.DataFrame({'ano': todos_anos, 'key': 1}), on='key').drop('key', axis=1)

df_pesos_full = pd.merge(grid, df_pesos_raw, on=["cod_ibge", "ano"], how="left")
df_pesos_full = pd.merge(df_pesos_full, pesos_medios, on="cod_ibge", how="left")
df_pesos_full["peso_final"] = df_pesos_full["peso_fgts"].fillna(df_pesos_full["peso_medio"])

# 4. Regionalização por Tipo de Financiamento e Cenário
all_results = []
for prefixo in ['fgts', 'sbpe']:
    for cenario in ['Tendencial', 'Otimista', 'Pessimista']:
        col_name = f'{prefixo}_{cenario}'
        temp_proj = df_fgts_sbpe[['Cenario', col_name]].rename(columns={'Cenario': 'ano', col_name: 'total_nacional'})
        
        # Consolida histórico e projeção (priorizando o dado projetado em caso de sobreposição)
        combined_total = pd.concat([df_totals_cleaned, temp_proj]).drop_duplicates('ano', keep='last')
        
        df_reg = pd.merge(df_pesos_full, combined_total, on="ano", how="inner")
        df_reg["valor_regionalizado"] = df_reg["peso_final"] * df_reg["total_nacional"]
        df_reg["cenario"] = cenario
        df_reg["tipo_financiamento"] = prefixo
        
        all_results.append(df_reg)

df_final = pd.concat(all_results, ignore_index=True)
df_final["cod_ibge"] = df_final["cod_ibge"].astype(int).astype(str)

# 5. Integração com PIB e Cálculo de Intensidade
df_gdp = pd.read_csv("inputs/br_ibge_pib_municipio.csv").rename(columns={"pib": "gdp"})
df_gdp["ano"] = df_gdp["ano"].astype(int)
df_gdp["cod_ibge"] = df_gdp["id_municipio"].astype(str).str[:6] # Padronização 6 dígitos
df_gdp["gdp"] *= 1000 # Ajuste de escala do PIB

df_pivot = df_final.pivot_table(
    index=["cenario", "ano", "cod_ibge"],
    columns="tipo_financiamento",
    values="valor_regionalizado"
).reset_index().rename(columns={"fgts": "recursos_fgts", "sbpe": "recursos_sbpe"})

# 6. Projeção de PIB Futuro e Exportação Final
for cenario in df_pivot["cenario"].unique():
    df_cen = df_pivot[df_pivot["cenario"] == cenario].copy()
    df_export = pd.merge(df_cen, df_gdp[["ano", "cod_ibge", "gdp"]], on=["ano", "cod_ibge"], how="left").sort_values(['cod_ibge', 'ano'])

    # Projeção de PIB baseada no último dado real + crescimento de 2% a.a.
    df_export['gap'] = df_export.groupby('cod_ibge')['gdp'].transform(lambda x: x.isnull().cumsum())
    df_export['last_gdp'] = df_export.groupby('cod_ibge')['gdp'].ffill()
    mask_null = df_export['gdp'].isna()
    df_export.loc[mask_null, 'gdp'] = df_export['last_gdp'] * (1.02 ** df_export['gap'])
    
    # Cálculo Final: Proporção do Financiamento em relação ao PIB
    df_export["recursos_fgts"] /= df_export["gdp"]
    # Recurso do SBPE está em milhares
    df_export.loc[df_export['ano']>=2026,"recursos_sbpe"] *=1000
    df_export["recursos_sbpe"] /= df_export["gdp"]
    
    filename = f"outputs/fgts_sbpe_pct_{cenario.lower()}.csv"
    df_export[['recursos_fgts',"recursos_sbpe"]] = df_export[['recursos_fgts',"recursos_sbpe"]].fillna(0).apply( lambda x:np.clip(x,0,100))
    df_export[['cod_ibge','ano','recursos_fgts','recursos_sbpe']].to_csv(filename, index=False)

pass