import pandas as pd

# 1. CARREGAR O MAPEAMENTO (Crosswalk)
# Use o arquivo CSV que geramos com as 68 linhas
df_crosswalk = pd.read_csv("crosswalk_SNA68_GIC42_ISIC12_v2.csv")

# 2. CARREGAR SEUS DADOS REAIS
# Substitua 'seus_coeficientes.csv' pelo arquivo que contém os 42 coeficientes de Alvarenga
# Deve ter colunas: ['GIC_42_Code', 'Intensity']
df_coefs = pd.read_csv("emission_intensity.csv",sep=';',decimal=',',encoding='latin-1') 
df_coefs['GIC_42_Code'] = df_coefs['GIC_code'].astype(str).str.zfill(2)


# Substitua 'seu_vbp_ibge.csv' pelo arquivo com o VBP dos 68 setores do SNA
# Deve ter colunas: ['SNA_68_Code', 'VBP']
df_vbp = pd.read_csv("valor_producao_42_setores.csv",sep=';',decimal=',',encoding='latin-1')
df_vbp = df_vbp.dropna()
# Certificar que os códigos são strings para evitar erros de merge
df_crosswalk['SNA_68_Code'] = df_crosswalk['SNA_68_Code'].astype(str).str.zfill(4)
df_crosswalk['GIC_42_Code'] = df_crosswalk['GIC_42_Code'].astype(str).str.zfill(2)

df_vbp['GIC_42_Code'] = df_vbp['GIC_code'].astype(int).astype(str).str.zfill(2)

# ---------------------------------------------------------
# PROCESSO DE CÁLCULO
# ---------------------------------------------------------

# a. Unir o mapeamento com os coeficientes (GIC 42) e depois com o VBP (SNA 68)
df_master = df_crosswalk.merge(df_coefs, on='GIC_42_Code', how='left')
df_master = df_master.merge(df_vbp, on='GIC_42_Code', how='left')

# b. Calcular a emissão total absoluta de cada linha (SNA 68)
# Emissão Total = Coeficiente (tCO2e/R$) * VBP (R$)
df_master['Emissao_Absoluta'] = df_master['com_LUC_2019'] * df_master['Demanda Total']

# c. Agrupar pela sua classificação de 12 setores
# Somamos as emissões totais e os VBPs totais de cada grupo
df_agregado = df_master.groupby('Aggregation_12_Sectors').agg({
    'Emissao_Absoluta': 'sum',
    'Demanda Total': 'sum'
}).reset_index()

# d. Calcular a Intensidade Ponderada Final
# Intensidade Final = Soma das Emissões / Soma do VBP
df_agregado['eco'] = df_agregado['Emissao_Absoluta'] / df_agregado['Demanda Total']

# 3. SALVAR RESULTADO
df_agregado.to_csv("emissions_12_sectors_2019.csv", index=False)
print("Cálculo concluído com sucesso!")
print(df_agregado[['Aggregation_12_Sectors', 'eco']])