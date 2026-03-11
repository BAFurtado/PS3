import pandas as pd
import os

os.chdir('auxiliary\\housing_data')
df = pd.read_csv("inputs/tabela_municipio_juros_emprestimo.csv")

df["ano"] = df['ano_assinatura']
df['cod_ibge'] = df['cod_mun_ibge']
df = df[['ano', 'cod_ibge', 'total_emprestimo']]


# Calculate total FGTS per year
total_fgts_per_year = df.groupby(["ano"])["total_emprestimo"].transform("sum")

# Create a new column for the weight (share)
df["peso_fgts"] = df["total_emprestimo"] / total_fgts_per_year
df["peso_fgts"] = df["peso_fgts"].fillna(0)
print(sum(df.loc[df["ano"] == 2010]['peso_fgts']))
print(len(df['cod_ibge'].unique()))
df = df.loc[df["ano"] >= 2010]
# Optional: save to a new CSV
df[['ano', 'cod_ibge','peso_fgts']].dropna().to_csv("outputs/pesos_fgts.csv", index=False)