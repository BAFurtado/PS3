import pandas as pd

# ==========================
# 1. Ler os arquivos
# ==========================
homens = pd.read_csv("/home/furtado/Downloads/Pessoa11_RR.csv", sep=";", dtype={"Cod_setor": str})
mulheres = pd.read_csv("/home/furtado/Downloads/Pessoa12_RR.csv", sep=";", dtype={"Cod_setor": str})


# ==========================
# 2. Função para transformar
# ==========================
def transformar(df, gender):
    # Garantir string
    df["Cod_setor"] = df["Cod_setor"].astype(str)

    # Criar AREAP (13 primeiros dígitos)
    df["AREAP"] = df["Cod_setor"].str[:13]
    # Criar município (7 primeiros dígitos)
    df["mun"] = df["Cod_setor"].str[:7]
    # Selecionar colunas de idade
    col_idades = ['V022'] + list(df.loc[:, "V035":"V134"].columns)

    # Substituir "X" por 0
    df[col_idades] = df[col_idades].replace("X", 0)

    # Converter para numérico (segurança extra)
    df[col_idades] = df[col_idades].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Transformar para formato longo
    df_long = df.melt(
        id_vars=["AREAP", "mun"],
        value_vars=col_idades,
        var_name="age_var",
        value_name="num_people"
    )

    # Criar idade numérica
    df_long["age"] = df_long["age_var"].apply(
        lambda x: 0 if x == "V022" else int(x[1:]) - 34
    )
    # Ajustar 100+
    df_long.loc[df_long["age"] > 100, "age"] = 100

    # Adicionar gênero
    df_long["gender"] = gender

    # Selecionar ordem final
    df_final = df_long[["AREAP", "gender", "age", "num_people", "mun"]]

    return df_final


# ==========================
# 3. Aplicar função
# ==========================
df_homens = transformar(homens, gender=1)
df_mulheres = transformar(mulheres, gender=2)

# ==========================
# 4. Concatenar
# ==========================
df_final = pd.concat([df_homens, df_mulheres], ignore_index=True)

df_final = (
    df_final
    .groupby(["AREAP", "gender", "age", "mun"], as_index=False)
    .agg({"num_people": "sum"})
)


# ==========================
# 5. Salvar
# ==========================
df_final.to_csv("resultado_final.csv", sep=";", index=False)
