import pandas as pd


# Defina as larguras das colunas com base nas posições
colunas = [
    (0, 4),    # Ano
    (46, 53),  # Código do município
    (54, 95),  # Nome do município
    (95, 95 + 85), # Código RM
    (412, 419), # Código concentração urbana
    (896, 915),  # valor adicionado a preços correntes
    (934, 953), # pib a preços correntes
    (953, 971) # pib per capita
    ]

# Nomes das colunas
nomes_colunas = [
    "ano",
    "cod_mun",
    "nome_mun",
    "rm",
    "cod_conc_urb",
    "va_corrente",
    "pib_corrente",
    "pib_percapita_corrente"
]


if __name__ == '__main__':
    f = 'pib_municipios_2009_2021_original.txt'
    d = pd.read_fwf(f, colspecs=colunas, names=nomes_colunas, encoding='latin-1')
    d = d[d.ano == 2021]
    d.to_csv('pib_municipios2021.csv', index=False)
