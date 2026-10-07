"""National real GDP index, 2010 = 1, for exports and the programme funds (input/national_real_gdp.csv).

Source: IBGE annual real GDP growth, as published in the Banco Central SGS series 7326 ("PIB - taxa de variação real
no ano"), <https://api.bcb.gov.br/dados/serie/bcdata.sgs.7326/dados?formato=json>. Years after the last one published
are not in the file; the model holds the index at its last value.

Usage: python auxiliary/national_real_gdp.py
"""
import pandas as pd
import requests

URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.7326/dados?formato=json"


def main():
    df = pd.DataFrame(requests.get(URL, timeout=60).json())
    df["year"] = pd.to_datetime(df.data, dayfirst=True).dt.year
    growth = df.set_index("year").valor.astype(float) / 100
    growth = growth[growth.index >= 2000]
    index = (1 + growth).cumprod()
    index = index / index.loc[2010]
    index.rename("index").to_frame().to_csv("input/national_real_gdp.csv", sep=";", float_format="%.6f")
    print(index.round(4).to_string())


if __name__ == "__main__":
    main()
