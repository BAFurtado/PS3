"""Real housing credit rates for INTEREST_HOUSING = 'real' (input/planhab_funds/interest_housing_real.csv).

Monthly rates, deflated by expected inflation: the monthly geometric mean of IPCA over the last 12 months (BCB SGS
433, <https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json>), and after the last published month the
inflation target, 3 %/yr. Floored at 0.
- sbpe, fgts: input/planhab_funds/interest_housing_media.csv (nominal regulated rates).
- mortgage: market-rate real estate financing, BCB SGS 25497 (monthly), held at its first value before it starts
  (as in input/interest_*.csv); after its last published month, the mortgage column of input/interest_real.csv.

Usage: SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt python auxiliary/real_housing_rates.py
"""
import pandas as pd
import requests

SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json&dataInicial=01/01/1999"
TARGET = 1.03 ** (1 / 12) - 1


def sgs(code):
    df = pd.DataFrame(requests.get(SGS.format(code), timeout=60).json())
    return pd.Series(df.valor.astype(float).values / 100, index=pd.to_datetime(df.data, dayfirst=True))


def main():
    nominal = pd.read_csv("input/planhab_funds/interest_housing_media.csv", parse_dates=["date"]).set_index("date")
    real_file = pd.read_csv("input/interest_real.csv", parse_dates=["date"]).set_index("date")
    ipca = sgs(433)
    mortgage = sgs(25497)

    expected = (1 + ipca).rolling(12).apply(lambda x: x.prod() ** (1 / 12), raw=True) - 1
    expected = expected.reindex(nominal.index)
    expected[nominal.index > ipca.index.max()] = TARGET
    expected = expected.bfill()

    mortgage_nominal = mortgage.reindex(nominal.index)
    mortgage_nominal[nominal.index < mortgage.index.min()] = mortgage.iloc[0]

    out = pd.DataFrame(index=nominal.index)
    for col in ("sbpe", "fgts"):
        out[col] = (1 + nominal[col]) / (1 + expected) - 1
    out["mortgage"] = (1 + mortgage_nominal) / (1 + expected) - 1
    after = nominal.index > mortgage.index.max()
    out.loc[after, "mortgage"] = real_file.mortgage.reindex(nominal.index)[after]
    out = out.clip(lower=0)
    out.to_csv("input/planhab_funds/interest_housing_real.csv", float_format="%.6f")
    annual = (1 + out).groupby(out.index.year).prod() - 1
    print(annual.loc[2008:2030].round(3).to_string())


if __name__ == "__main__":
    main()
