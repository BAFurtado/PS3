"""MCMV Faixa 1 price per unit by ACP and year, in R$ of 2010 (input/mcmv_ceiling.csv), and the state construction
cost over consumer prices by year, 2010 = 1 (input/sinapi_ipca.csv). Both are read under MCMV_COMMISSION.

Contracts: Ministério das Cidades, MCMV Subsidiado open data (one row per contract signed, position of 30/06/2026),
modality FAR. Price per unit = val_contratado_total / qtd_uh. Contracts whose units were all cancelled
(qtd_uh_distratadas = qtd_uh) are dropped, and so are contracts whose price per unit lies outside half to twice the
national median of their year of signature. The ACP's price of a year is the median price per unit of its contracts
signed that year, weighted by units not cancelled, deflated by the IPCA annual mean. Years without contracts are
interpolated linearly within the ACP. Years after the ACP's last contract take its last price times the growth of the
national median since that year. Years before the first contract take the first price. After LAST_YEAR every ACP
holds the mean of its LAST_YEAR - 2 to LAST_YEAR prices.

Sinapi over IPCA: annual means of the Sinapi index of the state (Sidra 2296, variable 49 chained across the 2012
rebase with the monthly change, variable 1196) over the IPCA index (Sidra 1737, variable 2266), 2010 = 1. After
LAST_YEAR every state holds the mean of its LAST_YEAR - 2 to LAST_YEAR ratios. The current year is the mean of the
months published.

Rows: ACPs;year;ceiling (R$ of 2010 per unit)   and   uf;year;factor

Usage: python auxiliary/mcmv_ceiling.py <MCMV Subsidiado csv or zip>
"""
import sys

import numpy as np
import pandas as pd
import requests

SIDRA = 'https://apisidra.ibge.gov.br/values/t/{t}/{geo}/v/{v}/p/all'
UF = {11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO', 21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN',
      25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA', 31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP', 41: 'PR', 42: 'SC',
      43: 'RS', 50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'}
FIRST_YEAR, LAST_YEAR, END_YEAR = 2009, 2026, 2060


def sidra(t, geo, v):
    d = pd.DataFrame(requests.get(SIDRA.format(t=t, geo=geo, v=v), timeout=300).json()[1:])
    d['V'] = pd.to_numeric(d.V, errors='coerce')
    d['month'] = pd.PeriodIndex(d.D3C, freq='M')
    return d


def ipca_annual():
    d = sidra(1737, 'n1/all', 2266)
    d = d[d.month.dt.year >= FIRST_YEAR]
    annual = d.groupby(d.month.dt.year).V.mean()
    return annual / annual[2010]


def sinapi_annual():
    """Annual mean of the Sinapi index by state, chained across the 2012 rebase"""
    index, change = sidra(2296, 'n3/all', 49), sidra(2296, 'n3/all', 1196)
    out = {}
    for code, uf in UF.items():
        i = index[index.D1C.astype(int) == code].set_index('month').V
        v = change[change.D1C.astype(int) == code].set_index('month').V
        old = i[i > 1e6]
        last = old.index.max()
        chained = [old[last]]
        months = v[v.index > last].dropna().index
        for m in months:
            chained.append(chained[-1] * (1 + v[m] / 100))
        series = pd.concat([old, pd.Series(chained[1:], index=months)])
        series = series[series.index.year >= FIRST_YEAR]
        out[uf] = series.groupby(series.index.year).mean()
    return pd.DataFrame(out)


def number(s):
    return pd.to_numeric(s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')


def weighted_median(x, w):
    order = np.argsort(x)
    x, w = np.asarray(x)[order], np.asarray(w)[order]
    cum = np.cumsum(w)
    return float(x[np.searchsorted(cum, cum[-1] / 2)])


def contracts(path):
    d = pd.read_csv(path, sep=';', dtype=str)
    d = d[d.txt_modalidade == 'FAR'].copy()
    for c in ('qtd_uh', 'qtd_uh_distratadas', 'val_contratado_total'):
        d[c] = number(d[c])
    d['units'] = d.qtd_uh - d.qtd_uh_distratadas.fillna(0)
    d['year'] = pd.to_datetime(d.dt_assinatura, dayfirst=True, errors='coerce').dt.year
    d = d[(d.units > 0) & (d.val_contratado_total > 0) & d.year.notna()].copy()
    d['year'] = d.year.astype(int)
    d['price'] = d.val_contratado_total / d.qtd_uh
    national = {y: weighted_median(g.price, g.units) for y, g in d.groupby('year')}
    ratio = d.price / d.year.map(national)
    out = ~ratio.between(.5, 2)
    print('dropped as outliers: %d of %d contracts' % (out.sum(), len(d)))
    print(d[out][['txt_nome_municipio', 'txt_sigla_uf', 'year', 'qtd_uh', 'price']].sort_values('year').to_string())
    d = d[~out]
    d['mun6'] = d.cod_ibge.str.replace('.', '', regex=False).str.zfill(6)
    return d


def hold_after(table, years):
    """Rows by year to END_YEAR, years after LAST_YEAR at the mean of the last three"""
    tail = table.loc[LAST_YEAR - 2:LAST_YEAR].mean()
    later = pd.DataFrame([tail] * (END_YEAR - LAST_YEAR), index=range(LAST_YEAR + 1, END_YEAR + 1))
    return pd.concat([table.loc[years[0]:LAST_YEAR], later])


def ceilings(d, ipca):
    acps = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';').drop_duplicates('cod_mun')
    acps['mun6'] = acps.cod_mun.astype(str).str[:6]
    d = d.merge(acps[['mun6', 'ACPs']], on='mun6')
    years = list(range(FIRST_YEAR, LAST_YEAR + 1))
    d['real'] = d.price / d.year.map(ipca)
    national = pd.Series({y: weighted_median(g.real, g.units) for y, g in d.groupby('year')}).reindex(years)
    national = national.interpolate().bfill().ffill()
    out = {}
    for acp, g in d.groupby('ACPs'):
        s = pd.Series({y: weighted_median(h.real, h.units) for y, h in g.groupby('year')}).reindex(years)
        first, last = s.first_valid_index(), s.last_valid_index()
        s = s.interpolate(limit_area='inside')
        s[s.index > last] = s[last] * national[s.index > last] / national[last]
        s[s.index < first] = s[first]
        out[acp] = s
    table = hold_after(pd.DataFrame(out), years)
    return table.stack().rename('ceiling').rename_axis(['year', 'ACPs']).reset_index()[['ACPs', 'year', 'ceiling']]


def main():
    ipca = ipca_annual()
    d = contracts(sys.argv[1])
    ceiling = ceilings(d, ipca)
    ceiling['ceiling'] = ceiling.ceiling.round(0)
    ceiling.sort_values(['ACPs', 'year']).to_csv('input/mcmv_ceiling.csv', sep=';', index=False)
    sinapi = sinapi_annual()
    years = list(range(2010, LAST_YEAR + 1))
    factor = sinapi.div(sinapi.loc[2010]).div(ipca, axis=0).loc[years]
    factor = hold_after(factor, years)
    factor = factor.stack().rename('factor').rename_axis(['year', 'uf']).reset_index()[['uf', 'year', 'factor']]
    factor['factor'] = factor.factor.round(4)
    factor.sort_values(['uf', 'year']).to_csv('input/sinapi_ipca.csv', sep=';', index=False)
    print(ceiling[ceiling.ACPs == 'GOIANIA'].set_index('year').ceiling.loc[2009:2027].to_string())
    print(factor[factor.uf == 'GO'].set_index('year').factor.loc[2010:2027].to_string())


if __name__ == '__main__':
    main()
