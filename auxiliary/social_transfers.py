"""Federal social transfers paid to residents, by municipality, 2010, for SOCIAL_TRANSFERS 'data'
(input/social_transfers_2010.csv): beneficiaries and mean monthly value in 2010 R$ of RGPS benefits, BPC and Bolsa
Família, and the Census population.

Sources:
- Bolsa Família and BPC: MDS, MI Social (aplicacoes.mds.gov.br/sagi/servicos/misocial), monthly per municipality,
  2010: valor_repassado_bolsa_familia, qtd_familias_beneficiarias_bolsa_familia, bpc_val, bpc_ben (elderly and
  disabled). Means over the twelve months.
- RGPS: INSS, "Valor líquido dos benefícios emitidos ... segundo grupos de espécies", 2019 (gov.br/previdencia,
  ben_municipios_especie_2019.xlsx; the earliest year by municipality is 2017): "Total de benefícios previdenciários",
  annual value and December quantity. Taken to 2010 with the national totals of IFI, Relatório de Acompanhamento
  Fiscal 112 (May 2026), Tabela 2 (BEPS, Siga Brasil, RTN): benefit spending R$ 585 bn (2010) and 858 bn (2019) at
  December 2025 prices, stock 24.4 and 30.9 million; and the IPCA annual mean (Sidra 1737, variable 2266), 2019 over
  2010. Each municipality keeps its 2019 share. The annual value includes the 13th payment; it is spread over 12 months.
- Population: IBGE, Censo Demográfico 2010, Sidra 1378, variable 93.

Usage: python auxiliary/social_transfers.py
"""
import io
import json
import subprocess
import statistics

import pandas as pd
import requests

MI = ('https://aplicacoes.mds.gov.br/sagi/servicos/misocial?q=*:*&fq=anomes_s:2010{month:02d}&wt=json&rows=6000'
      '&fl=codigo_ibge,valor_repassado_bolsa_familia_f,qtd_familias_beneficiarias_bolsa_familia_i,bpc_val_f,bpc_ben_i')
INSS = ('https://www.gov.br/previdencia/pt-br/assuntos/previdencia-social/arquivos/'
        'ben_municipios_especie_2019.xlsx')
IPCA = ('https://servicodados.ibge.gov.br/api/v3/agregados/1737/periodos/201001-201012|201901-201912/variaveis/2266'
        '?localidades=N1[all]')
POP = 'https://servicodados.ibge.gov.br/api/v3/agregados/1378/periodos/2010/variaveis/93?localidades=N6[{codes}]'
RGPS_VALUE_2010_2019 = 585 / 858
RGPS_STOCK_2010_2019 = 24.4 / 30.9
CHUNK = 100


def curl(url):
    # Python's certificate store rejects the MDS certificate
    return subprocess.run(['curl', '-s', '-g', '-m', '600', url], check=True, capture_output=True).stdout


def mi_social():
    rows = []
    for month in range(1, 13):
        for d in json.loads(curl(MI.format(month=month)))['response']['docs']:
            rows.append((int(d['codigo_ibge']), d.get('valor_repassado_bolsa_familia_f') or 0.0,
                         d.get('qtd_familias_beneficiarias_bolsa_familia_i') or 0, d.get('bpc_val_f') or 0.0,
                         d.get('bpc_ben_i') or 0))
    df = pd.DataFrame(rows, columns=['cod6', 'pbf_val', 'pbf_fam', 'bpc_val', 'bpc_ben'])
    return df.groupby('cod6').sum() / 12


def rgps():
    book = io.BytesIO(curl(INSS))
    sheets = {}
    for name, col in (('Valor (R$) - 2019', 'value'), ('Quantidade - dez19', 'ben')):
        s = pd.read_excel(book, name, header=None, skiprows=7)
        s = s[pd.to_numeric(s[0], errors='coerce').notna()]
        sheets[col] = pd.Series(pd.to_numeric(s[10]).values, index=s[0].astype(int).values)
    series = requests.get(IPCA, timeout=300).json()[0]['resultados'][0]['series'][0]['serie']
    ipca = {y: statistics.mean(float(v) for k, v in series.items() if k.startswith(y)) for y in ('2010', '2019')}
    value = sheets['value'] / 12 * RGPS_VALUE_2010_2019 * ipca['2010'] / ipca['2019']
    return pd.DataFrame({'rgps_val': value, 'rgps_ben': sheets['ben'] * RGPS_STOCK_2010_2019})


def population(codes):
    pop = {}
    for i in range(0, len(codes), CHUNK):
        url = POP.format(codes=','.join(str(c) for c in codes[i:i + CHUNK]))
        for s in requests.get(url, timeout=300).json()[0]['resultados'][0]['series']:
            pop[int(s['localidade']['id'])] = pd.to_numeric(s['serie']['2010'], errors='coerce')
    return pd.Series(pop)


def main():
    codes = sorted(pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';').cod_mun.unique())
    df = pd.DataFrame(index=pd.Index(codes, name='cod_mun'))
    df['pop'] = population(codes)
    df = df.join(rgps())
    mi = mi_social()
    for col in mi.columns:
        df[col] = (df.index // 10).map(mi[col])
    df = df.fillna(0.0)
    missing = df[(df['pop'] > 0) & ((df.rgps_ben == 0) | (df.pbf_fam == 0))].index.tolist()
    if missing:
        print(f'no RGPS or Bolsa Família data for {missing}')
    df[['pop', 'rgps_ben', 'rgps_val', 'bpc_ben', 'bpc_val', 'pbf_fam', 'pbf_val']].to_csv(
        'input/social_transfers_2010.csv', sep=';', float_format='%.2f')
    tot = df.sum()
    print(f'{len(df)} municipalities, per resident a month: RGPS R$ {tot.rgps_val / tot["pop"]:.1f} '
          f'({tot.rgps_ben / tot["pop"]:.3f} beneficiaries), BPC R$ {tot.bpc_val / tot["pop"]:.1f}, '
          f'Bolsa Família R$ {tot.pbf_val / tot["pop"]:.1f}')


if __name__ == '__main__':
    main()
