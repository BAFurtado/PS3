"""FPM paid to each municipality a year, net of the FUNDEB retention (input/fpm/{UF}.csv: uf, ano, fpm, cod), from the
Secretaria do Tesouro Nacional, Transferências Obrigatórias da União - por Município, 'FPM por município'
(Tesouro Transparente, monthly values by SIAFI municipality code). Complete years only: all 12 months published.
Municipalities are matched to IBGE codes by name and UF (IBGE localidades API).

Usage: python auxiliary/fpm_stn.py
"""
import io
import unicodedata

import pandas as pd
import requests

URL = ('https://www.tesourotransparente.gov.br/ckan/dataset/3b5a779d-78f5-4602-a6b7-23ece6d60f27/resource/'
       'd69ff32a-6681-4114-81f0-233bb6b17f58/download/fpm-por-municipio.csv')
IBGE = 'https://servicodados.ibge.gov.br/api/v1/localidades/municipios'
FIRST = 2000
# (STN spelling, UF) -> IBGE spelling, where the two differ beyond accents, case and punctuation
NAMES = {
    ('Muquém de São Francisco', 'BA'): 'Muquém do São Francisco', ('Santa Teresinha', 'BA'): 'Santa Terezinha',
    ('Itapagé', 'CE'): 'Itapajé', ('Barão de Monte Alto', 'MG'): 'Barão do Monte Alto',
    ('Brasópolis', 'MG'): 'Brazópolis', ('Dona Eusébia', 'MG'): 'Dona Euzébia',
    ('São Thomé das Letras', 'MG'): 'São Tomé das Letras', ('Poxoréo', 'MT'): 'Poxoréu',
    ('Santo Antônio do Leverger', 'MT'): 'Santo Antônio de Leverger',
    ('Eldorado dos Carajás', 'PA'): 'Eldorado do Carajás', ('Santa Isabel do Pará', 'PA'): 'Santa Izabel do Pará',
    ('São Domingos de Pombal', 'PB'): 'São Domingos', ('Seridó', 'PB'): 'São Vicente do Seridó',
    ('Belém de São Francisco', 'PE'): 'Belém do São Francisco', ('Iguaraci', 'PE'): 'Iguaracy',
    ('Lagoa do Itaenga', 'PE'): 'Lagoa de Itaenga', ('Parati', 'RJ'): 'Paraty',
    ('Trajano de Morais', 'RJ'): 'Trajano de Moraes', ('Açu', 'RN'): 'Assú', ('Arês', 'RN'): 'Arez',
    ('Augusto Severo', 'RN'): 'Campo Grande', ('Presidente Juscelino', 'RN'): 'Serra Caiada',
    ('São Luiz', 'RR'): 'São Luiz do Anauá', ('Amparo de São Francisco', 'SE'): 'Amparo do São Francisco',
    ('Gracho Cardoso', 'SE'): 'Graccho Cardoso', ('Embu', 'SP'): 'Embu das Artes', ('Florínia', 'SP'): 'Florínea',
    ('Moji Mirim', 'SP'): 'Mogi Mirim', ('São Luís do Paraitinga', 'SP'): 'São Luiz do Paraitinga',
    ('Couto de Magalhães', 'TO'): 'Couto Magalhães', ('Fortaleza do Tabocão', 'TO'): 'Tabocão',
    ('São Valério da Natividade', 'TO'): 'São Valério'}


def key(name, uf):
    s = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode().lower()
    return ''.join(c for c in s if c.isalnum()) + '|' + uf


def ibge_codes():
    out = {}
    for m in requests.get(IBGE, timeout=300).json():
        uf = m['regiao-imediata']['regiao-intermediaria']['UF']['sigla']
        out[key(m['nome'], uf)] = m['id']
    return out


def main():
    raw = requests.get(URL, timeout=600).content.decode('latin-1')
    d = pd.read_csv(io.StringIO(raw), sep=';', dtype=str).dropna(subset=['UF'])
    d.columns = [c.strip() for c in d.columns]
    years = [c for c in d.columns if c.isdigit() and int(c) >= FIRST]
    for y in years:
        d[y] = pd.to_numeric(d[y].str.strip().str.replace('.', '', regex=False).str.replace(',', '.', regex=False),
                             errors='coerce')
    d['uf'] = d.UF.str.strip()
    complete = [y for y in years if d.groupby('Mês')[y].count().min() > 0]
    names = [NAMES.get((n, uf), n) for n, uf in zip(d['Município'].str.strip(), d.uf)]
    codes = ibge_codes()
    d['cod'] = [codes.get(key(n, uf)) for n, uf in zip(names, d.uf)]
    missing = d[d.cod.isna()][['Município', 'uf']].drop_duplicates()
    if len(missing):
        raise SystemExit(f'No IBGE code for:\n{missing.to_string(index=False)}')
    annual = d.groupby(['uf', 'cod'], sort=False)[complete].sum(min_count=1)
    long = annual.stack().rename('fpm').round(2).reset_index().rename(columns={'level_2': 'ano'})
    long = long.astype({'cod': int, 'ano': int})
    for uf, g in long.groupby('uf'):
        g[['uf', 'ano', 'fpm', 'cod']].to_csv(f'input/fpm/{uf}.csv', index=False)
    print(f'{long.cod.nunique()} municipalities, years {complete[0]}-{complete[-1]}')
    print((long.groupby('ano').fpm.sum() / 1e9).round(2).to_string())


if __name__ == '__main__':
    main()
