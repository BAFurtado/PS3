"""Census 2022 home-to-work time bands by main transport mode, per municipality (IBGE Sidra table 10330: employed
persons aged 10+ working outside home and returning home 3 or more days a week; mode = the one in which the person
spends most time).

    python auxiliary/commute_mode_2022.py [table.json] [ACP ...]
With table.json (the API answer saved by curl), it reads that file instead of calling the API. writes input/commute_mode_2022.csv: acp;cod_mun;mode;band;persons
"""
import json
import sys
import urllib.request

import pandas as pd

OUT = 'input/commute_mode_2022.csv'
URL = ('https://servicodados.ibge.gov.br/api/v3/agregados/10330/periodos/2022/variaveis/13376?localidades=N6[{}]'
       '&classificacao=537[all]|2088[{}]|86[95251]|469[79176]')
MODES = {'79488': 'all', '79193': 'walk', '79194': 'bicycle', '79195': 'motorcycle', '79196': 'motorcycle_taxi',
         '79197': 'car', '79198': 'taxi', '79207': 'van', '79208': 'bus', '79209': 'brt', '79210': 'rail'}
ACPS = ['BRASILIA', 'FORTALEZA', 'BAIXADA SANTISTA', 'GOIANIA', 'MANAUS', 'TERESINA', 'SALVADOR', 'CURITIBA']
# Band midpoints in minutes, for means
MIDPOINT = {'Até cinco minutos': 3, 'De seis minutos até quinze minutos': 10.5,
            'Mais de quinze minutos até meia hora': 23, 'Mais de meia hora até uma hora': 45,
            'Mais de uma hora até duas horas': 90, 'Mais de duas horas até quatro horas': 180,
            'Mais de quatro horas': 300}


def main(*args):
    saved = [a for a in args if a.endswith('.json')]
    acps = [a for a in args if not a.endswith('.json')] or ACPS
    t = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    t = t[t.ACPs.isin(acps)]
    acp_of = dict(zip(t.cod_mun.astype(str), t.ACPs))
    if saved:
        with open(saved[0]) as f:
            data = json.load(f)
    else:
        with urllib.request.urlopen(URL.format(','.join(acp_of), ','.join(MODES)), timeout=120) as r:
            data = json.load(r)
    rows = []
    for res in data[0]['resultados']:
        cls = {c['id']: list(c['categoria'].values())[0] for c in res['classificacoes']}
        mode = [k for k, v in MODES.items() if any(list(c['categoria'].keys())[0] == k
                                                   for c in res['classificacoes'] if c['id'] == '2088')][0]
        for s in res['series']:
            value = s['serie']['2022']
            rows.append(dict(acp=acp_of[s['localidade']['id']], cod_mun=s['localidade']['id'], mode=MODES[mode],
                             band=cls['537'], persons=pd.to_numeric(value, errors='coerce')))
    out = pd.DataFrame(rows)
    out = out[out.band != 'Total']
    out.to_csv(OUT, sep=';', index=False)
    print('wrote', OUT, len(out))


if __name__ == '__main__':
    main(*sys.argv[1:])
