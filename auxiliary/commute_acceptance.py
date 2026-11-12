"""Commute acceptance P(t) = 1 / (1 + exp((t - T50) / T_SCALE)), t in one-way minutes, fitted per ACP to the Census
2010 home-to-work time bands.

Observed: persons aged 16-70 who work outside home (V0660 2, 3 or 5), return home daily (V0661 = 1) and answer the
usual home-to-work time (V0662), weighted. Bands: up to 30 min, 31-60, 61-120, over 120 (the census bands up to 5 and
6-30 are merged, as the model has no time below a region's own time).
Offered: every pair of PS3 regions (r, s) with weight E_r x J_s, E_r the census commuters living in r, J_s the firms in
s (input/firms_by_APs2010_t0_full.csv). A share c_r of E_r, the commuters of r living in a household with a car
or a motorcycle (V0222, V0221), travels in car minutes times the ACP's congestion factor
(input/car_congestion_2022.csv), the rest in transit base minutes (input/transport/travel_times_<ACP>__<weights>).
Fit: T50 and T_SCALE minimise the Kullback-Leibler divergence of the offered times weighted by P(t), binned, from
the observed bands.

    python auxiliary/commute_acceptance.py [weights] [ACP ...]
writes input/commute_acceptance_2010.csv: acp;weights;t50;t_scale;band;observed;offered;fitted
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA = '/home/furtado/MyModels/censo2010/data/amostra'
OUT = 'input/commute_acceptance_2010.csv'
BANDS = [0, 30, 60, 120, np.inf]
LABELS = ['0-30', '31-60', '61-120', '120+']
CENSUS_BAND = {'1': 0, '2': 0, '3': 1, '4': 2, '5': 3}
ACPS = ['BRASILIA', 'FORTALEZA', 'BAIXADA SANTISTA', 'GOIANIA', 'MANAUS', 'TERESINA']
HOUSE = {'mun': (1, 7), 'ctrl': (21, 28), 'moto': (102, 102), 'car': (103, 103)}
PERSON = {'mun': (1, 7), 'area': (8, 20), 'ctrl': (21, 28), 'w': (29, 44), 'age': (62, 64), 'place': (328, 328),
          'daily': (350, 350), 'time': (351, 351)}


def cut(line, cols):
    return {k: line[a - 1:b] for k, (a, b) in cols.items()}


def census(mun_acp):
    """Commuters: acp, area, weight, band, car"""
    states = pd.read_csv('input/STATES_ID_NUM.csv', sep=';')
    uf = dict(zip(states.nummun.astype(str), states.codmun))
    rows = []
    for code in sorted({m[:2] for m in mun_acp}):
        folder = os.path.join(DATA, uf[code] if uf[code] != 'SP' else 'SP1')
        for f in sorted(os.listdir(folder)):
            if not f.startswith('Amostra_Domicilios_'):
                continue
            car = {}
            with open(os.path.join(folder, f), encoding='latin-1') as h:
                for line in h:
                    if line[0:7] in mun_acp:
                        r = cut(line, HOUSE)
                        car[(r['mun'], r['ctrl'])] = r['car'] == '1' or r['moto'] == '1'
            with open(os.path.join(folder, f.replace('Domicilios', 'Pessoas')), encoding='latin-1') as p:
                for line in p:
                    if line[0:7] not in mun_acp:
                        continue
                    r = cut(line, PERSON)
                    if r['place'] not in '235' or r['daily'] != '1' or r['time'] not in CENSUS_BAND:
                        continue
                    if not 16 <= int(r['age']) <= 70:
                        continue
                    rows.append((mun_acp[r['mun']], r['area'], float(r['w']) / 1e13, CENSUS_BAND[r['time']],
                                 car.get((r['mun'], r['ctrl']), False)))
    return pd.DataFrame(rows, columns=['acp', 'area', 'w', 'band', 'car'])


def acceptance(t, t50, scale):
    return 1.0 / (1.0 + np.exp(np.clip((t - t50) / scale, -50, 50)))


def offered(acp, weights, people):
    m = pd.read_parquet('input/transport/travel_times_%s__%s.parquet' % (acp.replace(' ', '_'), weights))
    m = m.pivot_table(index=['ap_orig', 'ap_dest'], columns='state', values='minutes').reset_index()
    kappa = pd.read_csv('input/car_congestion_2022.csv', sep=';')
    m['car'] = m.car * float(kappa[(kappa.acp == acp) & (kappa.weights == weights)].kappa.iloc[-1])
    firms = pd.read_csv('input/firms_by_APs2010_t0_full.csv', sep=';')
    jobs = dict(zip(firms.AP.astype(str), firms.num_firms))
    by_area = people.groupby('area').apply(lambda g: pd.Series({'E': g.w.sum(), 'c': np.average(g.car, weights=g.w)}),
                                           include_groups=False)
    m['E'] = m.ap_orig.map(by_area.E).fillna(0.0)
    m['c'] = m.ap_orig.map(by_area.c).fillna(0.0)
    m['J'] = m.ap_dest.map(jobs).fillna(0.0)
    w = m.E * m.J
    times = np.concatenate([m.car.to_numpy(), m.base.to_numpy()])
    wt = np.concatenate([(w * m.c).to_numpy(), (w * (1 - m.c)).to_numpy()])
    keep = wt > 0
    return times[keep], wt[keep], float(m.E.groupby(m.ap_orig).first().gt(0).mean())


def bands(times, wt):
    idx = np.digitize(times, BANDS[1:-1], right=True)
    s = np.bincount(idx, weights=wt, minlength=4)
    return s / s.sum()


def fit(times, wt, observed):
    def loss(x):
        pred = bands(times, wt * acceptance(times, x[0], np.exp(x[1])))
        return float(np.sum(observed * np.log(observed / np.maximum(pred, 1e-12))))
    best = min((minimize(loss, [t0, np.log(s0)], method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-10})
                for t0 in (30, 60, 90, 150) for s0 in (5, 15, 40)), key=lambda r: r.fun)
    return best.x[0], float(np.exp(best.x[1])), best.fun


def main(weights='trips', *acps):
    acps = list(acps) or ACPS
    t = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    t = t[t.ACPs.isin(acps)]
    people = census(dict(zip(t.cod_mun.astype(str), t.ACPs)))
    out = []
    for acp in acps:
        p = people[people.acp == acp]
        observed = np.bincount(p.band, weights=p.w, minlength=4) / p.w.sum()
        times, wt, matched = offered(acp, weights, p)
        t50, scale, kl = fit(times, wt, observed)
        off = bands(times, wt)
        fitted = bands(times, wt * acceptance(times, t50, scale))
        print('%-17s %s commuters %d, areas matched %.2f | T50 %.1f T_SCALE %.1f KL %.5f | observed %s offered %s'
              ' fitted %s' % (acp, weights, len(p), matched, t50, scale, kl, np.round(observed, 3), np.round(off, 3),
                              np.round(fitted, 3)), flush=True)
        for b, label in enumerate(LABELS):
            out.append(dict(acp=acp, weights=weights, t50=round(t50, 2), t_scale=round(scale, 2), band=label,
                            observed=round(observed[b], 4), offered=round(off[b], 4), fitted=round(fitted[b], 4)))
    new = pd.DataFrame(out)
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, sep=';')
        old = old[~(old.weights.eq(weights) & old.acp.isin(acps))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(OUT, sep=';', index=False)
    print('wrote', OUT)


if __name__ == '__main__':
    main(*sys.argv[1:])
