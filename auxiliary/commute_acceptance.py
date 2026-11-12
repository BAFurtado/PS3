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

By mode, Census 2022: one P(t) for private-vehicle commuters (car, motorcycle, taxi, motorcycle taxi) fitted to their
bands against the offered car minutes, one for transit commuters (bus, BRT, van, rail) against the offered transit
minutes. Observed: Sidra table 10330 (input/commute_mode_2022.csv, auxiliary/commute_mode_2022.py), employed persons
aged 10+ working outside home, bands up to 30 min, 31-60, 61-120, over 120, walking and cycling left out. Offered: as
above, weights E_r x c_r x J_s for private and E_r x (1 - c_r) x J_s for transit.

    python auxiliary/commute_acceptance.py 2022 [weights] [ACP ...]
writes input/commute_acceptance_2022.csv: acp;weights;mode;t50;t_scale;band;observed;offered;fitted
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA = '/home/furtado/MyModels/censo2010/data/amostra'
OUT = 'input/commute_acceptance_2010.csv'
OUT_2022 = 'input/commute_acceptance_2022.csv'
MODE_2022 = 'input/commute_mode_2022.csv'
MODE_GROUP = {'car': 'private', 'motorcycle': 'private', 'taxi': 'private', 'motorcycle_taxi': 'private',
              'bus': 'transit', 'brt': 'transit', 'van': 'transit', 'rail': 'transit'}
BAND_2022 = {'Até cinco minutos': 0, 'De seis minutos até quinze minutos': 0, 'Mais de quinze minutos até meia hora': 0,
             'Mais de meia hora até uma hora': 1, 'Mais de uma hora até duas horas': 2,
             'Mais de duas horas até quatro horas': 3, 'Mais de quatro horas': 3}
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
    by_mode, matched = offered_by_mode(acp, weights, people)
    times = np.concatenate([by_mode['private'][0], by_mode['transit'][0]])
    wt = np.concatenate([by_mode['private'][1], by_mode['transit'][1]])
    return times, wt, matched


def offered_by_mode(acp, weights, people):
    """{'private': (car minutes, weights), 'transit': (transit minutes, weights)}, positive weights only, and the share
    of the ACP's regions with census commuters"""
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
    out = {}
    for mode, times, wt in (('private', m.car.to_numpy(), (w * m.c).to_numpy()),
                            ('transit', m.base.to_numpy(), (w * (1 - m.c)).to_numpy())):
        keep = wt > 0
        out[mode] = (times[keep], wt[keep])
    return out, float(m.E.groupby(m.ap_orig).first().gt(0).mean())


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


def observed_2022(acp):
    """Census 2022 bands of the ACP's motorised commuters, by mode group"""
    t = pd.read_csv(MODE_2022, sep=';')
    t = t[(t.acp == acp) & t['mode'].isin(MODE_GROUP)]
    # Cells Sidra leaves empty hold no persons
    t = t.assign(group=t['mode'].map(MODE_GROUP), b=t.band.map(BAND_2022), persons=t.persons.fillna(0.0))
    out = {}
    for mode, g in t.groupby('group'):
        s = np.bincount(g.b, weights=g.persons, minlength=4)
        out[mode] = s / s.sum()
    return out


def main_2022(weights='trips', *acps):
    acps = list(acps) or ACPS
    t = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    t = t[t.ACPs.isin(acps)]
    people = census(dict(zip(t.cod_mun.astype(str), t.ACPs)))
    out = []
    for acp in acps:
        by_mode, matched = offered_by_mode(acp, weights, people[people.acp == acp])
        obs = observed_2022(acp)
        for mode in ('private', 'transit'):
            times, wt = by_mode[mode]
            t50, scale, kl = fit(times, wt, obs[mode])
            off = bands(times, wt)
            fitted = bands(times, wt * acceptance(times, t50, scale))
            print('%-17s %s %-7s areas matched %.2f | T50 %.1f T_SCALE %.1f KL %.5f | observed %s offered %s'
                  ' fitted %s' % (acp, weights, mode, matched, t50, scale, kl, np.round(obs[mode], 3),
                                  np.round(off, 3), np.round(fitted, 3)), flush=True)
            for b, label in enumerate(LABELS):
                out.append(dict(acp=acp, weights=weights, mode=mode, t50=round(t50, 2), t_scale=round(scale, 2),
                                band=label, observed=round(obs[mode][b], 4), offered=round(off[b], 4),
                                fitted=round(fitted[b], 4)))
    new = pd.DataFrame(out)
    if os.path.exists(OUT_2022):
        old = pd.read_csv(OUT_2022, sep=';')
        old = old[~(old.weights.eq(weights) & old.acp.isin(acps))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(OUT_2022, sep=';', index=False)
    print('wrote', OUT_2022)


if __name__ == '__main__':
    if sys.argv[1:2] == ['2022']:
        main_2022(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
