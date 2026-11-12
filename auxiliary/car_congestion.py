"""Car congestion factor per ACP: kappa = (C_private / C_transit) / (M_car / M_transit).

C: Census 2022 mean home-to-work minutes by main mode (input/commute_mode_2022.csv, band midpoints), private = car
and motorcycle, transit = bus, BRT and rail. M: mean minutes of the free-flow car state and of the transit base state
of input/transport/travel_times_<ACP>__<weights>.parquet over all region pairs, weighted by the population aged 18-65
of the origin (Census 2010) times the firms of the destination. Scaling car minutes by kappa makes the model's
car-to-transit time ratio the Census one. It assumes car and transit commuters travel comparable distances.

    python auxiliary/car_congestion.py [weights]
writes input/car_congestion_2022.csv: acp;weights;census_private;census_transit;model_car;model_transit;kappa
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from commute_mode_2022 import MIDPOINT  # noqa: E402

OUT = 'input/car_congestion_2022.csv'
ACPS = ['BRASILIA', 'FORTALEZA', 'BAIXADA SANTISTA', 'GOIANIA', 'MANAUS', 'TERESINA']
PRIVATE = ['car', 'motorcycle']
TRANSIT = ['bus', 'brt', 'rail']


def census_means():
    t = pd.read_csv('input/commute_mode_2022.csv', sep=';').dropna(subset=['persons'])
    t['mid'] = t.band.map(MIDPOINT)
    out = {}
    for acp, g in t.groupby('acp'):
        p, q = g[g['mode'].isin(PRIVATE)], g[g['mode'].isin(TRANSIT)]
        out[acp] = (np.average(p.mid, weights=p.persons), np.average(q.mid, weights=q.persons))
    return out


def main(weights='trips'):
    firms = pd.read_csv('input/firms_by_APs2010_t0_full.csv', sep=';')
    jobs = dict(zip(firms.AP.astype(str), firms.num_firms))
    pop = pd.read_csv('input/num_people_age_gender_AP_2010.csv', sep=';')
    pop = pop[pop.age.between(18, 65)].groupby(pop.AREAP.astype(str)).num_people.sum()
    census = census_means()
    rows = []
    for acp in ACPS:
        m = pd.read_parquet('input/transport/travel_times_%s__%s.parquet' % (acp.replace(' ', '_'), weights))
        m = m.pivot_table(index=['ap_orig', 'ap_dest'], columns='state', values='minutes').reset_index()
        w = m.ap_orig.map(pop).fillna(0.0) * m.ap_dest.map(jobs).fillna(0.0)
        car, transit = np.average(m.car, weights=w), np.average(m.base, weights=w)
        c_private, c_transit = census[acp]
        kappa = (c_private / c_transit) / (car / transit)
        rows.append(dict(acp=acp, weights=weights, census_private=round(c_private, 1),
                         census_transit=round(c_transit, 1), model_car=round(car, 1), model_transit=round(transit, 1),
                         kappa=round(kappa, 3)))
    new = pd.DataFrame(rows)
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, sep=';')
        new = pd.concat([old[old.weights != weights], new], ignore_index=True)
    new.to_csv(OUT, sep=';', index=False)
    print(new.to_string(index=False))


if __name__ == '__main__':
    main(*sys.argv[1:])
