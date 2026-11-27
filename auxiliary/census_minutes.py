"""Census 2022 commute minutes as the base level of the ENMU travel-time matrices (TRANSPORT_BASE 'census2022').

Census: home-work flows of the controlled-access sample (census2022_od.csv, built from the IBGE 2022 microdata):
employed persons working away from home (place of work 2 or 3) who return home daily and report minutes (P1190),
mode groups as auxiliary/commute_acceptance.py, private (motorcycle, motorcycle taxi, car, taxi) and transit (van,
bus, BRT, rail). Unit: home 2010 weighting area o x work municipality m x mode.
Model: the matrix minutes from o to m, mean over m's weighting areas weighted by RAIS 2010 firms
(input/firms_by_APs2010_t0_full.csv), transit state 'base', private state 'car' (free flow, no congestion factor).
Ratio R(o, m, mode) = Census mean minutes / model minutes, at the first level of the cell with at least MIN_RECORDS
sample persons: (o, m), then (municipality of o, m), then (ACP, o and m in the same municipality or not), then the ACP.
TransportNetwork multiplies every state of a pair (o, d) by R(o, municipality of d, transit) and its car minutes by
R(o, municipality of d, private). A flow with zero matrix minutes gets the Census mean minutes of its level instead
(ratio empty, level suffixed '_minutes').

    python auxiliary/census_minutes.py OD_CSV [weights] [ACP ...]   (weights default 'trips', ACPs the six ENMU areas)

writes input/transport/census_ratio_2022__<weights>.parquet: acp, ap_orig, work_mun, mode, ratio, level, records,
census, model. The file is derived from controlled-access microdata: keep it out of any public release."""
import sys

import numpy as np
import pandas as pd

ACPS = ['BAIXADA SANTISTA', 'BRASILIA', 'FORTALEZA', 'GOIANIA', 'MANAUS', 'TERESINA']
GROUP = {'03': 'private', '04': 'private', '05': 'private', '06': 'private',
         '07': 'transit', '08': 'transit', '09': 'transit', '10': 'transit'}
STATE = {'transit': 'base', 'private': 'car'}
# Sample persons behind a cell mean before it is used
MIN_RECORDS = 30
OUT = 'input/transport/census_ratio_2022__%s.parquet'


def census(path, acp):
    od = pd.read_csv(path, sep=';', dtype={'home_mun': str, 'home_ap2010': str, 'work_mun': str, 'place': str,
                                           'returns': str, 'mode': str})
    od = od[(od.acp == acp) & od.place.isin(['2', '3']) & (od.returns == '1') & od['mode'].isin(GROUP)
            & (od.timed > 0)]
    return od.assign(mode=od['mode'].map(GROUP)).rename(columns={'home_ap2010': 'ap_orig'})


def model_minutes(acp, weights):
    """Firm-weighted model minutes from each origin area to each municipality, by mode"""
    m = pd.read_parquet('input/transport/travel_times_%s__%s.parquet' % (acp.replace(' ', '_'), weights))
    m = m[m.state.isin(STATE.values())].assign(mode=lambda x: x.state.map({v: k for k, v in STATE.items()}))
    firms = pd.read_csv('input/firms_by_APs2010_t0_full.csv', sep=';', dtype={'AP': str})
    m = m.merge(firms.rename(columns={'AP': 'ap_dest'}), on='ap_dest', how='inner')
    m['work_mun'] = m.ap_dest.str[:7]
    m['wm'] = m.minutes * m.num_firms
    g = m.groupby(['mode', 'ap_orig', 'work_mun'])[['wm', 'num_firms']].sum()
    return (g.wm / g.num_firms).rename('model').reset_index()


def ratios(od, model, acp):
    """R at the first level with MIN_RECORDS records, for every (origin area, municipality, mode) of the matrix"""
    c = od.merge(model, on=['mode', 'ap_orig', 'work_mun'], how='inner')
    c['home_mun'] = c.ap_orig.str[:7]
    c['same'] = c.home_mun == c.work_mun
    # Census minutes and model minutes summed over the same persons, so each level's ratio is persons-weighted
    c['cm'], c['mm'] = c.minutes, c.model * c.timed
    out = model.copy()
    out['home_mun'] = out.ap_orig.str[:7]
    out['same'] = out.home_mun == out.work_mun
    out['ratio'], out['level'], out['records'], out['census'] = np.nan, '', 0, np.nan
    for level, keys in (('area', ['ap_orig', 'work_mun']), ('municipality', ['home_mun', 'work_mun']),
                        ('acp_same', ['same']), ('acp', [])):
        keys = ['mode'] + keys
        g = c.groupby(keys)[['cm', 'mm', 'timed', 'records']].sum().reset_index()
        g = g[g.records >= MIN_RECORDS]
        g = g.assign(r=g.cm / g.mm, cen=g.cm / g.timed)[keys + ['r', 'records', 'cen']]
        j = out.merge(g, on=keys, how='left', suffixes=('', '_g'))
        fill = out.ratio.isna().to_numpy() & j.r.notna().to_numpy()
        out.loc[fill, 'ratio'] = j.r.to_numpy()[fill]
        out.loc[fill, 'level'] = level
        out.loc[fill, 'records'] = j.records_g.to_numpy()[fill]
        out.loc[fill, 'census'] = j.cen.to_numpy()[fill]
    if out.ratio.isna().any():
        raise ValueError('%s: %d cells without a ratio' % (acp, out.ratio.isna().sum()))
    # A flow the matrix gives zero minutes takes the Census mean minutes of its cell's level
    zero = out.model <= 0
    if out.census[zero].isna().any():
        raise ValueError('%s: zero model minutes without Census minutes' % acp)
    out.loc[zero, 'ratio'] = np.nan
    out.loc[zero, 'level'] = out.level[zero] + '_minutes'
    out.insert(0, 'acp', acp)
    return out[['acp', 'ap_orig', 'work_mun', 'mode', 'ratio', 'level', 'records', 'census', 'model']]


def main(path, weights='trips', *acps):
    acps = list(acps) or ACPS
    parts = []
    for acp in acps:
        od = census(path, acp)
        r = ratios(od, model_minutes(acp, weights), acp)
        persons = od.groupby(['mode', 'ap_orig', 'work_mun']).timed.sum().rename('p').reset_index()
        x = r.merge(persons, on=['mode', 'ap_orig', 'work_mun'], how='left').fillna({'p': 0.0})
        for mode in ('private', 'transit'):
            y = x[x['mode'] == mode]
            share = y.groupby('level').p.sum() / y.p.sum()
            y = y[y.ratio.notna()]
            print('%-17s %-7s persons-weighted ratio %.2f | commuters by level %s | cells by level %s'
                  % (acp, mode, np.average(y.ratio, weights=y.p), share.round(3).to_dict(),
                     y.level.value_counts().to_dict()), flush=True)
        parts.append(r)
    new = pd.concat(parts, ignore_index=True)
    out = OUT % weights
    try:
        old = pd.read_parquet(out)
        new = pd.concat([old[~old.acp.isin(acps)], new], ignore_index=True)
    except FileNotFoundError:
        pass
    new.to_parquet(out, index=False)
    print('wrote', out)


if __name__ == '__main__':
    main(*sys.argv[1:])
