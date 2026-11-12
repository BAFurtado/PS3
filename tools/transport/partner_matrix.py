"""ENMU partner matrices (2054, morning peak, public transport minutes) from traffic zones to PS3 regions.

Input: input/bndes/matrizes (the partners' zip, private, not in git), one '... v2.xlsx' per region with the zone
system. Each PS3 region (tools/transport/regions.py, its urban part) is cut by the zones. A piece weighs, within its
region, by its area ('area') or by its share of the zone's area times the zone's trips ('trips': trips produced on
the origin side, attracted on the destination side, TI + TPC, Cenário Padrão). The region-to-region time is the
weighted mean of zone-to-zone times over the zone pairs that have a time. 9999 and zero times off the diagonal count
as missing.

A pair still without a time takes the time of the reverse pair, and failing that the car time (input/transport/car,
tools/transport/car_skim.py) times the ACP's median ratio of transit to car minutes over pairs with both. State 'car'
is the car time, with missing pairs (a region to itself, pairs the car skim did not reach) the transit base time
divided by that ratio. A region still without a time to itself takes the median of the other regions' times to
themselves.

    python -m tools.transport.partner_matrix [ACP ...]
writes input/transport/travel_times_<ACP>__<weights>.parquet (states base, padrao, otimizado, car), private, not in
git.
"""
import glob
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from tools.transport.regions import regions, METRIC

SRC = 'input/bndes/matrizes/25 - Matrizes de viagens'
OUT = 'input/transport'
STATES = ['base', 'padrao', 'otimizado']
COLUMNS = ['o', 'd', 'ti_p', 'ti_o', 'tpc_p', 'tpc_o', 'base', 'padrao', 'otimizado']
# ACP -> (region folder prefix, zone file, zone id column)
ZONES = {'BRASILIA': ('01', 'input/bndes/SIG_RIDEDF/SG_MACROZONA_G1BRA.gpkg', 'CodigoMacrozona'),
         'FORTALEZA': ('06', 'Zoneamento_RMF.shp', 'NAME'),
         'BAIXADA SANTISTA': ('07', 'Zoneamento_RMBS.shp', 'ZONA_ID'),
         'GOIANIA': ('13', 'Zoneamento_RMG.shp', 'ZONA_ID'),
         'MANAUS': ('17', 'Zoneamento_RMM.shp', 'ZONA_ID'),
         'TERESINA': ('19', 'ZONAS_RIDE_TERESINA.shp', 'ZONA_ID')}


def read_matrix(prefix):
    folder = glob.glob(os.path.join(SRC, prefix + ' *'))[0]
    files = sorted(glob.glob(folder + '/**/*v2.xlsx', recursive=True)) or \
        sorted(glob.glob(folder + '/**/*.xlsx', recursive=True))
    m = pd.read_excel(files[0], header=None)
    first = next(i for i in range(6) if pd.notna(pd.to_numeric(m.iat[i, 0], errors='coerce')))
    m = m.iloc[first:, :9].apply(pd.to_numeric, errors='coerce')
    m.columns = COLUMNS
    m = m.dropna(subset=['o', 'd'])
    m[['o', 'd']] = m[['o', 'd']].astype(int)
    for s in STATES:
        bad = (m[s] >= 9999) | ((m[s] <= 0) & (m.o != m.d)) | (m[s] < 0)
        m.loc[bad, s] = np.nan
    return m, folder


def read_zones(folder, zone_file, column):
    path = zone_file if os.path.exists(zone_file) else glob.glob(folder + '/**/' + zone_file, recursive=True)[0]
    z = gpd.read_file(path).to_crs(METRIC)
    z['zone'] = pd.to_numeric(z[column], errors='coerce')
    z = z.dropna(subset=['zone'])
    z['zone'] = z.zone.astype(int)
    return z.dissolve(by='zone').reset_index()[['zone', 'geometry']]


def weights(reg, zones, zone_trips):
    """Region x zone weight matrices (area, trips), rows summing to 1 over covered pieces"""
    r = gpd.GeoDataFrame({'region_id': reg.region_id}, geometry=reg['urban'].values, crs=METRIC)
    z = zones.copy()
    z['zone_area'] = z.area
    cut = gpd.overlay(r, z, how='intersection', keep_geom_type=True)
    cut['area'] = cut.area
    cut['trips'] = cut.area / cut.zone_area * cut.zone.map(zone_trips).fillna(0.0)
    out = {}
    for w in ('area', 'trips'):
        t = cut.pivot_table(index='region_id', columns='zone', values=w, aggfunc='sum', fill_value=0.0)
        t = t.reindex(index=reg.region_id, columns=zones.zone, fill_value=0.0)
        out[w] = t.div(t.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    return out


def region_times(w_o, w_d, m, state, zone_ids):
    idx = {z: i for i, z in enumerate(zone_ids)}
    n = len(zone_ids)
    t = np.zeros((n, n))
    mask = np.zeros((n, n))
    ok = m[m.o.isin(idx) & m.d.isin(idx) & m[state].notna()]
    oi, di = ok.o.map(idx).to_numpy(), ok.d.map(idx).to_numpy()
    t[oi, di] = ok[state].to_numpy()
    mask[oi, di] = 1.0
    wo, wd = w_o.to_numpy(), w_d.to_numpy()
    num = wo @ (t * mask) @ wd.T
    den = wo @ mask @ wd.T
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def fill(wide, car):
    """Fill missing transit pairs (reverse pair, then car time x ratio) and add the 'car' state. Returns counts."""
    counts = {}
    car = car.reindex(wide.index)
    off = wide.index.get_level_values(0) != wide.index.get_level_values(1)
    ratio = float(np.nanmedian((wide.base / car)[off & car.notna().to_numpy()]))
    for s in STATES:
        missing = wide[s].isna()
        reverse = wide[s].copy()
        reverse.index = reverse.index.swaplevel()
        reverse = reverse.reindex(wide.index)
        wide.loc[missing, s] = reverse[missing]
        still = wide[s].isna()
        wide.loc[still, s] = car[still] * ratio
        diagonal = wide.index.get_level_values(0) == wide.index.get_level_values(1)
        left = wide[s].isna() & diagonal
        wide.loc[left, s] = float(wide.loc[diagonal, s].median())
        counts[s] = (int(missing.sum()), int(still.sum()), int(left.sum()))
    missing_car = car.isna()
    wide['car'] = car.where(~missing_car, wide.base / ratio)
    counts['car'] = int(missing_car.sum())
    counts['ratio'] = round(ratio, 2)
    return counts


def convert(acp):
    prefix, zone_file, column = ZONES[acp]
    m, folder = read_matrix(prefix)
    zones = read_zones(folder, zone_file, column)
    reg = regions(acp)
    m['trips'] = m.ti_p.fillna(0) + m.tpc_p.fillna(0)
    produced, attracted = m.groupby('o').trips.sum(), m.groupby('d').trips.sum()
    w_o = weights(reg, zones, produced)
    w_d = weights(reg, zones, attracted)
    ids = reg.region_id.tolist()
    car = pd.read_parquet(os.path.join(OUT, 'car', 'car_%s.parquet' % acp.replace(' ', '_')))
    car = car.set_index(['ap_orig', 'ap_dest']).minutes
    covered = int((w_o['area'].sum(axis=1) > 0).sum())
    for w in ('area', 'trips'):
        wide = pd.DataFrame({s: region_times(w_o[w], w_d[w], m, s, zones.zone.tolist()).ravel() for s in STATES},
                            index=pd.MultiIndex.from_product([ids, ids], names=['ap_orig', 'ap_dest']))
        raw_missing = wide.base.isna().mean()
        counts = fill(wide, car)
        long = wide.reset_index().melt(id_vars=['ap_orig', 'ap_dest'], var_name='state', value_name='minutes')
        long.insert(0, 'acp', acp)
        os.makedirs(OUT, exist_ok=True)
        long.to_parquet(os.path.join(OUT, 'travel_times_%s__%s.parquet' % (acp.replace(' ', '_'), w)), index=False)
        d = wide.padrao - wide.base
        print('%-17s %-5s regions %d covered %d pairs %d missing base %.3f | base mean %.1f, padrao - base %.1f'
              ' (faster %.2f), otimizado - base %.1f, car mean %.1f | filled (missing, by car, left) %s' %
              (acp, w, len(ids), covered, len(wide), raw_missing, wide.base.mean(), d.mean(),
               (d < -.5).mean(), (wide.otimizado - wide.base).mean(), wide.car.mean(), counts), flush=True)


if __name__ == '__main__':
    for a in (sys.argv[1:] or ZONES):
        convert(a)
