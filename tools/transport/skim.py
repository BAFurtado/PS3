"""
AP-to-AP transit travel-time matrices with r5py (text_bndes).

Origins and destinations are the pieces of each AP cut by ENMU's traffic zones (SG_ZONATRAFEGO_1), each placed at a
point inside it and weighted by the zone's 2022 population times the piece's share of zone area. The AP-to-AP time is
the population-weighted mean over origin pieces x destination pieces; a piece to itself is excluded, so an AP with one
piece has no diagonal. Routing: transit + walk, weekday, departures every minute from DEPARTURE over WINDOW, median
over departures, MAX_TIME cap. Pairs unreachable within the cap are left out of the mean and counted in `unreached`.

    python -m tools.transport.skim current base nec
writes input/bndes/skims/pieces_<state>.parquet (piece level) and input/bndes/skims/ap_<state>.parquet.
"""
import datetime
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

OSM = 'input/bndes/osm/ride_df_roads.osm.pbf'
GTFS = 'input/bndes/gtfs'
OUT = 'input/bndes/skims'
PARTNER = 'input/bndes/travel_times_areapond_DF.parquet'
METRIC = 31983
DEPARTURE = datetime.datetime(2026, 9, 30, 6, 0)
WINDOW = datetime.timedelta(hours=2)
MAX_TIME = datetime.timedelta(minutes=180)
WALK_KMH = 4.5
MIN_PIECE_POP = 50


def ap_polygons():
    """The partner matrix's 112 AP codes. PS3 AP shapes where they exist (all 84 of ACP BRASILIA); other codes are
    municipalities outside the ACP, represented by the whole municipality (several codes can share one polygon)."""
    codes = sorted(pd.read_parquet(PARTNER).code_weighting_orig.unique())
    shp = pd.concat([gpd.read_file('input/shapes/2010/areas/%s.shp' % uf) for uf in ('DF', 'GO', 'MG')])
    shp['ap'] = shp.id.astype(str)
    shp = shp[shp.ap.isin(codes)][['ap', 'geometry']].to_crs(METRIC)
    mun = gpd.read_file('input/bndes/SIG_RIDEDF/SG_MUNICIPIO_G1BRA.gpkg').to_crs(METRIC)
    mun['Codigo'] = mun.Codigo.astype(str)
    rest = [(c, mun.geometry[mun.Codigo == c[:7]].iloc[0]) for c in codes if c not in set(shp.ap)]
    rest = gpd.GeoDataFrame(rest, columns=['ap', 'geometry'], crs=METRIC)
    rest['whole_municipality'] = True
    out = pd.concat([shp.assign(whole_municipality=False), rest], ignore_index=True)
    missing = set(codes) - set(out.ap)
    if missing:
        raise ValueError('no polygon for %s' % sorted(missing))
    return out


def pieces():
    aps = ap_polygons()
    z = gpd.read_file('input/bndes/SIG_RIDEDF/SG_ZONATRAFEGO_1_G1BRA.gpkg').to_crs(METRIC)
    z['zone'] = np.arange(len(z))
    z['zone_area'] = z.area
    # Municipality stand-ins can repeat a polygon; cut once per distinct polygon and copy.
    aps['geom_key'] = aps.geometry.apply(lambda g: g.wkb)
    uniq = aps.drop_duplicates('geom_key')[['geom_key', 'geometry']]
    cut = gpd.overlay(uniq, z[['zone', 'zone_area', 'Pop_22', 'geometry']], how='intersection', keep_geom_type=True)
    cut['pop'] = cut.Pop_22 * cut.area / cut.zone_area
    cut = cut.merge(aps[['ap', 'geom_key']], on='geom_key').drop(columns='geom_key')
    cut = cut[cut['pop'] >= MIN_PIECE_POP].reset_index(drop=True)
    cut['geometry'] = cut.representative_point()
    cut['id'] = ['%s_%d' % (a, zz) for a, zz in zip(cut.ap, cut.zone)]
    lost = set(aps.ap) - set(cut.ap)
    if lost:
        raise ValueError('APs with no populated piece: %s' % sorted(lost))
    return cut[['id', 'ap', 'zone', 'pop', 'geometry']].to_crs(4326)


def network(state):
    import r5py
    feeds = [os.path.join(GTFS, 'df_semob_bus.zip'), os.path.join(GTFS, 'df_rail_%s.zip' % state)]
    return r5py.TransportNetwork(OSM, feeds)


def skim_state(state, pts):
    import r5py
    net = network(state)
    # Each piece point is routed once; pieces shared between municipality stand-in codes are deduplicated.
    uniq = pts.drop_duplicates('geometry')[['geometry']].copy()
    uniq['id'] = ['p%d' % i for i in range(len(uniq))]
    ttm = r5py.TravelTimeMatrix(net, origins=uniq, destinations=uniq, departure=DEPARTURE,
                                departure_time_window=WINDOW, transport_modes=[r5py.TransportMode.TRANSIT],
                                max_time=MAX_TIME, speed_walking=WALK_KMH, snap_to_network=True, percentiles=[50])
    ttm = pd.DataFrame(ttm)
    key = pts[['id', 'geometry']].merge(uniq, on='geometry', suffixes=('', '_u'))[['id', 'id_u']]
    ttm = ttm.merge(key.rename(columns={'id': 'from', 'id_u': 'from_id'}), on='from_id')
    ttm = ttm.merge(key.rename(columns={'id': 'to', 'id_u': 'to_id'}), on='to_id')
    return ttm[['from', 'to', 'travel_time']]


def aggregate(piece_ttm, pts):
    w = pts.set_index('id')[['ap', 'pop']]
    t = piece_ttm[piece_ttm['from'] != piece_ttm['to']]
    t = t.join(w.add_suffix('_o'), on='from').join(w.add_suffix('_d'), on='to')
    t['w'] = t.pop_o * t.pop_d
    t['reached'] = t.travel_time.notna()
    g = t.groupby(['ap_o', 'ap_d'])
    r = t[t.reached].assign(wt=lambda x: x.w * x.travel_time).groupby(['ap_o', 'ap_d'])
    out = pd.DataFrame({'minutes': r.wt.sum() / r.w.sum(),
                        'unreached': 1 - t[t.reached].groupby(['ap_o', 'ap_d']).w.sum() / g.w.sum()})
    out['unreached'] = out.unreached.fillna(1.0)
    return out.rename_axis(['ap_orig', 'ap_dest']).reset_index()


def main(states):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'pieces.gpkg')
    if os.path.exists(path):
        pts = gpd.read_file(path)
    else:
        pts = pieces()
        pts.to_file(path)
    print('%d pieces, %d APs, pop %.0f' % (len(pts), pts.ap.nunique(), pts['pop'].sum()), flush=True)
    for state in states:
        t0 = datetime.datetime.now()
        piece = skim_state(state, pts)
        piece.to_parquet(os.path.join(OUT, 'pieces_%s.parquet' % state))
        ap = aggregate(piece, pts)
        ap.to_parquet(os.path.join(OUT, 'ap_%s.parquet' % state))
        print(state, 'done in', datetime.datetime.now() - t0, 'pairs', len(ap), 'reached share %.3f' %
              piece.travel_time.notna().mean(), flush=True)


if __name__ == '__main__':
    sys.argv += ['--max-memory', '6G'] if '--max-memory' not in sys.argv else []
    states = [a for a in sys.argv[1:] if not a.startswith('--') and not a.endswith('G')]
    main(states or ['current', 'base', 'nec'])
