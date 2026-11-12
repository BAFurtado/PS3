"""Region-to-region car travel times with r5py, no congestion.

Network: the Geofabrik extract for the ACP's macro-region, cut to the ACP bounding box plus BUFFER_KM and to ways
tagged highway. One point per PS3 region, inside its urban part (tools/transport/regions.py). Departure
DEPARTURE, CAR mode, MAX_TIME cap. A region to itself gets no time.

    python -m tools.transport.car_skim <geofabrik dir> <osmium binary> ACP [ACP ...]
writes input/transport/car/car_<ACP>.parquet with columns acp, ap_orig, ap_dest, minutes.
"""
import datetime
import os
import subprocess
import sys

import pandas as pd

from tools.transport.regions import regions

OUT = 'input/transport/car'
DEPARTURE = datetime.datetime(2026, 11, 11, 7, 0)
MAX_TIME = datetime.timedelta(minutes=240)
BUFFER_KM = 10
GEOFABRIK = {'BRASILIA': 'centro-oeste', 'GOIANIA': 'centro-oeste', 'SALVADOR': 'nordeste', 'FORTALEZA': 'nordeste',
             'TERESINA': 'nordeste', 'MANAUS': 'norte', 'BAIXADA SANTISTA': 'sudeste', 'CURITIBA': 'sul'}


def name(acp):
    return acp.replace(' ', '_')


def cut_osm(acp, reg, geofabrik_dir, osmium):
    os.makedirs(os.path.join(OUT, 'osm'), exist_ok=True)
    roads = os.path.join(OUT, 'osm', '%s_roads.osm.pbf' % name(acp))
    if os.path.exists(roads):
        return roads
    box = reg.to_crs(4326).total_bounds
    pad = BUFFER_KM / 100.0
    bbox = '%f,%f,%f,%f' % (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
    src = os.path.join(geofabrik_dir, '%s.osm.pbf' % GEOFABRIK[acp])
    box_file = os.path.join(OUT, 'osm', '%s_box.osm.pbf' % name(acp))
    subprocess.run([osmium, 'extract', '-b', bbox, '-s', 'smart', '-O', '-o', box_file, src], check=True)
    subprocess.run([osmium, 'tags-filter', '-O', '-o', roads, box_file, 'w/highway'], check=True)
    os.remove(box_file)
    return roads


def skim(acp, geofabrik_dir, osmium):
    import geopandas as gpd
    import r5py
    reg = regions(acp)
    roads = cut_osm(acp, reg, geofabrik_dir, osmium)
    pts = gpd.GeoDataFrame({'id': reg.region_id}, geometry=reg['point'].values, crs=reg.crs).to_crs(4326)
    net = r5py.TransportNetwork(roads)
    ttm = r5py.TravelTimeMatrix(net, origins=pts, destinations=pts, departure=DEPARTURE,
                                transport_modes=[r5py.TransportMode.CAR], max_time=MAX_TIME, snap_to_network=True)
    t = pd.DataFrame(ttm).rename(columns={'from_id': 'ap_orig', 'to_id': 'ap_dest', 'travel_time': 'minutes'})
    t = t[t.ap_orig != t.ap_dest]
    t.insert(0, 'acp', acp)
    os.makedirs(OUT, exist_ok=True)
    t.to_parquet(os.path.join(OUT, 'car_%s.parquet' % name(acp)), index=False)
    print(acp, 'regions', len(reg), 'pairs', len(t), 'unreached %.3f' % t.minutes.isna().mean(),
          'median %.1f min' % t.minutes.median(), flush=True)


if __name__ == '__main__':
    sys.argv += ['--max-memory', '6G'] if '--max-memory' not in sys.argv else []
    args = [a for a in sys.argv[1:] if not a.startswith('--') and a != '6G']
    for a in args[2:]:
        skim(a, args[0], args[1])
