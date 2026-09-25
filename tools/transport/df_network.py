"""
DF transit network as GTFS, built from open SEMOB data plus synthetic ENMU projects (text_bndes, W2 on DF).

    python -m tools.transport.df_network            # writes input/bndes/gtfs/*.zip

Outputs
- df_semob_bus.zip: every SEMOB bus departure valid on SERVICE_DATE (a Wednesday). Shapes are the SEMOB
  itineraries and stops are the active 2025 stops that lie within SNAP_M of an itinerary, ordered along it.
  Stop times are interpolated along the itinerary from the scheduled trip duration (`tempo_percurso`).
- df_rail_<state>.zip: Metrô-DF plus the ENMU projects open in that state, as explicit trips at ENMU headways.
  States: 'current' (Metrô today), 'base' (+ ENMU Rede Base: S1F), 'nec' (+ Rede Necessária Padrão: S3F),
  and 'only_<code>' (base + a single Necessária project).

Sources for every synthetic parameter are in PROJECTS and METRO below. What is not modelled:
- ANTT semiurban buses in the Entorno (not in SEMOB).
- Express lines are allowed to board at every stop they pass (SEMOB does not say which stops they skip).
- Projects that extend one another (VLT 20104 on 20105, BRT 20107/20110 on BRT Oeste) run as separate lines,
  so riders pay a transfer that ENMU's through-running would not charge.
- No congestion or crowding: timetable times only.
"""
import os

from itertools import combinations

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.ops import linemerge

from tools.transport.gtfs_write import Feed, offsets_from_speed

SEMOB = 'input/bndes/semob'
SIG = 'input/bndes/SIG_RIDEDF'
OUT = 'input/bndes/gtfs'
METRIC = 31983  # SIRGAS 2000 / UTM 23S
SERVICE_DATE = pd.Timestamp('2026-09-30')  # Wednesday
WEEKDAY_CHAR = SERVICE_DATE.dayofweek      # position in SEMOB's 'SSSSSNN' (Monday first)
SNAP_M = 40         # stop-to-itinerary distance for a stop to be served by a line
REVISIT_GAP_M = 300  # a stop passed again after this much line distance is a second visit
NODE_SNAP_M = 25     # project-layer endpoints closer than this are one node
TERMINAL_M = 200     # add a terminal stop at a line end with no SEMOB stop this close
SYNTH_START, SYNTH_END = 5 * 3600, 11 * 3600  # synthetic service window (queries depart 06:00-08:00)

# Metrô-DF today. R2 Tab. 3 p.21 (Metrô-DF data): Ceilândia branch 10.5 and Samambaia branch 5.3 departures/h in the
# morning peak, operating speed 34.7 km/h.
METRO = {'speed': 34.7, 'branches': {'verde': ('bln_verde', 60 / 10.5), 'laranja': ('bln_laranja', 60 / 5.3)}}

# ENMU projects. Padrão parameters come from ProjetosSIG (stations, `Intervalo na hora pico`,
# `Velocidade operacional estimada`). The Rede Base projects are absent from ProjetosSIG: 10104 and 10103 from
# R2 Tab. 3 p.21, 20109 from R2 Tab. 4 p.25 (BRT Eixo Sul) with the station count from the exec report p.35.
# Metro extensions take the branch headway. `extends` names the metro branch ('trunk' = both).
PROJECTS = pd.DataFrame([
    # code, set, gpkg, mode, stations, headway_min, speed_kmh, extends
    (10104, 'base', 'SG_PROJ_B_BRT_1', 'brt', 41, 2.0, 27.6, None),
    (20109, 'base', 'SG_PROJ_B_BRT_4', 'brt', 3, 0.37, 29.6, None),
    (10103, 'base', 'SG_PROJ_B_METRO_1', 'metro', 2, None, 34.7, 'laranja'),
    (20101, 'nec', 'SG_PROJ_P_METRO_2', 'metro', 2, None, 34.7, 'verde'),
    (20102, 'nec', 'SG_PROJ_P_METRO_3', 'metro', 1, None, 34.7, 'trunk'),
    (20103, 'nec', 'SG_PROJ_P_VLT_1', 'vlt', 33, 3.2, 25, None),
    (20104, 'nec', 'SG_PROJ_P_VLT_2', 'vlt', 8, 3.2, 25, None),
    (20105, 'nec', 'SG_PROJ_P_VLT_3', 'vlt', 25, 3.3, 25, None),
    (20106, 'nec', 'SG_PROJ_P_VLT_4', 'vlt', 27, 12, 25, None),
    (20107, 'nec', 'SG_PROJ_P_BRT_2', 'brt', 7, 0.9, 25, None),
    (20108, 'nec', 'SG_PROJ_P_BRT_3', 'brt', 16, 0.8, 30, None),
    (20110, 'nec', 'SG_PROJ_P_BRT_5', 'brt', 15, 1.0, 25, None),
    # 20111 BRT Sudoeste is the alternative to VLT Linha 2 (20103) and is left out of the Necessária (B2 p.22).
    (20111, 'alt', 'SG_PROJ_P_BRT_6', 'brt', 12, 1.1, 25, None),
    (20112, 'nec', 'SG_PROJ_P_BRT_7', 'brt', 10, 0.9, 30, None),
    (20113, 'nec', 'SG_PROJ_P_BRT_8', 'brt', 8, 1.0, 30, None),
    (20114, 'nec', 'SG_PROJ_P_BRT_9', 'brt', 21, 0.9, 25, None),
], columns=['code', 'set', 'gpkg', 'mode', 'stations', 'headway', 'speed', 'extends'])
ROUTE_TYPE = {'metro': 1, 'vlt': 0, 'brt': 3}


def wgs(points_metric):
    xy = gpd.GeoSeries(points_metric, crs=METRIC).to_crs(4326)
    return [(p.x, p.y) for p in xy]


# ----------------------------------------------------------------------------------------------- SEMOB buses

def load_semob():
    h = pd.DataFrame(gpd.read_file(os.path.join(SEMOB, 'Horários_das_Linhas.geojson')).drop(columns='geometry'))
    start = pd.to_datetime(h.dt_inicio_vigencia)
    end = pd.to_datetime(h.dt_final_vigencia)
    h = h[(h.dias_semana.str[WEEKDAY_CHAR] == 'S') & (start <= SERVICE_DATE) & (end.isna() | (end >= SERVICE_DATE))]
    it = gpd.read_file(os.path.join(SEMOB, 'itinerario_espacial.geojson')).to_crs(METRIC)
    it['sentido'] = it.lin_sentido.str[0]
    stops = gpd.read_file(os.path.join(SEMOB, 'ponto_parada_v2025.geojson'))
    stops = stops[stops.parada_ativa].to_crs(METRIC).reset_index(drop=True)
    return h, it, stops


def stop_visits(line, tree, stop_ids):
    """(position along line in m, stop id) for every pass of the line within SNAP_M of a stop."""
    d = np.append(np.arange(0, line.length, 10.0), line.length)
    pts = shapely.line_interpolate_point(line, d)
    xy = shapely.get_coordinates(pts)
    near = tree.query_ball_point(xy, r=SNAP_M)
    rec = [(s, i) for i, ss in enumerate(near) for s in ss]
    if not rec:
        return []
    rec = pd.DataFrame(rec, columns=['s', 'i'])
    rec['dist'] = np.hypot(*(xy[rec.i] - tree.data[rec.s]).T)
    visits = []
    gap = REVISIT_GAP_M / 10
    for s, g in rec.sort_values('i').groupby('s'):
        cluster = (g.i.diff() > gap).cumsum()
        for _, c in g.groupby(cluster):
            visits.append((d[c.i.iloc[c.dist.argmin()]], stop_ids[s]))
    visits.sort()
    out = []
    for v in visits:
        if not out or out[-1][1] != v[1]:
            out.append(v)
    return out


def build_bus(feed):
    h, it, stops = load_semob()
    tree = cKDTree(shapely.get_coordinates(stops.geometry))
    stop_ids = ['S%d' % c for c in stops.cod_parada_v2025]
    for sid, (lon, lat) in zip(stop_ids, wgs(stops.geometry)):
        feed.add_stop(sid, lon, lat, sid)
    # Fallback duration for trips with tempo_percurso == 0: the median scheduled speed.
    speeds = []
    report = []
    groups = h.groupby(['id_linha', 'sentido'])
    it_idx = it.set_index(['id_linha', 'sentido'])
    lines = {}
    for (id_linha, sentido), g in groups:
        if (id_linha, sentido) not in it_idx.index:
            report.append((id_linha, sentido, 'no itinerary'))
            continue
        row = it_idx.loc[(id_linha, sentido)]
        line = row.geometry
        visits = stop_visits(line, tree, stop_ids)
        term = []
        if not visits or visits[0][0] > TERMINAL_M:
            term.append((0.0, 'T%d%s_a' % (id_linha, sentido)))
        if not visits or line.length - visits[-1][0] > TERMINAL_M:
            term.append((line.length, 'T%d%s_b' % (id_linha, sentido)))
        for pos, sid in term:
            (lon, lat), = wgs([line.interpolate(pos)])
            feed.add_stop(sid, lon, lat, 'terminal %d%s' % (id_linha, sentido))
        visits = sorted(visits + term)
        lines[(id_linha, sentido)] = (line.length, visits, g)
        ok = g.tempo_percurso > 0
        speeds += list(line.length / (g.tempo_percurso[ok] * 60))
    v_med = float(np.median(speeds))
    routes = set()
    n_fallback = 0
    for (id_linha, sentido), (length, visits, g) in lines.items():
        route_id = 'B%d' % id_linha
        if route_id not in routes:
            routes.add(route_id)
            feed.add_route(route_id, str(g.cd_linha.iloc[0]), '', 3)
        pos = np.array([v[0] for v in visits])
        ids = [v[1] for v in visits]
        for k, r in enumerate(g.itertuples()):
            dur = r.tempo_percurso * 60.0
            if dur <= 0:
                dur, n_fallback = length / v_med, n_fallback + 1
            hh, mm = map(int, r.hr_prevista.split(':')[:2])
            offsets = pos / length * dur
            feed.add_trip(route_id, 'B%d%s_%s_%d' % (id_linha, sentido, r.hr_prevista.replace(':', ''), k),
                          {'I': 0, 'V': 1, 'C': 0}[sentido], ids, offsets, hh * 3600 + mm * 60)
    served = {s for _, v, _ in lines.values() for _, s in v}
    stats = dict(line_directions=len(lines), departures=int(sum(len(g) for *_, g in lines.values())),
                 median_speed_kmh=round(v_med * 3.6, 1), zero_duration_trips=n_fallback,
                 semob_stops_served=len({s for s in served if s.startswith('S')}), semob_stops=len(stops),
                 unmatched=report)
    return stats


# ----------------------------------------------------------------------------------------------- rail + projects

def chain(parts):
    """One LineString through all parts, joining nearest endpoints greedily (for layers with no leaf pair)."""
    parts = sorted(parts, key=lambda p: -p.length)
    coords = list(parts.pop(0).coords)
    while parts:
        best = None
        for i, p in enumerate(parts):
            c = list(p.coords)
            for rev in (False, True):
                cc = c[::-1] if rev else c
                for at_end in (True, False):
                    a = coords[-1] if at_end else coords[0]
                    b = cc[0] if at_end else cc[-1]
                    dist = np.hypot(a[0] - b[0], a[1] - b[1])
                    if best is None or dist < best[0]:
                        best = (dist, i, cc, at_end)
        _, i, cc, at_end = best
        parts.pop(i)
        coords = coords + cc if at_end else cc + coords
    return LineString(coords)


def project_graph(gpkg):
    """The project layer as a graph: nodes are part endpoints snapped within NODE_SNAP_M, edges carry geometry."""
    g = gpd.read_file(os.path.join(SIG, gpkg + '_G1BRA.gpkg')).to_crs(METRIC)
    merged = shapely.union_all(g.geometry.values)  # also nodes crossing lines
    if merged.geom_type != 'LineString':
        merged = linemerge(merged)
    parts = [merged] if merged.geom_type == 'LineString' else list(merged.geoms)
    nodes = []

    def node(xy):
        for i, n in enumerate(nodes):
            if np.hypot(xy[0] - n[0], xy[1] - n[1]) < NODE_SNAP_M:
                return i
        nodes.append(xy)
        return len(nodes) - 1

    G = nx.MultiGraph()
    for p in parts:
        u, v = node(p.coords[0]), node(p.coords[-1])
        if u == v and p.length < 2 * NODE_SNAP_M:
            continue
        G.add_edge(u, v, geom=p, length=p.length, u=u)
    for i, xy in enumerate(nodes):
        if i in G:
            G.nodes[i]['xy'] = xy
    return G, parts


def project_routes(gpkg, n_stations):
    """Routes of a (possibly branched) project and their stations.

    One route per pair of leaves (shortest path). Stations sit on every node plus evenly inside each edge at the
    spacing that gives n_stations over the whole layer, so routes sharing a trunk share its stations.
    Returns [(LineString, [(station_id, position_m)])]; a layer with fewer than two leaves is chained into one route.
    """
    G, parts = project_graph(gpkg)
    leaves = [n for n in G if G.degree(n) == 1]
    total = sum(d['length'] for *_, d in G.edges(data=True))
    spacing = total / max(n_stations - 1, 1)
    if len(leaves) < 2:
        line = chain(parts)
        return [(line, [('s%02d' % k, p) for k, p in enumerate(np.linspace(0, line.length, n_stations))])]
    H = nx.Graph()
    for u, v, k, d in G.edges(keys=True, data=True):
        if not H.has_edge(u, v) or H[u][v]['length'] > d['length']:
            H.add_edge(u, v, length=d['length'], key=(u, v, k), geom=d['geom'], u=d['u'])
    routes = []
    for a, b in combinations(leaves, 2):
        path = nx.shortest_path(H, a, b, weight='length')
        coords, stops, offset = [], [('n%d' % a, 0.0)], 0.0
        for x, y in zip(path[:-1], path[1:]):
            e = H[x][y]
            forward = e['u'] == x
            geom = e['geom'] if forward else e['geom'].reverse()
            n_in = max(0, int(round(e['length'] / spacing)) - 1)
            for j in range(1, n_in + 1):
                jj = j if forward else n_in + 1 - j  # same physical station whichever way the edge is run
                stops.append(('e%d_%d_%d_%d' % (*e['key'], jj), offset + e['length'] * j / (n_in + 1)))
            offset += e['length']
            stops.append(('n%d' % y, offset))
            coords += list(geom.coords)[1 if coords else 0:]
        routes.append((LineString(coords), stops))
    return routes


def project_line(gpkg):
    """A single-line project (metro extensions)."""
    routes = project_routes(gpkg, 2)
    if len(routes) != 1:
        raise ValueError('%s is branched' % gpkg)
    return routes[0][0]


def metro_branches(extensions):
    """{branch: [(name, Point)]} ordered outer end -> Central, with extension stations added."""
    segs = gpd.read_file(os.path.join(SEMOB, 'linha_metro.geojson')).to_crs(METRIC)
    st = gpd.read_file(os.path.join(SEMOB, 'estacoes_metro.geojson')).to_crs(METRIC)
    st = st[st.bln_ativo]
    out = {}
    for branch, (col, _) in METRO['branches'].items():
        line = linemerge(shapely.union_all(segs[segs[col]].geometry.values))
        on = st[st.geometry.distance(line) < 50]
        on = on.assign(pos=on.geometry.apply(line.project)).sort_values('pos')
        if on.nom_estacao.iloc[-1] != 'Central':
            on = on.iloc[::-1]
        out[branch] = [(n, p) for n, p in zip(on.nom_estacao, on.geometry)]
    for code, n_st, extends in extensions:
        ext = project_line(PROJECTS.set_index('code').gpkg[code])
        for branch in out:
            if extends not in (branch, 'trunk'):
                continue
            at_central = extends == 'trunk'
            anchor = out[branch][-1 if at_central else 0][1]
            e = ext if ext.boundary.geoms[0].distance(anchor) <= ext.boundary.geoms[1].distance(anchor) else ext.reverse()
            gap = e.boundary.geoms[0].distance(anchor)
            if gap > 500:
                raise ValueError('metro extension %d starts %.0f m from %s' % (code, gap, branch))
            new = [('%d_%d' % (code, k), e.interpolate(e.length * k / n_st)) for k in range(1, n_st + 1)]
            out[branch] = out[branch] + new if at_central else new[::-1] + out[branch]
    return out


def build_rail(feed, codes):
    """Metrô-DF plus the given ENMU projects."""
    sel = PROJECTS[PROJECTS.code.isin(codes)]
    ext = [(r.code, r.stations, r.extends) for r in sel.itertuples() if r.mode == 'metro']
    branches = metro_branches(ext)
    for branch, stations in branches.items():
        headway = METRO['branches'][branch][1]
        ids = ['M_' + n.replace(' ', '_') for n, _ in stations]
        for sid, (lon, lat), (n, _) in zip(ids, wgs([p for _, p in stations]), stations):
            feed.add_stop(sid, lon, lat, n)
        pts = [p for _, p in stations]
        pos = np.cumsum([0] + [a.distance(b) for a, b in zip(pts[:-1], pts[1:])])
        off = offsets_from_speed(list(pos), METRO['speed'])
        rid = 'METRO_' + branch
        feed.add_route(rid, branch, 'Metrô-DF ' + branch, 1)
        feed.add_headway_line(rid, ids, off, headway, SYNTH_START, SYNTH_END, 0, rid)
        feed.add_headway_line(rid, ids[::-1], off, headway, SYNTH_START, SYNTH_END, 1, rid)
    for r in sel[sel['mode'] != 'metro'].itertuples():
        routes = project_routes(r.gpkg, r.stations)
        headway = r.headway * len(routes)  # the corridor headway is shared by the project's routes
        for i, (line, stops) in enumerate(routes):
            ids = ['P%d_%s' % (r.code, sid) for sid, _ in stops]
            pos = [p for _, p in stops]
            for sid, (lon, lat) in zip(ids, wgs([line.interpolate(p) for p in pos])):
                feed.add_stop(sid, lon, lat, sid)
            off = offsets_from_speed(pos, r.speed)
            rid = 'P%d_%d' % (r.code, i)
            feed.add_route(rid, str(r.code), r.gpkg, ROUTE_TYPE[r.mode])
            feed.add_headway_line(rid, ids, off, headway, SYNTH_START, SYNTH_END, 0, rid)
            feed.add_headway_line(rid, ids[::-1], off, headway, SYNTH_START, SYNTH_END, 1, rid)


def state_codes(state):
    base = list(PROJECTS.code[PROJECTS.set == 'base'])
    if state == 'current':
        return []
    if state == 'base':
        return base
    if state == 'nec':
        return base + list(PROJECTS.code[PROJECTS.set == 'nec'])
    if state.startswith('only_'):
        return base + [int(state[5:])]
    raise ValueError(state)


def new_feed(name):
    return Feed(name, name, SERVICE_DATE.replace(month=1, day=1).strftime('%Y%m%d'),
                SERVICE_DATE.replace(month=12, day=31).strftime('%Y%m%d'))


def main(states=('current', 'base', 'nec')):
    os.makedirs(OUT, exist_ok=True)
    feed = new_feed('SEMOB')
    stats = build_bus(feed)
    print('bus:', feed.write(os.path.join(OUT, 'df_semob_bus.zip')), {k: v for k, v in stats.items() if k != 'unmatched'})
    if stats['unmatched']:
        print('  no itinerary for', stats['unmatched'])
    for state in states:
        feed = new_feed('RAIL')
        build_rail(feed, state_codes(state))
        print(state + ':', feed.write(os.path.join(OUT, 'df_rail_%s.zip' % state)))


if __name__ == '__main__':
    import sys
    main(sys.argv[1:] or ('current', 'base', 'nec'))
