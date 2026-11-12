"""PS3 regions of an ACP as polygons, as world/shapes.prepare_shapes_2010 builds them: the 2010 áreas de ponderação
of the ACP municipalities that appear in the AP population file. Each region also gets its urban part (the
municipality's 2010 urban footprint) and a point inside it for routing."""
import geopandas as gpd
import pandas as pd

METRIC = 5880


def acp_municipalities(acp):
    t = pd.read_csv('input/ACPs_MUN_CODES.csv', sep=';')
    return sorted(t.loc[t.ACPs == acp, 'cod_mun'].astype(str))


def regions(acp):
    """GeoDataFrame (metric CRS) with columns region_id, mun, geometry, urban, point"""
    muns = acp_municipalities(acp)
    ap_pops = pd.read_csv('input/num_people_age_gender_AP_2010.csv', sep=';', usecols=['AREAP'])
    with_aps = {str(c)[:7] for c in ap_pops.AREAP.unique()}
    muns = [m for m in muns if m in with_aps]
    states = pd.read_csv('input/STATES_ID_NUM.csv', sep=';')
    uf = dict(zip(states.nummun.astype(str), states.codmun))
    shp = pd.concat([gpd.read_file('input/shapes/2010/areas/%s.shp' % uf[s]) for s in sorted({m[:2] for m in muns})])
    shp = shp[shp.mun_code.astype(str).isin(muns)].to_crs(METRIC)
    out = gpd.GeoDataFrame({'region_id': shp.id.astype(str).values, 'mun': shp.mun_code.astype(str).values},
                           geometry=shp.geometry.values, crs=METRIC)
    urban = gpd.read_file('input/shapes/2010/urban_mun_2010.shp').to_crs(METRIC)
    urban = urban[urban.CD_MUN.isin(muns)].set_index('CD_MUN').geometry
    parts = []
    for geom, mun in zip(out.geometry, out.mun):
        u = geom.intersection(urban[mun]) if mun in urban.index else None
        parts.append(u if u is not None and not u.is_empty and u.area > 0 else geom)
    out['urban'] = gpd.GeoSeries(parts, crs=METRIC)
    out['point'] = out['urban'].representative_point()
    return out
