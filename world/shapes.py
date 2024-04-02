from collections import defaultdict

import geopandas as gpd
import pandas as pd


def prepare_shapes_2010(geo):
    # TODO. Get the shapefiles for AREAS PONDERAÇÃO new shapefiles
    urban = pd.DataFrame()
    temp = gpd.read_file('input/shapes/2010/urban_mun_2010.shp')
    for mun in geo.mun_codes:
        temp1 = temp[temp.CD_MUN == str(mun)]
        urban = pd.concat([temp1, urban])

    urban = {
        mun: urban[urban.CD_MUN == mun]['geometry'].item()
        for mun in urban.CD_MUN
    }

    codes = [str(code) for code in geo.mun_codes]

    my_shapes = gpd.GeoDataFrame(columns=['id', 'geometry'])
    states = list()
    for uf in geo.states_on_process:
        states.append(gpd.read_file(f'input/shapes/2010/areas/{uf}.shp'))

    for mun_id in codes:
        for state in states:
            for index, row in state.iterrows():
                if row['mun_code'] == mun_id:
                    shap_data = gpd.GeoDataFrame({'id': row['id'], 'geometry': row.geometry}, index=[index])
                    my_shapes = pd.concat([my_shapes, shap_data], ignore_index=True)
    return urban, my_shapes


def prepare_shapes(geo):
    """Loads shape data for municipalities"""

    # list of States codes in Brazil
    states_codes = pd.read_csv('input/STATES_ID_NUM.csv', sep=';', header=0, decimal=',')

    # creating a list of code number for each state to use in municipalities selection
    processing_states_code_list = []
    for item in geo.states_on_process:
        processing_states_code_list.append((states_codes['nummun'].loc[states_codes['codmun'] == item]).values[0])

    # load the shapefiles
    if geo.year == 2010:
        return prepare_shapes_2010(geo)
    full_region = gpd.read_file('input/shapes/mun_ACPS_ibge_2014_latlong_wgs1984_fixed.shp')
    urban_region = gpd.read_file('input/shapes/URBAN_IBGE_ACPs.shp')
    aps_region = gpd.read_file('input/shapes/APs.shp')

    urban = dict()
    urban_mun_codes = []
    # selecting the urban areas for each municipality
    for state in processing_states_code_list:
        for acp in geo.processing_acps:
            for index, row in urban_region.iterrows():
                if row['ACP'] == str(acp) and row['CD_GEOS'] == str(state):
                    urban[row['GEOCODI']] = row['geometry']
                    urban_mun_codes.append(row['GEOCODI'])

    # map municipal codes to constituent AP shapes
    mun_codes_to_ap_shapes = defaultdict(list)
    for index, row in aps_region.iterrows():
        code = row['AP']
        mun_code = code[:7]
        row['id'] = code
        row['mun_code'] = mun_code
        mun_codes_to_ap_shapes[mun_code].append(row)

    my_shapes = []
    # selection of municipalities boundaries
    # running over the states in the list
    for mun_id in urban_mun_codes:
        # for all states different from Federal district (53 code)
        # if we have AP shapes for this municipality, use those
        if mun_id in mun_codes_to_ap_shapes:
            my_shapes.extend(mun_codes_to_ap_shapes[mun_id])
        else:
            for index, row in full_region.iterrows():
                if row['CD_GEOCMU'] == mun_id:
                    shap = row
                    shap['id'] = row['CD_GEOCMU']
                    my_shapes.append(shap)

    my_shapes = gpd.GeoDataFrame(my_shapes)
    return urban, my_shapes
