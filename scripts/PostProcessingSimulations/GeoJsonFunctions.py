import numpy as np
from OSMConstants import *
import os
import osmnx as ox
def Mph2Kmh(Mph):
    return Mph * 1.60934

def StrMph2IntMph(StrMph):
    if type(StrMph) == list:
        StrMph = StrMph[0]
        if type(StrMph) == str:
            if len(StrMph.split(' '))==1:
                Mph = int(StrMph)
                if np.isnan(Mph):
                    Mph = 50

            elif len(StrMph.split(' '))==2:
                Mph = int(StrMph.split(' ')[0])
            else:
                Mph = 0
        elif type(StrMph) == int:
            Mph = StrMph
    elif type(StrMph) == str:
        if len(StrMph.split(' '))==1:
            Mph = int(StrMph)
            if np.isnan(Mph):
                Mph = 50

        elif len(StrMph.split(' '))==2:
            Mph = int(StrMph.split(' ')[0])
        else:
            Mph = 0
    
    elif type(StrMph) == int:
        Mph = StrMph
    elif type(StrMph) == float:
        if np.isnan(StrMph):
            Mph = 0
        else:
            Mph = int(StrMph)
    else:
        Mph = 0

    return Mph

def ExtractUniqueValuesHighwayFromGeojson(Geojson):
    ListUnique = np.array([])
    for i in Geojson['highway'].to_numpy():
        if isinstance(i,list):
            if i[0] not in ListUnique:
                ListUnique = np.append(ListUnique,i[0])
            else:
                pass
        else:
            if i not in ListUnique:
                ListUnique = np.append(ListUnique,i)
            else:
                pass
    return ListUnique

def extract_lanes_recursively(lanes, symbols):
    for symbol in symbols:
        if symbol in lanes:
            lanes = lanes.split(symbol)[0]
            return extract_lanes_recursively(lanes, symbols)
    return int(lanes)

def LanesInt(Lanes):
    Symbols = [' ','|',';','/','\\',',']
    # If Lanes is a Str, 
    if type(Lanes) == str:
        # "Number ..." -> int("Number")
        Lanes = extract_lanes_recursively(Lanes, Symbols)
        if type(Lanes) == int:
            pass
    elif type(Lanes) == list:
        Lanes = Lanes[0]
        Lanes = extract_lanes_recursively(Lanes, Symbols)
    else:
        Lanes = 0
    return Lanes

def Highway2Capacity(Highway,Lanes,osm_road_capacities_us):
    if type(Highway) == list:
        Highway = Highway[0]
    if Highway in osm_road_capacities_us.keys():
        Capacity = osm_road_capacities_us[Highway]*Lanes
    else:
        Capacity = 0
    return Capacity

def CleanGeojson(Geojson):
    """
        Description:
            - Clean the Geojson
        Args:
            - Geojson: gpd.DataFrame -> u,v, uv, geometry, highway, lanes, maxspeed, capacity
        Returns:
            - Geojson: gpd.DataFrame -> u,v, uv, geometry, highway, lanes, maxspeed, capacity
        Columns:
            u	v	osmid	lanes	ref	name	highway	maxspeed	width	oneway	reversed	length	geometry	bridge	access	junction	tunnel	uv	maxspeed_int	capacity
    """
    Geojson['maxspeed_int'] = Geojson['maxspeed'].apply(StrMph2IntMph)
    Geojson["maxspeed_kmh"] = Geojson["maxspeed_int"].apply(Mph2Kmh)
    Geojson["maxspeed_kmh"] = Geojson["maxspeed_kmh"].apply(lambda x: 35* 1.60934 if x == 0 else x)
    Geojson['lanes'] = Geojson['lanes'].apply(LanesInt)
    Geojson["capacity"] = Geojson.apply(lambda x: Highway2Capacity(x['highway'],x['lanes'],osm_road_capacities_us),axis=1)
    if "u" in Geojson.columns:
        Geojson = Geojson.drop(["u","v","osmid","ref","name","highway","maxspeed","width","oneway","reversed","bridge","access","junction","tunnel"],axis = 1)
    return Geojson

def GetGeopandas(File):
    """
        Input:
            File: Path to the Graphml File
        Usage: OutputStats.GetGeopandas() for each R,UCI of the simulation.
        NOTE: So far is waste of resources since the GeoJson is the same for all the simulations.

    """
    if os.path.isfile(File):
        G = ox.load_graphml(File)
#        GeoJson = ox.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
        GeoJsonNodes = ox.graph_to_gdfs(G, nodes=True, edges=False, node_geometry=True, fill_edge_geometry=True)
        GeoJsonEdges = ox.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True).reset_index().set_index("uniqueid")#.drop(columns = ["key"])
        GeoJsonEdges["uv"] = GeoJsonEdges.index
        Bool = True
    else:
        GeoJsonNodes = None
        GeoJsonEdges = None
        Bool = False
    return GeoJsonNodes,GeoJsonEdges,Bool
