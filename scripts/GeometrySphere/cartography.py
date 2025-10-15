"""


"""
from osmnx import ox
import geopandas as gpd
from typing import Dict,List
from networkx import Graph
import ast
import pandas as pd
import statistics

## ------------- CONSTANTS ----------------------##
SPEED_DEFAULTS = {'tertiary' : {1 : 20, 2 : 20, 3 : 20, 4 : 20, -1 : 20},
            'tertiary_link' : {1 : 20, 2 : 20, 3 : 20, 4 : 20, -1 : 20},
            'secondary' : {1 : 25, 2 : 25, 3 : 25, 4 : 25, -1 : 25},
            'secondary_link' : {1 : 25, 2 : 25, 3 : 25, 4 : 25, -1 : 25},
            'primary' : {1 : 30, 2 : 30, 3 : 30, 4 : 30, -1 : 30},
            'primary_link' : {1 : 30, 2 : 30, 3 : 30, 4 : 30, -1 : 30},
            'trunk' : {1 : 45, 2 : 45, 3 : 45, 4 : 45, -1 : 45},
            'trunk_link' : {1 : 45, 2 : 45, 3 : 45, 4 : 45, -1 : 45},
            'motorway' : {1 : 50, 2 : 50, 3 : 65, 4 : 65, -1 : 57.5},
            'motorway_link' : {1 : 50, 2 : 50, 3 : 65, 4 : 65, -1 : 57.5},
            'unclassified' : {1 : 20, 2 : 20, 3 : 20, 4 : 20, -1 : 20},
            'road' : {1 : 30, 2 : 30, 3 : 30, 4 : 30, -1 : 30}}

# define per-lane capacity defaults for each hwy type and number of lanes, so we can infer when lacking data
CAPACITY_DEFAULTS = {'tertiary' : {1 : 900, 2 : 900, 3 : 900, 4 : 900, -1 : 900},
                'tertiary_link' : {1 : 900, 2 : 900, 3 : 900, 4 : 900, -1 : 900},
                'secondary' : {1 : 900, 2 : 900, 3 : 900, 4 : 900, -1 : 900},
                'secondary_link' : {1 : 900, 2 : 900, 3 : 900, 4 : 900, -1 : 900},
                'primary' : {1 : 1000, 2 : 1000, 3 : 1000, 4 : 1000, -1 : 1000},
                'primary_link' : {1 : 1000, 2 : 1000, 3 : 1000, 4 : 1000, -1 : 1000},
                'trunk' : {1 : 1900, 2 : 2000, 3 : 2000, 4 : 2000, -1 : 1975},
                'trunk_link' : {1 : 1900, 2 : 2000, 3 : 2000, 4 : 2000, -1 : 1975},
                'motorway' : {1 : 1900, 2 : 2000, 3 : 2000, 4 : 2200, -1 : 2025},
                'motorway_link' : {1 : 1900, 2 : 2000, 3 : 2000, 4 : 2200, -1 : 2025},
                'unclassified' : {1 : 800, 2 : 800, 3 : 800, 4 : 800, -1 : 800},
                'road' : {1 : 900, 2 : 900, 3 : 900, 4 : 900, -1 : 900}}

LANE_DEFAULTS = {'motorway': 4.0,
                'motorway_link': 1.0,
                'primary': 3.0,
                'primary_link': 1.0,
                'road': 1.0,
                'secondary': 3.0,
                'secondary_link': 1.0,
                'tertiary': 2.0,
                'tertiary_link': 1.0,
                'trunk': 3.0,
                'trunk_link': 1.0,
                'unclassified': 2.0}




## ------------- FROM POLYGON TO ROAD NETWORK ---------------------- ## 


def extract_graph_from_gdf_polygon(gdf_polygon:gpd.GeoDataFrame,
                                   str_network_type_ox: str = "drive"):
    """
        Description:
            From a shapefile: Usually this shapefile contain polygon or multipolygon as geometry, such multipolygons come from census usually but they can be whatever
            Extract the convex hull
            Extract the graph inside the convex hull with the properties of str_network_type_ox 
        Return:
            Graph of the road network inside the hull with not simplified edges.

    """
    # Ensure all geometries are Polygon or MultiPolygon
    if not all(g.geom_type in ["Polygon", "MultiPolygon"] for g in gdf_polygon.geometry):
        raise ValueError("All geometries in the GeoDataFrame must be Polygon or MultiPolygon.")
    # Extract polygon countour
    polygon = gdf_polygon.unary_union
    # Take the convex hull containing it
    polygon_hull = polygon.convex_hull
    # Project it in space
    polygon_hull_proj, crs = ox.projection.project_geometry(polygon_hull)
    # Consider 1 mile from the exterior
    polygon_hull_proj_buff = polygon_hull_proj.buffer(1600) #1 mile in meters
    # Project it back in the lat,lon format
    polygon_hull_buff, crs = ox.projection.project_geometry(polygon_hull_proj_buff, crs=crs, to_latlong=True)    
    # Extract the drive network that is not simplified yet
    G2 = ox.graph_from_polygon(polygon_hull_buff, network_type=str_network_type_ox, simplify=False)
    return G2

def simplify_graph_from_polygon_gdf(gdf_polygon:gpd.GeoDataFrame,
                                    types:List,
                                    index2osmid: Dict,
                                    str_network_type_ox: str = "drive",
                                    str_highway_ox_name: str = 'highway',
                                    str_unique_index_edges: str = "uniqueid",
                                    str_start_node: str = "u",
                                    str_end_node: str = "v",
                                    str_start_node_osmid: str = "osmid_u",
                                    str_end_node_osmid: str = "osmid_v"
                                    ):        
    """
        Obtain graph from 
    """
    if not gdf_polygon is None:
        # Extract the graph from the
        G2 = extract_graph_from_gdf_polygon(gdf_polygon = gdf_polygon,
                                            str_network_type_ox = str_network_type_ox)
        # Compute the minor streets to simplify the network (NOTE: specify what it means this operation)
        minor_streets = [(u, v, k) for u, v, k, d in G2.edges(keys=True, data=True) if d[str_highway_ox_name] not in types]
        # Remove minor streets
        G2.remove_edges_from(minor_streets)
        # Remove isolated nodes
        G2 = ox.utils_graph.remove_isolated_nodes(G2)
        # Choose just the largest component of the road network
        G2_connected = ox.utils_graph.get_largest_component(G2, strongly=True)
        # Simplify the network: simplifying the intersection so that at least 3 roads are leaving a node
        G2_simp = ox.simplify_graph(G2_connected, strict=True)  
        i = 0
        for u, v, k, d in G2_simp.edges(data=True, keys=True):
            d[str_unique_index_edges] = i
            # Fill the starting node with matrix index
            d[str_start_node] = u 
            # Fill the ending node with matrix index
            d[str_end_node] = v 
            # Fill the start node with osmid index
            d[str_start_node_osmid] = index2osmid[u]
            # Fill the ending node with osmid index
            d[str_end_node_osmid] = index2osmid[v]
            i += 1
    else:
        raise ValueError('The Polygon is None, need to upload a valid shape file')
    return G2_simp




## ----------------- Graph to edges, nodes -------------------- #
def from_graph_to_edges_nodes_format(G:Graph):
    if G is not None:
        nodes, edges = ox.graph_to_gdfs(G, node_geometry=False, fill_edge_geometry=False)
    else:
        raise ValueError('Graph is None')
    return nodes,edges


## ---------------------- Nodes ------------------------ #
def reindex_nodes(nodes,
                column_osmid = "osmid",
                column_x = "x",
                column_y = "y",
                column_ref = "ref",
                column_highway = "highway",
                column_index = "index"
                ):
    if nodes is not None:
        nodes = nodes.reset_index()
        nodes = nodes.reindex(columns=[column_osmid, column_x, column_y, column_ref, column_highway])
        nodes[column_index] = nodes.index
    else:
        raise ValueError('Nodes is None, need to upload a valid graph')
    return nodes


## --------------------- Edges ---------------------- ##

def process_highway(edges,
                    useful_cols,
                    column_highway = "highway",
                    column_lanes = "lanes",
                    column_max_speed = "maxspeed"):
    """
        Ensures the following standard for edge files:
        - avoid the multiple edges -> just one highway value
        -                          -> just one number of lanes

    """
    if edges is not None:
        # Adjust the multi-edges highway such that the slowest is chosen
        edges[column_highway] = edges[column_highway].map(convert_lists)
        edges[column_highway] = edges[column_highway].map(collapse_multiple_hwy_values)
        # Adjust lane columns such that just integers are there
        edges[column_lanes] = edges[column_lanes].map(convert_lists)
        edges[column_lanes] = edges[column_lanes].map(collapse_multiple_lane_values)
        edges[column_lanes] = edges[column_lanes].map(remove_hyphens)
        edges[column_lanes] = edges[column_lanes].map(remove_semicolons)
        edges[column_lanes] = edges[column_lanes].astype(float)
        # Lanes
        lane_defaults = edges.groupby(column_highway)[column_lanes].median()
        lane_defaults = lane_defaults.fillna(value=2).to_dict() #'road' type is null
        # Compute the number of lanes
        edges[column_lanes] = edges.apply(impute_lanes, axis=1).astype(int)
        edges[column_lanes] = edges.apply(allocate_lanes, axis=1)
        edges.loc[edges[column_lanes] < 1, column_lanes] = 1
        edges['lanes_capped'] = edges[column_lanes]
        edges.loc[edges['lanes_capped'] > 4, 'lanes_capped'] = 4
        edges[column_max_speed] = edges[column_max_speed].map(convert_lists)
        edges[column_max_speed] = edges[column_max_speed].map(collapse_multiple_maxspeed_values)
        edges[column_max_speed] = edges[column_max_speed].map(parse_speed_strings)
        known_speeds = edges[pd.notnull(edges[column_max_speed])][column_max_speed]
        known_speeds = known_speeds.astype(int)           
        inferred_speeds = edges[pd.isnull(edges[column_max_speed])].apply(infer_speed, axis=1)
        # merge known speeds with inferred speeds to get a free-flow speed for each edge
        edges['speed_mph'] = known_speeds._append(inferred_speeds, ignore_index=False, verify_integrity=True)
        edges['capacity_lane_hour'] = edges.apply(infer_capacity, axis=1)
        edges['capacity_hour'] = edges['capacity_lane_hour'] * edges[column_lanes]
        edges['length'] = edges['length'].round(1)
        edges.drop(columns=['u','v','key'], inplace=True)
        edges.reset_index(inplace=True)
        edges_save = edges[useful_cols]
#            edges_save['osmid_u'] = edges_save['u']
#            edges_save['osmid_v'] = edges_save['v']            
    else:
        raise ValueError('Edges is None, cannot process highway')
    return edges_save
        
def collapse_multiple_hwy_values(hwy):
    """
        Description:

    """
    if isinstance(hwy, list):
        # if we find an item in our defaults dict, use that value
        # otherwise, just use the zeroth item in the list
        for item in hwy:
            if item in SPEED_DEFAULTS.keys():
                return item
        return hwy[0]
    else:
        return hwy


def convert_lists(value):
    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        return ast.literal_eval(value) #parse str -> list
    else:
        return value
    
def collapse_multiple_lane_values(value):
    if isinstance(value, list):
        #remove the elements with semicolons
        list_ = [str(x) for x in list(range(10))]
        numeric_values = []
        for x in value:
            if x in list_:
                numeric_values.append(x)
#        numeric_values = [x for x in value if ';' not in x or ',' not in x]
        # return the mean of the values in the list
        numeric_values = [int(x) for x in numeric_values]
        return int(statistics.mean(numeric_values))
    else:
        return value

def remove_hyphens(x):
    val = x
    if isinstance(x, str):
        if '-' in x:
            val = x.split('-')[0]
    return val

def remove_semicolons(x):
    val = x
    if isinstance(x, str):
        if ';' in x:
            val = x.split(';')[0]
    return val
    
def impute_lanes(row,column_lanes = "lanes"):
    if pd.notnull(row[column_lanes]):
        return row[column_lanes]
    else:
        return LANE_DEFAULTS[row['highway']]
# convert string representation of multiple highway types to a list
def allocate_lanes(row,column_lanes = "lanes"):
    if row['oneway']:
        return row[column_lanes]
    else:
        return int(row[column_lanes] / 2)


def collapse_multiple_maxspeed_values(value):
    if isinstance(value, list):
        try:
            # strip non-numeric " mph" from each value in the list then take the mean
            values = [int(x.replace(' mph', '')) for x in value]
            return statistics.mean(values)
        except:
            # if exception, return null (a few have multiple values like "35 mph;40 mph")
            return None
    else:
        return value

def parse_speed_strings(value):
    if isinstance(value, str):
        if (value == 'signals') or (value == 'cyclestreet'):
            value = value.replace('signals', 'nan')
        else:
            value = value.replace(' mph', '')
        # sometimes multiple speeds are semicolon-delimited -- collapse to a single value
        if ';' in value:
            # return the mean of the values if it has that semicolon
            values = [int(x) for x in value.split(';')]
            return statistics.mean(values)
        else:
            try:
                return int(value)
            except:
                return 10
    else:
        return value
    
    
def infer_speed(row):
    """
        Extract the dault values from SPEED_DEFAULTS
    """
    hwy = row['highway']
    lanes = row['lanes_capped']
    return SPEED_DEFAULTS[hwy][lanes]
    
def infer_capacity(row):
    """
        Extract default value from the CAPACITY_DEFAULTS
    """
    hwy = row['highway']
    lanes = row['lanes_capped']
    return CAPACITY_DEFAULTS[hwy][lanes]
