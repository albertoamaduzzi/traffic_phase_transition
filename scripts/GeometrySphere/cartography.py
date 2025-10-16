"""


"""
import osmnx as ox
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

from tqdm import tqdm
def extract_graph_from_gdf_polygon(gdf_polygon: gpd.GeoDataFrame,
                                   str_network_type_ox: str = "drive",
                                   show_progress: bool = True) -> Graph:
    """
        Description:
            From a shapefile: Usually this shapefile contain polygon or multipolygon as geometry, 
            such multipolygons come from census usually but they can be whatever
            Extract the convex hull
            Extract the graph inside the convex hull with the properties of str_network_type_ox 
        Return:
            Graph of the road network inside the hull with not simplified edges.
    """
    with tqdm(total=5, desc="Extracting graph from polygon", disable=not show_progress) as pbar:
        # Step 1: Validate geometries
        if not all(g.geom_type in ["Polygon", "MultiPolygon"] for g in gdf_polygon.geometry):
            raise ValueError("All geometries in the GeoDataFrame must be Polygon or MultiPolygon.")
        pbar.update(1)
        pbar.set_description("2/5: Extracting polygon contour")
        # Step 2: Extract polygon contour
        polygon = gdf_polygon.unary_union
        polygon_hull = polygon.convex_hull
        pbar.update(1)
        pbar.set_description("3/5: Projecting in space")
        # Step 3: Project in space
        polygon_hull_proj, crs = ox.projection.project_geometry(polygon_hull)
        pbar.update(1)
        pbar.set_description("4/5: Buffering")
        # Step 4: Buffer (1 mile in meters)
        polygon_hull_proj_buff = polygon_hull_proj.buffer(1600)
        polygon_hull_buff, crs = ox.projection.project_geometry(polygon_hull_proj_buff, crs=crs, to_latlong=True)
        pbar.update(1)
        # Step 5: Extract the drive network (this is slow - OSM query)
        pbar.set_description("5/5 Querying OSM (this may take a while)")
        G2 = ox.graph_from_polygon(polygon_hull_buff, network_type=str_network_type_ox, simplify=False)
        pbar.update(1)
    
    return G2



def compute_index2osmid_osmid2idx(G:Graph):
    """
        Compute the mapping from index to osmid
    """
    index2osmid = {}
    i = 0
    for n, _ in tqdm(G.nodes(data=True), desc="Computing index2osmid mapping"):
        index2osmid[i] = int(n) 
        i += 1
    osmid2index = {int(v): int(k) for k, v in index2osmid.items()}    
    return index2osmid, osmid2index


def simplify_graph_from_polygon_gdf(gdf_polygon: gpd.GeoDataFrame,
                                    types: List,
                                    str_network_type_ox: str = "drive",
                                    str_highway_ox_name: str = 'highway',
                                    str_unique_index_edges: str = "uniqueid",
                                    str_start_node: str = "u",
                                    str_end_node: str = "v",
                                    str_start_node_osmid: str = "osmid_u",
                                    str_end_node_osmid: str = "osmid_v",
                                    show_progress: bool = True) -> Graph:
    """
        Optimized version with progress tracking and fixed node mapping
    """
    if gdf_polygon is None:
        raise ValueError('The Polygon is None, need to upload a valid shape file')
    
    with tqdm(total=7, desc="Simplifying graph", disable=not show_progress) as pbar:
        # Step 1: Extract the graph from the polygon
        pbar.set_description("Step 1/7: Extracting graph from polygon")
        G2 = extract_graph_from_gdf_polygon(
            gdf_polygon=gdf_polygon,
            str_network_type_ox=str_network_type_ox,
            show_progress=True
        )
        pbar.update(1)
        
        # Step 2: Filter edges
        pbar.set_description("Step 2/7: Filtering by road types")
        minor_streets = [(u, v, k) for u, v, k, d in G2.edges(keys=True, data=True) 
                         if d[str_highway_ox_name] not in types]
        pbar.update(1)
        
        # Step 3: Remove minor streets
        pbar.set_description("Step 3/7: Removing minor streets")
        G2.remove_edges_from(minor_streets)
        pbar.update(1)
        
        # Step 4: Remove isolated nodes
        pbar.set_description("Step 4/7: Removing isolated nodes")
        G2 = ox.utils_graph.remove_isolated_nodes(G2)
        pbar.update(1)
        
        # Step 5: Get largest connected component
        pbar.set_description("Step 5/7: Extracting largest component")
        G2_connected = ox.utils_graph.get_largest_component(G2, strongly=True)
        pbar.update(1)
        
        # Step 6: Simplify the network
        pbar.set_description("Step 6/7: Simplifying network topology")
        G2_simp = ox.simplify_graph(G2_connected, strict=True)
        pbar.update(1)
        
        # Step 7: Add edge attributes
        # FIX: Build a mapping from current node IDs to osmid from the simplified graph
        pbar.set_description("Step 7/7: Adding edge attributes")
        
        # Extract osmid from node attributes in the simplified graph
        index2osmid, osmid2index = compute_index2osmid_osmid2idx(G = G2_simp)
        i = 0
        for u_osm, v_osm, _, d in tqdm(G2_simp.edges(data=True, keys=True), desc="Processing edges", leave=False, disable=not show_progress):
            if i == 0:
                print(f"First edge nodes: u_osm={u_osm}, v_osm={v_osm}, d: {d}")
            d[str_unique_index_edges] = i
            u = osmid2index.get(u_osm)
            v = osmid2index.get(v_osm)
            if u is None or v is None:
                raise ValueError(f"Node {u} or {v} not found in index2osmid mapping.")
            else:
                d[str_start_node] = u 
                d[str_end_node] = v 
            d[str_start_node_osmid] = u_osm
            d[str_end_node_osmid] = v_osm
            i += 1
        pbar.update(1)
    return G2_simp, index2osmid, osmid2index
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
                    column_highway="highway",
                    column_lanes="lanes",
                    column_max_speed="maxspeed",
                    show_progress: bool = True):
    """
        Ensures the following standard for edge files with progress tracking:
        - avoid the multiple edges -> just one highway value
        -                          -> just one number of lanes
    """
    if edges is None:
        raise ValueError('Edges is None, cannot process highway')
    
    with tqdm(total=8, desc="Processing highway data", disable=not show_progress) as pbar:
        # Step 1: Process highway column
        pbar.set_description("Step 1/8: Converting highway lists")
        edges[column_highway] = edges[column_highway].map(convert_lists)
        pbar.update(1)
        
        pbar.set_description("Step 2/8: Collapsing highway values")
        edges[column_highway] = edges[column_highway].map(collapse_multiple_hwy_values)
        pbar.update(1)
        
        # Step 3-4: Process lanes column
        pbar.set_description("Step 3/8: Processing lanes")
        edges[column_lanes] = edges[column_lanes].map(convert_lists)
        edges[column_lanes] = edges[column_lanes].map(collapse_multiple_lane_values)
        edges[column_lanes] = edges[column_lanes].map(remove_hyphens)
        edges[column_lanes] = edges[column_lanes].map(remove_semicolons)
        edges[column_lanes] = edges[column_lanes].astype(float)
        pbar.update(1)
        
        # Step 5: Compute lane defaults
        pbar.set_description("Step 4/8: Computing lane defaults")
        lane_defaults = edges.groupby(column_highway)[column_lanes].median()
        lane_defaults = lane_defaults.fillna(value=2).to_dict()
        edges[column_lanes] = edges.apply(impute_lanes, axis=1).astype(int)
        edges[column_lanes] = edges.apply(allocate_lanes, axis=1)
        edges.loc[edges[column_lanes] < 1, column_lanes] = 1
        edges['lanes_capped'] = edges[column_lanes]
        edges.loc[edges['lanes_capped'] > 4, 'lanes_capped'] = 4
        pbar.update(1)
        
        # Step 6: Process maxspeed
        pbar.set_description("Step 5/8: Processing max speeds")
        edges[column_max_speed] = edges[column_max_speed].map(convert_lists)
        edges[column_max_speed] = edges[column_max_speed].map(collapse_multiple_maxspeed_values)
        edges[column_max_speed] = edges[column_max_speed].map(parse_speed_strings)
        pbar.update(1)
        
        # Step 7: Merge known and inferred speeds
        pbar.set_description("Step 6/8: Inferring missing speeds")
        known_speeds = edges[pd.notnull(edges[column_max_speed])][column_max_speed]
        known_speeds = known_speeds.astype(int)
        inferred_speeds = edges[pd.isnull(edges[column_max_speed])].apply(infer_speed, axis=1)
        edges['speed_mph'] = known_speeds._append(inferred_speeds, ignore_index=False, verify_integrity=True)
        pbar.update(1)
        
        # Step 8: Compute capacities
        pbar.set_description("Step 7/8: Computing capacities")
        edges['capacity_lane_hour'] = edges.apply(infer_capacity, axis=1)
        edges['capacity_hour'] = edges['capacity_lane_hour'] * edges[column_lanes]
        edges['length'] = edges['length'].round(1)
        pbar.update(1)
        
        # Step 9: Final cleanup
        pbar.set_description("Step 8/8: Final cleanup")
        edges.drop(columns=['u', 'v', 'key'], inplace=True)
        edges.reset_index(inplace=True)
        edges_save = edges[useful_cols]
        pbar.update(1)
    
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
