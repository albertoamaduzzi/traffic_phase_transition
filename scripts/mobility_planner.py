import geopandas as gpd
import numpy as np
import os
import pandas as pd
from global_functions import ifnotexistsmkdir
import osmnx as ox
from collections import defaultdict
from shapely.geometry import Point,Polygon
import matplotlib.pyplot as plt
import json
import ast
import statistics

class mobility_planner:
    '''
        This class takes as input direction to .shp files. It converts them automatically to nodes.csv and edges.csv

    '''
    def __init__(self,name_city):
        self.speed_defaults = {'tertiary' : {1 : 20, 2 : 20, 3 : 20, 4 : 20, -1 : 20},
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
        self.capacity_defaults = {'tertiary' : {1 : 900, 2 : 900, 3 : 900, 4 : 900, -1 : 900},
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
        self.ecols = ['uniqueid', 'u', 'v', 'key', 'oneway', 'highway', 'name', 'length',
                'lanes', 'width', 'est_width', 'maxspeed', 'access', 'service',
                'bridge', 'tunnel', 'area', 'junction', 'osmid', 'ref']
        self.useful_cols = ['uniqueid', 'u', 'v', 'length', 'maxspeed', 'lanes', 'highway', 'oneway']
        self.types = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 
         'primary', 'primary_link', 'secondary', 'secondary_link',
         'tertiary', 'tertiary_link', 'unclassified', 'road']
        self.name = name_city
        self.data_dir,self.carto_base = set_environment()
        self.gpd_polygon = read_file_gpd(os.path.join(self.data_dir,self.name,self.name + '.shp'))
        self.save_dir = os.path.join(self.carto_base,self.name)
        self.simplify_polygon()
        self.Area = self.gpd_polygon.area
        self.radius = np.sqrt(self.Area/np.pi)

##------------------------------- SIMPLIFY OX CARTO -------------------------------------##
    
    def simplify_polygon(self):
        print('Input dir: ',self.data_dir)
        print('Creating polygon for: ',self.name)
        print('saving dir: ',self.save_dir)
        if not os.path.isfile(os.path.join(self.save_dir,self.name + '_new_tertiary_simplified.graphml')) == False:
            print('simplify graph:')
            self.simplify_graph_from_polygon_gdf()
            # create a unique ID for each edge because osmid can hold multiple values due to topology simplification
            print('save graphml')
            ox.io.save_graphml(self.G2_simp, filepath= os.path.join(self.save_dir,self.name + '_new_tertiary_simplified.graphml'))
            print('save shapefile')
            ox.io.save_graph_shapefile(self.G2_simp, filepath=os.path.join(self.save_dir,self.name + '_new_tertiary_simplified'))
            fig, ax = ox.plot.plot_graph(self.G2_simp, node_size=0, edge_linewidth=0.2)
            plt.savefig(os.path.join(self.save_dir,self.name + '_new_tertiary_simplified.png'), dpi=300, bbox_inches='tight')
            print('simplify node edges:')
            self.get_nodes_edges_from_graph_simplified()
            print('save nodes')
            self.reindex_nodes_and_save()
            print('save edges')
            self.handle_edges_and_save()
        else:
            print('File {}_new_tertiary_simplified already exists'.format(self.name))
            self.G2_simp = ox.load_graphml(filepath=os.path.join(self.save_dir,self.name + '_new_tertiary_simplified.graphml'))
            print('save shapefile: ',os.path.join(self.save_dir,self.name + '_new_tertiary_simplified'))
            if not os.path.isfile(os.path.join(self.save_dir,self.name + '_new_tertiary_simplified','nodes.shp')):
                ox.io.save_graph_geopackage(self.G2_simp, filepath=os.path.join(self.save_dir,self.name + '_new_tertiary_simplified'))
                fig, ax = ox.plot.plot_graph(self.G2_simp, node_size=0, edge_linewidth=0.2)
                plt.savefig(os.path.join(self.save_dir,self.name + '_new_tertiary_simplified.png'), dpi=300, bbox_inches='tight')
            if not os.path.isfile(os.path.join(self.save_dir,'nodes.csv')) and not os.path.isfile(os.path.join(self.save_dir,'edges.csv')):
                print('simplify node edges:')
                self.get_nodes_edges_from_graph_simplified()
                print('save nodes')
                self.reindex_nodes_and_save()
                print('save edges')
                self.handle_edges_and_save()            

##------------------------------- SIMPLIFY OX CARTO -------------------------------------##

    def simplify_graph_from_polygon_gdf(self):        
        if not self.gpd_polygon is None:
            polygon = self.gpd_polygon.unary_union
            polygon_hull = polygon.convex_hull
            polygon_hull_proj, crs = ox.projection.project_geometry(polygon_hull)
            polygon_hull_proj_buff = polygon_hull_proj.buffer(1600) #1 mile in meters
            polygon_hull_buff, crs = ox.projection.project_geometry(polygon_hull_proj_buff, crs=crs, to_latlong=True)    
            G2 = ox.graph_from_polygon(polygon_hull_buff, network_type='drive', simplify=False)
            minor_streets = [(u, v, k) for u, v, k, d in G2.edges(keys=True, data=True) if d['highway'] not in self.types]
            G2.remove_edges_from(minor_streets)
            G2 = ox.utils_graph.remove_isolated_nodes(G2)
            G2_connected = ox.utils_graph.get_largest_component(G2, strongly=True)
            self.G2_simp = ox.simplify_graph(G2_connected, strict=True)  
            i = 0
            for u, v, k, d in self.G2_simp.edges(data=True, keys=True):
                d['uniqueid'] = i
                d['u'] = u
                d['v'] = v
                i += 1
        else:
            raise ValueError('gpd_polygon is None, need to upload a valid shape file')


    def get_nodes_edges_from_graph_simplified(self):
        if self.G2_simp is not None:
            self.nodes, self.edges = ox.graph_to_gdfs(self.G2_simp, node_geometry=False, fill_edge_geometry=False)
        else:
            raise ValueError('Graph is None')

    def reindex_nodes_and_save(self):
        if self.nodes is not None:
            print('nodes')
            print(self.nodes)
            self.nodes = self.nodes.reset_index()
            self.nodes = self.nodes.reindex(columns=['osmid', 'x', 'y', 'ref', 'highway'])
            self.nodes['index'] = self.nodes.index
            print('nodes after reset index',self.nodes)
            print('nodes dir:\n',os.path.join(self.save_dir,'nodes.csv'))
            self.nodes.to_csv(os.path.join(self.save_dir,'nodes.csv'), index=False, encoding='utf-8')
        else:
            raise ValueError('Nodes is None, need to upload a valid graph')
    def handle_edges_and_save(self):
        if not self.edges is None:
            self.edges = self.edges.drop(columns=['geometry']).reindex(columns=self.ecols)
            self.process_highway()
        else:
            raise ValueError('Edges is None, need to upload a valid graph')

    def process_highway(self):
        print(self.edges)
        if self.edges is not None:
            self.edges['u'] = self.edges['u'].astype(int)
            self.edges['v'] = self.edges['v'].astype(int)
            self.edges['highway'] = self.edges['highway'].map(convert_lists)
            self.edges['highway'] = self.edges['highway'].map(collapse_multiple_hwy_values)
            self.edges['lanes'] = self.edges['lanes'].map(convert_lists)
            self.edges['lanes'] = self.edges['lanes'].map(collapse_multiple_lane_values)
            self.edges['lanes'] = self.edges['lanes'].map(remove_hyphens)
            self.edges['lanes'] = self.edges['lanes'].map(remove_semicolons)
            self.edges['lanes'] = self.edges['lanes'].astype(float)
            lane_defaults = self.edges.groupby('highway')['lanes'].median()
            lane_defaults = lane_defaults.fillna(value=2).to_dict() #'road' type is null
            self.edges['lanes'] = self.edges.apply(impute_lanes, axis=1).astype(int)
            self.edges['lanes'] = self.edges.apply(allocate_lanes, axis=1)
            self.edges.loc[self.edges['lanes'] < 1, 'lanes'] = 1
            self.edges['lanes_capped'] = self.edges['lanes']
            self.edges.loc[self.edges['lanes_capped'] > 4, 'lanes_capped'] = 4
            self.edges['maxspeed'] = self.edges['maxspeed'].map(convert_lists)
            self.edges['maxspeed'] = self.edges['maxspeed'].map(collapse_multiple_maxspeed_values)
            self.edges['maxspeed'] = self.edges['maxspeed'].map(parse_speed_strings)
            known_speeds = self.edges[pd.notnull(self.edges['maxspeed'])]['maxspeed']
            known_speeds = known_speeds.astype(int)           
            inferred_speeds = self.edges[pd.isnull(self.edges['maxspeed'])].apply(infer_speed, axis=1)
            # merge known speeds with inferred speeds to get a free-flow speed for each edge
            self.edges['speed_mph'] = known_speeds._append(inferred_speeds, ignore_index=False, verify_integrity=True)
            self.edges['capacity_lane_hour'] = self.edges.apply(infer_capacity, axis=1)
            self.edges['capacity_hour'] = self.edges['capacity_lane_hour'] * self.edges['lanes']
            self.edges['length'] = self.edges['length'].round(1)
            self.edges_save = self.edges[self.useful_cols]
            print('edges dir:\n',os.path.join(self.save_dir,'edges.csv'))
            self.edges_save.to_csv(os.path.join(self.save_dir,'edges.csv'), index=False, encoding='utf-8')
        else:
            raise ValueError('Edges is None, cannot process highway')
        
def collapse_multiple_hwy_values(hwy):
    speed_defaults = {'tertiary' : {1 : 20, 2 : 20, 3 : 20, 4 : 20, -1 : 20},
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
    if isinstance(hwy, list):
        # if we find an item in our defaults dict, use that value
        # otherwise, just use the zeroth item in the list
        for item in hwy:
            if item in speed_defaults.keys():
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
    
def impute_lanes(row):
    lane_defaults = {'motorway': 4.0,
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
    if pd.notnull(row['lanes']):
        return row['lanes']
    else:
        return lane_defaults[row['highway']]
# convert string representation of multiple highway types to a list
def allocate_lanes(row):
    if row['oneway']:
        return row['lanes']
    else:
        return int(row['lanes'] / 2)


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
        if ';' in value and value != 'cyclestreet':
            # return the mean of the values if it has that semicolon
            values = [int(x) for x in value.split(';')]
            return statistics.mean(values)
        else:
            return int(value)
    else:
        return value
    
    
def infer_speed(row):
    speed_defaults = {'tertiary' : {1 : 20, 2 : 20, 3 : 20, 4 : 20, -1 : 20},
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
    hwy = row['highway']
    lanes = row['lanes_capped']
    return speed_defaults[hwy][lanes]
    
def infer_capacity(row):
    capacity_defaults = {'tertiary' : {1 : 900, 2 : 900, 3 : 900, 4 : 900, -1 : 900},
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
    hwy = row['highway']
    lanes = row['lanes_capped']
    return capacity_defaults[hwy][lanes]


def set_environment():
    base_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
    data_dir = os.path.join(base_dir,'data')
    carto = os.path.join(data_dir,'carto')
    ifnotexistsmkdir(data_dir)
    ifnotexistsmkdir(carto)
    return data_dir,carto


def read_file_gpd(file_name):
    '''
        Input:
            file_name: string [name of the file to read]
        Output:
            df: dataframe [dataframe with the data of the file]
        Description:
            Read the file and return the dataframe
    '''
    df = gpd.read_file(file_name)
    return df


if __name__ =='__main__':
    
    data_dir,carto_base = set_environment()
    dirs = ['BOS','LAX','SFO','RIO','LIS']#['LAX','RIO','LIS']
    for dir_ in dirs:
        save_dir = os.path.join(carto_base,dir_)            
        mobility_planner(dir_)