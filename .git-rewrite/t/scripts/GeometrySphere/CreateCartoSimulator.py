'''
    The unique use of this script is to build and save nodes.csv and edges.csv that are needed to run the GPU simulation.
    - REQUIRED:
        city_shapefile.shp
    - RUN:
        python3 ./CreateCartoSimulator.py

    - CITY AVAILBLE:
        ['BOS','LAX','SFO','RIO','LIS']
    To modify the city could be added a configuration set of steps to automatically retrieve the shapefiles, save and have a list of names of saved shapefile.
    NOTE: In this script the shape file are saved as data_dir/NAME/NAME.shp (RULE: 2 step below the dir of this file (data_dir/scripts/PreProcessing) )
'''

import geopandas as gpd
import numpy as np
import os
import pandas as pd
import osmnx as ox
from collections import defaultdict
from shapely.geometry import Point,Polygon
import matplotlib.pyplot as plt
import json
import ast
import statistics
import sys
import socket
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if socket.gethostname()=='artemis.ist.berkeley.edu':
    TRAFFIC_DIR = '/home/alberto/LPSim/traffic_phase_transition'
else:
    TRAFFIC_DIR = os.getenv('TRAFFIC_DIR')
sys.path.append(os.path.join(TRAFFIC_DIR,'GenerationNet'))
from global_functions import ifnotexistsmkdir
from cartography import simplify_graph_from_polygon_gdf,from_graph_to_edges_nodes_format,reindex_nodes,process_highway,CAPACITY_DEFAULTS,SPEED_DEFAULTS


"""
config = {"name_city": {
                        }
         }

"""


class mobility_planner:
    '''


        This class takes as input direction to .shp files. It converts them automatically to nodes.csv and edges.csv

    '''
    def __init__(self,name_city):
        # Lane: speed -> dict
        self.speed_defaults = SPEED_DEFAULTS
        # Lane: capacity -> dict
        self.capacity_defaults = CAPACITY_DEFAULTS
        self.ecols = ['uniqueid', 'u', 'v', 'key', 'oneway', 'highway', 'name', 'length',
                'lanes', 'width', 'est_width', 'maxspeed', 'access', 'service',
                'bridge', 'tunnel', 'area', 'junction', 'osmid', 'ref']
        self.useful_cols = ['uniqueid', 'u', 'v', 'length', 'speed_mph', 'lanes'] #, 'highway', 'oneway'
        self.types = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 
         'primary', 'primary_link', 'secondary', 'secondary_link',
         'tertiary', 'tertiary_link', 'unclassified', 'road']
        self.name = name_city
        self.data_dir = os.path.join(TRAFFIC_DIR,'data','carto',self.name,'shape_files')
        self.gpd_polygon = read_file_gpd(os.path.join(self.data_dir,self.name + '.shp'))
        self.save_dir = os.path.join(TRAFFIC_DIR,'data','carto',self.name)
        with open(os.path.join(self.save_dir,"idx2osmid.json"),"r") as f:
            self.index2osmid = json.load(f)
        self.index2osmid = {int(k):int(v) for k,v in self.index2osmid.items()}
        self.simplify_polygon()
        self.Area = self.gpd_polygon.area
        self.radius = np.sqrt(self.Area/np.pi)

##------------------------------- SIMPLIFY OX CARTO -------------------------------------##
    
    def simplify_polygon(self):
        """
            @description: Simplify the polygon to create the graph
        """
        logger.info("SympifyPolygon: {}".format(os.path.join(self.data_dir,self.name,self.name + '.shp')))
        if not os.path.isfile(os.path.join(self.save_dir,self.name + '_new_tertiary_simplified.graphml')):
            # Extract Simplified Network
            self.G2_simp = simplify_graph_from_polygon_gdf(self.gpd_polygon,
                                                            self.types,
                                                            self.index2osmid,
                                                            str_network_type_ox = "drive",
                                                            str_highway_ox_name = 'highway',
                                                            str_unique_index_edges = "uniqueid",
                                                            str_start_node = "u",
                                                            str_end_node = "v",
                                                            str_start_node_osmid = "osmid_u",
                                                            str_end_node_osmid = "osmid_v"
                                                            )
            # create a unique ID for each edge because osmid can hold multiple values due to topology simplification
            if not os.path.isfile(os.path.join(self.save_dir,self.name + '_new_tertiary_simplified.graphml')):
                print('save graphml')
                ox.io.save_graphml(self.G2_simp, filepath= os.path.join(self.save_dir,self.name + '_new_tertiary_simplified.graphml'))
            if not os.path.isfile(os.path.join(self.save_dir,self.name + '_new_tertiary_simplified.shp')):
                print('save shapefile')
                ox.io.save_graph_shapefile(self.G2_simp, filepath=os.path.join(self.save_dir,self.name + '_new_tertiary_simplified'))
            fig, ax = ox.plot.plot_graph(self.G2_simp, node_size=0, edge_linewidth=0.2)
            plt.savefig(os.path.join(self.save_dir,self.name + '_new_tertiary_simplified.png'), dpi=200, bbox_inches='tight')
            logger.info('SympifyPolygon: Save nodes.csv and edges.csv')
        else:
            logger.info('SympifyPolygon: File {}_new_tertiary_simplified already exists'.format(self.name))
            self.G2_simp = ox.load_graphml(filepath=os.path.join(self.save_dir,self.name + '_new_tertiary_simplified.graphml'))
            fig, ax = ox.plot.plot_graph(self.G2_simp, node_size=0, edge_linewidth=0.2)
            plt.savefig(os.path.join(self.save_dir,self.name + '_new_tertiary_simplified.png'), dpi=200, bbox_inches='tight')

            if not os.path.isfile(os.path.join(self.save_dir,self.name + '_new_tertiary_simplified')):
                logger.info('SympifyPolygon: {} Plot Graph'.format(os.path.join(self.save_dir,self.name + '_new_tertiary_simplified')))
                ox.io.save_graph_geopackage(self.G2_simp, filepath=os.path.join(self.save_dir,self.name + '_new_tertiary_simplified'))
            if not os.path.isfile(os.path.join(self.save_dir,'nodes.csv')) and not os.path.isfile(os.path.join(self.save_dir,'edges.csv')):
                logger.info('SympifyPolygon: Save nodes.csv and edges.csv')
        # Extract nodes and edges dataframe from the networkx graph
        self.nodes, self.edges = from_graph_to_edges_nodes_format(self.G2_simp)
        # add the index column [0,...,N_nodes]
        self.nodes = reindex_nodes(self.nodes)
        self.nodes.to_csv(os.path.join(self.save_dir,'nodes.csv'), index=False, encoding='utf-8')
        # Modify the edge file to have compliant edge file
        self.edges = self.edges.drop(columns=['geometry']).reindex(columns=self.ecols)
        self.edges = process_highway(self.edges,
                                     self.useful_cols)
        self.edges.to_csv(os.path.join(self.save_dir,'edges.csv'), index=False, encoding='utf-8')
        logger.info('SympifyPolygon: Done')          


def set_environment():
    base_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
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
    dirs = ['BOS','LAX','SFO','RIO','LIS']#['BOS','LAX']['SFO','RIO','LIS']
    for dir_ in dirs:
        save_dir = os.path.join(carto_base,dir_)            
        mobility_planner(dir_)