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

from operator import is_
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
from cartography import simplify_graph_from_polygon_gdf,from_graph_to_edges_nodes_format,reindex_nodes,process_highway,CAPACITY_DEFAULTS,SPEED_DEFAULTS, compute_index2osmid_osmid2idx


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
#        with open(os.path.join(self.save_dir,"idx2osmid.json"),"r") as f:
#            self.index2osmid = json.load(f)
#        self.index2osmid = {int(k):int(v) for k,v in self.index2osmid.items()}
        self.simplify_polygon()
        self.Area = self.gpd_polygon.area
        self.radius = np.sqrt(self.Area/np.pi)

##------------------------------- SIMPLIFY OX CARTO -------------------------------------##
   

    def simplify_polygon(self):
        """
            @description: Simplify the polygon to create the graph
        """
        logger.info("SympifyPolygon: {}".format(os.path.join(self.data_dir,self.name,self.name + '.shp')))
        # NOTE: Check existence new tertiary_simplified.graphml -> is the one that contains the regional network
        graphml_name = os.path.join(self.save_dir,self.name + '_new_tertiary_simplified.graphml')
        is_graphml_file = os.path.isfile(graphml_name)
        gpkg_name = os.path.join(self.save_dir,self.name + '_new_tertiary_simplified.gpkg')
        is_gpkg_file = os.path.isfile(gpkg_name)
        # NOTE: Check if nodes.csv and edges.csv already exist
        edges_name = os.path.join(self.save_dir,'edges.csv')
        nodes_name = os.path.join(self.save_dir,'nodes.csv')
        is_edges_file = os.path.isfile(edges_name)
        is_nodes_file = os.path.isfile(nodes_name)
        # NOTE: idx2osmid and osmid2idx
        idx2osmid_name = os.path.join(self.save_dir,'idx2osmid.json')
        osmid2idx_name = os.path.join(self.save_dir,'osmid2idx.json')
        is_idx2osmid_file = os.path.isfile(idx2osmid_name)
        is_osmid2idx_file = os.path.isfile(osmid2idx_name)
        # Extract Simplified Network
        self.G2_simp, self.index2osmid, self.osmid2index = simplify_graph_from_polygon_gdf(gdf_polygon = self.gpd_polygon,
                                                                                            types = self.types,
                                                                                            str_network_type_ox = "drive",
                                                                                            str_highway_ox_name = 'highway',
                                                                                            str_unique_index_edges = "uniqueid",
                                                                                            str_start_node = "u",
                                                                                            str_end_node = "v",
                                                                                            str_start_node_osmid = "osmid_u",
                                                                                            str_end_node_osmid = "osmid_v"
                                                                                            )
        ox.io.save_graphml(self.G2_simp, filepath= graphml_name)
        ox.io.save_graph_geopackage(self.G2_simp, filepath=gpkg_name)
        self.osmid2index = {int(k):int(v) for k,v in self.osmid2index.items()}
        self.index2osmid = {int(k):int(v) for k,v in self.index2osmid.items()}
        with open(osmid2idx_name,"w") as f:
            json.dump(self.osmid2index,f, indent=4)
        with open(idx2osmid_name,"w") as f:
            json.dump(self.index2osmid,f, indent=4)
        # Extract nodes and edges dataframe from the networkx graph
        self.nodes, self.edges = from_graph_to_edges_nodes_format(self.G2_simp)
        # add the index column [0,...,N_nodes]
        self.nodes = reindex_nodes(self.nodes)
        # Modify the edge file to have compliant edge file
        self.edges = self.edges.drop(columns=['geometry']).reindex(columns=self.ecols)
        self.edges = process_highway(self.edges,
                                     self.useful_cols)
        
        self.nodes.to_csv(nodes_name, index=False, encoding='utf-8')
        self.edges.to_csv(edges_name, index=False, encoding='utf-8')

        # If not available compute the simplified graph
        if not os.path.isfile(is_graphml_file):
            logger.info(f"SympifyPolygon: Computing Simplified Graph {graphml_name} ")
            pass
        else:
            logger.info('SympifyPolygon: File {}_new_tertiary_simplified already exists'.format(self.name))
            self.G2_simp = ox.load_graphml(filepath=os.path.join(self.save_dir,self.name + '_new_tertiary_simplified.graphml'))            
            self.index2osmid, self.osmid2index = compute_index2osmid_osmid2idx(G = self.G2_simp)
            # Extract nodes and edges dataframe from the networkx graph
            self.nodes, self.edges = from_graph_to_edges_nodes_format(self.G2_simp)
            # add the index column [0,...,N_nodes]
            self.nodes = reindex_nodes(self.nodes)
            # Modify the edge file to have compliant edge file
            self.edges = self.edges.drop(columns=['geometry']).reindex(columns=self.ecols)
            self.edges = process_highway(self.edges,
                                        self.useful_cols)

        # NOTE: Saving section

        if is_graphml_file:
            logger.info(f"SimplifyPolygon: File {graphml_name} already exists. Skip save.")
        else:
            logger.info(f"SimplifyPolygon: {graphml_name} Save graphml")
            ox.io.save_graphml(self.G2_simp, filepath=graphml_name)

        # Replace shapefile with GeoPackage
        if is_gpkg_file:
            logger.info(f"SimplifyPolygon: File {gpkg_name} already exists. Skip save.")
        else:
            logger.info(f"SimplifyPolygon: {gpkg_name} Save geopackage")
            ox.io.save_graph_geopackage(self.G2_simp, filepath=gpkg_name)

        if is_idx2osmid_file:
            logger.info(f"SympifyPolygon: Load: {idx2osmid_name}")
        else:
            logger.info(f"SympifyPolygon: Save: {idx2osmid_name}")
            with open(idx2osmid_name, "w") as f:
                json.dump(self.index2osmid, f, indent=4)

        # NOTE: Map idx2osmid
        if is_osmid2idx_file:
            logger.info(f"SympifyPolygon: {osmid2idx_name} exists. Skip saving it.")
        else:
            logger.info(f"SympifyPolygon: Save: {osmid2idx_name}")
            with open(osmid2idx_name, "w") as f:
                json.dump(self.osmid2index, f, indent=4)

        if is_edges_file and is_nodes_file:
            logger.info(f"SympifyPolygon: {edges_name} and {nodes_name} already exist. Skip saving them.")
        else:
            logger.info(f"SympifyPolygon: {edges_name} and {nodes_name} do not exist. Save them.")
            self.nodes.to_csv(nodes_name, index=False, encoding='utf-8')
            self.edges.to_csv(edges_name, index=False, encoding='utf-8')

        logger.info("SympifyPolygon: Done")


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