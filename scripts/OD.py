import json
import numpy as np
import os
import graph_tool as gt
import pandas as pd


def lognormal(mean, std):
    '''
        Distribution of distances for each trip (the OD pair must be generated in this way)
        For policentric cities the variance is bigger. [ maybe we can invent some way of defining the variance according to the policentricity]
    '''
    return np.random.lognormal(mean, std)

def Weibull(shape, scale):
    '''
        Distribution of distances of trapped cars after 1 hour
    '''
    return np.random.weibull(shape, scale)

class OD:
    def __init__(self,config):
        tuple = os.walk('.', topdown=True)
        root = tuple.__next__()[0]
        self.config_dir = os.path.join(root,'config')
        self.config_name = os.listdir(self.config_dir)[conf.index() for conf in os.listdir(self.config_dir) if 'OD' in conf]
        with open(os.path.join(self.config_dir,self.config_name),'r') as f:
            self.config = json.load(f)
        self.graphdf = pd.read_csv(config['file_nodes'],index_col=0)

    def create_graph(self):
        self.graph = gt.Graph()
        self.graph.add_vertex(self.graphdf.shape[0])
        self.graph.vp['pos'] = self.graph.new_vertex_property('vector<double>')
        self.graph.vp['pos'].a = self.graphdf[['x','y']].values
        self.graph.vp['unique_id'] = self.graph.new_vertex_property('int')
        self.graph.vp['unique_id'].a = self.graphdf['unique_id'].values
        self.graph.ep['weight'] = self.graph.new_edge_property('double')
        self.graph.ep['weight'].a = self.graphdf['weight'].values
        self.graph.ep['capacity'] = self.graph.new_edge_property('double')
        self.graph.ep['capacity'].a = self.graphdf['capacity'].values
        self.graph.ep['length'] = self.graph.new_edge_property('double')
        self.graph.ep['length'].a = self.graphdf['length'].values
        self.graph.ep['free_flow_time'] = self.graph.new_edge_property('double')
        self.graph.ep['free_flow_time'].a = self.graphdf['free_flow_time'].values
        self.graph.ep['B'] = self.graph.new_edge_property('double')
        self.graph.ep['B'].a = self.graphdf['B'].values
        self.graph.ep['power'] = self.graph.new_edge_property('double')
        self.graph.ep['power'].a = self.graphdf['power'].values
        self.graph.ep['speed_limit'] = self.graph.new_edge_property('double')
        self.graph.ep['speed_limit'].a = self.graphdf['speed_limit'].values
        self.graph.ep['toll'] = self.graph.new_edge_property('double')
        self.graph.ep['toll'].a = self.graphdf['toll'].values
        self.graph.ep['type'] = self.graph.new_edge_property('string')
        self.graph.ep['type'].a = self.graphdf['type'].values
        self.graph.ep['modes'] = self.graph.new_edge_property('string')
        self.graph.ep['modes'].a = self.graphdf['modes'].values
        self.graph.ep['osm_id'] = self.graph.new_edge_property('string')
        self.graph.ep['osm_id'].a = self.graphdf['osm_id'].values
        self.graph.ep['geometry'] = self.graph.new_edge_property('string')
        self.graph.ep['geometry'].a = self.graphdf

