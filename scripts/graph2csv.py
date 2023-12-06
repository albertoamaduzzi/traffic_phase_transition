import geopandas as gpd
from shapely.geometry import Point, LineString
import pandas as pd
import numpy as np
import os
def nodes2csv(planar_graph):
    '''
        Creates the csv file with the nodes to be given to artemis.
    '''
    vertices = [planar_graph.graph.vp['id'][v] for v in planar_graph.graph.vertices() if (planar_graph.graph.vp['important_node'][v] == True) or (planar_graph.graph.vp['intersection'][v] == True)]
    col_id = pd.DataFrame(vertices,name='osmid')
    col_x = pd.DataFrame([planar_graph.graph.vp['x'][planar_graph.graph.vertex(v)] for v in vertices],name='x')
    col_y = pd.DataFrame([planar_graph.graph.vp['y'][planar_graph.graph.vertex(v)] for v in vertices],name='y')
    ref = pd.DataFrame([None for v in vertices],name='ref')
    highway = pd.DataFrame([None for v in vertices],name='highway')
    df = pd.concat([col_id,col_x,col_y,ref,highway],join='inner',ignore_index=True)
    df.to_csv(os.path.join(planar_graph.base_dir,'nodes.csv'),index=False)

def edges2csv(planar_graph):
    '''
        For each node I check the road associated with that v -> this means that must be initialized with an empty list every time I 
        add a new node in the graph.
        NOTE: So fare I have not added the possibility of assigning the velocity of the road. One way to do that could be:
            1) Put in order the roads according to their length.
            2) Take 5% that are longest 
            3) These are the highways 
    '''
    uniqueid = []
    u = []
    v = []
    length = [] 
    lanes = []
    speed_mph = []
    for v in planar_graph.graph.vertices():
        for list_r in planar_graph.graph.vp['roads'][v]:
            if len(list_r)!=0:
                for r in list_r:
                    uniqueid.append(r.id)
                    u.append(r.initial_node)
                    v.append(r.end_point)
                    length.append(r.length)
                    lanes.append(1)
                    speed_mph.append(30)

def edges2geojson(planar_graph):
    uniqueid = []
    u = []
    v = []
    length = [] 
    lanes = []
    speed_mph = []
    for r in planar_graph.list_roads:
