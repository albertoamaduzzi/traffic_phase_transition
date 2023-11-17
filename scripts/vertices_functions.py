from planar_graph import planar_graph
from scipy.spatial import Voronoi
import numpy as np
##-------------------------------------- IS STATEMENTS FOR VERTICES ------------------------------------
def is_in_graph(planar_graph,vertex):
    return planar_graph.graph.vp['is_in_graph'][vertex]

def is_end_point(planar_graph,vertex):
    return planar_graph.graph.vp['end_point'][vertex]

def is_active(planar_graph,vertex):
    return planar_graph.graph.vp['is_active'][vertex]

def is_newly_added(planar_graph,vertex):    
    return planar_graph.graph.vp['newly_added_center'][vertex]

def is_important_node(planar_graph,vertex):
    return planar_graph.graph.vp['important_node'][vertex]

def is_intersection(planar_graph,vertex):
    return planar_graph.graph.vp['intersection'][vertex]
    
##--------------------------------------- SPATIAL GEOMETRY, NEIGHBORHOODS, RELATIVE NEIGHBORS ------------------------------------
##----------------------------------------  SET FUNCTIONS ------------------------ (ACTING ON SINGLE VERTEX PROPERTY MAPS) -----------------------
    
##### "IMPORTANT NODE" FUNCTIONS #####

def set_id(planar_graph,vertex,id_):
    planar_graph.graph.vp['id'][vertex] = id_

def set_active_vertex(planar_graph,vertex,boolean):
    planar_graph.graph.vp['is_active'][vertex] = boolean

def set_important_node(planar_graph,vertex,boolean):
    planar_graph.graph.vp['important_node'][vertex] = boolean

def set_newly_added(planar_graph,vertex,boolean):
    planar_graph.graph.vp['newly_added_center'][vertex] = boolean

def set_end_point(planar_graph,vertex,boolean):
    planar_graph.graph.vp['end_point'][vertex] = boolean

def set_empty_relative_neighbors(planar_graph,vertex):
    planar_graph.graph.vp['relative_neighbors'][vertex] = []

def set_in_graph(planar_graph,vertex,boolean):
    planar_graph.graph.vp['is_in_graph'][vertex] = boolean

def set_empty_road(planar_graph,vertex):
    planar_graph.graph.vp['roads'][vertex] = []

def set_is_intersection(planar_graph,vertex,boolean):
    planar_graph.graph.vp['intersection'][vertex] = boolean
#### "IN GRAPH NODE" FUNCTIONS #####
def set_activate_in_graph(planar_graph,vertex):
    planar_graph.graph.vp['is_in_graph'][vertex] = True

def set_initialize_x_y_pos(planar_graph,vertex,x,y,point_idx):
    planar_graph.graph.vp['x'][vertex] = x[point_idx]
    planar_graph.graph.vp['y'][vertex] = y[point_idx]
    planar_graph.graph.vp['pos'][vertex] = np.array([x[point_idx],y[point_idx]])



def get_voronoi(planar_graph,vertex):
    planar_graph.graph.vp['voronoi'][vertex] = Voronoi(np.array([planar_graph.graph.vp['x'][vertex],planar_graph.graph.vp['y'][vertex]]).T)


