from scipy.spatial import Voronoi
import numpy as np
# FROM PROJECT
from output import *
from geometric_features import road
from edges_functions import *

##-------------------------------------------- ADD/GET FUNCTIONS --------------------------------------------####

def add_vertex(planar_graph):
    '''
        Adds a vertex to the graph
    '''
    planar_graph.graph.add_vertex()

def get_last_vertex(planar_graph):
    '''
        Returns the number of vertices in the graph
    '''
    return planar_graph.graph.vertex(planar_graph.graph.num_vertices()-1)

def get_id_last_vertex(planar_graph):
    '''
        Returns the id of the last vertex in the graph
    '''
    return planar_graph.graph.num_vertices()-1



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

def set_relative_neighbors(planar_graph,vertex,list_relative_neighbors):
    '''
        Input:
            list_relative_neighbors: list dtype = vertex
        Description:
            1) Set the relative neighbors
    '''
    planar_graph.graph.vp['relative_neighbors'][vertex] = list_relative_neighbors

def set_in_graph(planar_graph,vertex,boolean):
    '''
        Input:
            boolean: bool
        Description:
            1) Set the relative neighbors
    '''
    planar_graph.graph.vp['is_in_graph'][vertex] = boolean

def set_empty_road(planar_graph,vertex):
    planar_graph.graph.vp['roads'][vertex] = []

def set_is_intersection(planar_graph,vertex,boolean):
    planar_graph.graph.vp['intersection'][vertex] = boolean

#### "IN GRAPH NODE" FUNCTIONS #####

def set_initialize_x_y_pos(planar_graph,vertex,x,y,point_idx):
    planar_graph.graph.vp['x'][vertex] = x[point_idx]
    planar_graph.graph.vp['y'][vertex] = y[point_idx]
    planar_graph.graph.vp['pos'][vertex] = np.array([x[point_idx],y[point_idx]])

## VORONOI

def get_voronoi(planar_graph,vertex):
    planar_graph.graph.vp['voronoi'][vertex] = Voronoi(np.array([planar_graph.graph.vp['x'][vertex],planar_graph.graph.vp['y'][vertex]]).T)

##--------------------------------------------- UPDATES ---------------------------------------------------- NEXT STEP -> COMPUTE DELAUNEY TRIANGULATION (new added,old) attracting vertices    
## LISTS

def update_list_important_vertices(planar_graph):
    planar_graph.important_vertices = [v for v in planar_graph.graph.vertices() if planar_graph.graph.vp['important_node'][v] == True]

def update_list_in_graph_vertices(planar_graph):
    '''
        List of points to consider when deciding old attracting vertices relative neighbor
    '''
    planar_graph.is_in_graph_vertices = [v for v in planar_graph.graph.vertices() if planar_graph.graph.vp['is_in_graph'][v] == True]

def update_list_end_points(planar_graph):
    '''
        List of points to consider when deciding old attracting vertices relative neighbor
    '''
    planar_graph.end_points = [v for v in planar_graph.graph.vertices() if planar_graph.graph.vp['end_point'][v] == True]

## ACTIVE VERTICES  

def update_list_old_attracting_vertices(planar_graph):
    '''
        These are the vertices that attract just end points. 
            -Delauney triangulation just among these and the end points in graph
    '''
    planar_graph.old_attracting_vertices = [v for v in planar_graph.graph.vertices() if is_active(planar_graph,v) and not is_newly_added(planar_graph,v)]

def update_list_newly_added_attracting_vertices(planar_graph):
    planar_graph.newly_added_attracting_vertices = [v for v in planar_graph.graph.vertices() if is_active(planar_graph,v) and is_newly_added(planar_graph,v)]

def update_list_intersection_vertices(planar_graph):
    planar_graph.intersection_vertices = [v for v in planar_graph.graph.vertices() if planar_graph.graph.vp['intersection'][v] == True]

def update_list_plausible_starting_point_of_roads(planar_graph):
    planar_graph.plausible_starting_road_vertices = [v for v in planar_graph.graph.vertices() if is_important_node(planar_graph,v) or is_intersection(planar_graph,v)]

## ROADS

def update_list_roads(planar_graph):
    '''
        Updates the list of roads in the graph:
        NOTE: call update_list_active_vertices after -> as each road is activated by an active vertex
    '''
    for v in planar_graph.graph.vertices():
        for r in planar_graph.graph.vp['roads'][v]:
            planar_graph.list_roads.append(r)    

## LIST ACTIVATING POINT

def update_list_active_vertices(planar_graph):
    '''
        For each road in the network, if the road is not closed, add the points that are activating it, given they are not already in the list
    '''
    for r in planar_graph.list_roads:
        for activating_point in r.activated_by:
            if activating_point not in planar_graph.active_vertices and not r.is_closed:
                planar_graph.active_vertices.append(r.activating_node())

## Single property

def update_active_vertices(planar_graph):
    '''
        An important node is either:
            1) Active: 
                    Is attracting some road that is not closed
            2) Passive: 
                    Is attracting some road that is closed 
    '''
    for v in planar_graph.important_vertices:
        if v in planar_graph.active_vertices:
            planar_graph.graph.vp['is_active'][v] = True
        else:
            planar_graph.graph.vp['is_active'][v] = False


# This piece must be inserted to take update the roads, as I need each point that is evolving to have an attraction set

def update_attracted_by(planar_graph):
    '''
        For each vertex update the attraction:
            1) The end_points are attracted by the relative neighbors that are not in the graph
            2) The growing points are attracted by new added vertices (stop)
    '''
    print('updating attracted by: ')
    list_considered_vertices = []
    for v in planar_graph.graph.vertices():
        print('considering vertex: ',planar_graph.graph.vp['id'][v])
        planar_graph.graph.vp['attracted_by'][v] = []
        if is_end_point(planar_graph,v):
            print('is end point')
            list_considered_vertices.append(v)
            list_rn = planar_graph.graph.vp['relative_neighbors'][v]
            print('relative neighbors: ',list_rn)
            starting_vertex,local_idx,found = find_road_vertex(planar_graph,v)
            if found:
                for relative_neighbor in list_rn:
                    if relative_neighbor not in planar_graph.graph.vp['roads'][starting_vertex][local_idx].list_nodes:
                        planar_graph.graph.vp['attracted_by'][v].append(relative_neighbor)
            else:
                pass
        elif is_growing_and_not_attracting(planar_graph,v):
            list_considered_vertices.append(v)
            list_rn = planar_graph.graph.vp['relative_neighbors'][v]
            starting_vertex,local_idx,found = find_road_vertex(planar_graph,v)
            if found:
                for relative_neighbor in list_rn:
                    if relative_neighbor not in planar_graph.graph.vp['roads'][starting_vertex][local_idx].list_nodes:
                        planar_graph.graph.vp['attracted_by'][v].append(relative_neighbor)
            else:
                pass
        elif is_attracting_and_not_growing(planar_graph,v):
            list_considered_vertices.append(v)
            planar_graph.graph.vp['attracted_by'][v] = []
            pass
        elif is_growing_and_attracting(planar_graph,v):
            list_considered_vertices.append(v)
            planar_graph.graph.vp['attracted_by'][v] = []
            pass
    not_considered = [v for v in planar_graph.graph.vertices() if v not in list_considered_vertices]
    print_not_considered_vertices(planar_graph,not_considered)


def update_intersections(planar_graph):
    '''
        Description:
            newly attracted vertices can attract points that are growing:
            2 cases:
                1) Node attracted: growing, not end_point
                2) Node attracted: growing, end_point
        In case 1 the node gets attracted and becomes intersection
        NOTE:
            This function is called after:
                1) adding new attracting vertices
                2) compute_newly_added_centers_rng
    '''
    ## For each newly added attracting vertex
    for n_attracting_vertex in planar_graph.newly_added_attracting_vertices:
        for attracted_vertex_idx in planar_graph.graph.vp['relative_neighbors'][n_attracting_vertex]:
            attracted_vertex = planar_graph.graph.vertex(attracted_vertex_idx)
            ## If the attracted vertex (belonging to the relative neighbor) is not an end point, and is not an important node (then It can be just growing)
            if not is_end_point(planar_graph,attracted_vertex) and not is_important_node(planar_graph,attracted_vertex):
                starting_vertex,local_idx,found = planar_graph.find_road_vertex(attracted_vertex)
                if found:
                    planar_graph.global_counting_roads += 1
                    planar_graph.graph.vp['roads'][starting_vertex].append(road(starting_vertex,planar_graph.global_counting_roads,n_attracting_vertex))
                    planar_graph.graph.vp['roads'][starting_vertex][-1].copy_road_specifics(planar_graph,planar_graph.graph.vp['roads'][starting_vertex][local_idx])
                else:
                    pass
