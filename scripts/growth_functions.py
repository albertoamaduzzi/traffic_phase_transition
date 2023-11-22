from global_functions import *
from pyhull.delaunay import DelaunayTri
from scipy.spatial import Delaunay
from scipy.spatial import distance_matrix
import numpy as np
import time
# FROM PROJECT
from vertices_functions import *
from geometric_features import road
from vector_operations import *
from output import *
from plots import *

##----------------------------------------- ADDING POINTS, EDGES AND ROADS -------------------------------------------- <- SUBSTEPS OF EVOLVE STREETS                            
    ## STARTS WITH CONNECTED POINTS (DEPRECATED)




def add_initial_points2graph(planar_graph,r0,number_nodes,side_city):
    '''
        This functions constructs an initially connected graph -> I am leaving it, but it is deprecated
        NOTE: Once I add points I need to set new point to end_point, the previous point not anymore end_points,
        1) Check if important point is still active
        2) Add road
    '''
    x,y = generate_exponential_distribution_nodes_in_space_square(r0,number_nodes,side_city)
    initialize_distance_matrix(planar_graph,x,y)
    for point_idx in range(len(x)):
        add_vertex(planar_graph)
        vertex = get_last_vertex(planar_graph)
        id_ = get_id_last_vertex(planar_graph)
        set_id(planar_graph,vertex,id_)
        set_initialize_x_y_pos(planar_graph,vertex,x,y,point_idx)
        set_important_node(planar_graph,vertex,True)
        set_active_vertex(planar_graph,vertex,True)
        set_id(planar_graph,vertex,id_)
        set_relative_neighbors(planar_graph,vertex,[])
        set_end_point(planar_graph,vertex,True)
        set_in_graph(planar_graph,vertex,True)
        set_is_intersection(planar_graph,vertex,False)
    for vertex in planar_graph.graph.vertices():
        for vertex1 in planar_graph.graph.vertices():
            if vertex != vertex1 and is_graph_connected(planar_graph.graph) == False:        
                edge = get_edge(planar_graph,vertex,vertex1)
                set_length(planar_graph,edge)
                set_direction(planar_graph,edge)
                set_real_edge(planar_graph,edge,False)

## STARTS WITH DETACHED CENTERS
def add_centers2graph(planar_graph,r0,number_nodes,side_city):
    '''
        For each center generated:
            1) Add to graph
            2) Give it a name
            3) Initialize position
            4) Set it as important node (each important node attracts points in graph)
            5) Set it as active (It will deactivate just when all the attracted road reach it)

    '''
    print_geometrical_info(planar_graph)
    x,y = generate_exponential_distribution_nodes_in_space_square(r0,number_nodes,side_city)
#        x,y = planar_graph.city_box.contains_vector_points(np.array([x,y]).T)
    if planar_graph.distance_matrix_ is None:
        initialize_distance_matrix(planar_graph,x,y)
        for point_idx in range(len(x)):
            add_vertex(planar_graph)
            id_ = get_id_last_vertex(planar_graph)
            vertex = get_last_vertex(planar_graph)
            set_id(planar_graph,vertex,id_)
            set_newly_added(planar_graph,vertex,True)
            set_initialize_x_y_pos(planar_graph,vertex,x,y,point_idx)
            set_important_node(planar_graph,vertex,True)
            set_active_vertex(planar_graph,vertex,True)
            ## Will they be attracted? Yes As they are the first created centers
            set_in_graph(planar_graph,vertex,True)
            set_end_point(planar_graph,vertex,True)
            ## RELATIVE NEIGHBOR, ROADS starting from it.                
            set_relative_neighbors(planar_graph,vertex,[])
            set_road(planar_graph,vertex,[])
            ## Intersection
            set_is_intersection(planar_graph,vertex,False)
            print_properties_vertex(planar_graph,vertex)            
    else:
        for point_idx in range(len(x)):
            update_distance_matrix(planar_graph,np.array([planar_graph.graph.vp['x'].a,planar_graph.graph.vp['y'].a]).T,np.array([[x[point_idx],y[point_idx]]]))
            add_vertex(planar_graph)
            vertex = get_last_vertex(planar_graph)
            id_ = get_id_last_vertex(planar_graph)
            set_id(planar_graph,vertex,id_)
            set_newly_added(planar_graph,vertex,True)
            set_initialize_x_y_pos(planar_graph,vertex,x,y,point_idx)
            set_important_node(planar_graph,vertex,True)
            set_active_vertex(planar_graph,vertex,True)
            ## Will they be attracted? No
            set_in_graph(planar_graph,vertex,False)
            set_end_point(planar_graph,vertex,False)
            ## RELATIVE NEIGHBOR, ROADS starting from it.
            set_relative_neighbors(planar_graph,vertex,[])
            set_road(planar_graph,vertex,[])
            ## Intersection
            set_is_intersection(planar_graph,vertex,False)
            print_properties_vertex(planar_graph,vertex)            


def add_point2graph(planar_graph,source_vertex,dx,dy):
    '''
        Adds a point from the graph and initializes x,y with the vector dx,dy
        The added node is:
            1) Not important
            2) Not active
            3) End point
            4) With an empty set of relative neighbors
            5) In graph
            6) Added to a road
    '''
    x_new_node = [planar_graph.graph.vp['x'][source_vertex]+dx]
    y_new_node = [planar_graph.graph.vp['y'][source_vertex]+dy]
    ## INITIALIZE NEW VERTEX 
    add_vertex(planar_graph)
    id_ = get_id_last_vertex(planar_graph)
    vertex = get_last_vertex(planar_graph)
    set_id(planar_graph,vertex,id_)
    set_initialize_x_y_pos(planar_graph,vertex,x_new_node,y_new_node,0)
    set_important_node(planar_graph,vertex,False)
    set_active_vertex(planar_graph,vertex,False)
    set_relative_neighbors(planar_graph,vertex,[])
    set_end_point(planar_graph,vertex,True)
    set_in_graph(planar_graph,vertex,True)
    add_road(planar_graph,source_vertex,vertex)
    set_road(planar_graph,vertex,[])
    ## CHANGE INFO SOURCE VERTEX
    set_end_point(planar_graph,source_vertex,False)
    return get_last_vertex(planar_graph)


##-------------------------------------- DISTANCE MATRIX ------------------------------------------

def initialize_distance_matrix(planar_graph,x,y):
    planar_graph.distance_matrix_ = distance_matrix(np.array([x,y]).T,np.array([x,y]).T) #distance_matrix(np.array([planar_graph.graph.vp['x'],planar_graph.graph.vp['y']]).T,np.array([planar_graph.graph.vp['x'],planar_graph.graph.vp['y']]).T)

def update_distance_matrix(planar_graph,old_points,new_point):
    '''
        Description:
            For each new point added in the graph, adds the distance of this point to all the other points.
            In this way I do have a reduction of calculation from N_nodes*N_nodes to N_nodes
        new_point is a vector of dimension (1,2) -> the bottleneck of this function resides in np.vstack as I am adding the 0 of the 
        concatenation of the column and the row one after the other
    '''
#        print('expected distance matrix shape {0},{1}'.format(len(old_points),len(old_points)),np.shape(planar_graph.distance_matrix_))        
    dm_row = distance_matrix(new_point,old_points)
#        print('expected row matrix shape {0},{1}'.format(1,len(old_points)),np.shape(dm_row))
    dm_col = distance_matrix(old_points,new_point) # dimension (1,starting_points)
    dm_col = np.vstack((dm_col,[0]))        
#        print('expected col matrix shape {0},{1}'.format(len(old_points)+1,1),np.shape(dm_col))
    dm_conc = np.concatenate((planar_graph.distance_matrix_,dm_row),axis = 0) # dimension (starting_points+1,starting_popints)
#        print('expected conc matrix shape {0},{1}'.format(len(old_points)+1,len(old_points)),np.shape(dm_conc))
    planar_graph.distance_matrix_ = np.concatenate((dm_conc,dm_col),axis = 1)





## UPDATE 

def close_roads(planar_graph):
    '''
        Attach the edge if the growing points are close enough to their relative neighbors,
        in this way the relative neighbor becomes a growing point as well
    '''
    
    already_existing_edges = [[e.source(),e.target()] for e in planar_graph.graph.edges()]
    for r in planar_graph.list_roads:
        for active_v in planar_graph.active_vertices:
                ep = planar_graph.graph.vp['id'][r.end_point]
                if planar_graph.distance_matrix_[planar_graph.graph.vp['id'][active_v],ep] < planar_graph.rate_growth and [active_v,ep] not in already_existing_edges:
                    planar_graph.graph.add_edge(ep,active_v)
                    r.is_closed = True
                    r.end_point = active_v
                    print('closing road: ',r.id,' with end point: ',ep)
                    print_property_road(r)
                else:
                    pass


##---------------------------------------- EVOLUTION ----------------------------------------------------##

## ---------------------------------------- EVOLVE VERTEX ----------------------------------------------------##

def evolve_uniquely_attracted_vertex(planar_graph,growing_node,available_vertices,attracting_node):
    '''
        Evolve street from (in graph) points that have a unique attracting vertex 
    '''
    print_delauney_neighbors(planar_graph,growing_node)
    print('available attracting vertices: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.graph.vp['relative_neighbors'][growing_node]])
    old_points = np.array([planar_graph.graph.vp['x'].a,planar_graph.graph.vp['y'].a]).T
    vertex_relative_neighbor = available_vertices[0] # planar_graph.graph.vertex(int_idx_relative_neighbor)
    dx,dy = coordinate_difference(planar_graph.graph.vp['x'][vertex_relative_neighbor],planar_graph.graph.vp['y'][vertex_relative_neighbor],planar_graph.graph.vp['x'][growing_node],planar_graph.graph.vp['y'][growing_node])
    dx,dy = normalize_vector(dx,dy)
    dx,dy = scale(dx,dy,planar_graph.rate_growth)
    new_point = np.array([[planar_graph.graph.vp['x'][growing_node] +dx,planar_graph.graph.vp['y'][growing_node] +dy]])
    update_distance_matrix(planar_graph,old_points,new_point)
    added_vertex = add_point2graph(planar_graph,growing_node,dx,dy)
    add_edge2graph(planar_graph,planar_graph.graph.vertex(growing_node),added_vertex,vertex_relative_neighbor)
    ## Find the road where the points belongs to (looking at the starting vertex it is generated from)
    add_point2road(planar_graph,growing_node,added_vertex,attracting_node)
    plot_relative_neighbors(planar_graph,growing_node,attracting_node,added_vertex,available_vertices)
    print('EVOLVING UNIQUE ATTRACTOR')
    print('direction: ',planar_graph.graph.vp['id'][vertex_relative_neighbor])
    print(' dx: ',dx,' dy: ',dy) 
    print('added vertex: ',planar_graph.graph.vp['id'][added_vertex],' coords: ',planar_graph.graph.vp['pos'][added_vertex])

def evolve_multiply_attracted_vertices(planar_graph,growing_node,available_vertices,attracting_node):
    '''
        Evolve street from (in graph) points that have a multiple attracting vertices
    '''
    print_delauney_neighbors(planar_graph,growing_node)
    print('available attracting vertices: ',[planar_graph.graph.vp['id'][v] for v in available_vertices])
    dx = 0
    dy = 0
    for neighbor_attracting_vertex in range(len(available_vertices)):
        old_points = np.array([planar_graph.graph.vp['x'].a,planar_graph.graph.vp['y'].a]).T
        vertex_relative_neighbor = available_vertices[neighbor_attracting_vertex]#planar_graph.graph.vertex(int_idx_relative_neighbor)
        x,y = coordinate_difference(planar_graph.graph.vp['x'][vertex_relative_neighbor],planar_graph.graph.vp['y'][vertex_relative_neighbor],planar_graph.graph.vp['x'][growing_node],planar_graph.graph.vp['y'][growing_node])
        dx += x 
        dy += y
    if np.sqrt(dx**2+dy**2)!=0: 
        dx,dy = normalize_vector(dx,dy)
        dx,dy = scale(dx,dy,planar_graph.rate_growth)
        new_point = np.array([[planar_graph.graph.vp['x'][growing_node] +dx,planar_graph.graph.vp['y'][growing_node] +dy]])
        update_distance_matrix(planar_graph,old_points,new_point)
        added_vertex = add_point2graph(planar_graph,growing_node,dx,dy)
        add_edge2graph(planar_graph,planar_graph.graph.vertex(growing_node),added_vertex,available_vertices)
        add_point2road(planar_graph,growing_node,added_vertex,attracting_node)
        plot_relative_neighbors(planar_graph,growing_node,attracting_node,added_vertex,available_vertices)                            
        print('EVOLVING SUM ATTRACTOR')
        print('direction sum of : ',[planar_graph.graph.vp['id'][vertex_relative_neighbor] for vertex_relative_neighbor in available_vertices])
        print(' dx: ',dx,' dy: ',dy) 
        print('added vertex: ',planar_graph.graph.vp['id'][added_vertex],' coords: ',planar_graph.graph.vp['pos'][added_vertex])
    else:
        print('EVOLVING IN ALL DIRECTIONS DUE TO DEGENERACY')
        for neighbor_attracting_vertex in range(len(available_vertices)):
            intersection = True
            vertex_relative_neighbor = available_vertices[neighbor_attracting_vertex]#planar_graph.graph.vertex(int_idx_relative_neighbor)
#                            vertex_relative_neighbor = planar_graph.graph.vertex(int_idx_relative_neighbor)
            dx,dy = coordinate_difference(planar_graph.graph.vp['x'][vertex_relative_neighbor],planar_graph.graph.vp['y'][vertex_relative_neighbor],planar_graph.graph.vp['x'][growing_node],planar_graph.graph.vp['y'][growing_node])
            dx,dy = normalize_vector(dx,dy)
            dx,dy = scale(dx,dy,planar_graph.rate_growth)
            new_point = np.array([[planar_graph.graph.vp['x'][growing_node] +dx,planar_graph.graph.vp['y'][growing_node] +dy]])
            update_distance_matrix(planar_graph,old_points,new_point)
            added_vertex = add_point2graph(planar_graph,growing_node,dx,dy)
            add_edge2graph(planar_graph,planar_graph.graph.vertex(growing_node),added_vertex,neighbor_attracting_vertex)  
            add_point2road(planar_graph,growing_node,added_vertex,attracting_node,intersection)
            plot_relative_neighbors(planar_graph,growing_node,attracting_node,added_vertex,available_vertices)
            print('direction sum of : ',planar_graph.graph.vp['id'][vertex_relative_neighbor])
            print(' dx: ',dx,' dy: ',dy) 
            print('added vertex: ',planar_graph.graph.vp['id'][added_vertex],' coords: ',planar_graph.graph.vp['pos'][added_vertex])

## ---------------------------------------- EVOLVE STREET ----------------------------------------------------##

def evolve_street_old_attractors(planar_graph):
    '''
        Evolve street for the old attractors
    '''
    ## TODO: differentiate between intersections,important nodes and normal nodes
    already_grown_vertices = []
    for attracting_node in planar_graph.old_attracting_vertices:
        print('------------------------')
        print_properties_vertex(planar_graph,attracting_node)
        print('attracting node: ',planar_graph.graph.vp['id'][attracting_node])
        print('attracted vertices: ',[planar_graph.graph.vp['id'][relative_neighbor] for relative_neighbor in planar_graph.graph.vp['relative_neighbors'][attracting_node]])
        ## if the relative neighbor is just one
        for growing_node in planar_graph.graph.vp['relative_neighbors'][attracting_node]:

            if growing_node not in already_grown_vertices:
                available_vertices = planar_graph.graph.vp['relative_neighbors'][attracting_node]
                number_relative_neighbors_growing_node = len(planar_graph.graph.vp['relative_neighbors'][attracting_node])
                print('growing node: ',planar_graph.graph.vp['id'][growing_node],' coords: ',planar_graph.graph.vp['pos'][growing_node])
                ## Take if relative neighbor of the growing node is just one  
                if number_relative_neighbors_growing_node==1:
                    evolve_uniquely_attracted_vertex(planar_graph,growing_node,available_vertices,attracting_node)
                elif number_relative_neighbors_growing_node>1: # a relative neighbor is also a neighbor -> kill the growth
                    evolve_multiply_attracted_vertices(planar_graph,growing_node,available_vertices,attracting_node)
            else: 
                pass
            if growing_node not in already_grown_vertices:
                already_grown_vertices.append(growing_node)
                vv = [planar_graph.graph.vp['id'][growing_node] for growing_node in already_grown_vertices]
                vv = np.sort(vv)


def evolve_street_newly_added_attractors(planar_graph):
    '''
        Needed prior steps:
            1) update_list_newly_added_attracting_vertices
            2) update_list_growing_vertices
            3) compute_rng

    '''
    already_grown_vertices = []
    for attracting_node in planar_graph.newly_added_attracting_vertices:
        ## if the relative neighbor is just one and it is not already attached tothe graph (neigbor['growing'] == False)
        print('------------------------')
        print('attracting node: ',planar_graph.graph.vp['id'][attracting_node])
        print('attracted vertices: ',[planar_graph.graph.vp['id'][relative_neighbor]for relative_neighbor in planar_graph.graph.vp['relative_neighbors'][attracting_node]])
        ## if the relative neighbor is just one
        for growing_node in planar_graph.graph.vp['relative_neighbors'][attracting_node]:
            if growing_node not in already_grown_vertices:
                available_vertices = planar_graph.graph.vp['relative_neighbors'][attracting_node]
                number_relative_neighbors_growing_node = len(planar_graph.graph.vp['relative_neighbors'][attracting_node])
                print('growing node: ',planar_graph.graph.vp['id'][growing_node],' coords: ',planar_graph.graph.vp['pos'][growing_node])
                ## Take if relative neighbor of the growing node is just one  
                if number_relative_neighbors_growing_node==1:
                    evolve_uniquely_attracted_vertex(planar_graph,growing_node,available_vertices,attracting_node)
                elif number_relative_neighbors_growing_node>1: # a relative neighbor is also a neighbor -> kill the growth
                    evolve_multiply_attracted_vertices(planar_graph,growing_node,available_vertices,attracting_node)
            else: 
                pass
            if growing_node not in already_grown_vertices:
                already_grown_vertices.append(growing_node)
                vv = [planar_graph.graph.vp['id'][growing_node] for growing_node in already_grown_vertices]
                vv = np.sort(vv)
    
