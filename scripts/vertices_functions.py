from scipy.spatial import Voronoi
import numpy as np
from shapely import LineString,Point
from termcolor import cprint
# FROM PROJECT
from output import *
from geometric_features import road
## ------------------------------------- VERTICES -------------------------------------

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

def set_road(planar_graph,vertex,road):
    '''
        It adds the road id to the property map roads of the vertex if it is not growing but just starting. 
        NOTE: THAT IS THE BEHAVIOR WHENEVER I EVOLVE_STREET
    '''
    if road == []:
        planar_graph.graph.vp['roads'][vertex] = []
    else:
        planar_graph.graph.vp['roads'][vertex].append(road)

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

def update_list_important_vertices(planar_graph,debug = False):
    planar_graph.important_vertices = [v for v in planar_graph.graph.vertices() if planar_graph.graph.vp['important_node'][v] == True]
    if debug:
        print('\t\tUpdating list important vertices')
        print('IMPORTANT VERTICES: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.important_vertices])
def update_list_in_graph_vertices(planar_graph,debug=False):
    '''
        List of points to consider when deciding old attracting vertices relative neighbor
    '''
    planar_graph.is_in_graph_vertices = [v for v in planar_graph.graph.vertices() if planar_graph.graph.vp['is_in_graph'][v] == True]
    if debug:
        print('\t\tUpdating list in graph vertices')
        print('IN GRAPH VERTICES: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.is_in_graph_vertices])
def update_list_end_points(planar_graph,debug = False):
    '''
        List of points to consider when deciding old attracting vertices relative neighbor
        NOTE: already_evolved_end_points is used to avoid to consider the end points that have already evolved, in particular
        every time evolve_(new,old)_road is called I add the vertex that evolved.
        And before evolving I check that the vertex is not on the list. Every time we end the evolution
        of all the roads the update_lists_rng will be called and the list emptied
    '''
    planar_graph.already_evolved_end_points = []
    planar_graph.end_points = [v for v in planar_graph.graph.vertices() if planar_graph.graph.vp['end_point'][v] == True]
    if debug:
        print('\t\tUpdating list end points')
        print('END POINTS: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.end_points])
## ACTIVE VERTICES  

def update_list_old_attracting_vertices(planar_graph,debug = False):
    '''
        These are the vertices that attract just end points. 
            -Delauney triangulation just among these and the end points in graph
    '''
    planar_graph.old_attracting_vertices = [v for v in planar_graph.graph.vertices() if is_active(planar_graph,v) and not is_newly_added(planar_graph,v)]
    if debug:
        print('\t\tUpdating list old attracting vertices')
        print('OLD ATTRACTING VERTICES: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.old_attracting_vertices])
def update_list_newly_added_attracting_vertices(planar_graph,debug = False):
    planar_graph.newly_added_attracting_vertices = [v for v in planar_graph.graph.vertices() if is_active(planar_graph,v) and is_newly_added(planar_graph,v)]
    if debug:   
        print('\t\tUpdating list newly added attracting vertices')
        print('NEWLY ADDED ATTRACTING VERTICES: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.newly_added_attracting_vertices])
def update_list_intersection_vertices(planar_graph,debug=False):
    planar_graph.intersection_vertices = [v for v in planar_graph.graph.vertices() if planar_graph.graph.vp['intersection'][v] == True]
    if debug:
        print('\t\tUpdating list intersection vertices')
        print('INTERSECTION VERTICES: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.intersection_vertices])
def update_list_plausible_starting_point_of_roads(planar_graph,debug = False):
    planar_graph.plausible_starting_road_vertices = [v for v in planar_graph.graph.vertices() if (is_important_node(planar_graph,v) or is_intersection(planar_graph,v)) and is_in_graph(planar_graph,v)]
    if debug:
        print('\t\tUpdating list plausible starting point of roads')
        print('PLAUSIBLE STARTING POINT OF ROADS: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.plausible_starting_road_vertices])

## LIST ACTIVATING POINT

def update_list_active_vertices(planar_graph,debug = False):
    '''
        For each road in the network, if the road is not closed, add the points that are activating it, given they are not already in the list
    '''
    planar_graph.active_vertices = [v for v in planar_graph.important_vertices if planar_graph.graph.vp['is_active'][v]]
    if debug:
        print('\t\tUpdating list active vertices')
        print('ACTIVE VERTICES: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.active_vertices])

##
def update_list_vertices_starting_roads(planar_graph,debug = False):
    '''
        For each road in the network, if the road is not closed, add the points that are activating it, given they are not already in the list
    '''
    planar_graph.vertices_starting_roads = [v for v in planar_graph.graph.vertices() if is_intersection(planar_graph,v) or (is_important_node(planar_graph,v) and is_in_graph(planar_graph,v))]
    if debug:
        print('\t\tUpdating list vertices starting roads')
        print('VERTICES STARTING ROADS: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.vertices_starting_roads])

# This piece must be inserted to take update the roads, as I need each point that is evolving to have an attraction set

##--------------------------------- SET FUNCTION ---------------------------------##

def assign_type_road(planar_graph,initial_node,r,debug=False):
    '''
        Description:
            Assign the type of road. This function will be useful to define the capacity and the velocity of a road. If
            the road is secondary or tertiary etc. it will have smaller velocity. type means how many intersection from the 
            starting important nodes we have.
        Input:
            initial_node: vertex
    '''
    if debug:
        print('\t\t\t\t\tASSIGNING TYPE ROAD')        
    if is_important_node(planar_graph,initial_node):
        r.previous_roads = []
        pass
    elif is_intersection(planar_graph,initial_node):
        backward_initial_node = initial_node
        r.type_ = 0
        r.previous_roads = []
        while(not is_important_node(planar_graph,backward_initial_node)):
            r.type_ += 1
            backward_initial_node,_,found,id_ = find_road_vertex(planar_graph,backward_initial_node) 
            if found==False or id_ is None:
                raise ValueError('ERROR: New road has started from nowhere')
            r.previous_roads.append(id_)
    else:
        raise ValueError('ERROR: New road started neither from intersection nor important road: assign_type_road')
    ## CAPACITY_LEVEL
    r.capacity_level = 0
    if debug:
        print('ROAD TYPE: ',r.type_)
        print('ROAD PREVIOUS ROADS: ',r.previous_roads)        


##----------------------------- SHAPELY FUNCTIONS -----------------------------##
def get_linestring(planar_graph,r):
    '''
        Produce the linestring of the road
    '''
    r.line = []
    for v in r.list_nodes:
        r.line.append(Point(planar_graph.graph.vp['pos'][v]))
    r.line = LineString(r.line)
    return r.line


## ------------------------------------- VERtiCES AND ROADS ------------------------------------- ##




def find_road_vertex(planar_graph,vertex,debug = False):
    '''
        Description:
            Find the road that starts from vertex
        Output:
            starting_vertex of the road
            local_idx_road: index of the road in the list of roads starting from starting_vertex
        Complexity:
            O(number_vertices)
    '''
    if debug:
        print('FINDING ROAD VERTEX: {}'.format(planar_graph.graph.vp['id'][vertex]))
    found = False
    for starting_vertex in planar_graph.plausible_starting_road_vertices:
        if debug:
            print('\tPlausible starting vertex: {}'.format(planar_graph.graph.vp['id'][starting_vertex]))
        for r in planar_graph.graph.vp['roads'][starting_vertex]:
            if debug:
                print('\t\tRoad: {}'.format(r.id))
            local_idx_road = 0
            if r.in_road(vertex):
                found = True
                if debug:
                    print('\t\t\tFOUND ROAD')
                    print('\t\t\tRoad: ',r.id,' local_idx_road: ',local_idx_road,)
                return starting_vertex,local_idx_road,found,r.id
            else:
                print(planar_graph.graph.vp['id'][vertex],' not in road')
            local_idx_road += 1
    return starting_vertex,None,found,None
'''
def create_road(planar_graph,source_vertex,target_vertex,activation_vertices,type_=0):
    
        Creates the road and adds to the list of available roads on the graph.
        The roads are properties of vertices.
        The global_counting_roads is updated just here, as other 
        cases are just updating the existing roads.
    
    new_road = road(source_vertex,target_vertex,planar_graph.global_counting_roads,activation_vertices)
    planar_graph.graph.vp['roads'][source_vertex].append(new_road)
    planar_graph.global_counting_roads += 1

    return planar_graph.graph.vp['roads'][source_vertex][-1]
'''

## ------------------------------------------ UPDATE FUNCTIONS ------------------------------------------ ##

'''def update_intersections(planar_graph):
    
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
    
    ## For each newly added attracting vertex
    for n_attracting_vertex in planar_graph.newly_added_attracting_vertices:
        for attracted_vertex_idx in planar_graph.graph.vp['relative_neighbors'][n_attracting_vertex]:
            attracted_vertex = planar_graph.graph.vertex(attracted_vertex_idx)
            ## If the attracted vertex (belonging to the relative neighbor) is not an end point, and is not an important node (then It can be just growing)
            if not is_end_point(planar_graph,attracted_vertex) and not is_important_node(planar_graph,attracted_vertex):
                starting_vertex,local_idx,found,id_ = find_road_vertex(planar_graph,attracted_vertex)
                if found:
                    planar_graph.graph.vp['roads'][starting_vertex].append(road(starting_vertex,planar_graph.global_counting_roads,n_attracting_vertex))
                    planar_graph.graph.vp['roads'][starting_vertex][-1].copy_road_specifics(planar_graph,planar_graph.graph.vp['roads'][starting_vertex][local_idx])
                    planar_graph.global_counting_roads += 1

                else:
                    pass
'''

## ------------------------------------------ EDGES ------------------------------------------ ##

####-------------------------------------------- SET FUNCTIONS --------------------------------------------####
def set_length(planar_graph,edge):
    planar_graph.graph.ep['length'][edge] = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][edge.source()],planar_graph.graph.vp['id'][edge.target()]]

def set_direction(planar_graph,edge):
    planar_graph.graph.ep['direction'][edge] = planar_graph.graph.vp['pos'][edge.target()].a - planar_graph.graph.vp['pos'][edge.source()].a

def set_real_edge(planar_graph,edge,boolean):
    planar_graph.graph.ep['real_edge'][edge] = boolean

##-------------------------------------------- GET FUNCTIONS --------------------------------------------##
def get_edge(planar_graph,vertex,vertex1):
    '''
        Returns the edge between vertex and vertex1
    '''
    return planar_graph.graph.edge(vertex,vertex1)



##---------------------------------------- ROAD OPERATIONS ---------------------------------------------

def add_edge2graph(planar_graph,source_vertex,target_vertex,attracting_vertices,intersection,debug = False):
    '''
        source_idx: Vertex
        target_idx: Vertex
        Description:
            1) Add the edge
            2) Update the ep of the edge
            3) Add a new road if the starting vertex is important or intersection
            4) Upgrade the existing road otherwise
            5) Control that the source vertex is an important vertex or an intersection:
                NOTE: These quantities are controlled in the evolving step in either (old and new)
                NOTE: If the attractor is old -> then no possibility of having intersection (the evaluation of intersection does that.)
                NOTE: If the attractor is new -> then I need to check if the source vertex is an intersection
                NOTE: An intersection happen, iff starting vertex is not an end point (it can be attracted JUST by a new attracting vertex for geometric reasons)
                5a) Yes: Means that a new road is starting
                5b) No: Means that the road is already started and I need to find the road where the source vertex belongs to
                    5b1) I check if the source vertex is in the road and add the target vertex to it 
    '''
    if debug:
        print('ADDING EDGE TO GRAPH')
    source_idx = planar_graph.graph.vp['id'][source_vertex]
    target_idx = planar_graph.graph.vp['id'][target_vertex]
    planar_graph.graph.add_edge(source_idx,target_idx)
    planar_graph.graph.ep['distance'][planar_graph.graph.edge(source_idx,target_idx)] = planar_graph.distance_matrix_[source_idx,target_idx]
    planar_graph.graph.ep['direction'][planar_graph.graph.edge(source_idx,target_idx)] = planar_graph.graph.vp['pos'][target_vertex].a - planar_graph.graph.vp['pos'][source_vertex].a
    planar_graph.graph.ep['real_edge'][planar_graph.graph.edge(source_idx,target_idx)] = False
    if debug:
        print('source index: ',source_idx)
        print('target index: ',target_idx)
    if intersection:
        if debug:
            print('+++ ADDING INTERSECTION +++')
        set_is_intersection(planar_graph,source_vertex,True)
        new_road = road(source_vertex,target_vertex,planar_graph.global_counting_roads,attracting_vertices,type_ = 0,unit_length = planar_graph.rate_growth)
        planar_graph.graph.vp['roads'][source_vertex].append(new_road)
        planar_graph.global_counting_roads += 1
        assign_type_road(planar_graph,source_vertex,new_road,debug)
        if debug:
            print_property_road(planar_graph,new_road)        
    elif is_important_node(planar_graph,source_vertex):
        if debug:
            print('+++ ADDING NEW ROAD +++')
        new_road = road(source_vertex,target_vertex,planar_graph.global_counting_roads,attracting_vertices,type_ = 0,unit_length = planar_graph.rate_growth)
        planar_graph.graph.vp['roads'][source_vertex].append(new_road)
        planar_graph.global_counting_roads += 1
        if debug:
            print_property_road(planar_graph,new_road)
    else:
        ## If the vertex is neither important nor intersection, then I add the vertex 2 the road
        ## An ending point belongs just to one road, whose type will not change
        if debug:
            print('CONTINUING ROAD')
        starting_vertex_road,local_idx_road,found,id_ = find_road_vertex(planar_graph,source_vertex,debug)
        if found:
            planar_graph.graph.vp['roads'][starting_vertex_road][local_idx_road].add_node_in_road(source_vertex,target_vertex,planar_graph.distance_matrix_[source_idx,target_idx])
            if debug:
                print('+++ Added node in road +++')
                print_property_road(planar_graph,planar_graph.graph.vp['roads'][starting_vertex_road][local_idx_road])
        else:
            raise ValueError('ERROR: The point is growing, but has no road')
#TODO: Fix the generation of intersections, when a new center is added, and generates a new road, the point when the road starts, is 
#  Inersection, new kind of NODE, this node, is the beginning of a new road.
# I need a new variable in road() -> type_starting_point: ['important_node','intersection']] 



def get_list_nodes_in_roads_starting_from_v(planar_graph,v,debug=False):
    '''
        Output:
            List of vertices that are in the roads starting from v
            type: list vertex
    '''
    points_in_adjacent_roads_v = []
    for r in planar_graph.graph.vp['roads'][v]:
        for v_road in r.list_nodes:
            points_in_adjacent_roads_v.append(v_road)
    if debug:
        cprint('GET LIST NODES IN ROADS','red','on_white')
        cprint('POINTS IN ADJACENT ROADS TO {}: ' + str(v) + str([planar_graph.graph.vp['id'][v] for v in points_in_adjacent_roads_v]),'red','on_white')
    return points_in_adjacent_roads_v


## ROADS

def update_list_roads(planar_graph,debug = False):
    '''
        Updates the list of roads in the graph:
        NOTE: call update_list_active_vertices after -> as each road is activated by an active vertex
    '''
    if debug:
        print('UPDATING LIST ROADS')
    planar_graph.list_roads = []
    for v in planar_graph.graph.vertices():
        for r in planar_graph.graph.vp['roads'][v]:
            if debug:
                print('\tconsidering road:',r.id)
#                print_property_road(planar_graph,r) 
            planar_graph.list_roads.append(r)   
    if debug:
        print('\t\t\tList of roads')
#        for r in planar_graph.list_roads:
#            print_property_road(planar_graph,r) 


def update_list_active_roads(planar_graph,debug = False):
    '''
        Updates the list of active roads in the graph,
        These are the roads that are not closed.
        Among these I need to check the end points such that close_road 
    '''
    if debug:
        print('\t\tUpdating list active roads')
    planar_graph.list_active_roads = []
    planar_graph.plausible_starting_road_vertices = [v for v in planar_graph.graph.vertices() if (is_important_node(planar_graph,v) or is_intersection(planar_graph,v)) and is_in_graph(planar_graph,v)]
    for v in planar_graph.plausible_starting_road_vertices:
        for r in planar_graph.graph.vp['roads'][v]:
            if r.is_closed_ == False:
                planar_graph.list_active_roads.append(r)
    if debug:
        print('\t\t\tList of active roads')
        for r in planar_graph.list_active_roads:
            print_property_road(planar_graph,r)