import time
from collections import defaultdict
import graph_tool as gt
from graph_tool.all import label_components
from scipy.spatial import distance_matrix,Voronoi
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from gi.repository import Gtk, Gdk, GdkPixbuf, GObject, GLib
import multiprocessing as mp
from pyhull.delaunay import DelaunayTri
from scipy.spatial import Delaunay
from geometric_features import Grid,road
from vector_operations import normalize_vector,coordinate_difference,scale
'''
1) Generates the first centers [they are all of interest and they are all attracting and are all growing]
2) For each growing I compute the relative neighbor (if it is growing iself and not a center, then I give up, otherwise I continue)
'''

def ifnotexistsmkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
## GLOBAL

def initial_graph():
    '''
    graph whose vertex properties are:
        id: int
        x: double
        y: double
        relative_neighbors: vector<int>
        important_node: bool (if the node is a node True, else is a point)
        connected: bool (if the node needs to be attached to the net True,
                    once the road is completed, the point is added to the net)
    '''
    g = gt.Graph(directed=True)
    ## NODES
    ## ID: int
    g.vp['id'] = g.new_vertex_property('int')
    ## IMPORTANT: bool (These are the points of interest)
    g.vp['important_node'] = g.new_vertex_property('bool')
    g.vp['is_active'] = g.new_vertex_property('bool')
    g.vp['newly_added_center'] = g.new_vertex_property('bool')
    ## GROWING VERTICES 
    g.vp['end_point'] = g.new_vertex_property('bool')   
    g.vp['is_in_graph'] = g.new_vertex_property('bool')
    ## ATTRACTED: list attracted (Is the set of indices of vertices that are attracted by the node)
    g.vp['list_nodes_attracted'] = g.new_vertex_property('vector<int>')
    
    ## GROWING: bool (they are attached to the growing net, street evolve from this set)
    g.vp['growing'] = g.new_vertex_property('bool')
    # Maybe of no use
    g.vp['growing_and_attracting'] = g.new_vertex_property('bool')
    g.vp['growing_and_not_attracting'] = g.new_vertex_property('bool')
    g.vp['attracting_and_not_growing'] = g.new_vertex_property('bool')

    ## INTERSECTION: bool
    ## role: for growing not important nodes, when addnewcenter: compute_rng, if the vertex in rng is attracted by it, the open intersection
    g.vp['intersection'] = g.new_vertex_property('bool')
    g.vp['attracted_by'] = g.new_vertex_property('vector<int>')
    ## RELATIVE NEIGHBORS: list (Is the set of indices of vertices that are relative neighbors of the attracting node)
    g.vp['relative_neighbors'] = g.new_vertex_property('vector<int>')
    ## positions of the vertices
    g.vp['x'] = g.new_vertex_property('double')
    g.vp['y'] = g.new_vertex_property('double')
    g.vp['pos'] = g.new_vertex_property('vector<double>')
    
    ## VORONOI diagram of the vertices 
    g.vp['voronoi'] = g.new_vertex_property('object')
    g.vp['new_attracting_delauney_neighbors'] = g.new_vertex_property('vector<int>')
    g.vp['old_attracting_delauney_neighbors'] = g.new_vertex_property('vector<int>')
    ## Set of roads starting at the node
    g.vp['roads_belonging_to'] = g.new_vertex_property('vector<int>') # used to new nodes to the right road.
    g.vp['roads'] = g.new_vertex_property('object') # is a list of roads
    
    ## EDGES
    growth_unit_vect = g.new_edge_property('double')
    g.ep['distance'] = growth_unit_vect
    growth_unit_vect = g.new_edge_property('vector<double>')
    g.ep['direction'] = growth_unit_vect
    real_edges = g.new_edge_property('bool')
    g.ep['real_edge'] = real_edges
    capacity = g.new_edge_property('double')    
    
    ## Animation
    state = g.new_vertex_property('vector<double>')
    g.vp['state'] = state

    return g
    
    
def generate_exponential_distribution_nodes_in_space_square(r0,number_nodes,side_city):
    '''
        Citation:
            The spatial structure of networks: Michael T. Gastner and M. E. J. Newman
        Description:
            Generate a uniform distribution of points in a circle of radius radius_city
            Initialize postiions and distance matrix
            ALL in one step:        
    '''
    
    r = np.random.default_rng().exponential(scale = r0,size = int(number_nodes))*side_city
    theta = np.random.random(int(number_nodes))*side_city    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y


def is_graph_connected(graph):
    """
    Check if a graph is connected using graph-tool.

    Parameters:
    - graph: graph_tool.Graph
        The input graph.

    Returns:
    - bool
        True if the graph is connected, False otherwise.
    """
    # Label connected components
    components, _ = label_components(graph)
    # Check if there is only one component
    return len(set(components.a)) == 1

 ## BARTHELEMY GRAPH
class barthelemy_graph:
    def __init__(self,config:dict,r0):
        self.config = config
        self.starting_phase = True
        ## Geometrical parameters
        self.side_city = config['side_city'] # in km
        self.ratio_growth2size_city = config['ratio_growth2size_city'] # 1000 segments  to go from one place to the other (pace of 10 meters) 
        self.rate_growth = self.ratio_growth2size_city*self.side_city
        self.r0 = r0
        ## dynamical parameters
        self.number_iterations = config['number_iterations'] #
        self.tau_c = config['tau_c'] # 
        self.number_nodes_per_tau_c = config['number_nodes_per_tau_c'] 
        ## creation rates of nodes
        self.initial_number_points = config['initial_number_points']
        self.total_number_attraction_points = config['total_number_attraction_points'] # these are the auxins
        self.ratio_evolving2initial = config['ratio_evolving2initial']
        self.total_number_nodes = self.number_iterations*self.number_nodes_per_tau_c + self.initial_number_points
        self.distance_matrix_ = None
        ## Roads 
        self.global_counting_roads = 0
        ## relative neighbors
        self.delauney_ = None
        ## policentricity
        self.set_policentricity()
        ## Grid
        self.number_grids = config['number_grids']
        # 
        #  animation
        self.offscreen = config['offscreen']
        ##
        self.initialize_base_dir()

    def set_policentricity(self):
        if 'degree_policentricity' in self.config.keys() and 'policentricity' in self.config.keys() and self.config['policentricity'] == True:
            self.policentricity = config['policentricity']
            self.number_centers = config['degree_policentricity']            
        else:
            self.policentricity = False
            self.number_centers = 1


##------------------------------------------- INITIALIZE BASE DIR -----------------------------------------------------------
    def initialize_base_dir(self):
        for root,_,_ in os.walk('.', topdown=True):
            pass
        self.base_dir = os.path.join(root,'output')
        ifnotexistsmkdir(self.base_dir)
        self.base_dir = os.path.join(self.base_dir,'n_tot_pts_{}'.format(self.total_number_attraction_points))
        ifnotexistsmkdir(self.base_dir)
        self.base_dir = os.path.join(self.base_dir,'n_tau_c_{}'.format(self.number_nodes_per_tau_c))
        ifnotexistsmkdir(self.base_dir)

        
##------------------------------------------------------------- PRINTING ----------------------------------------------------------
    def print_properties_vertex(self,vertex):
        '''
            Flushes all the properties of the vertex
        '''
        print('vertex: ',self.graph.vp['id'][vertex])
        print('is_active: ',self.graph.vp['is_active'][vertex])
        print('important_node: ',self.graph.vp['important_node'][vertex])
        print('end_point: ',self.graph.vp['end_point'][vertex])
        print('is_in_graph: ',self.graph.vp['is_in_graph'][vertex])
        print('newly_added_center: ',self.graph.vp['newly_added_center'][vertex])
        print('relative neighbors: ',self.graph.vp['relative_neighbors'][vertex])
        print('x: ',self.graph.vp['x'][vertex])
        print('y: ',self.graph.vp['y'][vertex])
        print('pos: ',self.graph.vp['pos'][vertex])
        print('out neighbor: ',[self.graph.vp['id'][v] for v in vertex.out_neighbours()])
        print('in neighbor: ',[self.graph.vp['id'][v] for v in vertex.in_neighbours()])
        for r in self.graph.vp['roads'][vertex]:
            print('road: ',r.id)
            print('number_iterations: ',r.number_iterations)
            print('length: ',r.length)
            print('list_nodes: ',[self.graph.vp['id'][v] for v in r.list_nodes])
            print('list_edges: ',[[self.graph.vp['id'][v1],self.graph.vp['id'][v2]] for v1,v2 in r.list_edges])
            print('end_node: ',self.graph.vp['id'][r.end_node])
            print('is_closed: ',r.is_closed)
            print('activated_by: ',[self.graph.vp['id'][v] for v in r.activated_by])

    def print_geometrical_info(self):
        print('*****************************')
        print('Side bounding box: {} km'.format(self.side_city))
#        print('Number square grid: ',len(self.grid))
#        print('Side single square: {} km'.format(self.side_city/np.sqrt(len(self.grid))))
        print('----------------------')
        print('Initial number points: ',self.initial_number_points)
        print('Total number of POIs expected: ',self.total_number_nodes)
        print('*******************************')

    def print_not_considered_vertices(self,not_considered):
        '''
        Type: Debug
        '''
        print('XXXX   vertices whose attracted by is not updated   XXXX')
        for v in not_considered:
            print('id: ',self.graph.vp['id'][v])
            print('important by: ',self.graph.vp['important_node'][v])
            print('attracting: ',self.graph.vp['attracting'][v])
            print('growing: ',self.graph.vp['growing'][v])
            print('end_point: ',self.graph.vp['end_point'][v])

    def print_delauney_neighbors(self,vi):
        print('old: ')
        old_dn = [v for v in self.graph.vertices() if len(self.graph.vp['old_attracting_delauney_neighbors'][v])!=0]
        for vj in old_dn:
            print(self.graph.vp['id'][vj])
            print('neighbors: ',self.graph.vp['old_attracting_delauney_neighbors'][vj])
        print('new: ')
        new_dn = [v for v in self.graph.vertices() if len(self.graph.vp['new_attracting_delauney_neighbors'][v])!=0]
        for vj in new_dn:
            print(self.graph.vp['id'][vj])
            print('neighbors: ',self.graph.vp['new_attracting_delauney_neighbors'][vj])



    def ASSERT_PROPERTIES_VERTICES(self,v):
        '''
            I here control that:
                1) If a vertex is an end node -> it must be in the graph
                2) If a vertex is not in the graph -> it must be active
        '''
        if self.graph.vp['end_point'] and not self.graph.vp['is_in_graph']:
            self.print_properties_vertex(v)
            raise ValueError('The END NODE vertex {} is not in the graph'.format(self.graph.vp['id'][v]))
        if not self.graph.vp['is_in_graph'] and not self.graph['is_active']:
            self.print_properties_vertex(v)
            raise ValueError('The vertex {} that is NOT in graph must be ACTIVE'.format(self.graph.vp['id'][v]))


## ------------------------------------------- FIRST INITIALiZATION ---------------------------------------------
    def initialize_graph(self):
        self.graph = initial_graph()

##----------------------------------------  SET FUNCTIONS ------------------------ (ACTING ON SINGLE VERTEX PROPERTY MAPS) -----------------------
    
##### "IMPORTANT NODE" FUNCTIONS #####

    def set_id(self,vertex,id_):
        self.graph.vp['id'][vertex] = id_

    def set_active_vertex(self,vertex,boolean):
        self.graph.vp['is_active'][vertex] = boolean

    def set_important_node(self,vertex,boolean):
        self.graph.vp['important_node'][vertex] = boolean

    def set_newly_added(self,vertex,boolean):
        self.graph.vp['newly_added_center'][vertex] = boolean

    def set_end_point(self,vertex,boolean):
        self.graph.vp['end_point'][vertex] = boolean

    def set_empty_relative_neighbors(self,vertex):
        self.graph.vp['relative_neighbors'][vertex] = []

    def set_in_graph(self,vertex,boolean):
        self.graph.vp['is_in_graph'][vertex] = boolean

    def set_empty_road(self,vertex):
        self.graph.vp['roads'][vertex] = []

    def set_is_intersection(self,vertex,boolean):
        self.graph.vp['intersection'][vertex] = boolean
#### "IN GRAPH NODE" FUNCTIONS #####
    def set_activate_in_graph(self,vertex):
        self.graph.vp['is_in_graph'][vertex] = True

    def set_initialize_x_y_pos(self,vertex,x,y,point_idx):
        self.graph.vp['x'][vertex] = x[point_idx]
        self.graph.vp['y'][vertex] = y[point_idx]
        self.graph.vp['pos'][vertex] = np.array([x[point_idx],y[point_idx]])

#### "EDGES" FUNCTIONS #####
    def set_length(self,edge):
        self.graph.ep['length'][edge] = self.distance_matrix_[self.graph.vp['id'][edge.source()],self.graph.vp['id'][edge.target()]]

    def set_direction(self,edge):
        self.graph.ep['direction'][edge] = self.graph.vp['pos'][edge.target].a - self.graph.vp['pos'][edge.source()].a

    def set_real_edge(self,edge,boolean):
        self.graph.ep['real_edge'][edge] = boolean


##----------------------------------------- ADDING POINTS, EDGES AND ROADS -------------------------------------------- <- SUBSTEPS OF EVOLVE STREETS                            
    ## STARTS WITH CONNECTED POINTS (DEPRECATED)


    def add_initial_points2graph(self,r0,number_nodes,side_city):
        '''
            This functions constructs an initially connected graph -> I am leaving it, but it is deprecated
            NOTE: Once I add points I need to set new point to end_point, the previous point not anymore end_points,
            1) Check if important point is still active
            2) Add road
        '''
        x,y = generate_exponential_distribution_nodes_in_space_square(r0,number_nodes,side_city)
        self.initialize_distance_matrix(x,y)
        for point_idx in range(len(x)):
            self.graph.add_vertex()
            vertex = self.graph.vertex(self.graph.num_vertices()-1)
            id_ = self.graph.num_vertices()-1
            self.set_initialize_x_y_pos(self,vertex,x,y,point_idx)
            self.set_important_node(vertex,True)
            self.set_active_vertex(vertex,True)
            self.set_id(vertex,id_)
            self.set_empty_relative_neighbors(vertex)
            self.set_end_point(vertex,True)
            self.set_in_graph(vertex,True)
            self.set_is_intersection(vertex,False)
        for vertex in self.graph.vertices():
            for vertex1 in self.graph.vertices():
                if vertex != vertex1 and is_graph_connected(self.graph) == False:
                    self.graph.add_edge(vertex,vertex1)
                    edge = self.graph.edge(vertex,vertex1)
                    self.set_length(edge)
                    self.set_direction(edge)
                    self.set_real_edge(self,edge,False)

    ## STARTS WITH DETACHED CENTERS
    def add_centers2graph(self,r0,number_nodes,side_city):
        '''
            For each center generated:
                1) Add to graph
                2) Give it a name
                3) Initialize position
                4) Set it as important node (each important node attracts points in graph)
                5) Set it as active (It will deactivate just when all the attracted road reach it)

        '''
        self.print_geometrical_info()
        x,y = generate_exponential_distribution_nodes_in_space_square(r0,number_nodes,side_city)
#        x,y = self.city_box.contains_vector_points(np.array([x,y]).T)
        if self.distance_matrix_ is None:
            self.initialize_distance_matrix(x,y)
            for point_idx in range(len(x)):
                self.graph.add_vertex()
                vertex = self.graph.vertex(self.graph.num_vertices()-1)
                id_ = self.graph.num_vertices()-1
                self.set_id(vertex,id_)
                self.set_initialize_x_y_pos(self,vertex,x,y,point_idx)
                self.set_important_node(vertex,True)
                self.set_active_vertex(vertex,True)
                ## Will they be attracted? Yes As they are the first created centers
                self.set_in_graph(vertex,True)
                self.set_end_point(vertex,True)
                ## RELATIVE NEIGHBOR, ROADS starting from it.                
                self.set_empty_relative_neighbors(vertex)
                self.set_empty_road(vertex)
                ## Intersection
                self.set_is_intersection(vertex,False)
        else:
            for point_idx in range(len(x)):
                self.update_distance_matrix(np.array([self.graph.vp['x'].a,self.graph.vp['y'].a]).T,np.array([[x[point_idx],y[point_idx]]]))
                self.graph.add_vertex()
                vertex = self.graph.vertex(self.graph.num_vertices()-1)
                id_ = self.graph.num_vertices()-1
                self.set_id(vertex,id_)
                self.set_initialize_x_y_pos(self,vertex,x,y,point_idx)
                self.set_important_node(vertex,True)
                self.set_active_vertex(vertex,True)
                ## Will they be attracted? No
                self.set_in_graph(vertex,False)
                self.set_end_point(vertex,False)
                ## RELATIVE NEIGHBOR, ROADS starting from it.
                self.set_empty_relative_neighbors(vertex)
                self.set_empty_road(vertex)
                ## Intersection
                self.set_is_intersection(vertex,False)
    
    
    def add_point2graph(self,source_vertex,dx,dy):
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
        x_new_node = [self.graph.vp['x'][source_vertex]+dx]
        y_new_node = [self.graph.vp['y'][source_vertex]+dy]
        ## INITIALIZE NEW VERTEX 
        self.graph.add_vertex()
        vertex = self.graph.vertex(self.graph.num_vertices()-1)
        id_ = self.graph.num_vertices()-1
        self.set_id(vertex,id_)
        self.set_initialize_x_y_pos(vertex,x_new_node,y_new_node,0)
        self.set_important_node(vertex,False)
        self.set_active_vertex(vertex,False)
        self.set_empty_relative_neighbors(vertex)
        self.set_end_point(vertex,True)
        self.set_in_graph(vertex,True)
        self.add_road(source_vertex,vertex)
        ## CHANGE INFO SOURCE VERTEX
        self.set_end_point(source_vertex,False)
        return self.graph.vertex(self.graph.num_vertices()-1)


##-------------------------------------- DISTANCE MATRIX ------------------------------------------

    def initialize_distance_matrix(self,x,y):
        self.distance_matrix_ = distance_matrix(np.array([x,y]).T,np.array([x,y]).T) #distance_matrix(np.array([self.graph.vp['x'],self.graph.vp['y']]).T,np.array([self.graph.vp['x'],self.graph.vp['y']]).T)

    def update_distance_matrix(self,old_points,new_point):
        '''
            Description:
                For each new point added in the graph, adds the distance of this point to all the other points.
                In this way I do have a reduction of calculation from N_nodes*N_nodes to N_nodes
            new_point is a vector of dimension (1,2) -> the bottleneck of this function resides in np.vstack as I am adding the 0 of the 
            concatenation of the column and the row one after the other
        '''
#        print('expected distance matrix shape {0},{1}'.format(len(old_points),len(old_points)),np.shape(self.distance_matrix_))        
        dm_row = distance_matrix(new_point,old_points)
#        print('expected row matrix shape {0},{1}'.format(1,len(old_points)),np.shape(dm_row))
        dm_col = distance_matrix(old_points,new_point) # dimension (1,starting_points)
        dm_col = np.vstack((dm_col,[0]))        
#        print('expected col matrix shape {0},{1}'.format(len(old_points)+1,1),np.shape(dm_col))
        dm_conc = np.concatenate((self.distance_matrix_,dm_row),axis = 0) # dimension (starting_points+1,starting_popints)
#        print('expected conc matrix shape {0},{1}'.format(len(old_points)+1,len(old_points)),np.shape(dm_conc))
        self.distance_matrix_ = np.concatenate((dm_conc,dm_col),axis = 1)

##-------------------------------------- IS STATEMENTS FOR VERTICES ------------------------------------
    def is_in_graph(self,vertex):
        return self.graph.vp['is_in_graph'][vertex]
    
    def is_end_point(self,vertex):
        return self.graph.vp['end_point'][vertex]

    def is_active(self,vertex):
        return self.graph.vp['is_active'][vertex]

    def is_newly_added(self,vertex):    
        return self.graph.vp['newly_added_center'][vertex]

    def is_important_node(self,vertex):
        return self.graph.vp['important_node'][vertex]

    def is_intersection(self,vertex):
        return self.graph.vp['intersection'][vertex]
    
##--------------------------------------- SPATIAL GEOMETRY, NEIGHBORHOODS, RELATIVE NEIGHBORS ------------------------------------

    def get_voronoi(self,vertex):
        self.graph.vp['voronoi'][vertex] = Voronoi(np.array([self.graph.vp['x'][vertex],self.graph.vp['y'][vertex]]).T)

##---------------------------------------- ROAD OPERATIONS ---------------------------------------------

    def add_edge2graph(self,source_vertex,target_vertex):
        '''
            source_idx: Vertex
            target_idx: Vertex
            Description:
                1) Add the edge
                2) Add the distance
                3) Add the direction
                4) Add the real_edge property
                5) Control that the source vertex is an important vertex:
                    5a) Yes: Means that a new road is starting
                    5b) No: Means that the road is already started and I need to find the road where the source vertex belongs to
                        5b1) I check if the source vertex is in the road and add the target vertex to it 
        '''
        source_idx = self.graph.vp['id'][source_vertex]
        target_idx = self.graph.vp['id'][target_vertex]
        self.graph.add_edge(source_idx,target_idx)
        self.graph.ep['distance'][self.graph.edge(source_idx,target_idx)] = self.distance_matrix_[source_idx,target_idx]
        self.graph.ep['direction'][self.graph.edge(source_idx,target_idx)] = self.graph.vp['pos'][target_vertex].a - self.graph.vp['pos'][source_vertex].a
        self.graph.ep['real_edge'][self.graph.edge(source_idx,target_idx)] = False
        if self.graph.vp['important_node'][source_vertex] == True:
            self.global_counting_roads += 1
            new_road = road(source_vertex,self.global_counting_roads)            
            self.graph.vp['roads'][source_vertex].append(new_road)
            distance_ = self.distance_matrix_[self.graph.vp['id'][source_vertex],self.graph.vp['id'][target_vertex]]
            self.graph.vp['roads'][source_vertex][-1].add_node_in_road(source_vertex,target_vertex,distance_)
        else:
            ## Check that for each vertex that has got a road
            ## TODO: Add variable: starting road
            for initial_node in self.important_vertices:
                local_idx_road = 0
                for r in self.graph.vp['roads'][initial_node]:
                    _,found = r.in_road(source_vertex)
                    local_idx_road += 1
                if found:
                    self.graph.vp['roads'][initial_node][local_idx_road].append(target_vertex)
                    break

#TODO: Fix the generation of intersections, when a new center is added, and generates a new road, the point when the road starts, is 
#  Inersection, new kind of NODE, this node, is the beginning of a new road.
# I need a new variable in road() -> type_starting_point: ['important_node','intersection']] 
    def update_intersections(self):
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
        for n_attracting_vertex in self.newly_added_attracting_vertices:
            for attracted_vertex_idx in self.graph.vp['relative_neighbors'][n_attracting_vertex]:
                attracted_vertex = self.graph.vertex(attracted_vertex_idx)
                ## If the attracted vertex (belonging to the relative neighbor) is not an end point, and is not an important node (then It can be just growing)
                if not self.is_end_point(attracted_vertex) and not self.is_important_node(attracted_vertex):
                    starting_vertex,local_idx,found = self.find_road_vertex(attracted_vertex)
                    if found:
                        self.global_counting_roads += 1
                        self.graph.vp['roads'][starting_vertex].append(road(starting_vertex,self.global_counting_roads,n_attracting_vertex))
                        self.graph.vp['roads'][starting_vertex][-1].copy_road_specifics(self,self.graph.vp['roads'][starting_vertex][local_idx])
                    else:
                        pass


    def create_road(self,source_vertex,activation_vertices):
        self.graph.vp['roads'][source_vertex].append(road(source_vertex,self.global_counting_roads,activation_vertices))
        self.global_counting_roads += 1

    def add_road(self,source_vertex,vertex):
        '''
            Adds the vertex to the road
        '''
        if self.is_in_graph(source_vertex):
            if self.is_important_node(source_vertex):
                self.create_road(source_vertex,vertex)
            else:
                starting_vertex_road,local_idx_road,found = self.find_road_vertex(vertex)
                if found:
                    self.graph.vp['roads'][starting_vertex_road][local_idx_road].append(vertex)
                else:
                    pass
        else:
            print(self.print_properties_vertex(source_vertex))
            raise ValueError('The source vertex {} is not in the graph'.format(self.graph.vp['id'][source_vertex]))

    def add_point2road(self,growing_node,added_vertex):
        '''
            Description:
                Adds added_vertex to the road of growing node
        '''
        starting_vertex_road,local_idx_road,found = self.find_road_vertex(growing_node)
        if found:
            distance_ = self.distance_matrix_[self.graph.vp['id'][starting_vertex_road],self.graph.vp['id'][added_vertex]]
            self.graph.vp['roads'][starting_vertex_road][local_idx_road].add_node_in_road(growing_node,added_vertex,distance_)                            
        else:
            pass

    def find_road_vertex(self,vertex):
        '''
            Description:
                Find the road that starts from vertex
            Output:
                starting_vertex of the road
                local_idx_road: index of the road in the list of roads starting from starting_vertex
            Complexity:
                O(number_vertices)
        '''
        local_idx_road = 0
        found = False
        for starting_vertex in self.important_vertices:
            for r in self.graph.vp['roads'][starting_vertex]:
                local_idx_road += 1
                if r.in_road(vertex):
                    found = True
                    return starting_vertex,local_idx_road,found
                else:
                    print(self.graph.vp['id'][vertex],' not in road')
        return starting_vertex,0,found
    
    def get_list_nodes_in_roads_starting_from_v(self,v):
        '''
            Output:
                List of vertices that are in the roads starting from v
                type: list vertex
        '''
        points_in_adjacent_roads_v = []
        for r in self.graph.vp['roads'][v]:
            for v_road in r.list_nodes:
                points_in_adjacent_roads_v.append(v_road)
        return points_in_adjacent_roads_v


##--------------------------------------------- UPDATES ---------------------------------------------------- NEXT STEP -> COMPUTE DELAUNEY TRIANGULATION (new added,old) attracting vertices    
    def update_list_important_vertices(self):
        self.important_vertices = [v for v in self.graph.vertices() if self.graph.vp['important_node'][v] == True]
    def update_list_in_graph_vertices(self):
        self.is_in_graph_vertices = [v for v in self.graph.vertices() if self.graph.vp['is_in_graph'][v] == True]
    def update_list_end_points(self):
        '''
            List of points to consider when deciding old attracting vertices relative neighbor
        '''
        self.end_points = [v for v in self.graph.vertices() if self.graph.vp['end_point'][v] == True]
    ## ACTIVE VERTICES  
    def update_list_old_attracting_vertices(self):
        '''
            These are the vertices that attract just end points. 
                -Delauney triangulation just among these and the end points in graph
        '''
        self.old_attracting_vertices = [v for v in self.graph.vertices() if self.is_active(v) and not self.is_newly_added(v)]
    def update_list_newly_added_attracting_vertices(self):
        self.newly_added_attracting_vertices = [v for v in self.graph.vertices() if self.is_active(v) and self.is_newly_added(v)]
    
    def update_list_intersection_vertices(self):
        self.intersection_vertices = [v for v in self.graph.vertices() if self.graph.vp['intersection'][v] == True]
    
    def update_list_plausible_starting_point_of_roads(self):
        self.plausible_starting_road_vertices = [v for v in self.graph.vertices() if self.is_important_node(v) or self.is_intersection(v)]

    def update_list_active_vertices(self):
        '''
            For each important vertex check if it 
        '''
        self.active_vertices = []
        for attracting_vertex in self.important_vertices:
            for starting_road_vertex in self.important_vertices:
                for r in self.graph.vp['roads'][starting_road_vertex]:
                    if r.is_closed == False and attracting_vertex in r.activated_by:
                        self.graph.vp['is_active'][attracting_vertex] = True
                        break

# This piece must be inserted to take update the roads, as I need each point that is evolving to have an attraction set
    def update_attracted_by(self):
        '''
            For each vertex update the attraction:
                1) The end_points are attracted by the relative neighbors that are not in the graph
                2) The growing points are attracted by new added vertices (stop)
        '''
        print('updating attracted by: ')
        list_considered_vertices = []
        for v in self.graph.vertices():
            print('considering vertex: ',self.graph.vp['id'][v])
            self.graph.vp['attracted_by'][v] = []
            if self.is_end_point(v):
                print('is end point')
                list_considered_vertices.append(v)
                list_rn = self.graph.vp['relative_neighbors'][v]
                print('relative neighbors: ',list_rn)
                starting_vertex,local_idx,found = self.find_road_vertex(v)
                if found:
                    for relative_neighbor in list_rn:
                        if relative_neighbor not in self.graph.vp['roads'][starting_vertex][local_idx].list_nodes:
                            self.graph.vp['attracted_by'][v].append(relative_neighbor)
                else:
                    pass
            elif self.is_growing_and_not_attracting(v):
                list_considered_vertices.append(v)
                list_rn = self.graph.vp['relative_neighbors'][v]
                starting_vertex,local_idx,found = self.find_road_vertex(v)
                if found:
                    for relative_neighbor in list_rn:
                        if relative_neighbor not in self.graph.vp['roads'][starting_vertex][local_idx].list_nodes:
                            self.graph.vp['attracted_by'][v].append(relative_neighbor)
                else:
                    pass
            elif self.is_attracting_and_not_growing(v):
                list_considered_vertices.append(v)
                self.graph.vp['attracted_by'][v] = []
                pass
            elif self.is_growing_and_attracting(v):
                list_considered_vertices.append(v)
                self.graph.vp['attracted_by'][v] = []
                pass
        not_considered = [v for v in self.graph.vertices() if v not in list_considered_vertices]
        self.print_not_considered_vertices(not_considered)

## UPDATE DELAUNEY TRIANGULATION: NEXT STEP -> COMPUTE RNG (new added,old) attracting vertices      
    
    def update_delauney_newly_attracting_vertices(self):
        '''
            The dalueney graph will contain:
                1) The newly added attracting vertices
                2) All the in graph vertices in the graph
            NOTE: 
                "IN GRAPH" vertices can be attracted by new inserted vertices -> create new roads.
                I MAY CHANGE IT IF IN THE CENTER THE ROADS ARE TOO DENSE -> 
                [self.graph.vp['x'][v] for v in self.graph.vertices() if self.is_newly_added(v) or (self.is_in_graph(v) and not self.is_active(v))]

        '''
        x = [self.graph.vp['x'][v] for v in self.graph.vertices() if self.is_newly_added(v) or (self.is_in_graph(v))]
        y = [self.graph.vp['y'][v] for v in self.graph.vertices() if self.is_newly_added(v) or (self.is_in_graph(v))]   
        idx_xy = [self.graph.vp['id'][v] for v in self.graph.vertices() if self.is_newly_added(v) or (self.is_in_graph(v))]  
        self.delauneyid2idx_new_vertices = {i:self.graph.vp['id'][self.graph.vertex(idx_xy[i])] for i in range(len(idx_xy))} 
        if len(x)>3:
            tri = Delaunay(np.array([x,y]).T)
            # Iterate over all triangles in the Delaunay triangulation
            for simplex in tri.simplices:
                for i, j in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]:
                        vertex_i = self.graph.vertex(self.delauneyid2idx_new_vertices[i])   
                        vertex_j_idx = self.delauneyid2idx_new_vertices[j]
                        vertex_j = self.graph.vertex(self.delauneyid2idx_new_vertices[j])   
                        vertex_i_idx = self.delauneyid2idx_new_vertices[i]
                        if not j in self.graph.vp['new_attracting_delauney_neighbors'][vertex_i]:
                            self.graph.vp['new_attracting_delauney_neighbors'][vertex_i].append(vertex_j_idx)
                        if not i in self.graph.vp['new_attracting_delauney_neighbors'][vertex_j]:
                            self.graph.vp['new_attracting_delauney_neighbors'][vertex_j].append(vertex_i_idx)
        else:
            print('Do not have enough points for Delauney triangulation')
            raise ValueError
                            
    def update_delauney_old_attracting_vertices(self):
        '''
            Consider just "important" vertices that are 
                1) Still active
                2) Not newly added
            Consider just "not important" vertices:
                1) Still active
                2) Not newly added
            2) Compute the delauney triangulation for this subsets of points
            3) Save it on old_attracting_delauney_neighbor for each vertex
            NOTE: not in graph -> active  NOT active -> not in graph
        '''
        ## Filter vertices of the wanted graph
        x = [self.graph.vp['x'][v] for v in self.graph.vertices() if (not self.is_newly_added(v) and self.is_active(v)) or self.is_end_point(v)]
        y = [self.graph.vp['y'][v] for v in self.graph.vertices() if (not self.is_newly_added(v) and self.is_active(v)) or self.is_end_point(v)]         
        ## Take their indices
        idx_xy = [self.graph.vp['id'][v] for v in self.graph.vertices() if (self.graph.vp['attracting'][v] == True and self.graph.vp['newly_added_center'][v] == False) or (self.graph.vp['end_point'][v] == True)]
        ## Save the map of indices that I will use to retrieve the id of the vertex
        self.delauneyid2idx_old_attracting_vertices = {i:self.graph.vp['id'][self.graph.vertex(idx_xy[i])] for i in range(len(idx_xy))} 
        ## Compute delauney
        tri = Delaunay(np.array([x,y]).T)
        # Iterate over all triangles in the Delaunay triangulation
        self.edges_delauney_for_old_attracting_vertices = []
        for simplex in tri.simplices:
            for i, j in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]:
                ## Initialize the neighborhood
                vertex_i = self.graph.vertex(self.delauneyid2idx_old_attracting_vertices[i])   
                vertex_j = self.graph.vertex(self.delauneyid2idx_old_attracting_vertices[i])   
                vertex_j_idx = self.delauneyid2idx_old_attracting_vertices[j]
                vertex_i_idx = self.delauneyid2idx_old_attracting_vertices[j]
                if not j in self.graph.vp['old_attracting_delauney_neighbors'][vertex_i]:   
                    self.graph.vp['old_attracting_delauney_neighbors'][vertex_i].append(vertex_j_idx)
                if not i in self.graph.vp['old_attracting_delauney_neighbors'][self.delauneyid2idx_old_attracting_vertices[j]]:
                    self.graph.vp['old_attracting_delauney_neighbors'][vertex_j].append(vertex_i_idx)


## CALLING ALL UPDATES: NEXT STEP -> COMPUTE RNG (new added,old) attracting vertices
    def update_lists_next_rng(self):
        '''
            Recompute:
                1) growing_graph
                2) growing_vertices -> are the vertices to search from the newly added centers
                3) end_points -> are the vertices for which the old added points must look for in their relative neighborhood
                4) attracting_vertices
                5) newly_added_attracting_vertices
                6) old_attracting_vertices
        '''
        # GROWING VERTICES
        # for newly added centers
        self.update_list_end_points()
        self.update_list_newly_added_attracting_vertices()
        self.update_list_important_vertices()
        self.update_list_old_attracting_vertices()
        self.update_list_in_graph_vertices()
        self.update_list_intersection_vertices()
        self.update_list_plausible_starting_point_of_roads()

## COMPUTE RNG for NEWLY ADDED CENTERS: NEXT STEP -> EVOLVE STREET for (new,old) attracting vertices
    def compute_rng_newly_added_centers(self):
        '''
            Description:
                1) update_delauney:
                    1a) compute the Voronoi diagram for all vertices O(Nlog(N))
                    1b) compute the delauney triangulation  O(N)
                2) For each attracting node (vi) and a node from the delauney triangulation (vj)
                3) Check for if for any node vk in the graph max(d_ix, d_xj) < d_ij
                4) If it is not the case, then vj is a relative neighbor of vi

        '''
        self.update_lists_next_rng()
        self.update_delauney_newly_attracting_vertices()
        for vi in self.newly_added_attracting_vertices: # self.growing_graph.vertices()
            self.graph.vp['relative_neighbors'][vi] = []
            for vj in self.graph.vp['new_attracting_delauney_neighbors'][vi]: # self.growing_graph.vertices()
                try:
                    d_ij = self.distance_matrix_[self.graph.vp['id'][vi]][vj]
                except KeyError:
                    d_ij = None
                    continue
                for vx in self.graph.vertices(): 
                    try:
                        d_ix = self.distance_matrix_[self.graph.vp['id'][vi]][self.graph.vp['id'][vx]]
                    except KeyError:
                        d_ix = None
                        continue
                    try:
                        d_xj = self.distance_matrix_[self.graph.vp['id'][self.graph.vp['id'][vx]]][vj]
                    except KeyError:
                        d_xj = None
                        continue
                if max(d_ix, d_xj) < d_ij: break
                else:
                    if d_ij != 0:
                        self.graph.vp['relative_neighbors'][vi].append(vj)
                        if vi not in self.graph.vp['relative_neighbors'][self.graph.vertex(vj)]:
                            self.graph.vp['relative_neighbors'][self.graph.vertex(vj)].append(self.graph.vp['id'][vi])
                            if self.graph.vp['end_point'][self.graph.vertex(vj)] == False and self.graph.vp['important_node'][self.graph.vertex(vj)] == False:
                                self.graph.vp['intersection'][self.graph.vertex(vj)] = True
                                # FIND THE STARTING VERTEX OF THE ROAD THIS POINTS BELONGS TO
                                # ADD ROAD 
        self.update_intersections()
        self.update_list_intersection_vertices()


    def compute_rng_old_centers(self):
        '''
            Description:
                1) compute the delauney triangulation with just:
                    1a) End points
                    1b) old attracting vertices
                2) Compare them with just the nodes that are end points and are not in the road starting 
                    from the vertex vi I am considering.
        '''
        self.update_lists_next_rng()
        self.update_delauney_old_attracting_vertices()
        for vi in self.old_attracting_vertices: # self.growing_graph.vertices()
            list_nodes_road_vi = self.get_list_nodes_in_roads_starting_from_v(vi)
            self.graph.vp['relative_neighbors'][vi] = []
            for vj in self.graph.vp['old_attracting_delauney_neighbors'][vi]: # self.growing_graph.vertices()
                try:
                    d_ij = self.distance_matrix_[self.graph.vp['id'][vi]][vj]
                except KeyError:
                    d_ij = None
                    continue
                for vx in self.end_points: 
                    if vx not in list_nodes_road_vi:
                        try:
                            d_ix = self.distance_matrix_[self.graph.vp['id'][vi]][self.graph.vp['id'][vx]]
                        except KeyError:
                            d_ix = None
                            continue
                        try:
                            d_xj = self.distance_matrix_[self.graph.vp['id'][self.graph.vp['id'][vx]]][vj]
                        except KeyError:
                            d_xj = None
                            continue
                    if max(d_ix, d_xj) < d_ij: break
                else:
                    if d_ij != 0:
                        self.graph.vp['relative_neighbors'][vi].append(vj)
                        if vi not in self.graph.vp['relative_neighbors'][self.graph.vertex(vj)]:
                            self.graph.vp['relative_neighbors'][self.graph.vertex(vj)].append(self.graph.vp['id'][vi])


## UPDATE 

    def close_road(self):
        '''
            Attach the edge if the growing points are close enough to their relative neighbors,
            in this way the relative neighbor becomes a growing point as well
        '''
        already_existing_edges = [[e.source(),e.target()] for e in self.graph.edges()]
        for v in self.important_vertices:
            for u_idx in self.graph.vp['relative_neighbors'][v]:
                    if self.distance_matrix_[self.graph.vp['id'][v],u_idx] < self.rate_growth and [v,self.graph.vertex(u_idx)] not in already_existing_edges:
                        self.graph.add_edge(self.graph.vertex(u_idx),v)
                        if self.graph.vp['growing'][v] == False:
                            self.graph.vp['growing'][v] = True
                            print('closing road between: ',self.graph.vp['id'][v],self.graph.vp['id'][u_idx])
                        else:
                            pass



    def choose_just_growing_nodes(self,delauney_neighborhood):
        '''
            --- Attractor node function: ----
            When I take the delauney neighbor:
                I check if the neighbor is a growing node
                If it is a growing node, I add it to the list of relative neighbors
        '''
        relative_growing_neighbor = []
        for delauney_neighbor in delauney_neighborhood:
            if self.graph.vp['growing'][self.graph.vertex(delauney_neighbor)]:
                relative_growing_neighbor.append(self.graph.vertex(delauney_neighbor))
        return relative_growing_neighbor


    def choose_available_vertices_to_grow_into(self,growing_node,list_vert_relative_neighbor_per_growing_vertex):
        '''
            ---- Growing node function -----
            list_vert_relative_neighbor_per_growing_vertex: list of Vertex
            Rerurns:
                available_vertices: list of Vertex that are:
                    1) Not connected to the growing node
                    2) Attracting vertices
        '''
        available_vertices = []
        for relative_neighbor in list_vert_relative_neighbor_per_growing_vertex:
            is_not_out_neighbor = relative_neighbor not in list(growing_node.out_neighbors())
            is_not_in_neighbor = relative_neighbor not in list(growing_node.in_neighbors())
            is_attracting = self.graph.vp['attracting'][relative_neighbor]
            if  is_not_out_neighbor and is_not_in_neighbor and is_attracting:
                available_vertices.append(relative_neighbor)
        return available_vertices


    def check_available_vertices(self,available_vertices):
        '''
            ---- Growing node function: ----
            for each available node that my growing vertex is attracted by:
                (attracting,growing) -> accept
                (not attracting,growing) -> reject
                (attracting,not growing) -> accept
                (not attracting,not growing) -> impossible
        '''
        for av in available_vertices:
            if self.graph.vp['growing'][av] and not self.graph.vp['important_node'][av]:
                raise ValueError('The available vertex {} cannot be growing'.format(self.graph.vp['id'][av]))
            elif not self.graph.vp['attracting'][av]:
                raise ValueError('The available vertex {} must be attracting'.format(self.graph.vp['id'][av]))
                
    def evolve_street_old_attractors(self):
        '''
            Evolve street for the old attractors
        '''
        already_grown_vertices = []
        for attracting_node in self.old_attracting_vertices:
            list_vert_relative_neighbor_per_attracting_vertex = self.choose_just_growing_nodes(self.graph.vp['relative_neighbors'][attracting_node])
            print('------------------------')
            print('attracting node: ',self.graph.vp['id'][attracting_node])
            print('attracted vertices: ',[self.graph.vp['id'][relative_neighbor]for relative_neighbor in list_vert_relative_neighbor_per_attracting_vertex])
            ## if the relative neighbor is just one
            for growing_node in list_vert_relative_neighbor_per_attracting_vertex:
                if growing_node not in already_grown_vertices:
                    list_int_idx_relative_neighbor_per_growing_node = self.graph.vp['relative_neighbors'][growing_node]
                    list_vert_relative_neighbor_per_growing_vertex = [self.graph.vertex(int_idx) for int_idx in list_int_idx_relative_neighbor_per_growing_node]
                    ## choosing available nodes to grow towards, they must be:
                    ## 1) Attracting nodes
                    ## 2) Not edge attached to the growing node 
                    available_vertices = self.choose_available_vertices_to_grow_into(growing_node,list_vert_relative_neighbor_per_growing_vertex)
                    number_relative_neighbors_growing_node = len(available_vertices)
                    print('growing node: ',self.graph.vp['id'][growing_node],' coords: ',self.graph.vp['pos'][growing_node])
                    ## Take if relative neighbor of the growing node is just one  
                    if number_relative_neighbors_growing_node==1:
                        self.print_delauney_neighbors(growing_node)
                        print('available attracting vertices: ',[self.graph.vp['id'][v] for v in available_vertices])
                        self.check_available_vertices(available_vertices)
                        old_points = np.array([self.graph.vp['x'].a,self.graph.vp['y'].a]).T
                        vertex_relative_neighbor = available_vertices[0] # self.graph.vertex(int_idx_relative_neighbor)
                        dx,dy = coordinate_difference(self.graph.vp['x'][vertex_relative_neighbor],self.graph.vp['y'][vertex_relative_neighbor],self.graph.vp['x'][growing_node],self.graph.vp['y'][growing_node])
                        dx,dy = normalize_vector(dx,dy)
                        dx,dy = scale(dx,dy,self.rate_growth)
                        new_point = np.array([[self.graph.vp['x'][growing_node] +dx,self.graph.vp['y'][growing_node] +dy]])
                        self.update_distance_matrix(old_points,new_point)
                        added_vertex = self.add_point2graph(self.graph.vp['x'][growing_node],self.graph.vp['y'][growing_node],dx,dy)
                        self.add_edge2graph(self.graph.vertex(growing_node),added_vertex)
                        ## Find the road where the points belongs to (looking at the starting vertex it is generated from)
                        self.add_point2road(growing_node,added_vertex)
                        self.plot_relative_neighbors(growing_node,attracting_node,added_vertex,available_vertices)
                        print('EVOLVING UNIQUE ATTRACTOR')
                        print('direction: ',self.graph.vp['id'][vertex_relative_neighbor])
                        print(' dx: ',dx,' dy: ',dy) 
                        print('added vertex: ',self.graph.vp['id'][added_vertex],' coords: ',self.graph.vp['pos'][added_vertex])
                    elif number_relative_neighbors_growing_node>1: # a relative neighbor is also a neighbor -> kill the growth
                        self.print_delauney_neighbors(growing_node)
                        print('available attracting vertices: ',[self.graph.vp['id'][v] for v in available_vertices])
                        self.check_available_vertices(available_vertices)
                        dx = 0
                        dy = 0
                        for neighbor_attracting_vertex in range(len(available_vertices)):
                            old_points = np.array([self.graph.vp['x'].a,self.graph.vp['y'].a]).T
                            vertex_relative_neighbor = available_vertices[neighbor_attracting_vertex]#self.graph.vertex(int_idx_relative_neighbor)
                            x,y = coordinate_difference(self.graph.vp['x'][vertex_relative_neighbor],self.graph.vp['y'][vertex_relative_neighbor],self.graph.vp['x'][growing_node],self.graph.vp['y'][growing_node])
                            dx += x 
                            dy += y
                        if np.sqrt(dx**2+dy**2)!=0: 
                            dx,dy = normalize_vector(dx,dy)
                            dx,dy = scale(dx,dy,self.rate_growth)
                            new_point = np.array([[self.graph.vp['x'][growing_node] +dx,self.graph.vp['y'][growing_node] +dy]])
                            self.update_distance_matrix(old_points,new_point)
                            added_vertex = self.add_point2graph(self.graph.vp['x'][growing_node],self.graph.vp['y'][growing_node],dx,dy)
                            self.add_edge2graph(self.graph.vertex(growing_node),added_vertex)
                            self.add_point2road(growing_node,added_vertex)
                            self.plot_relative_neighbors(growing_node,attracting_node,added_vertex,available_vertices)                            
                            print('EVOLVING SUM ATTRACTOR')
                            print('direction sum of : ',[self.graph.vp['id'][vertex_relative_neighbor] for vertex_relative_neighbor in available_vertices])
                            print(' dx: ',dx,' dy: ',dy) 
                            print('added vertex: ',self.graph.vp['id'][added_vertex],' coords: ',self.graph.vp['pos'][added_vertex])
                        else:
                            print('EVOLVING IN ALL DIRECTIONS DUE TO DEGENERACY')
                            for neighbor_attracting_vertex in range(len(available_vertices)):
                                vertex_relative_neighbor = available_vertices[neighbor_attracting_vertex]#self.graph.vertex(int_idx_relative_neighbor)
    #                            vertex_relative_neighbor = self.graph.vertex(int_idx_relative_neighbor)
                                dx,dy = coordinate_difference(self.graph.vp['x'][vertex_relative_neighbor],self.graph.vp['y'][vertex_relative_neighbor],self.graph.vp['x'][growing_node],self.graph.vp['y'][growing_node])
                                dx,dy = normalize_vector(dx,dy)
                                dx,dy = scale(dx,dy,self.rate_growth)
                                new_point = np.array([[self.graph.vp['x'][growing_node] +dx,self.graph.vp['y'][growing_node] +dy]])
                                self.update_distance_matrix(old_points,new_point)
                                added_vertex = self.add_point2graph(self.graph.vp['x'][growing_node],self.graph.vp['y'][growing_node],dx,dy)
                                self.add_edge2graph(self.graph.vertex(growing_node),added_vertex)  
                                self.add_point2road(growing_node,added_vertex)
                                self.plot_relative_neighbors(growing_node,attracting_node,added_vertex,available_vertices)
                                print('direction sum of : ',self.graph.vp['id'][vertex_relative_neighbor])
                                print(' dx: ',dx,' dy: ',dy) 
                                print('added vertex: ',self.graph.vp['id'][added_vertex],' coords: ',self.graph.vp['pos'][added_vertex])
                else: 
                    pass
                if growing_node not in already_grown_vertices:
                    already_grown_vertices.append(growing_node)
                    vv = [self.graph.vp['id'][growing_node] for growing_node in already_grown_vertices]
                    vv = np.sort(vv)
#                    print('already grown nodes: ',vv)
        t0 =time.time()
        self.close_road()
        t1 = time.time()
        print('time to close road: ',t1-t0)


    def evolve_street_newly_added_attractors(self):
        '''
            Needed prior steps:
                1) update_list_newly_added_attracting_vertices
                2) update_list_growing_vertices
                3) compute_rng

        '''
        already_grown_vertices = []
        for attracting_node in self.newly_added_attracting_vertices:
            ## if the relative neighbor is just one and it is not already attached tothe graph (neigbor['growing'] == False)
            list_vert_relative_neighbor_per_attracting_vertex = self.choose_just_growing_nodes(self.graph.vp['relative_neighbors'][attracting_node])
            print('------------------------')
            print('attracting node: ',self.graph.vp['id'][attracting_node])
            print('attracted vertices: ',[self.graph.vp['id'][relative_neighbor]for relative_neighbor in list_vert_relative_neighbor_per_attracting_vertex])
            ## if the relative neighbor is just one
            for growing_node in list_vert_relative_neighbor_per_attracting_vertex:
                if growing_node not in already_grown_vertices:
                    list_int_idx_relative_neighbor_per_growing_node = self.graph.vp['relative_neighbors'][growing_node]
                    list_vert_relative_neighbor_per_growing_vertex = [self.graph.vertex(int_idx) for int_idx in list_int_idx_relative_neighbor_per_growing_node]
                    ## choosing available nodes to grow towards, they must be:
                    ## 1) Attracting nodes
                    ## 2) Not edge attached to the growing node 
                    available_vertices = self.choose_available_vertices_to_grow_into(growing_node,list_vert_relative_neighbor_per_growing_vertex)
                    number_relative_neighbors_growing_node = len(available_vertices)
                    print('growing node: ',self.graph.vp['id'][growing_node],' coords: ',self.graph.vp['pos'][growing_node])
                    ## Take if relative neighbor of the growing node is just one  
                    if number_relative_neighbors_growing_node==1:
                        self.print_delauney_neighbors(growing_node)
                        print('available attracting vertices: ',[self.graph.vp['id'][v] for v in available_vertices])
                        self.check_available_vertices(available_vertices)
                        old_points = np.array([self.graph.vp['x'].a,self.graph.vp['y'].a]).T
                        vertex_relative_neighbor = available_vertices[0] # self.graph.vertex(int_idx_relative_neighbor)
                        dx,dy = coordinate_difference(self.graph.vp['x'][vertex_relative_neighbor],self.graph.vp['y'][vertex_relative_neighbor],self.graph.vp['x'][growing_node],self.graph.vp['y'][growing_node])
                        dx,dy = normalize_vector(dx,dy)
                        dx,dy = scale(dx,dy,self.rate_growth)
                        new_point = np.array([[self.graph.vp['x'][growing_node] +dx,self.graph.vp['y'][growing_node] +dy]])
                        self.update_distance_matrix(old_points,new_point)
                        added_vertex = self.add_point2graph(self.graph.vp['x'][growing_node],self.graph.vp['y'][growing_node],dx,dy)
                        self.add_edge2graph(self.graph.vertex(growing_node),added_vertex)
                        ## Find the road where the points belongs to (looking at the starting vertex it is generated from)
                        self.add_point2road(growing_node,added_vertex)
                        self.plot_relative_neighbors(growing_node,attracting_node,added_vertex,available_vertices)
                        print('EVOLVING UNIQUE ATTRACTOR')
                        print('direction: ',self.graph.vp['id'][vertex_relative_neighbor])
                        print(' dx: ',dx,' dy: ',dy) 
                        print('added vertex: ',self.graph.vp['id'][added_vertex],' coords: ',self.graph.vp['pos'][added_vertex])
                    elif number_relative_neighbors_growing_node>1: # a relative neighbor is also a neighbor -> kill the growth
                        self.print_delauney_neighbors(growing_node)
                        print('available attracting vertices: ',[self.graph.vp['id'][v] for v in available_vertices])
                        self.check_available_vertices(available_vertices)
                        dx = 0
                        dy = 0
                        for neighbor_attracting_vertex in range(len(available_vertices)):
                            old_points = np.array([self.graph.vp['x'].a,self.graph.vp['y'].a]).T
                            vertex_relative_neighbor = available_vertices[neighbor_attracting_vertex]#self.graph.vertex(int_idx_relative_neighbor)
                            x,y = coordinate_difference(self.graph.vp['x'][vertex_relative_neighbor],self.graph.vp['y'][vertex_relative_neighbor],self.graph.vp['x'][growing_node],self.graph.vp['y'][growing_node])
                            dx += x 
                            dy += y
                        if np.sqrt(dx**2+dy**2)!=0: 
                            dx,dy = normalize_vector(dx,dy)
                            dx,dy = scale(dx,dy,self.rate_growth)
                            new_point = np.array([[self.graph.vp['x'][growing_node] +dx,self.graph.vp['y'][growing_node] +dy]])
                            self.update_distance_matrix(old_points,new_point)
                            added_vertex = self.add_point2graph(self.graph.vp['x'][growing_node],self.graph.vp['y'][growing_node],dx,dy)
                            self.add_edge2graph(self.graph.vertex(growing_node),added_vertex)
                            self.add_point2road(growing_node,added_vertex)
                            self.plot_relative_neighbors(growing_node,attracting_node,added_vertex,available_vertices)                            
                            print('EVOLVING SUM ATTRACTOR')
                            print('direction sum of : ',[self.graph.vp['id'][vertex_relative_neighbor] for vertex_relative_neighbor in available_vertices])
                            print(' dx: ',dx,' dy: ',dy) 
                            print('added vertex: ',self.graph.vp['id'][added_vertex],' coords: ',self.graph.vp['pos'][added_vertex])
                        else:
                            print('EVOLVING IN ALL DIRECTIONS DUE TO DEGENERACY')
                            for neighbor_attracting_vertex in range(len(available_vertices)):
                                vertex_relative_neighbor = available_vertices[neighbor_attracting_vertex]#self.graph.vertex(int_idx_relative_neighbor)
    #                            vertex_relative_neighbor = self.graph.vertex(int_idx_relative_neighbor)
                                dx,dy = coordinate_difference(self.graph.vp['x'][vertex_relative_neighbor],self.graph.vp['y'][vertex_relative_neighbor],self.graph.vp['x'][growing_node],self.graph.vp['y'][growing_node])
                                dx,dy = normalize_vector(dx,dy)
                                dx,dy = scale(dx,dy,self.rate_growth)
                                new_point = np.array([[self.graph.vp['x'][growing_node] +dx,self.graph.vp['y'][growing_node] +dy]])
                                self.update_distance_matrix(old_points,new_point)
                                added_vertex = self.add_point2graph(self.graph.vp['x'][growing_node],self.graph.vp['y'][growing_node],dx,dy)
                                self.add_edge2graph(self.graph.vertex(growing_node),added_vertex)  
                                self.add_point2road(growing_node,added_vertex)
                                self.plot_relative_neighbors(growing_node,attracting_node,added_vertex,available_vertices)
                                print('direction sum of : ',self.graph.vp['id'][vertex_relative_neighbor])
                                print(' dx: ',dx,' dy: ',dy) 
                                print('added vertex: ',self.graph.vp['id'][added_vertex],' coords: ',self.graph.vp['pos'][added_vertex])
                else: 
                    pass
                if growing_node not in already_grown_vertices:
                    already_grown_vertices.append(growing_node)
                    vv = [self.graph.vp['id'][growing_node] for growing_node in already_grown_vertices]
                    vv = np.sort(vv)
#                    print('already grown nodes: ',vv)
        t0 =time.time()
        self.close_road()
        t1 = time.time()
        print('time to close road: ',t1-t0)
    
    
    def evolve_street(self):
        '''
            1) For each attrcting point ('auxine')  
                1a) find relative neighbor in the delauney neighbor
                2) For each relative neighbor
                    2a) Check it is a growing node
                    2b) If more than one, grow in the 'sum' direction (that is the solution optimizing different roads)
            Both of these increment are of magnitude ratio_growth2size_city*side_city
        '''
        already_grown_vertices = []

        ## for each node attached to the graph (growing) I evolve the street
        for attracting_node in self.important_vertices:
            ## if the relative neighbor is just one and it is not already attached tothe graph (neigbor['growing'] == False)
            list_vert_relative_neighbor_per_attracting_vertex = self.choose_just_growing_nodes(self.graph.vp['relative_neighbors'][attracting_node])
#            list_vert_relative_neighbor_per_attracting_vertex = [self.graph.vertex(int_idx) for int_idx in list_int_idx_relative_neighbor_per_attracting_vertex if self.graph.vp['growing'][self.graph.vertex(int_idx)]]
            print('------------------------')
            print('attracting node: ',self.graph.vp['id'][attracting_node])
            print('attracted vertices: ',[self.graph.vp['id'][relative_neighbor]for relative_neighbor in list_vert_relative_neighbor_per_attracting_vertex])
            ## if the relative neighbor is just one
            for growing_node in list_vert_relative_neighbor_per_attracting_vertex:
                if growing_node not in already_grown_vertices:
                    list_int_idx_relative_neighbor_per_growing_node = self.graph.vp['relative_neighbors'][growing_node]
                    list_vert_relative_neighbor_per_growing_vertex = [self.graph.vertex(int_idx) for int_idx in list_int_idx_relative_neighbor_per_growing_node]
                    ## choosing available nodes to grow towards, they must be:
                    ## 1) Attracting nodes
                    ## 2) Not edge attached to the growing node 
                    available_vertices = self.choose_available_vertices_to_grow_into(growing_node,list_vert_relative_neighbor_per_growing_vertex)
                    number_relative_neighbors_growing_node = len(available_vertices)
                    print('growing node: ',self.graph.vp['id'][growing_node],' coords: ',self.graph.vp['pos'][growing_node])
                    ## Take if relative neighbor of the growing node is just one  
                    if number_relative_neighbors_growing_node==1:
                        self.print_delauney_neighbors(growing_node)
                        print('available attracting vertices: ',[self.graph.vp['id'][v] for v in available_vertices])
                        self.check_available_vertices(available_vertices)
                        old_points = np.array([self.graph.vp['x'].a,self.graph.vp['y'].a]).T
                        vertex_relative_neighbor = available_vertices[0] # self.graph.vertex(int_idx_relative_neighbor)
                        dx = self.graph.vp['x'][vertex_relative_neighbor] - self.graph.vp['x'][growing_node] 
                        dy = self.graph.vp['y'][vertex_relative_neighbor] - self.graph.vp['y'][growing_node] 
                        dx = dx/np.sqrt(dx**2+dy**2)*self.rate_growth
                        dy = dy/np.sqrt(dx**2+dy**2)*self.rate_growth
                        new_point = np.array([[self.graph.vp['x'][growing_node] +dx,self.graph.vp['y'][growing_node] +dy]])
                        self.update_distance_matrix(old_points,new_point)
                        added_vertex = self.add_point2graph(self.graph.vp['x'][growing_node],self.graph.vp['y'][growing_node],dx,dy)
                        self.add_edge2graph(self.graph.vertex(growing_node),added_vertex)
                        ## Find the road where the points belongs to (looking at the starting vertex it is generated from)
                        for starting_vertex_road in self.important_vertices:
                            if self.graph.vp['road'][starting_vertex_road].in_road(growing_node):
                                distance_ = self.distance_matrix_[self.graph.vp['id'][starting_vertex_road]][self.graph.vp['id'][added_vertex]]
                                self.graph.vp['road'][starting_vertex_road].add_node_in_road(starting_vertex_road,added_vertex,distance_)                            
                        self.plot_relative_neighbors(growing_node,attracting_node,added_vertex,available_vertices)
                        print('EVOLVING UNIQUE ATTRACTOR')
                        print('direction: ',self.graph.vp['id'][vertex_relative_neighbor])
                        print(' dx: ',dx,' dy: ',dy) 
                        print('added vertex: ',self.graph.vp['id'][added_vertex],' coords: ',self.graph.vp['pos'][added_vertex])
                    elif number_relative_neighbors_growing_node>1: # a relative neighbor is also a neighbor -> kill the growth
                        self.print_delauney_neighbors(growing_node)
                        print('available attracting vertices: ',[self.graph.vp['id'][v] for v in available_vertices])
                        self.check_available_vertices(available_vertices)
                        dx = 0
                        dy = 0
                        for neighbor_attracting_vertex in range(len(available_vertices)):
                            old_points = np.array([self.graph.vp['x'].a,self.graph.vp['y'].a]).T
                            vertex_relative_neighbor = available_vertices[neighbor_attracting_vertex]#self.graph.vertex(int_idx_relative_neighbor)
                            dx += self.graph.vp['x'][vertex_relative_neighbor] - self.graph.vp['x'][growing_node] 
                            dy += self.graph.vp['y'][vertex_relative_neighbor] - self.graph.vp['y'][growing_node]
                        if np.sqrt(dx**2+dy**2)!=0: 
                            dx = dx/np.sqrt(dx**2+dy**2)*self.rate_growth
                            dy = dy/np.sqrt(dx**2+dy**2)*self.rate_growth
                            new_point = np.array([[self.graph.vp['x'][growing_node] +dx,self.graph.vp['y'][growing_node] +dy]])
                            self.update_distance_matrix(old_points,new_point)
                            added_vertex = self.add_point2graph(self.graph.vp['x'][growing_node],self.graph.vp['y'][growing_node],dx,dy)
                            self.add_edge2graph(self.graph.vertex(growing_node),added_vertex)
                            for starting_vertex_road in self.important_vertices:
                                local_idx_road = 0
                                for ri in self.graph.vp['roads'][starting_vertex_road]:
                                    local_idx_road += 1
                                    if ri.in_road(growing_node):
                                        distance_ = self.distance_matrix_[self.graph.vp['id'][starting_vertex_road]][self.graph.vp['id'][added_vertex]]
                                        self.graph.vp['roads'][starting_vertex_road][local_idx_road].add_node_in_road(starting_vertex_road,added_vertex,distance_)                            

                            self.plot_relative_neighbors(growing_node,attracting_node,added_vertex,available_vertices)                            
                            print('EVOLVING SUM ATTRACTOR')
                            print('direction sum of : ',[self.graph.vp['id'][vertex_relative_neighbor] for vertex_relative_neighbor in available_vertices])
                            print(' dx: ',dx,' dy: ',dy) 
                            print('added vertex: ',self.graph.vp['id'][added_vertex],' coords: ',self.graph.vp['pos'][added_vertex])
                        else:
                            print('EVOLVING IN ALL DIRECTIONS DUE TO DEGENERACY')
                            for neighbor_attracting_vertex in range(len(available_vertices)):
                                vertex_relative_neighbor = available_vertices[neighbor_attracting_vertex]#self.graph.vertex(int_idx_relative_neighbor)
    #                            vertex_relative_neighbor = self.graph.vertex(int_idx_relative_neighbor)
                                dx = self.graph.vp['x'][vertex_relative_neighbor] - self.graph.vp['x'][growing_node] 
                                dy = self.graph.vp['y'][vertex_relative_neighbor] - self.graph.vp['y'][growing_node] 
                                dx = dx/np.sqrt(dx**2+dy**2)*self.rate_growth
                                dy = dy/np.sqrt(dx**2+dy**2)*self.rate_growth
                                new_point = np.array([[self.graph.vp['x'][growing_node] +dx,self.graph.vp['y'][growing_node] +dy]])
                                self.update_distance_matrix(old_points,new_point)
                                added_vertex = self.add_point2graph(self.graph.vp['x'][growing_node],self.graph.vp['y'][growing_node],dx,dy)
                                self.add_edge2graph(self.graph.vertex(growing_node),added_vertex)  
                                for starting_vertex_road in self.important_vertices:
                                    local_idx_road = 0
                                    for ri in self.graph.vp['roads'][starting_vertex_road]:
                                        local_idx_road += 1
                                        if ri.in_road(growing_node):
                                            distance_ = self.distance_matrix_[self.graph.vp['id'][starting_vertex_road]][self.graph.vp['id'][added_vertex]]                                            
                                            self.graph.vp['roads'][starting_vertex_road][local_idx_road].add_node_in_road(starting_vertex_road,added_vertex,distance_)                            
                                self.plot_relative_neighbors(growing_node,attracting_node,added_vertex,available_vertices)
                                print('direction sum of : ',self.graph.vp['id'][vertex_relative_neighbor])
                                print(' dx: ',dx,' dy: ',dy) 
                                print('added vertex: ',self.graph.vp['id'][added_vertex],' coords: ',self.graph.vp['pos'][added_vertex])

#                                    self.get_voronoi(arriving_vertex)
#                                    print('relative neighbor int index: ',self.graph.vp['id'][vertex_relative_neighbor],' dx: ',dx,' dy: ',dy) 
#                            print('starting node: ',self.graph.vp['id'][growing_node])
                        list_idx_attracting_nodes = [self.graph.vp['id'][v] for v in available_vertices]
#                            print('all directions: ',list_idx_attracting_nodes)

#                            print('---------------------------------')
                        
                else: 
                    pass
                if growing_node not in already_grown_vertices:
                    already_grown_vertices.append(growing_node)
                    vv = [self.graph.vp['id'][growing_node] for growing_node in already_grown_vertices]
                    vv = np.sort(vv)
#                    print('already grown nodes: ',vv)
        t0 =time.time()
        self.close_road()
        t1 = time.time()
        print('time to close road: ',t1-t0)

    def mask_real_edges(self):            
        for node in self.graph.vertices():
            in_edges = list(node.in_edges())
            out_edges = list(node.out_edges())
            print('node {} '.format(self.graph.vp['id'][node]),'in edges: ',len(in_edges),' out edges: ',len(out_edges))
            total_edge_count = len(in_edges) + len(out_edges)
            if total_edge_count > 2:
                for edge in self.graph.edges():
                    self.graph.ep['real_edge'][edge] = True
                for e in out_edges:
                    self.graph.ep['real_edge'][edge] = True
        self.graph_real_edges = self.graph.copy()
        self.graph_real_edges.set_edge_filter(self.graph.ep['real_edge'])
    def compute_total_length(self):
        self.total_length = 0
        for edge in self.graph.edges():
            self.total_length += self.graph.ep['distance'][edge]
    # GRID
    def set_grid(self):
        '''
        Description:
            Sets a grid of squares of self.side_city/np.sqrt(self.number_grids)
        '''
        ## Grid
        self.city_box = Grid(self.side_city,self.side_city/self.number_grids)
        self.grid = self.city_box.partition() # Mst expensive procedure in terms of time
        self.bounding_box = self.city_box.get_polygon()
        self.grid2point = defaultdict(list)




    ## SAVING GRAPH
    def save_custom_graph(self,filename):
        # Save the graph to the specified filename
        self.graph.save(filename)
        print(f"Graph saved to {filename}")


    ## PLOTTING PROCEDURES

    def plot_relative_neighbors(self,vi,attracting_vertex,new_added_vertex,available_vertices):
        fig,ax = plt.subplots(1,2,figsize = (20,20))
        ## All attracting vertices
        attracting_vertices = np.array([np.array([self.graph.vp['x'][v],self.graph.vp['y'][v]]) for v in self.graph.vertices() if self.graph.vp['attracting'][v] == True])
        attracting_vertices_indices = np.array([self.graph.vp['id'][v] for v in self.graph.vertices() if self.graph.vp['attracting'][v] == True])
        ## Attracting vertex whose growing relative neighbors are updated
        coords_attracting_vertex = np.array([self.graph.vp['x'][attracting_vertex],self.graph.vp['y'][attracting_vertex]])
        ## Growing node v
        coordinates_vi = np.array([self.graph.vp['x'][vi],self.graph.vp['y'][vi]])
        coordinates_new_added_vertex = np.array([self.graph.vp['x'][new_added_vertex],self.graph.vp['y'][new_added_vertex]])  
        ## vector toward attracting vertices
        coords_available_vertices = np.array([np.array([self.graph.vp['x'][vj],self.graph.vp['y'][vj]]) for vj in available_vertices])
        toward_attr_vertices = coords_available_vertices - coordinates_vi
        ua_plus_ub = np.sum(toward_attr_vertices,axis = 0)
        utoward_attr_vertices = ua_plus_ub/np.sqrt(np.sum(ua_plus_ub**2))
        print(np.shape(utoward_attr_vertices))
        vector_edge = self.graph.vp['pos'][new_added_vertex].a - self.graph.vp['pos'][vi].a   
        uvector_edge = vector_edge/np.sqrt(np.sum(vector_edge**2))
        ## plot (attracting vertices, attracting vertex, growing node)
        ax[0].scatter(self.graph.vp['x'].a,self.graph.vp['y'].a,color = 'black')
        ax[0].scatter(attracting_vertices[:,0],attracting_vertices[:,1],color = 'blue')
        for av in range(len(attracting_vertices_indices)):
            ax[0].text(attracting_vertices[av,0],attracting_vertices[av,1], f'({attracting_vertices_indices[av]})', verticalalignment='bottom', horizontalalignment='center', fontsize=10)                
        ax[0].scatter(coords_attracting_vertex[0],coords_attracting_vertex[1],color = 'yellow')
        ax[0].scatter(coordinates_vi[0],coordinates_vi[1],color = 'red')
        ## Plot relative neighbors
        ax[1].scatter(self.graph.vp['x'].a,self.graph.vp['y'].a,color = 'black')
        ax[1].scatter(attracting_vertices[:,0],attracting_vertices[:,1],color = 'white')
        for av in range(len(attracting_vertices_indices)):
            ax[1].text(attracting_vertices[av,0],attracting_vertices[av,1], f'({attracting_vertices_indices[av]})', verticalalignment='bottom', horizontalalignment='center', fontsize=10)                
        ax[1].scatter(coords_attracting_vertex[0],coords_attracting_vertex[1],color = 'yellow')
        ax[1].scatter(coordinates_vi[0],coordinates_vi[1],color = 'red')        
        ax[1].scatter(coordinates_new_added_vertex[0],coordinates_new_added_vertex[1],color = 'orange')
        ax[1].plot([coordinates_vi[0],coordinates_new_added_vertex[0]],[coordinates_vi[1],coordinates_new_added_vertex[1]],linestyle = '-',linewidth = 1.5,color = 'black')
        ax[1].plot(vector_edge[0],vector_edge[1],linestyle = '-',linewidth = 1.5,color = 'violet')
        ax[1].grid()
        print('growth line: ',uvector_edge)
        print('new added: ',utoward_attr_vertices)
        print('theta: ',np.arccos(np.dot(uvector_edge,utoward_attr_vertices)/np.sqrt(np.sum(vector_edge**2)))/np.pi*180)
        print('growth line is orthogonal to line (vi,new_node): ',np.dot(uvector_edge,utoward_attr_vertices))
        for vj in self.graph.vp['relative_neighbors'][vi]:
            coordinates_vj = np.array([self.graph.vp['x'][vj],self.graph.vp['y'][vj]])
            r = self.distance_matrix_[self.graph.vp['id'][vi]][vj]
            circle1 = plt.Circle(coordinates_vi, r, color='red', linestyle = '--',fill=True ,alpha = 0.1)
            circle2 = plt.Circle(coordinates_vj, r, color='green', linestyle = '--',fill=True ,alpha = 0.1)
            ax[1].add_artist(circle1)
            ax[1].add_artist(circle2)
#            intersection = plt.Circle((coordinates_vi + coordinates_vj) / 2, np.sqrt(r ** 2 - (1/(2*r)) ** 2), color='green',alpha = 0.2)
            # Add the intersection to the axis
#            ax.add_artist(intersection)            
            ax[1].scatter(self.graph.vp['x'][self.graph.vertex(vj)],self.graph.vp['y'][self.graph.vertex(vj)],color = 'green')
            ax[1].plot([self.graph.vp['x'][vi],self.graph.vp['x'][self.graph.vertex(vj)]],[self.graph.vp['y'][vi],self.graph.vp['y'][self.graph.vertex(vj)]],linestyle = '--',color = 'green')
        ax[0].legend(['any vertex','attracting vertices','responsable attracting vertex','growing node'])
        ax[1].legend(['any vertex','attracting vertices','responsable attracting vertex','growing node','new added vertex','connection new node','growth line','relative neighbors','circle growing','circle relative neighbor','line relative neighbor'])
        plt.title('attracting {0}, growing {1} '.format(self.graph.vp['id'][attracting_vertex],self.graph.vp['id'][vi]))
        plt.show()

    def plot_growing_roads(self):
        fig,ax = plt.subplots(1,1,figsize = (20,20))
        colors = ['red','blue','green','yellow','orange','violet','black','brown','pink','grey','cyan','magenta']
        for starting_vertex in self.important_vertices:
            for r in self.graph.vp['roads'][starting_vertex]: 
                for edge in r.list_edges:
                    if r.id < len(colors):
                        ax.plot([self.graph.vp['x'][edge.source()],self.graph.vp['x'][edge.target()]],[self.graph.vp['y'][edge.source()],self.graph.vp['y'][edge.target()]],linestyle = '-',color = colors[r.id])
                    else:
                        i = r.id%len(colors) 
                        ax.plot([self.graph.vp['x'][edge.source()],self.graph.vp['x'][edge.target()]],[self.graph.vp['y'][edge.source()],self.graph.vp['y'][edge.target()]],linestyle = '-',color = colors[i])
        plt.show()
    def plot_evolving_graph(self):
        fig,ax = plt.subplots(1,1,figsize = (20,20))
        for edge in self.graph.edges():
            if self.graph.vp['important_node'][edge.source()] == True:
                color_source = 'red'
                ax.scatter(self.graph.vp['x'][edge.source()],self.graph.vp['y'][edge.source()],color = color_source)
                ax.plot([self.graph.vp['x'][edge.source()],self.graph.vp['x'][edge.target()]],[self.graph.vp['y'][edge.source()],self.graph.vp['y'][edge.target()]],linestyle = '-',color = 'black')
        fig.show()


    def animate_growth(self):        
        # Setting general parameters
        black = [0, 0, 0, 1]           # Black color (attracting nodes)
        red = [1, 0, 0, 1]             # Red color (important nodes)
        green = [0, 1, 0, 1]           # Green color (growing nodes)
        blue = [0, 0, 1, 1]            # Blue color 
        pos = self.graph.vp['pos']
        # Generate path save 
        if not os.path.exists(os.path.join(root,'animation')):
            os.mkdir(os.path.join(root,'animation'))
        self.max_count = 500
        # Generate the graph window
        if not self.offscreen:
            win = gt.GraphWindow(self.graph, pos, geometry=(500, 400),
                            edge_color=[0.6, 0.6, 0.6, 1],
                            vertex_fill_color=self.graph.vp['important_node'],
                            vertex_halo=self.graph.vp['attracting'],
                            vertex_halo_color=[0.8, 0, 0, 0.6])
        else:
            win = Gtk.OffscreenWindow()
            win.set_default_size(500, 400)
            win.graph = gt.GraphWidget(self.graph, pos,
                                edge_color=[0.6, 0.6, 0.6, 1],
                                vertex_fill_color=self.graph.vp['state'],
                                vertex_halo=self.graph.vp['attracting'],
                                vertex_halo_color=[0.8, 0, 0, 0.6])
            win.add(win.graph)
        # Bind the function above as an 'idle' callback.
        cid = GLib.idle_add(self.update_state())
        # We will give the user the ability to stop the program by closing the window.
        win.connect("delete_event", Gtk.main_quit)
        # Actually show the window, and start the main loop.
        win.show_all()
        Gtk.main()

    def update_state(self,win):
        # Filter out the recovered vertices
        self.graph.set_vertex_filter(self.graph.vp['attracting'], inverted=True)
        # The following will force the re-drawing of the graph, and issue a

        # re-drawing of the GTK window.
        win.graph.regenerate_surface()
        win.graph.queue_draw()
        # if doing an offscreen animation, dump frame to disk

        if offscreen:
            global count
            pixbuf = win.get_pixbuf()
            pixbuf.savev(r'./frames/sirs%06d.png' % count, 'png', [], [])
            if count > max_count:
                sys.exit(0)
            count += 1
        # We need to return True so that the main loop will call this function more
        # than once.
        return True

    # Bind the function above as an 'idle' callback.

## MAIN
def build_planar_graph(config,r0):
    '''
        Runs the creation of the graph
    '''
    ## Initialization parameters (with r0 being the characteristic distance of the probability distribution generation)
    bg = barthelemy_graph(config,r0)
    ## Initializes the properties of graph
    bg.initialize_graph()
    ## Add initial centers and control they are in the bounding box
    bg.add_centers2graph(bg.r0,bg.ratio_evolving2initial*bg.initial_number_points,bg.side_city)
    ## Compute the relative neighbors of each attracting node
    bg.compute_rng_newly_added_centers()
    bg.evolve_street_newly_added_attractors()
    bg.starting_phase = False
    list_number_real_edges = []
    list_time = []
    for t in range(100): #bg.number_iterations
        print('iteration: ',t)
        if t%bg.tau_c == 0 and t!=0:
            # add nodes
            print('*********************************************')
            print('ADD CENTERS')
            bg.add_centers2graph(bg.r0,bg.ratio_evolving2initial*bg.initial_number_points,bg.side_city)
            print('UPDATE RELATIVE NEIGHBORS NEW CENTERS')
            t0 = time.time()
            bg.compute_rng_newly_added_centers()
            t1 = time.time()
            print('time update rng old: ',t1-t0)      
            print('EVOLVE STREET FOR NEW CENTERS')
            bg.evolve_street_newly_added_attractors()
        if t%bg.tau_c != 0 and t!=0:
            print('*********************************************')
        print('UPDATE RELATIVE NEIGHBORS OLD CENTERS')
        t0 = time.time()
        bg.compute_rng_old_centers()
        t1 = time.time()  
        print('time update rng old: ',t1-t0)      
        print('EVOLVE STREET OLD CENTERS')
        t0 = time.time()
        bg.evolve_street_old_attractors()
        t1 = time.time()
        print('time to evolve street: ',t1-t0)
        print('time to compute rng: ',t1-t0)
        bg.update_attracted_by()
        bg.close_road()
        bg.mask_real_edges()
        print('number of edges: ',bg.graph.num_edges())
        list_number_real_edges.append(bg.graph_real_edges.num_edges())
        list_time.append(t)
        print('number real edges: ',bg.graph_real_edges.num_edges())
        if t==10:
            bg.plot_evolving_graph()
            bg.plot_growing_roads()
            for edge in bg.graph.edges():
                print(edge)
    if not os.path.exists(os.path.join(root,'graphs')):
        os.mkdir(os.path.join(root,'graphs'))
    bg.save_custom_graph(os.path.join(root,'graphs','graph_r0_{0}.gt'.format(round(r0,2))))
    plt.scatter(list_time,list_number_real_edges)
    plt.xlabel('time')
    plt.ylabel('number of real edges')
    plt.plot() 


if __name__ == '__main__':
    ## Initialize parameters
    seed = np.random.seed(42)
    list_r0 = [0.1]#np.linspace(1,10,100) # r0 takes values with a frequency of 100 meter from 0 to 10 km
    tuple = os.walk('.', topdown=True)
    root = tuple.__next__()[0]
    print('root: ',root)
    config_dir = os.path.join('/home/aamad/Desktop/phd/berkeley/traffic_phase_transition','config')
    config_name = os.listdir(config_dir)[0]
    with open(os.path.join(config_dir,config_name),'r') as f:
        config = json.load(f)
    number_nodes = 4
    for r0 in list_r0:
        build_planar_graph(config,r0)
