from collections import defaultdict
import graph_tool as gt
from graph_tool.all import label_components
from scipy.spatial import distance_matrix,Voronoi
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import time
from gi.repository import Gtk, Gdk, GdkPixbuf, GObject, GLib
import multiprocessing as mp
from pyhull.delaunay import DelaunayTri
from scipy.spatial import Delaunay

'''
1) Generates the first centers [they are all of interest and they are all attracting and are all growing]
2) For each growing I compute the relative neighbor (if it is growing iself and not a center, then I give up, otherwise I continue)

'''
class attractor:
    def __init__(self):
        self.id: int # s
        self.current_attracted_vertices: list # V(s)
        self.current_attracted_roads: list # [starting_vertex,...,ending_vertex]



class evolving_node:
    def __init__(self):
        self.id: int # s
        self.current_


class road:
    '''
        Road will become useful when I need to update rng, indeed It is happening that if:
            Node i and Node j are attracted one another and they grow the first step of the road
            they stop, as now their relative neighbors are the points that have grown,
            for this reason I want to avoid the elements of the road in the rng calculation
        Need to add a condition not to growing element grow back to the starting point  
    '''
    # TODO: need to block the growing nodes to grow back to their starting point
    def __init__(self,initial_node):
        self.initial_node = initial_node
        self.number_iterations = 0
        self.length = 0
        self.list_nodes = [initial_node]
        self.list_edges = []
        self.evolution_attractors = defaultdict()#{t:[] for t in range()}
        self.final_node = []        

    def check_if_vertex_in_road(self,growing_vertex):
        '''
            Every time evolve_street, growing node mus also not return to the starting road
        '''
        return growing_vertex in self.list_nodes
    def add_node_in_road(self,starting_vertex,new_vertex):
        '''
            Every time add_point2graph I increase the read
        '''
        if starting_vertex in self.list_nodes or starting_vertex == self.initial_node:
            self.list_nodes.append(new_vertex)
            self.list_edges.append([starting_vertex,new_vertex])
            self.length += 1
            self.number_iterations += 1
        else:
            print('trying to attach a vertex that is not of this road')
            pass
    
    def in_road(self,vertex):
        return vertex in self.list_nodes


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
    ## id
    id_ = g.new_vertex_property('int')
    g.vp['id'] = id_
    ## Node centrality (These are the points of interest)
    important_node = g.new_vertex_property('bool')
    g.vp['important_node'] = important_node
    ## attracting (they are the "auxins", streets are attracted by them)
    attracting = g.new_vertex_property('bool')
    g.vp['attracting'] = attracting
    ## list_nodes_attracted (Is the set of indices of vertices that are attracted by the node)
    list_nodes_attracted = g.new_vertex_property('vector<int>')
    g.vp['list_nodes_attracted'] = list_nodes_attracted
    ## growing nodes (they are attached to the growing net, street evolve from this set)
    active = g.new_vertex_property('bool')
    g.vp['growing'] = active
    ## neighbors (Is the set of indices of vertices that are relative neighbors of the attracting node)
    relative_negihbors = g.new_vertex_property('vector<int>')
    g.vp['relative_neighbors'] = relative_negihbors
    ## positions of the vertices
    x = g.new_vertex_property('double')
    y = g.new_vertex_property('double')
    g.vp['x'] = x
    g.vp['y'] = y
    pos = g.new_vertex_property('vector<double>')
    g.vp['pos'] = pos
    ## Voronoi diagram of the vertices
    voronoi = g.new_vertex_property('object')
    g.vp['voronoi'] = voronoi
    g.vp['delauney_neighbors'] = g.new_vertex_property('vector<int>')
    ## Set of roads starting at the node
    g.vp['road'] = g.new_vertex_property('vector<object>')
    # EDGES
    growth_unit_vect = g.new_edge_property('double')
    g.ep['distance'] = growth_unit_vect
    growth_unit_vect = g.new_edge_property('vector<double>')
    g.ep['direction'] = growth_unit_vect
    real_edges = g.new_edge_property('bool')
    g.ep['real_edge'] = real_edges
    capacity = g.new_edge_property('double')    
    # Animation
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


class barthelemy_graph:
    def __init__(self,config:dict,r0):
        # Geometrical parameters
        self.side_city = config['side_city'] # in km
        self.ratio_growth2size_city = config['ratio_growth2size_city'] # 1000 segments  to go from one place to the other (pace of 10 meters) 
        self.rate_growth = self.ratio_growth2size_city*self.side_city
        self.r0 = r0
        # dynamical parameters
        self.number_iterations = config['number_iterations'] #
        self.tau_c = config['tau_c'] # 
        self.number_nodes_per_tau_c = config['number_nodes_per_tau_c'] 
        # creation rates of nodes
        self.initial_number_points = config['initial_number_points']
        self.total_number_attraction_points = config['total_number_attraction_points'] # these are the auxins
        self.ratio_evolving2initial = config['ratio_evolving2initial']
        self.total_number_nodes = self.number_iterations*self.number_nodes_per_tau_c + self.initial_number_points
        self.distance_matrix_ = None
        # relative neighbors
        self.delauney_ = None
        # policentricity
        self.number_centers = config['degree_policentricity']
        # animation
        self.offscreen = config['offscreen']

    def initialize_graph(self):
        self.graph = initial_graph()

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
#        print('expected distance matrix shape {0},{1}'.format(len(old_points)+1,len(old_points)+1),np.shape(self.distance_matrix_))

    def get_voronoi(self,vertex):
        self.graph.vp['voronoi'][vertex] = Voronoi(np.array([self.graph.vp['x'][vertex],self.graph.vp['y'][vertex]]).T)

    def add_initial_points2graph(self,r0,number_nodes,side_city):
        x,y = generate_exponential_distribution_nodes_in_space_square(r0,number_nodes,side_city)
        self.initialize_distance_matrix(x,y)
        for point_idx in range(len(x)):
            self.graph.add_vertex()
            self.graph.vp['x'][self.graph.vertex(self.graph.num_vertices()-1)] = x[point_idx]
            self.graph.vp['y'][self.graph.vertex(self.graph.num_vertices()-1)] = y[point_idx]
            self.graph.vp['pos'][self.graph.vertex(self.graph.num_vertices()-1)] = np.array([x[point_idx],y[point_idx]])
            self.graph.vp['important_node'][self.graph.vertex(self.graph.num_vertices()-1)] = True
            self.graph.vp['attracting'][self.graph.vertex(self.graph.num_vertices()-1)] = True
            self.graph.vp['id'][self.graph.vertex(self.graph.num_vertices()-1)] = self.graph.num_vertices()-1
            self.graph.vp['relative_neighbors'][self.graph.vertex(self.graph.num_vertices()-1)] = []
            self.graph.vp['growing'][self.graph.vertex(self.graph.num_vertices()-1)] = True
        for vertex in self.graph.vertices():
            for vertex1 in self.graph.vertices():
                if vertex != vertex1 and is_graph_connected(self.graph) == False:
                    self.graph.add_edge(vertex,vertex1)
                    self.graph.ep['distance'][self.graph.edge(vertex,vertex1)] = self.distance_matrix_[self.graph.vp['id'][vertex],self.graph.vp['id'][vertex1]]
                    self.graph.ep['direction'][self.graph.edge(vertex,vertex1)] = self.graph.vp['pos'][vertex1].a - self.graph.vp['pos'][vertex].a
                    self.graph.ep['real_edge'][self.graph.edge(vertex,vertex1)] = False


    def add_centers2graph(self,r0,number_nodes,side_city):
        '''
            Generates number_of_nodes:
                1) important
                2) from an exponential distribution (characteristic distance r0)
                3) attracting
                4) not growing

        '''
        x,y = generate_exponential_distribution_nodes_in_space_square(r0,number_nodes,side_city)
        if self.distance_matrix_ is None:
            self.initialize_distance_matrix(x,y)
            for point_idx in range(len(x)):
                self.graph.add_vertex()
                self.graph.vp['x'][self.graph.vertex(self.graph.num_vertices()-1)] = x[point_idx]
                self.graph.vp['y'][self.graph.vertex(self.graph.num_vertices()-1)] = y[point_idx]
                self.graph.vp['pos'][self.graph.vertex(self.graph.num_vertices()-1)] = np.array([x[point_idx],y[point_idx]])
                self.graph.vp['important_node'][self.graph.vertex(self.graph.num_vertices()-1)] = True
                self.graph.vp['attracting'][self.graph.vertex(self.graph.num_vertices()-1)] = True
                self.graph.vp['id'][self.graph.vertex(self.graph.num_vertices()-1)] = self.graph.num_vertices()-1
                self.graph.vp['relative_neighbors'][self.graph.vertex(self.graph.num_vertices()-1)] = []
                self.graph.vp['growing'][self.graph.vertex(self.graph.num_vertices()-1)] = True
                self.graph.vp['road'][self.graph.vertex(self.graph.num_vertices()-1)] = road(self.graph.vertex(self.graph.num_vertices()-1))
        else:
            for point_idx in range(len(x)):
                self.update_distance_matrix(np.array([self.graph.vp['x'].a,self.graph.vp['y'].a]).T,np.array([[x[point_idx],y[point_idx]]]))
                self.graph.add_vertex()
                self.graph.vp['x'][self.graph.vertex(self.graph.num_vertices()-1)] = x[point_idx]
                self.graph.vp['y'][self.graph.vertex(self.graph.num_vertices()-1)] = y[point_idx]
                self.graph.vp['pos'][self.graph.vertex(self.graph.num_vertices()-1)] = np.array([x[point_idx],y[point_idx]])
                self.graph.vp['important_node'][self.graph.vertex(self.graph.num_vertices()-1)] = True
                self.graph.vp['attracting'][self.graph.vertex(self.graph.num_vertices()-1)] = True
                self.graph.vp['id'][self.graph.vertex(self.graph.num_vertices()-1)] = self.graph.num_vertices()-1
                self.graph.vp['relative_neighbors'][self.graph.vertex(self.graph.num_vertices()-1)] = []
                self.graph.vp['growing'][self.graph.vertex(self.graph.num_vertices()-1)] = False
                self.graph.vp['road'][self.graph.vertex(self.graph.num_vertices()-1)] = road(self.graph.vertex(self.graph.num_vertices()-1))
    def update_delauney(self):
        self.delauney_graph = self.graph.copy()
        self.delauney_graph.clear_edges()
        x = self.graph.vp['x'].a
        y = self.graph.vp['y'].a
        tri = Delaunay(np.array([x,y]).T)
        # Iterate over all triangles in the Delaunay triangulation
        for simplex in tri.simplices:
            for i, j in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]:
                if (self.graph.vertex(i),self.graph.vertex(j)) not in self.delauney_graph.edges():
                    self.delauney_graph.add_edge(self.graph.vertex(i),self.graph.vertex(j))
                    if not j in self.graph.vp['delauney_neighbors'][self.graph.vertex(i)]:   
                        self.graph.vp['delauney_neighbors'][self.graph.vertex(i)].append(j)
                    if not i in self.graph.vp['delauney_neighbors'][self.graph.vertex(j)]:
                        self.graph.vp['delauney_neighbors'][self.graph.vertex(j)].append(i)

    def print_delauney_neighbors(self,vi):
        print('delauney neighbors: ')
        for vj in self.graph.vp['delauney_neighbors'][vi]:
            print(vj)

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

    def plot_evolving_graph(self):
        fig,ax = plt.subplots(1,1,figsize = (20,20))
        for edge in self.graph.edges():
            if self.graph.vp['important_node'][edge.source()] == True:
                color_source = 'red'
                ax.scatter(self.graph.vp['x'][edge.source()],self.graph.vp['y'][edge.source()],color = color_source)
                ax.plot([self.graph.vp['x'][edge.source()],self.graph.vp['x'][edge.target()]],[self.graph.vp['y'][edge.source()],self.graph.vp['y'][edge.target()]],linestyle = '-',color = 'black')
        fig.show()

    def compute_rng(self): 
        '''
            Description:
                1) update_delauney:
                    1a) compute the Voronoi diagram for all vertices O(Nlog(N))
                    1b) compute the delauney triangulation  O(N)
                2) For each attracting node (vi) and a node from the delauney triangulation (vj)
                3) Check for if for any node vk in the graph max(d_ix, d_xj) < d_ij
                4) If it is not the case, then vj is a relative neighbor of vi

        '''
        self.growing_graph  = self.graph.copy()
        self.growing_vertices = [v for v in self.growing_graph.vertices() if self.growing_graph.vp['growing'][v] == True]
#        print('growing vertices: ',len(self.growing_vertices),self.growing_graph.vp['growing'].a)
        self.attracting_graph = self.graph.copy()
        self.attracting_vertices = [v for v in self.attracting_graph.vertices() if self.attracting_graph.vp['attracting'][v] == True]
        self.update_delauney()

#        print('attracting vertices: ',len(self.attracting_vertices),self.attracting_graph.vp['attracting'].a)
#        print('available vertices: ',self.graph.vp['id'].a)
        if self.growing_graph is None:
            pass
        else:
            for vi in self.attracting_vertices: # self.growing_graph.vertices()
                list_nodes_road_vi = self.graph.vp['road'].list_nodes
                self.graph.vp['relative_neighbors'][vi] = []
                for vj in self.graph.vp['delauney_neighbors'][vi]: # self.growing_graph.vertices()
                    try:
                        d_ij = self.distance_matrix_[self.graph.vp['id'][vi]][vj]
                    except KeyError:
                        d_ij = None
                        continue
                    for vx in self.graph.vertices(): # self.attracting_graph.vertices() 
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


    def close_road(self):
        '''
            Attach the edge if the growing points are close enough to their relative neighbors,
            in this way the relative neighbor becomes a growing point as well
        '''
        already_existing_edges = [[e.source(),e.target()] for e in self.graph.edges()]
        for v in self.attracting_vertices:
            for u_idx in self.graph.vp['relative_neighbors'][v]:
                    if self.distance_matrix_[self.graph.vp['id'][v],u_idx] < self.rate_growth and [v,self.graph.vertex(u_idx)] not in already_existing_edges:
                        self.graph.add_edge(self.graph.vertex(u_idx),v)
                        if self.graph.vp['growing'][v] == False:
                            self.graph.vp['growing'][v] = True
                            print('closing road between: ',self.graph.vp['id'][v],self.graph.vp['id'][u_idx])
                        else:
                            pass
                            
    def add_point2graph(self,source_x,source_y,dx,dy,degenerate = False):
        '''
            Adds a point from the graph and initializes x,y with the vector dx,dy
            The added node is:
                1) not important
                2) attracting
                3) growing
                4) with an empty set of relative neighbors
        '''
#        if self.graph.vp['important_node'][self.graph.vertex(self.graph.num_vertices()-1)]==False: 
#            self.graph.vp['growing'][self.graph.vertex(self.graph.num_vertices()-1)] = False 
        if not degenerate:
            self.graph.add_vertex()
            self.graph.vp['x'][self.graph.vertex(self.graph.num_vertices()-1)] = source_x + dx 
            self.graph.vp['y'][self.graph.vertex(self.graph.num_vertices()-1)] = source_y + dy
            self.graph.vp['pos'][self.graph.vertex(self.graph.num_vertices()-1)] = np.array([source_x + dx ,source_y + dy ])
            self.graph.vp['important_node'][self.graph.vertex(self.graph.num_vertices()-1)] = False
            self.graph.vp['attracting'][self.graph.vertex(self.graph.num_vertices()-1)] = False
            self.graph.vp['growing'][self.graph.vertex(self.graph.num_vertices()-1)] = True
            self.graph.vp['id'][self.graph.vertex(self.graph.num_vertices()-1)] = self.graph.num_vertices()-1
            self.graph.vp['relative_neighbors'][self.graph.vertex(self.graph.num_vertices()-1)] = []            
        return self.graph.vertex(self.graph.num_vertices()-1)


    def add_edge2graph(self,source_vertex,target_vertex):
        '''
            source_idx: Vertex
            target_idx: Vertex
        '''
        source_idx = self.graph.vp['id'][source_vertex]
        target_idx = self.graph.vp['id'][target_vertex]
        self.graph.add_edge(source_idx,target_idx)
        self.graph.ep['distance'][self.graph.edge(source_idx,target_idx)] = self.distance_matrix_[source_idx,target_idx]
        self.graph.ep['direction'][self.graph.edge(source_idx,target_idx)] = self.graph.vp['pos'][target_vertex].a - self.graph.vp['pos'][source_vertex].a
        self.graph.ep['real_edge'][self.graph.edge(source_idx,target_idx)] = False

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
        for attracting_node in self.attracting_vertices:
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
                        for starting_vertex_road in self.attracting_vertices:
                            if self.graph.vp['road'][starting_vertex_road].in_road(growing_node):
                                self.graph.vp['road'][starting_vertex_road].add_node_in_road(starting_vertex_road,added_vertex)                            
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
                            for starting_vertex_road in self.attracting_vertices:
                                if self.graph.vp['road'][starting_vertex_road].in_road(growing_node):
                                    self.graph.vp['road'][starting_vertex_road].add_node_in_road(starting_vertex_road,added_vertex)                            

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
                                for starting_vertex_road in self.attracting_vertices:
                                    if self.graph.vp['road'][starting_vertex_road].in_road(growing_node):
                                        self.graph.vp['road'][starting_vertex_road].add_node_in_road(starting_vertex_road,added_vertex)                                                      
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

    def save_custom_graph(self,filename):
        # Save the graph to the specified filename
        self.graph.save(filename)
        print(f"Graph saved to {filename}")

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

 
if __name__ == '__main__':
    np.random.seed(42)
    list_r0 = [10]#np.linspace(1,10,100) # r0 takes values with a frequency of 100 meter from 0 to 10 km
    tuple = os.walk('.', topdown=True)
    root = tuple.__next__()[0]
    config_dir = os.path.join(root,'config')
    config_name = os.listdir(config_dir)[0]
    with open(os.path.join(config_dir,config_name),'r') as f:
        config = json.load(f)
    number_nodes = 4
    for r0 in list_r0:
        bg = barthelemy_graph(config,r0)
        bg.initialize_graph()
#        bg.add_initial_points2graph(r0,number_nodes,bg.side_city)
        bg.add_centers2graph(bg.r0,bg.ratio_evolving2initial*bg.initial_number_points,bg.side_city)
        bg.compute_rng()
        list_number_real_edges = []
        list_time = []
        for t in range(100): #bg.number_iterations
            print('iteration: ',t)
            if t%bg.tau_c == 0 and t!=0:
                # add nodes
                print('*********************************************')
                print('ADD CENTERS')
                t0 = time.time()
                bg.add_centers2graph(bg.r0,bg.ratio_evolving2initial*bg.initial_number_points,bg.side_city)
#                bg.initialize_delauney_graph()
                t1 = time.time()
                print('time to add centers: ',t1-t0)
#                t0 = time.time()
#                bg.compute_rng()
#                t1 = time.time()
#                print('time to compute rng: ',t1-t0)
#                bg.close_road()
#                print('number of nodes graph: ',bg.graph.num_vertices())
#                bg.graph.set_vertex_filter(bg.graph.vp['important_node'])
#                print('number of important nodes in graph:',bg.graph.num_vertices())
#                bg.graph.set_vertex_filter(None)
#                print('number of edges: ',bg.graph.num_edges())
#                bg.mask_real_edges()
#                list_number_real_edges.append(bg.graph_real_edges.num_edges())
#                list_time.append(t)
#                print('number real edges: ',bg.graph_real_edges.num_edges())                
            else:
                print('*********************************************')
                print('EVOLVE STREET')
                t0 = time.time()
                bg.evolve_street()
                t1 = time.time()
                print('time to evolve street: ',t1-t0)
            t0 = time.time()
            # multiprocessing to compute rng
#            pool = mp.Pool(mp.cpu_count())
#            vi = list(bg.graph.vp['id'].a)
#            pool.map(bg.compute_rng_parallel,vi)
            bg.compute_rng()
            t1 = time.time()
            print('time to compute rng: ',t1-t0)
            bg.close_road()
            print('number of nodes graph: ',bg.graph.num_vertices())
            bg.graph.set_vertex_filter(bg.graph.vp['important_node'])
            print('number of important nodes in graph:',bg.graph.num_vertices())
            bg.graph.set_vertex_filter(None)
            bg.mask_real_edges()
            print('number of edges: ',bg.graph.num_edges())
            list_number_real_edges.append(bg.graph_real_edges.num_edges())
            list_time.append(t)
            print('number real edges: ',bg.graph_real_edges.num_edges())
            if t==10:
                bg.plot_evolving_graph()
                for edge in bg.graph.edges():
                    print(edge)
        if not os.path.exists(os.path.join(root,'graphs')):
            os.mkdir(os.path.join(root,'graphs'))
        bg.save_custom_graph(os.path.join(root,'graphs','graph_r0_{0}.gt'.format(round(r0,2))))
    plt.scatter(list_time,list_number_real_edges)
    plt.xlabel('time')
    plt.ylabel('number of real edges')
    plt.plot()