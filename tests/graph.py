import numpy as np
from graph-tools.all import *
from plotting_procedures import xy_plot
import scipy.spatial as sp
import pyhull



class graph_gt:
    '''
        Properties graph:
            Nodes:
                Mandatory: (at initialization)
                    'id': int
                    'position': np.array(dtype=float)
                Optional: (computed by some function)
                    'topological_distance': vector<int>
                    'metric_distance': vector<float>
            Edges:
                Mandatory: (at initialization)
                    'length': float
                    'unweighted': float 
                Optional: (computed by some function)
                    'time percorrence': float

        Description:
            This class creates a graph with the library graph-tools. The functions of the class are created for the scope of reproducing
            directed percolation universality classes for different planar networks according to the papers cited in them.
            The hope is to verify the existence of the dependence of the phase transition exponent with the distribution of
            centers (sinks of mobility) in the network.
    '''
    def __init__(self,
                 library_used = 'graph-tools',
                 type_graph = 'lattice',
                 reading_mode = False,
                 geojson_path = None):
        '''
        library_used = 'networkx','pytorch','graph-tools'
        Usage:
            instantiate graph (different type for the different purpose)  
        '''
        self.library_used = library_used
        self.list_type_graphs = ['lattice','newman','real_data']
        if reading_mode == False:
            self.graph = self.create_graph(type_graph,reading_mode)
        else:
            self.type_graph = 'real_data'
            self.graph = self.create_graph(type_graph,reading_mode,geojson_path)               

    def create_graph(self,
                     reading_mode = False,
                     number_nodes = 223330,
                     number_edges = 547696,
                     probability_connection = 0.5,
                     total_length = 48000,
                     area = 181000000,
                     populatoin = 715000000,
                     geojson_path = None):
        '''
            type_graph = 'lattice','newman-barthelemy','real_data'
            reading_mode = False, True (False if you want to create a graph from an ensamble of graph_type)
            if reading_mode = False:
                number_nodes = 223330, Default: San Francisco from OSM
                number_edges = 547696, Default: San Francisco from OSM
                total_length = 48000, is the total (m) of the road network Default: San Francisco Bay Area
                area = 181000000, is the total (m^2) of the road network Default: San Francisco Bay Area
                population = 715000000, is the total population of the road network Default: San Francisco Bay Area 
                probability_connection = 0.5, if reading_mode = False (probability of connection between two nodes)
        '''
        self.nuber_nodes = number_nodes
        if reading_mode == False:
            if self.type_graph == 'lattice':
                from graph_tools.all import lattice
                self.graph = lattice([number_nodes,number_nodes])
            elif self.type_graph == 'newman':
                self.create_newman_graph()
            elif self.type_graph == 'barthelemy':
                self.graph = self._create_barthelemy_graph()
        else:
            self.read_graph(geojson_path)

                                
                                
    ################################################ GENERATION PLANAR GRAPHS ##################################################
    
    ############## BARtHELEMY GRAPHS ##########################################################

    def _create_bartehelemy_graph(self,characteristic_length,tau_c = 3):
        '''
            Citation:
                Modeling Urban Street Patterns: Marc Barthélemy and Alessandro Flammini
            Input:
                characteristic_length: float = side(radius) length of the square(circle) of around city (m)
                tau_c: int = time of creation of a new center 
            Definitions:
                centers : houses, commercial centers, etc.
            Description:
                New centers are created at a constant rate and are connected to the existing network by new roads.   
            Drawbacks:
                1) The initial position of centers is random     
        '''
        self.points_in_network = []
        self.radius_city = characteristic_length
        unit_length = average_degree*self.radius_city*np.sqrt(self.number_nodes)/(2*number_edges)
        for tau_r in range(self.number_nodes):
            if tau_r%tau_c == 0:
                point = self.create_random_center()
                self.points_in_network.append(point)
                self.add_barthelemy_node_graph(tau_r,tau_c,point)

    def add_barthelemy_node_graph(self,tau_r,tau_c,point):
        self.graph.add_vertex(tau_r/tau_c)
        self.graph.vertex_properties['pos'][tau_r/tau_c] = point
        

    def _create_random_center(self):
        '''
            Description:
                Create a random center in the space
        '''
        center_x = np.random.random()*self.radius_city
        center_y = np.random.random()*self.radius_city
        center = np.array([center_x,center_y])
        return center
    
    def _update_distance_matrix(self):
        coords = (np.array([self.graph.vp['x'].a,self.graph.vp['y'].a]).T) 
        self.distance_matrix = sp.distance_matrix(coords,coords)

    def _compute_relative_neighbor(self):
        for vertex_idx in self.distance_matrix:
            idx_s = np.where(self.distance_matrix[vertex_idx,:] == np.min(self.distance_matrix[vertex_idx,:]))                
            # get vertex TODO
            vertex = # vertex with vertex_idx
            # ad the relative neighbor idx_s
            vertex.vp['relative_neighbor'] = # vertex with index = idx_s 

    
    ############### NEWMAN GRAPHS #############################################################
    def _create_newman_graph(self):
        '''
            Citation:
                The spatial structure of networks: Michael T. Gastner and M. E. J. Newman
            Description:
        '''
        self._generate_uniform_distribution_nodes_in_space_circle(self.radius_city)
        self.graph = gt.Graph(directed=False)
        for i in range(len(self.x)):
            self.graph.add_vertex()
        
            self.graph.vertex_properties['pos'][i] = np.array([self.x[i],self.y[i]])

    def _simulated_anneahling(self):
        self.Tmax = 25000
        self.number_iterations = 100
        self.alpha = 0.95
        self.energy = 0
        self.temperature_min = 0.00001
    def _generate_uniform_distribution_nodes_in_space_circle(self,radius_city):  
        '''
            Citation:
                The spatial structure of networks: Michael T. Gastner and M. E. J. Newman
            Description:
                Generate a uniform distribution of points in a circle of radius radius_city
                Initialize postiions and distance matrix
                ALL in one step:
        '''  
        from scipy.spatial import distance_matrix
        self.radius_city = radius_city
        nodes_r = np.random.random(self.number_nodes)
        nodes_costeta = 2*np.random.random(self.number_nodes) - 1        
        self.x = nodes_r*(self.radius_city)*nodes_costeta
        self.y = nodes_r*(self.radius_city)*np.sqrt(1-nodes_costeta**2)    
        self.distance_matrix = distance_matrix(np.array[self.x,self.y].T,np.array[self.x,self.y].T)

    def _generate_uniform_distribution_nodes_in_space_square(self,side_city):
        '''
            Citation:
                The spatial structure of networks: Michael T. Gastner and M. E. J. Newman
            Description:
                Generate a uniform distribution of points in a circle of radius radius_city
                Initialize postiions and distance matrix
                ALL in one step:        
        '''
        from scipy.spatial import distance_matrix
        self.radius_city = side_city
        self.x = np.random.random(self.number_nodes)*side_city
        self.y = np.random.random(self.number_nodes)*side_city    
        self.distance_matrix = distance_matrix(np.array[self.x,self.y].T,np.array[self.x,self.y].T)


    def _connect_nodes(self,probability_connection):
        '''
            Connect nodes up to a cost = $sum_{i \diff j} d_{ij} $ like in:
                The spatial structure of networks: Michael T. Gastner and M. E. J. Newman
                arXiv:cond-mat/0407680v1 [cond-mat.stat-mech] 26 Jul 2004
                $lim_{r \to \infty} d\frac{log(N_v(r))}{dlog(r)} = d$

        '''




        self.adj_matrix = self.graph.get_adjacency_matrix()


    def _control_dimension_graph(self):
        '''
            Test of planarity like in:
                The spatial structure of networks: Michael T. Gastner and M. E. J. Newman
                arXiv:cond-mat/0407680v1 [cond-mat.stat-mech] 26 Jul 2004
                $lim_{r \to \infty} d\frac{log(N_v(r))}{dlog(r)} = d$
        '''
        diameter = self.graph.diameter()
        self.degree_distribution = self.graph.degree_distribution() # degree list for each node
        steps = np.linspace(0,diameter,diameter)
        N_v = np.zeros(diameter)
        list_power_adj_matrix = [self.adj_matrix**r for r in steps] 
        for vertex in self.graph.vertices():
            for r in steps:
                N_v[r] += list_power_adj_matrix[vertex.index()][vertex.index()]
        avg_Nv = np.array(N_v)/np.array(self.degree_distribution)
        xy_plot(avg_Nv,steps,'$r$','$N_v(r)$','Average number of nodes at distance r')

    

    def _create_origin_destination(self,type_centricity):
        '''
            Description:
                This function creates the origin destination matrix for the monocentric case.
        '''
        
        return self.OD
    
    def _rewire_OD(self):
        assert self.OD!=None:
        if self.library_used == 'pytorch':
            pass
        elif self.library_used == 'graph-tools':
            if self.OD!=None:
                self.OD = self.OD.rewire()
        else:
            print('No OD matrix to rewire')
            break

    def _compute_centrality_index(self):
        '''
            Description:
                This function returns the centrality index of the graph.
        '''
        if self.library_used == 'networkx':
            pass
        elif self.library_used == 'pytorch':
            pass
        elif self.library_used == 'graph-tools':
            self.centrality_index = centrality.closeness(self.graph)
        return self.centrality_index

        

    def _save_origin_destination(self,OD_path):
        '''
            #TODO
            Description:
                This function saves the origin destination matrix in a csv format.
        '''
        number_trips_OD = np
        for i in range(self.population.number_people):
            for origin_idx in range(len(self.OD)):
                for destination_idx in range(len(self.OD[origin_idx])):

        df['SAMPN']
        df['PERNO']
        df['origin']
        df['destination']
        df.to_csv(OD_path, self.OD, delimiter=",")

    def _check_moran_index(self):
        '''
            #TODO
            Description:
                This function checks the moran index of the graph.
        '''
        self.moran_index = spatial.moran(self.graph)
        return self.moran_index

    def _compute_centrality_index(self):
        '''
            #TODO
            Description:
                This function returns the centrality index of the graph.
        '''
        self.centrality_index = centrality.closeness(self.graph)
        return self.centrality_index



    def _save_graph(self,graph_path):
        '''
            Description:
                This function saves the graph in a csv format (nodes.csv,edges.csv,origin_dest.csv).
        '''
        self.save_node_file(os.path.join(graph_path,'nodes.csv'))
        self.save_edge_file(os.path.join(graph_path,'edges.csv'))

        self.graph.save(graph_path)

    def save_node_file(self,total_graph_path):

    def save_edge_file(self,total_graph_path):
        
        'uniqueid','osmid_u','osmid_v','length','lanes','speed_mph','u','v'








    def draw_graph(self, edge_weights):
        # Create a graph layout
        pos = sfdp_layout(self.graph,cooling_step = 0.95, epsilon = 1e-2)
        graph_draw(self.graph, pos=pos, vertex_size=5, edge_pen_width=edge_weights,output = os.path.join(dir_'graph.png'))

        # Draw the graph with edge weights
        graph_draw(self.graph, pos=pos, vertex_size=5, edge_pen_width=edge_weights)

    def animate(self):
        # Create a list of edge weights
        edge_weights = np.zeros(self.graph.num_edges())

        # Create a graph layout
        pos = sfdp_layout(self.graph)

        # Create a figure and axis for the animation
        fig, ax = plt.subplots()

        # Define the animation function
        def update(frame):
            # Update the edge weights
            for i, e in enumerate(self.graph.edges()):
                edge_weights[i] = frame[i]

            # Draw the updated graph
            self.draw_graph(edge_weights)

            # Return the updated graph
            return ax

        # Create the animation
        anim = FuncAnimation(fig, update, frames=self.dc.get_model_vars_dataframe()['Speed'], interval=50)

        # Show the animation
        plt.show()





class city:
    '''
        Growth Tensor of Plant Organs: Z. HEJNOWICZ, JOHN A. ROMBERGER
    '''
    def __init__(self):
        self.growth_tensor: np.ndarray(dtype=float) # [number_barthelemy_vertices,spatial_dimension(2)]
        self.Area: float
        self.length: float

class barthelemy_vertex:
    '''
        Citation:
            Modeling Urban Street Patterns: Marc Barthélemy and Alessandro Flammini
            Modeling and visualization of leaf venation patterns: Adam Runions, Martin Fuhrer, Brendan Lane, Pavol Federl,
                                                                  Anne-Ga¨elle Rolland-Lagan, Przemyslaw Prusinkiewicz
            Growth Tensor of Plant Organs: Z. HEJNOWICZ, JOHN A. ROMBERGER
        Description:
            Wants to build an object whose attributes are able to describe the properties needed to create that graph

    '''
    def __init__(self,position,time_creation):
        self.position: np.array(dtype=float) = position
        self.all_vertices: np.ndarray(dtype=float)
        self.time_creation: int = time_creation
        self.relative_neighbors = []
        self.distance_from_all_vertices = []
        self.growth_direction: np.array(dtype=float)
        self.velocity_growth: float
        self.velocity_gradient_direction: np.array(dtype=float)
        self.velocity_gradient: float

    
    def _update_position(self):
        self.position += self.velocity_growth*self.growth_direction

    def _compute_distance_from_all_vertices(self):
        '''
            all_vertices: (np.ndarray) [number_vertices,spatial_dimension(2)]
            NOTE: the index of the vertices must be in order with the index of the distance matrix
        '''
        self.distance_matrix = sp.distance_matrix(self.position,self.all_vertices)

    def _choose_closest_neighbor(self):
        d1 = np.ma.masked_array(self.distance_matrix,mask = np.eye(len(d)))
        nd = np.unravel_index(np.argmin(d1, axis=None), d1.shape)


    def add_relative_neighbor(self,relative_neighbor):
        self.relative_neighbors.append(relative_neighbor)
        


    def add_edge(self,edge):
        self.edges.append(edge)

    def get_edges(self):
        return self.edges

    def get_position(self):
        return self.position

    def get_time_creation(self):
        return self.time_creation


def growing_city():
    for vertex in vertices:
        vertex._compute_distance_from_all_vertices()
        vertex._compute_growth_direction()



class partition:
    '''
    Measures of interest:
        eta_i : economic attractiviness partition i
        W_ij : jobs at partition i
        
    Indices range
        i = 1,..,number_partitions
        j = 1,.., number_jobs_given_i_
    Description:
        This class creates a map of partitions.
        Essentially it defines eta_i:
    eta depends on different factors and can be different for different users. As different users can access informations, or can 
    access to work in different ways.
    The most general objject, from which we take a semplification is W_j, the wage job j offers. 
    c = 1,..., average_wage.
    In general I have 
    '''
    def __init__(self):



def generate_graph_from_points(points,add_vertex_property = True,vertex_property_name = 'id',vertex_property = 'int'):
    G = gt.Graph(directed=False)
    for i in range(len(points)):
        G.add_vertex()
    if add_vertex_property:
        vprop = G.new_vertex_property(vertex_property)
        G.vertex_properties[vertex_property_name] = vprop
    return G    
def add_vertex_property(G,vertex_property_name = 'id',vertex_property = 'int'):
    vprop = G.new_vertex_property(vertex_property)
    G.vertex_properties[vertex_property_name] = vprop
    return G
def add_edge_property(G,edge_property_name = 'id',edge_property = 'int'):
    eprop = G.new_edge_property(edge_property)
    G.edge_properties[edge_property_name] = eprop
    return G