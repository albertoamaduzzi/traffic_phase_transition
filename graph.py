import numpy as np
from graph-tools.all import *
from plotting_procedures import xy_plot
import pyhull



class graph_gt:
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
            if self.type_graph == 'newman':

        else:
            self.read_graph(geojson_path)

    def _generate_uniform_distribution_nodes_in_space(self,radius_city):  
        '''
            Generate a uniform distribution of points in a circle of radius radius_city
            Initialize postiions and distance matrix
        '''  
        from scipy.spatial import distance_matrix
        self.radius_city = radius_city
        nodes_r = np.random.random(self.number_nodes)
        nodes_costeta = 2*np.random.random(self.number_nodes) - 1        
        self.x = nodes_r*(self.radius_city)*nodes_costeta
        self.y = nodes_r*(self.radius_city)*np.sqrt(1-nodes_costeta**2)    
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

    def _edges_over_nodes(self):
        '''
            Modeling Urban Street Patterns Marc Barthélemy and Alessandro Flammini
            Expected E(N) = aplha*N  (alpha in [1,2] [tree-like,regular lattice])
        '''

    def _create_origin_destination(self,type_centricity):
        '''
            Description:
                This function creates the origin destination matrix for the monocentric case.
        '''
        
        return self.OD
    
    def _rewire_OD(self,rewire_probability):
        '''
        Polycentricity according to:
            Transport on Coupled Spatial Networks R. G. Morris and M. Barthelemy
        Description:
            For each origin-destination pair, we rewire the destination with a probability rewire_probability
            Starting from a monocentric case, we can obtain a polycentric case.
        '''
        assert self.OD!=None:
        self.rewire_probability = rewire_probability
        for vertex in self.graph.vertices():
            if np.random.random() < self.rewire_probability:
                pass
            else:
                 
                D1 = np.randint(0,self.number_nodes)
                while(D1 == vertex.index()):
                    D1 = np.randint(0,self.number_nodes)
                self.OD[vertex.index()][D1] = self.OD[vertex.index()][self.center_index]
                self.OD[vertex.index()][self.center_index] = 0
        else:
            print('No OD matrix to rewire')
            break

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
    def __init__(self,graph,typegraph):
        if typegraph == 'networkx':

        elif typegraph == 'pytorch':
            



