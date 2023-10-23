import numpy as np
from graph-tools.all import *
from plotting_procedures import xy_plot



class graph:
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
        self.availble_libraries = ['networkx','pytorch','graph-tools','real_data']
        self.list_type_graphs = ['lattice','newman','real_data']
        self.library2type_graph = {u:v for u,v in zip(self.availble_libraries,self.list_type_graph)}
        if reading_mode == False:
            self.type_graph = type_graph
            if library_used == 'networkx':
                self.graph = self.create_graph(type_graph,reading_mode)
            elif library_used == 'pytorch':
                self.graph = self.create_graph(type_graph,reading_mode)
            elif library_used == 'graph-tools':
                self.graph = self.create_graph(type_graph,reading_mode)
        else:
            self.type_graph = 'real_data'
            self.graph = self.create_graph(type_graph,reading_mode,geojson_path)               

    def create_graph(self,
                     reading_mode = False,
                     number_nodes = 100,
                     number_edges = 1000,
                     probability_connection = 0.5,
                     geojson_path = None):
        '''
            type_graph = 'lattice','newman'
            reading_mode = False, True (False if you want to create a graph from an ensamble of graph_type)
            number_nodes = 100, if reading_mode = False (number of nodes in the graph)
            number_edges = 1000, if reading_mode = False (number of edges in the graph)
            probability_connection = 0.5, if reading_mode = False (probability of connection between two nodes)
        '''
        self.nuber_nodes = number_nodes
        if reading_mode == False:
            if self.type_graph == 'lattice':
                if self.library_used == 'networkx':
                    import networkx as nx
                    self.graph = nx.watts_strogatz_graph(number_nodes,number_edges,probability_connection)
                elif self.library_used == 'pytorch':
                    import pytorch as pt
                    self.graph = pt.watts_strogatz_graph(number_nodes,number_edges,probability_connection)
                elif self.library_used == 'graph-tools':
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

    

    def _create_origin_destination(self,type_centricity):
        '''
            Description:
                This function creates the origin destination matrix for the monocentric case.
        '''
        
        return self.OD
    
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



    def _save_graph(self,graph_path):
        '''
            Description:
                This function saves the graph in a csv format (nodes.csv,edges.csv,origin_dest.csv).
        '''

        self.graph.save(graph_path)










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
            



