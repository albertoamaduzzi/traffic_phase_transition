import geopandas as gpd
import networkx as nx
import graph-tools as gt
'''
In Remy model, what is as an assumption, is the fact that W(j) is equivalent to the energy of an atom, where j is the level. And the random variables 
are in one case positions an  
'''

class city:
    '''
        City is the object that must contain all different perspectives and so scales at which we look.

    '''
    def __init__(self,name):
        self.name = name
    def _set_population(self,population):
        self.population = population
    def _get_population(self):
        return self.population 
    def _generate_graph(self):
        '''
            This function will generate the graph of the city.
        '''
        
        self.adj_graph_weighted_time : np.array 
        self.adj_graph_weighted_distance : np.array
        self.node_gdf2node_adj : dict # this is done to access array from polies from geopandas object.
        self.critical_road = [] # the bigger the number the best lower tau.
        self.dict_distribution = {
            'gauss':,
            'levy':,
            'uniform':,
            'maxwell':
            'lognorm':,
            'weibull':,
            'poisson':,
            'bernoulli':
        }
        self.dict_default_parameters_graph_generator = {
            'erdos-renyi':{
                'n':100,
                'p':0.1
            },
            'dense':{
                'n':100,
                'm':100
            },
            'small-world':{
                'n':100,
                'k':10,
                'p':0.1
            },
            'connected-small-world':{
                'n':100,
                'k':10,
                'p':0.1
            }
        }
        self.dict_graph_generator = {
            'erdos-renyi':nx.fast_gnp_random_graph(self.dict_default_parameters_graph_generator['erdos-renyi'].values()), # n nodes, p probability link [,seed]
            'dense':nx.dense_gnm_random_graph(self.dict_default_parameters_graph_generator['dense'].values()), # n nodes, m edges [,seed]
            'small-world': nx.newman_watts_strogatz_graph(self.dict_default_parameters_graph_generator['small-world'].values()),# n nodes, k edges, p probabilty [,seed]
            'connected-small-world':nx.connected_watts_strogatz_graph(self.dict_default_parameters_graph_generator['connected-small-world'].values())# n nodes, k edges, p probabilty [,seed]
        }

# GET GRAPH
    def create_random_graph(self,mode = 'random',randomicity = 'erdos-renyi'):
        '''
            This functions create the graph to work on.
                1) creates random (default erdos-renyi)
                2) load from file
        '''
        self.mode = mode
        self.randomicity_graph = randomicity
        self.G = self.dict_graph_generator[randomicity]()

    def graph_from_geojson(self,filename,node_col_front,node_col_tail,edge_col,prop_edge_col):
        '''
            Input:
                filename: string -> name.geojson
                name of node column front: string
                name of node column tail: string
                name of edge column
                name of property column
            Output:
                graph-tool graph
        '''
        if 'geojson' not in filename:
            G = load_graph_from_csv(filename,eprop_types = ('int','string'),(edge_col,prop_edge_col))
        else:
            gdf = gpd.read_file(filename)
            self.G = Graph()
            self.G.add_edge_list(gdf[[node_col_front,node_col_tail]].values)
            for e in self.G.edges():
                e.new_ep({'id':gdf[gdf[node_col_front] == e.source()][gdf[node_col_tail] == e.target()][edge_col].values[0],
                        'length':gdf[gdf[node_col_front] == e.source()][gdf[node_col_tail] == e.target()][prop_edge_col].values[0]})
        return self.G

    def edge2localid(self):
        '''
            This function will give a local id to the edges.
        '''
        for i,edge in enumerate(self.city_graph.edges):
            self.city_graph.edge.data['local_id'] = i

        pass
        

    def _get_critical_roads():
        '''
            This function will return the critical roads that are responsible for the congestion.
        '''
        for edge in self.city_graph.edges:
            if edge.critical:
                self.critical_roads.append(edge.data['local_id'])
        pass

    def _load_nodes(self,node_load_distribution):
        self.dict_distribution[node_load_distribution]()
            

    def _compute_distance_matrix(self):
        if self.percolated == False:
            distance_matrix = nx.floyd_warshall_numpy(self.G,weight='lenght')
            dir = ifnotexistsmkdir(os.path.join(self.base,'matrices'))
            distance_matrix.save(os.path.join(dir,'distance_matrix'))
        else:
            nodelist = self.G
            distance_matrix = nx.floyd_warshall_numpy(self.G,weight='lenght')
    def _get_betweennes_centrality(self):

    def _compute_crititcal_values(self):
        '''
            This function will compute the critical values for the percolation.
        '''
        q = np.linspace(0,1,100)

        for qu in q:
            for na,nb,dat in self.G.edges(data=True):
                if dat.get(fract_velocity) > qu:
                    dat.critical = True
                else:
                    dat.critical = False


        pass

    def _get_max_velocity_edges(self):
        '''
            This function will compute the velocity of the edges.
        '''
        pass    

    def _compute_fract_velocity(self):
        '''
            This function will compute the fractional velocity of the edges. v_fract = v_ij/v_max_ij
        '''

        pass




class TASEP_single_road:

    '''
        Control parameters are those from Havlin: (for connected components)
        road: edge of the graph
    '''

    def __init__(self,road,initiation_rate,exit_rate,hop_right,hop_left):
        self.id_road = road['id_local']
        self.length = road['length']
        self.initiation_rate = initiation_rate
        self.exit_rate = exit_rate
        self.hop_right = hop_right
        self.hop_left = hop_left
        self.control_parameters = {
            'q': 0,
            'R/R_c': 0,
        }