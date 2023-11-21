
import time
from collections import defaultdict
import graph_tool as gt
from scipy.spatial import distance_matrix
import numpy as np
import os
import json
import matplotlib.pyplot as plt
# FROM PROJECT
from geometric_features import road
from grid import Grid
from edges_functions import *
from vertices_functions import *
from relative_neighbors import *
from growth_functions import *
#from output import *    
from global_functions import *
from vector_operations import normalize_vector,coordinate_difference,scale
from plots import *
'''
1) Generates the first centers [they are all of interest and they are all attracting and are all growing]
2) For each growing I compute the relative neighbor (if it is growing iself and not a center, then I give up, otherwise I continue)
'''


 ## BARTHELEMY GRAPH
class planar_graph:
    def __init__(self,config:dict,r0):
        self.config = config
        self.starting_phase = True
        ## Geometrical parameters
        self.side_city = config['side_city'] # in km
        self.ratio_growth2size_city = config['ratio_growth2size_city'] # 1000 segments  to go from one place to the other (pace of 10 meters) 
        self.rate_growth = self.ratio_growth2size_city*self.side_city
        self.r0 = r0
        ## dynamical parameters
        self.number_iterations = config['number_iterations'] #import multiprocessing as mp
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
        ## LISTS
        self.important_vertices = []
        self.is_in_graph_vertices = []
        self.end_points = []
        self.old_attracting_vertices = []
        self.newly_added_attracting_vertices = []
        self.intersection_vertices = []
        self.plausible_starting_road_vertices = []
        self.list_roads = []
        self.active_vertices = []
        ## DYNAMICAL MEASURES
        self.time = []
        self.length_total_roads = []
        self.count_roads = []

    def set_policentricity(self):
        if 'degree_policentricity' in self.config.keys() and 'policentricity' in self.config.keys() and self.config['policentricity'] == True:
            self.policentricity = config['policentricity']
            self.number_centers = config['degree_policentricity']            
        else:
            self.policentricity = False
            self.number_centers = 1

## ------------------------------------------- FIRST INITIALiZATION ---------------------------------------------
    def initialize_graph(self):
        self.graph = initial_graph()

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

    def update_total_length_road(self):
        total_length = 0
        for r in self.list_roads:
            total_length += r.length
        self.length_total_roads.append(total_length)

    def update_count_roads(self):
        self.count_roads.append(self.global_counting_roads)

    def update_time(self,t):
        self.time.append(t)






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

 
    # Bind the function above as an 'idle' callback.

## MAIN
def build_planar_graph(config,r0):
    '''
        Runs the creation of the graph.
        Steps:
            1) initialize graph
            2) add initial centers (every time I add a center I assign all the boolean properties to it)
            3) evaluate the centers (in network,active,end points) -> list_in_net_vertices,list_end_vertices, list_active_vertices
            4) compute the relative neighbor for old attracting vertices if there are any
            5) compute the relative neighbor for new attracting vertices if there are any
            6) evolve the street for old attracting vertices if there are any
                6a)
            7) evolve the street for new attracting vertices if there are any
                7a)

            NEXT: 
                1) Define: Number of people in each node
                2) Define: Capacity of a road
                3) Define: Velocity of road
                4) Define: Capacity of a node
                5) Define: Distribution of areas in the city
    '''

    ## Initialization parameters (with r0 being the characteristic distance of the probability distribution generation)
    t0 = time.time()
    bg = planar_graph(config,r0)
    ## Initializes the properties of graph    
    bg.initialize_graph()
    t1  = time.time()
    print('0) INITIALIZATION: ',t1-t0)
    ## Add initial centers and control they are in the bounding box 
    for t in range(100): #bg.number_iterations
        print('iteration: ',t)
        if t%bg.tau_c == 0:
            # add nodes
            print('*********************************************')
            t0 = time.time()
            add_centers2graph(bg,bg.r0,bg.ratio_evolving2initial*bg.initial_number_points,bg.side_city)
            t1 = time.time()
            print('1) ADD CENTERS: ',t1-t0)
            t0 = time.time()
            update_lists_next_rng(bg)
            t1 = time.time()
            print('2) UPDATE LISTS OF VERTICES: ',t1-t0)
            t0 = time.time()
            update_delauney_newly_attracting_vertices(bg)
            t1 = time.time()
            print('3) UPDATE DELAUNEY NEW CENTERS: ',t1-t0)
            t0 = time.time()
            compute_rng_newly_added_centers(bg)
            t1 = time.time()
            print('4) COMPUTE RELATIVE NEIGHBORS NEWLY ADDED CENTERS: ',t1-t0)      
            t0 = time.time()
            evolve_street_newly_added_attractors(bg)
            t1 = time.time()
            print('5) EVOLVE STREET FOR NEW CENTERS: ',t1-t0)
        if t%bg.tau_c != 0 and t!=0:
            t0 = time.time()
            update_lists_next_rng(bg)
            t1 = time.time()
            print('1) UPDATE LISTS OF VERTICES: ',t1-t0)
        if bg.starting_phase == False:
            t0 = time.time()
            update_delauney_old_attracting_vertices(bg)
            t1 = time.time()
            print('3) UPDATE DELAUNEY OLD CENTERS: ',t1-t0)
            t0 = time.time()
            compute_rng_old_centers(bg)
            t1 = time.time()
            print('4) COMPUTE RELATIVE NEIGHBORS OLD ADDED CENTERS: ',t1-t0)      
            t0 = time.time()
            evolve_street_old_attractors(bg)
            t1 = time.time()
            print('5) EVOLVE STREET FOR OLD CENTERS: ',t1-t0)
            print_all_lists(bg)
            ## ROAD SECTION
            update_attracted_by(bg)
            close_roads(bg)
            print('number of edges: ',bg.graph.num_edges())
            bg.update_total_length_road()
            bg.update_count_roads()
            bg.update_time(t)
            update_lists_next_rng(bg)
        if False or t==10:
            plot_evolving_graph(bg)
            plot_growing_roads(bg)
        if bg.starting_phase:
            bg.starting_phase = False

    if not os.path.exists(os.path.join(root,'graphs')):
        os.mkdir(os.path.join(root,'graphs'))
    bg.save_custom_graph(os.path.join(root,'graphs','graph_r0_{0}.gt'.format(round(r0,2))))
    return bg

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
        bg = build_planar_graph(config,r0)
        plot_number_roads_time(bg)
        plot_total_length_roads_time(bg)