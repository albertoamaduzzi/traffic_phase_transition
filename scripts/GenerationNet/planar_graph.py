import sys
sys.path.append('/home/aamad/')
from termcolor import cprint
import time
from collections import defaultdict
import graph_tool as gt
import numpy as np
import os
import json
import keyboard
#import matplotlib.pyplot as plt
# FROM PROJECT
from grid import Grid
from vertices_functions import *
from relative_neighbors import *
from growth_functions import *
#from output import *    
from global_functions import *
from plots import *
from output import *
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
        self.tau_c = config['tau_c'] # 
        self.number_nodes_per_tau_c = config['number_nodes_per_tau_c'] 
        self.iteration_count = 0
        ## creation rates of nodes
        self.initial_number_points = config['initial_number_points']
        self.total_number_attraction_points = config['total_number_attraction_points'] # these are the auxins
        self.number_added_nodes = 0
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
        self.initial_graph()
        ## DEBUGGING
        self.debug_initial_points = config['debug_initial_points']
        self.debug_update_lists = config['debug_update_lists']
        self.debug_add_centers = config['debug_add_centers']
        self.debug_delauney_old = config['debug_delauney_old']
        self.debug_delauney_new = config['debug_delauney_new']
        self.debug_rng_old = config['debug_rng_old']
        self.debug_rng_new = config['debug_rng_new']
        self.debug_evolve_old = config['debug_evolve_old']
        self.debug_evolve_new = config['debug_evolve_new']
        self.debug_close_roads = config['debug_close_roads']
        self.debug_new2old = config['debug_new2old']
        self.debug_intersection = config['debug_intersection']
        self.iteration_plot = config['iteration_plot']
        self.debug_evolution_roads = config['debug_evolution_roads']
        self.debug_unique_evolution = config['debug_unique_evolution']
        self.debug_degenerate_evolution = config['debug_degenerate_evolution']
        self.debug_sum_evolution = config['debug_sum_evolution']
        self.debug_multiple_evolution = self.debug_degenerate_evolution or self.debug_sum_evolution

## ------------------------------------------- FIRST INITIALiZATION ---------------------------------------------

    def initial_graph(self):
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
        self.graph = gt.Graph(directed=True)
        ## NODES
        ## ID: int
        id_ = self.graph.new_vertex_property('int')
        self.graph.vp['id'] = id_
        ## IMPORTANT: bool (These are the points of interest)
        important_node = self.graph.new_vertex_property('bool')
        self.graph.vp['important_node'] = important_node
        is_active = self.graph.new_vertex_property('bool')
        self.graph.vp['is_active'] = is_active
        newly_added_center = self.graph.new_vertex_property('bool')
        self.graph.vp['newly_added_center'] = newly_added_center
        ## GROWING VERTICES 
        end_point = self.graph.new_vertex_property('bool')
        self.graph.vp['end_point'] = end_point
        is_in_graph = self.graph.new_vertex_property('bool')   
        self.graph.vp['is_in_graph'] = is_in_graph
        ## ATTRACTED: list attracted (Is the set of indices of vertices that are attracted by the node)
        list_nodes_attracted = self.graph.new_vertex_property('vector<int>')
        self.graph.vp['list_nodes_attracted'] = list_nodes_attracted
        ## INTERSECTION: bool
        ## role: for growing not important nodes, when addnewcenter: compute_rng, if the vertex in rng is attracted by it, the open intersection
        intersection = self.graph.new_vertex_property('bool')
        self.graph.vp['intersection'] = intersection
        attracted_by = self.graph.new_vertex_property('vector<int>')
        self.graph.vp['attracted_by'] = attracted_by
        ## RELATIVE NEIGHBORS: list (Is the set of indices of vertices that are relative neighbors of the attracting node)
        relative_neighbors = self.graph.new_vertex_property('vector<int>')
        self.graph.vp['relative_neighbors'] = relative_neighbors
        ## positions of the vertices
        x = self.graph.new_vertex_property('double')
        self.graph.vp['x'] = x
        y = self.graph.new_vertex_property('double')
        self.graph.vp['y'] = y
        pos = self.graph.new_vertex_property('vector<double>')
        self.graph.vp['pos'] = pos
        
        ## VORONOI diagram of the vertices 
        voronoi = self.graph.new_vertex_property('object')
        self.graph.vp['voronoi'] = voronoi
        new_attracting_delauney_neighbors = self.graph.new_vertex_property('vector<int>')
        self.graph.vp['new_attracting_delauney_neighbors'] = new_attracting_delauney_neighbors
        old_attracting_delauney_neighbors = self.graph.new_vertex_property('vector<int>')
        self.graph.vp['old_attracting_delauney_neighbors'] = old_attracting_delauney_neighbors
        ## Set of roads starting at the node
        roads_belonging_to = self.graph.new_vertex_property('vector<int>')
        self.graph.vp['roads_belonging_to'] = roads_belonging_to # used to new nodes to the right road.
        roads = self.graph.new_vertex_property('object') # is a list of roads
        self.graph.vp['roads'] = roads 
        self.graph.vp['roads_activated'] = self.graph.new_vertex_property('vector<int>') # is a list of roads
        ## EDGES
        growth_unit_vect = self.graph.new_edge_property('double')
        self.graph.ep['distance'] = growth_unit_vect
        growth_unit_vect = self.graph.new_edge_property('vector<double>')
        self.graph.ep['direction'] = growth_unit_vect
        real_edges = self.graph.new_edge_property('bool')
        self.graph.ep['real_edge'] = real_edges
        capacity = self.graph.new_edge_property('double')    
        self.graph.ep['capacity'] = capacity
        ## Animation
        state = self.graph.new_vertex_property('vector<double>')
        self.graph.vp['state'] = state
        

        return self.graph


    def set_policentricity(self):
        if 'degree_policentricity' in self.config.keys() and 'policentricity' in self.config.keys() and self.config['policentricity'] == True:
            self.policentricity = config['policentricity']
            self.number_centers = config['degree_policentricity']            
        else:
            self.policentricity = False
            self.number_centers = 1


##------------------------------------------- INITIALIZE BASE DIR -----------------------------------------------------------
    def initialize_base_dir(self):
        self.dir_proj = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
        self.base_dir = os.path.join(self.dir_proj,'output')
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
            cprint('node {} '.format(self.graph.vp['id'][node]),'in edges: ',len(in_edges),' out edges: ',len(out_edges))
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
        cprint(f"Graph saved to {filename}")


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
    t = 0
    ## Initialization parameters (with r0 being the characteristic distance of the probability distribution generation)
    t0 = time.time()
    cprint('0) INITIALIZATION: ','yellow')
    bg = planar_graph(config,r0)
    ## Initializes the properties of graph    
    t1  = time.time()
    cprint(str(t1-t0),'yellow')
    print_geometrical_info(bg)    
    ## Add initial centers and control they are in the bounding box 
    t0 = time.time()
    cprint('1) ADD CENTERS: ','yellow')
    add_centers2graph(bg,debug = bg.debug_initial_points)
    t1 = time.time()
    cprint(str(t1-t0),'yellow')
    t0 = time.time()
    cprint('2) UPDATE LISTS OF VERTICES: ','yellow')
    update_lists_next_rng(bg,debug = bg.debug_update_lists)
#    cprint('\t2a) Print all lists:')
#    print_all_lists(bg)
    t1 = time.time()
    cprint(str(t1-t0),'yellow')
    t0 = time.time()
    cprint('3) UPDATE DELAUNEY NEW CENTERS: ','yellow')
    update_delauney_newly_attracting_vertices(bg,debug=bg.debug_delauney_new)
    t1 = time.time()
    cprint(str(t1-t0),'yellow')
    t0 = time.time()
    cprint('4) COMPUTE RELATIVE NEIGHBORS NEWLY ADDED CENTERS: ','yellow')      
    compute_rng_newly_added_centers(bg,debug=bg.debug_rng_new)
    t1 = time.time()
    cprint(str(t1-t0),'yellow')      
    t0 = time.time()
    cprint('5) EVOLVE STREET FOR NEW CENTERS: ','yellow')
    evolve_street_newly_added_attractors(bg,debug=bg.debug_evolve_new)
    t1 = time.time()
    cprint(str(t1-t0),'yellow')
    cprint('\t5a) Update total length road:','yellow')
    bg.update_total_length_road()
    cprint('\t5b) Update count roads:','yellow')
    bg.update_count_roads()
    cprint('\t5c) Update time:','yellow')
    bg.update_time(t)
    ## UPDATING LISTS AFTER UPDAtING GRAPH
    cprint('6) UPDATING LISTS AFTER UPDATING GRAPH:','yellow')
    cprint('\t6a) New to old:','yellow')
    new2old(bg,debug=False)     
    cprint('\t6b) Update lists next rng:','yellow')
    update_lists_next_rng(bg,debug = bg.debug_update_lists)
#    cprint('\t6c) Print all lists:')
#    print_all_lists(bg)
    cprint('******************** END INITIALIZATION *************************','yellow')
    while(len(bg.list_active_roads)!=0):
        cprint('iteration: '+str(t),'light_blue','on_white')
        if t%bg.tau_c == 0 and t!=0 and bg.number_added_nodes < bg.total_number_attraction_points:
            cprint('xxxxxxxxxxxxx ADDING NEW CENTERS xxxxxxxxxxxxxxxxx','grey','on_green')
            # add nodes
            if bg.debug_add_centers:
                t0 = time.time()
                cprint('1) ADD CENTERS: ','grey','on_green')
                add_centers2graph(bg,debug = bg.debug_add_centers)
                t1 = time.time()
                cprint(str(t1-t0),'grey','on_green')
            else:
                add_centers2graph(bg)
            if bg.debug_update_lists:
                t0 = time.time()
                cprint('2) UPDATE LISTS OF VERTICES: ','grey','on_green')
                update_lists_next_rng(bg,debug = bg.debug_update_lists)
                t1 = time.time()
                cprint(str(t1-t0),'grey','on_green')
            else:
                update_lists_next_rng(bg)
            if bg.debug_delauney_new:
                t0 = time.time()
                cprint('3a) UPDATE DELAUNEY NEW CENTERS: ','grey','on_green')
                update_delauney_newly_attracting_vertices(bg,debug=bg.debug_delauney_new)
                t1 = time.time()
                cprint(str(t1-t0),'grey','on_green')
            else:
                update_delauney_newly_attracting_vertices(bg)
            if bg.debug_delauney_old:
                t0 = time.time()
                cprint('3b) UPDATE DELAUNEY OLD CENTERS: ','green')
                update_delauney_old_attracting_vertices(bg,debug=bg.debug_delauney_old)
                t1 = time.time()
                cprint(str(t1-t0),'green')
            else:
                update_delauney_old_attracting_vertices(bg)
            if bg.debug_rng_new:
                t0 = time.time()
                cprint('4s) COMPUTE RELATIVE NEIGHBORS NEWLY ADDED CENTERS: ','grey','on_green')      
                compute_rng_newly_added_centers(bg,debug=bg.debug_rng_new)
                t1 = time.time()
                cprint(str(t1-t0),'grey','on_green')      
            else:
                compute_rng_newly_added_centers(bg)
            if bg.debug_rng_old:
                t0 = time.time()
                cprint('4b) COMPUTE RELATIVE NEIGHBORS OLD ADDED CENTERS: ','green')      
                compute_rng_old_centers(bg,debug=bg.debug_rng_old)
                t1 = time.time()
                cprint(str(t1-t0),'green')      
            else:
                compute_rng_old_centers(bg,debug=bg.debug_rng_old)
            if bg.debug_evolve_new:
                t0 = time.time()
                cprint('5a) EVOLVE STREET FOR NEW CENTERS: ','grey','on_green')
                evolve_street_newly_added_attractors(bg,debug=bg.debug_evolve_new)
                t1 = time.time()
                cprint(str(t1-t0),'grey','on_green')
            else:
                evolve_street_newly_added_attractors(bg)
            if bg.debug_evolve_old:
                t0 = time.time()
                cprint('5b) EVOLVE STREET FOR OLD CENTERS: ','green')
                evolve_street_old_attractors(bg,debug=bg.debug_evolve_old)
                t1 = time.time()
                cprint(str(t1-t0),'green')
            else:
                evolve_street_old_attractors(bg,debug=bg.debug_evolve_old)
#            cprint('\t7c) Print all lists:')
#            print_all_lists(bg)
        elif (t%bg.tau_c != 0 and bg.starting_phase == False) or bg.number_added_nodes > bg.total_number_attraction_points:
            cprint('------------------- UPDATING OLD CENTERS -----------------------','green')
            update_lists_next_rng(bg)
            if bg.debug_delauney_old:
                t0 = time.time()
                cprint('3) UPDATE DELAUNEY OLD CENTERS: ','green')
                update_delauney_old_attracting_vertices(bg,debug=bg.debug_delauney_old)
                t1 = time.time()
                cprint(str(t1-t0),'green')
            else:
                update_delauney_old_attracting_vertices(bg)
            if bg.debug_rng_old:
                t0 = time.time()
                cprint('4) COMPUTE RELATIVE NEIGHBORS OLD ADDED CENTERS: ','green')      
                compute_rng_old_centers(bg,debug=bg.debug_rng_old)
                print('END POINTS')
                for v in bg.end_points:
                    print(bg.graph.vp['id'][v],' relative_neighbor: ',bg.graph.vp['relative_neighbors'][v])
                t1 = time.time()
                cprint(str(t1-t0),'green')      
            else:
                compute_rng_old_centers(bg)
            if bg.debug_evolve_old:
                t0 = time.time()
                cprint('5) EVOLVE STREET FOR OLD CENTERS: ','green')
                evolve_street_old_attractors(bg,debug=bg.debug_evolve_old)
                t1 = time.time()
                cprint(str(t1-t0),'green')
            else:
                evolve_street_old_attractors(bg)
        else:
            if t!=0:
                print('t: ',t)
                print('quotient tauc: ',t%bg.tau_c == 0)
                print('total n nodes: ',bg.total_number_attraction_points)
                print('added nodes: ',bg.number_added_nodes)
                pass
#                raise ValueError('This case must not have happened')
        if bg.starting_phase:
            bg.starting_phase = False
        if bg.debug_evolution_roads:
            cprint('\t5a) Update total length road:','cyan')
            bg.update_total_length_road()
            cprint('\t5b) Update count roads:','cyan')
            bg.update_count_roads()
            cprint('\t5c) Update time:','cyan')
            bg.update_time(t)
        else:
            bg.update_total_length_road()
            bg.update_count_roads()
            bg.update_time(t)
        if bg.debug_close_roads:
            t0 = time.time()
            cprint('6) CLOSING ROADS: ','cyan')
            close_roads(bg,bg.debug_close_roads)
            t1 = time.time()
        else:
            close_roads(bg)
        if bg.debug_new2old:
            cprint('7) UPDATING LISTS AFTER UPDATING GRAPH:','cyan')
            cprint('\t7a) New to old:','cyan')
            new2old(bg,debug=bg.debug_new2old)
        else:
            new2old(bg)
        if bg.debug_update_lists:
            cprint('\t7b) Update lists next rng:','cyan')
            update_lists_next_rng(bg,debug = bg.debug_update_lists)
        else:
            update_lists_next_rng(bg)
        for v in bg.active_vertices:
            print(bg.graph.vp['id'][v],' is active: ',bg.graph.vp['is_active'][v],' roads activated: ',bg.graph.vp['roads_activated'][v])
        print('ACTIVE ROADS')
        for r in bg.list_active_roads:
            print('road: ',r.id,' activated by vertex(s): ',r.activated_by)
        print('ACTIVE VERTICES')
        for v in bg.active_vertices:
            print('vertex: ',bg.graph.vp['id'][v],' attracts road(s): ',bg.graph.vp['roads_activated'][v],' in graph: ',bg.graph.vp['is_in_graph'][v])
        fig,ax = plt.subplots(1,1,figsize = (10,10))
        plot_number_roads_time(bg,ax)
        plt.close(fig)
        fig,ax = plt.subplots(1,1,figsize = (10,10))
        plot_total_length_roads_time(bg,ax)
        plt.close(fig)
        t+=1
        bg.iteration_count = t
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
    cprint('root: {}'.format(root),'black','on_green')
    config_dir = os.path.join('/home/aamad/Desktop/phd/berkeley/traffic_phase_transition','config')
    config_name = 'barthelemy_graph.json'#os.listdir(config_dir)[0]
    with open(os.path.join(config_dir,config_name),'r') as f:
        config = json.load(f)
    for r0 in list_r0:
        bg = build_planar_graph(config,r0)
#        plot_number_roads_time(bg,ax)
#        plot_total_length_roads_time(bg)