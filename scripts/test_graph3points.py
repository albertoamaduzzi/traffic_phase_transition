from planar_graph import planar_graph
import sys
sys.path.append('/home/aamad/')

import time
from collections import defaultdict
import graph_tool as gt
import numpy as np
import os
import json
import logging
#import matplotlib.pyplot as plt
# FROM PROJECT
from grid import Grid
from vertices_functions import *
from relative_neighbors import *
from growth_functions import *
#from output import *    
from global_functions import *
from plots import *
def main(config,r0):
    t0 = time.time()
    bg = planar_graph(config,r0)
    ## Initializes the properties of graph    
    t1  = time.time()
    print('0) INITIALIZATION: ',t1-t0)
    ## Add initial centers and control they are in the bounding box 
    t0 = time.time()
    add_centers2graph(bg,bg.r0,bg.ratio_evolving2initial*bg.initial_number_points,bg.side_city)
    t1 = time.time()
    print('1) ADD CENTERS: ',t1-t0)
    t0 = time.time()
    update_lists_next_rng(bg)
    print_all_lists(bg)
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
    t = 0
    bg.update_total_length_road()
    bg.update_count_roads()
    bg.update_time(t)
    ## UPDATING LISTS AFTER UPDAtING GRAPH
    print('6) UPDATING LISTS AFTER UPDATING GRAPH:')
    new2old(bg)     
    update_lists_next_rng(bg)
    print_all_lists(bg)
    while(len(bg.list_active_roads)!=0): #bg.number_iterations
        print('iteration: ',t)
        t0 = time.time()
        update_lists_next_rng(bg)
        t1 = time.time()
        print('2) UPDATE LISTS OF VERTICES: ',t1-t0)
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
        ## 
        close_roads(bg)
        print('number of edges: ',bg.graph.num_edges())
        bg.update_total_length_road()
        bg.update_count_roads()
        bg.update_time(t)
        print('6) UPDATING LISTS AFTER UPDATING GRAPH:')
        new2old(bg)
        update_lists_next_rng(bg)
        print_all_lists(bg)
    if False or t==10:
        plot_evolving_graph(bg)
        plot_growing_roads(bg)
    if bg.starting_phase:
        bg.starting_phase = False
    t+=1
    if not os.path.exists(os.path.join(root,'graphs')):
        os.mkdir(os.path.join(root,'graphs'))
    bg.save_custom_graph(os.path.join(root,'graphs','graph_r0_{0}.gt'.format(round(r0,2))))
    return bg

if __name__=='__main__':
    seed = np.random.seed(42)
    list_r0 = [0.1]#np.linspace(1,10,100) # r0 takes values with a frequency of 100 meter from 0 to 10 km
    tuple = os.walk('.', topdown=True)
    root = tuple.__next__()[0]
    print('root: ',root)
    config_dir = os.path.join('/home/aamad/Desktop/phd/berkeley/traffic_phase_transition','config')
    config_name = os.listdir(config_dir)[0]
    with open(os.path.join(config_dir,config_name),'r') as f:
        config = json.load(f)
    number_nodes = 3
    for r0 in list_r0:
        bg = main(config,r0)

