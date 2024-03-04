import os
#import graph_tool as gt
import numpy as np  
#from graph_tool.all import label_components
from scipy.spatial import distance_matrix

def ifnotexistsmkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
## GLOBAL

def check_distance_above_threshold(x,y,x1,y1,threshold):
    distance_matrix_ = distance_matrix(np.array([x,y]).T,np.array([x1,y1]).T)
    indices = np.where(distance_matrix_ > threshold)
    selected_x = [x[xi] for xi in range(len(x)) if xi in np.unique(indices[0])]
    selected_y = [y[yi] for yi in range(len(y)) if yi in np.unique(indices[0])]
    return selected_x,selected_y    

def generate_uniform_distribution_nodes_in_space_square(planar_graph,initial_points = True,debug=False):
    '''
        Citation:
            The spatial structure of networks: Michael T. Gastner and M. E. J. Newman
        Description:
            Generate a uniform distribution of points in a circle of radius radius_city
            Initialize postiions and distance matrix
            ALL in one step:        
    '''
    threshold = planar_graph.rate_growth
    side_city = planar_graph.side_city
    if debug:
        print('\t\tGenerate uniform distribution of nodes in space')
    number_nodes = planar_graph.initial_number_points      
    r = np.random.random(int(number_nodes))*side_city
    theta = np.random.random(int(number_nodes))*side_city    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    if int(number_nodes) == 1:
        x = np.array(x).reshape(1)
        y = np.array(y).reshape(1)  
    else:
        x,y = check_distance_above_threshold(x,y,x,y,threshold)  
    planar_graph.number_added_nodes += len(x)  
    return x,y

def generate_exponential_distribution_nodes_in_space_square(planar_graph,initial_points = True,debug=False):
    '''
        Citation:
            The spatial structure of networks: Michael T. Gastner and M. E. J. Newman
        Description:
            Generate a uniform distribution of points in a circle of radius radius_city
            Initialize postiions and distance matrix
            ALL in one step:        
    '''
    threshold = planar_graph.rate_growth
    side_city = planar_graph.side_city
    r0 = planar_graph.r0
    if initial_points:
        if debug:
            print('\t\tGenerate exponential distribution of nodes in space')
        number_nodes = planar_graph.initial_number_points      
        r = np.random.default_rng().exponential(scale = r0,size = int(number_nodes))*side_city
        theta = np.random.random(int(number_nodes))*side_city    
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        if int(number_nodes) == 1:
            x = np.array(x).reshape(1)
            y = np.array(y).reshape(1)  
        else:
            x,y = check_distance_above_threshold(x,y,x,y,threshold)  
        planar_graph.number_added_nodes += len(x)  
    elif not initial_points and planar_graph.iteration_count%planar_graph.tau_c==0:
        if debug:
            print('\t\tGenerate exponential distribution of nodes in space')
        number_nodes = planar_graph.number_nodes_per_tau_c
        r = np.random.default_rng().exponential(scale = r0,size = int(number_nodes))*side_city
        theta = np.random.random(int(number_nodes))*side_city    
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        x1 = planar_graph.graph.vp['x'].a
        y1 = planar_graph.graph.vp['y'].a
        x,y = check_distance_above_threshold(x,y,x1,y1,threshold)   
        planar_graph.number_added_nodes += len(x)
    else:
        raise ValueError('I cannot create centers if not in multiple times of tau_c')     
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

