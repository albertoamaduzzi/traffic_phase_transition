import os
import graph_tool as gt
import numpy as np  
from graph_tool.all import label_components
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
    
def generate_exponential_distribution_nodes_in_space_square(r0,number_nodes,side_city,planar_graph,initial_points = True,debug=False):
    '''
        Citation:
            The spatial structure of networks: Michael T. Gastner and M. E. J. Newman
        Description:
            Generate a uniform distribution of points in a circle of radius radius_city
            Initialize postiions and distance matrix
            ALL in one step:        
    '''
    threshold = planar_graph.rate_growth
    if initial_points:
        if debug:
            print('\t\tGenerate exponential distribution of nodes in space')
        r = np.random.default_rng().exponential(scale = r0,size = int(number_nodes))*side_city
        theta = np.random.random(int(number_nodes))*side_city    
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        if int(number_nodes) == 1:
            x = np.array(x).reshape(1)
            y = np.array(y).reshape(1)  
        else:
            x,y = check_distance_above_threshold(x,y,x,y,threshold)    
    else:
        if debug:
            print('\t\tGenerate exponential distribution of nodes in space')
        r = np.random.default_rng().exponential(scale = r0,size = int(number_nodes))*side_city
        theta = np.random.random(int(number_nodes))*side_city    
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        x1 = planar_graph.graph.vp['x'].a
        y1 = planar_graph.graph.vp['y'].a
        x,y = check_distance_above_threshold(x,y,x1,y1,threshold)        
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

