import os
import graph_tool as gt
import numpy as np  
from graph_tool.all import label_components

def ifnotexistsmkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
## GLOBAL

    
    
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
    if int(number_nodes) == 1:
        x = np.array(x).reshape(1)
        y = np.array(y).reshape(1)        
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

