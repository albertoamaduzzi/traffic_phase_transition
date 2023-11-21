import os
import graph_tool as gt
import numpy as np  
from graph_tool.all import label_components

def ifnotexistsmkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
## GLOBAL

def initial_graph():
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
    g = gt.Graph(directed=True)
    ## NODES
    ## ID: int
    id_ = g.new_vertex_property('int')
    g.vp['id'] = id_
    ## IMPORTANT: bool (These are the points of interest)
    important_node = g.new_vertex_property('bool')
    g.vp['important_node'] = important_node
    is_active = g.new_vertex_property('bool')
    g.vp['is_active'] = is_active
    newly_added_center = g.new_vertex_property('bool')
    g.vp['newly_added_center'] = newly_added_center
    ## GROWING VERTICES 
    end_point = g.new_vertex_property('bool')
    g.vp['end_point'] = end_point
    is_in_graph = g.new_vertex_property('bool')   
    g.vp['is_in_graph'] = is_in_graph
    ## ATTRACTED: list attracted (Is the set of indices of vertices that are attracted by the node)
    list_nodes_attracted = g.new_vertex_property('vector<int>')
    g.vp['list_nodes_attracted'] = list_nodes_attracted
    ## INTERSECTION: bool
    ## role: for growing not important nodes, when addnewcenter: compute_rng, if the vertex in rng is attracted by it, the open intersection
    intersection = g.new_vertex_property('bool')
    g.vp['intersection'] = intersection
    attracted_by = g.new_vertex_property('vector<int>')
    g.vp['attracted_by'] = attracted_by
    ## RELATIVE NEIGHBORS: list (Is the set of indices of vertices that are relative neighbors of the attracting node)
    relative_neighbors = g.new_vertex_property('vector<int>')
    g.vp['relative_neighbors'] = relative_neighbors
    ## positions of the vertices
    x = g.new_vertex_property('double')
    g.vp['x'] = x
    y = g.new_vertex_property('double')
    g.vp['y'] = y
    pos = g.new_vertex_property('vector<double>')
    g.vp['pos'] = pos
    
    ## VORONOI diagram of the vertices 
    voronoi = g.new_vertex_property('object')
    g.vp['voronoi'] = voronoi
    new_attracting_delauney_neighbors = g.new_vertex_property('vector<int>')
    g.vp['new_attracting_delauney_neighbors'] = new_attracting_delauney_neighbors
    old_attracting_delauney_neighbors = g.new_vertex_property('vector<int>')
    g.vp['old_attracting_delauney_neighbors'] = old_attracting_delauney_neighbors
    ## Set of roads starting at the node
    roads_belonging_to = g.new_vertex_property('vector<int>')
    g.vp['roads_belonging_to'] = roads_belonging_to # used to new nodes to the right road.
    roads = g.new_vertex_property('object') # is a list of roads
    g.vp['roads'] = roads    
    ## EDGES
    growth_unit_vect = g.new_edge_property('double')
    g.ep['distance'] = growth_unit_vect
    growth_unit_vect = g.new_edge_property('vector<double>')
    g.ep['direction'] = growth_unit_vect
    real_edges = g.new_edge_property('bool')
    g.ep['real_edge'] = real_edges
    capacity = g.new_edge_property('double')    
    g.ep['capacity'] = capacity
    ## Animation
    state = g.new_vertex_property('vector<double>')
    g.vp['state'] = state

    return g
    
    
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

