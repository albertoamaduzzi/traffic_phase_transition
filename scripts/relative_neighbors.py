from pyhull.delaunay import DelaunayTri
from scipy.spatial import Delaunay
from scipy.spatial import distance_matrix
import numpy as np
# FROM PROJECT
from vertices_functions import *
from output import *
## UPDATE DELAUNEY TRIANGULATION: NEXT STEP -> COMPUTE RNG (new added,old) attracting vertices      

def update_delauney_newly_attracting_vertices(planar_graph):
    '''
        The dalueney graph will contain:
            1) The newly added attracting vertices
            2) All the in graph vertices in the graph
        NOTE: 
            "IN GRAPH" vertices can be attracted by new inserted vertices -> create new roads.
            I MAY CHANGE IT IF IN THE CENTER THE ROADS ARE TOO DENSE -> 
            [planar_graph.graph.vp['x'][v] for v in planar_graph.graph.vertices() if planar_graph.is_newly_added(v) or (planar_graph.is_in_graph(v) and not planar_graph.is_active(v))]

    '''
    x = [planar_graph.graph.vp['x'][v] for v in planar_graph.graph.vertices() if is_newly_added(planar_graph,v) or (is_in_graph(planar_graph,v))]
    y = [planar_graph.graph.vp['y'][v] for v in planar_graph.graph.vertices() if is_newly_added(planar_graph,v) or (is_in_graph(planar_graph,v))]   
    idx_xy = [planar_graph.graph.vp['id'][v] for v in planar_graph.graph.vertices() if is_newly_added(planar_graph,v) or (is_in_graph(planar_graph,v))]  
    planar_graph.delauneyid2idx_new_vertices = {i:planar_graph.graph.vp['id'][planar_graph.graph.vertex(idx_xy[i])] for i in range(len(idx_xy))} 
    if len(x)>3:
        tri = Delaunay(np.array([x,y]).T)
        # Iterate over all triangles in the Delaunay triangulation
        for simplex in tri.simplices:
            for i, j in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]:
                    vertex_i = planar_graph.graph.vertex(planar_graph.delauneyid2idx_new_vertices[i])   
                    vertex_j_idx = planar_graph.delauneyid2idx_new_vertices[j]
                    vertex_j = planar_graph.graph.vertex(planar_graph.delauneyid2idx_new_vertices[j])   
                    vertex_i_idx = planar_graph.delauneyid2idx_new_vertices[i]
                    if not j in planar_graph.graph.vp['new_attracting_delauney_neighbors'][vertex_i]:
                        planar_graph.graph.vp['new_attracting_delauney_neighbors'][vertex_i].append(vertex_j_idx)
                    if not i in planar_graph.graph.vp['new_attracting_delauney_neighbors'][vertex_j]:
                        planar_graph.graph.vp['new_attracting_delauney_neighbors'][vertex_j].append(vertex_i_idx)
    else:
        print('Do not have enough points for Delauney triangulation')
        raise ValueError
                        
def update_delauney_old_attracting_vertices(planar_graph):
    '''
        Consider just "important" vertices that are 
            1) Still active
            2) Not newly added
        Consider just "not important" vertices:
            1) Still active
            2) Not newly added
        2) Compute the delauney triangulation for this subsets of points
        3) Save it on old_attracting_delauney_neighbor for each vertex
        NOTE: not in graph -> active  NOT active -> not in graph
    '''
    ## Filter vertices of the wanted graph
    x = [planar_graph.graph.vp['x'][v] for v in planar_graph.graph.vertices() if (not is_newly_added(planar_graph,v) and is_active(planar_graph,v)) or is_end_point(planar_graph,v)]
    y = [planar_graph.graph.vp['y'][v] for v in planar_graph.graph.vertices() if (not is_newly_added(planar_graph,v) and is_active(planar_graph,v)) or is_end_point(planar_graph,v)]         
    ## Take their indices
    idx_xy = [planar_graph.graph.vp['id'][v] for v in planar_graph.graph.vertices() if (planar_graph.graph.vp['is_active'][v] == True and planar_graph.graph.vp['newly_added_center'][v] == False) or (planar_graph.graph.vp['end_point'][v] == True)]
    ## Save the map of indices that I will use to retrieve the id of the vertex
    planar_graph.delauneyid2idx_old_attracting_vertices = {i:planar_graph.graph.vp['id'][planar_graph.graph.vertex(idx_xy[i])] for i in range(len(idx_xy))} 
    ## Compute delauney
    tri = Delaunay(np.array([x,y]).T)
    # Iterate over all triangles in the Delaunay triangulation
    planar_graph.edges_delauney_for_old_attracting_vertices = []
    for simplex in tri.simplices:
        for i, j in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]:
            ## Initialize the neighborhood
            vertex_i = planar_graph.graph.vertex(planar_graph.delauneyid2idx_old_attracting_vertices[i])   
            vertex_j = planar_graph.graph.vertex(planar_graph.delauneyid2idx_old_attracting_vertices[i])   
            vertex_j_idx = planar_graph.delauneyid2idx_old_attracting_vertices[j]
            vertex_i_idx = planar_graph.delauneyid2idx_old_attracting_vertices[j]
            if not j in planar_graph.graph.vp['old_attracting_delauney_neighbors'][vertex_i]:   
                planar_graph.graph.vp['old_attracting_delauney_neighbors'][vertex_i].append(vertex_j_idx)
            if not i in planar_graph.graph.vp['old_attracting_delauney_neighbors'][planar_graph.delauneyid2idx_old_attracting_vertices[j]]:
                planar_graph.graph.vp['old_attracting_delauney_neighbors'][vertex_j].append(vertex_i_idx)


## CALLING ALL UPDATES: NEXT STEP -> COMPUTE RNG (new added,old) attracting vertices
def update_lists_next_rng(planar_graph):
    '''
        Recompute:
            1) growing_graph
            2) growing_vertices -> are the vertices to search from the newly added centers
            3) end_points -> are the vertices for which the old added points must look for in their relative neighborhood
            4) attracting_vertices
            5) newly_added_attracting_vertices
            6) old_attracting_vertices
    '''
    # GROWING VERTICES
    # for newly added centers
    update_list_end_points(planar_graph)
    update_list_newly_added_attracting_vertices(planar_graph)
    update_list_important_vertices(planar_graph)
    update_list_old_attracting_vertices(planar_graph)
    update_list_in_graph_vertices(planar_graph)
    update_list_intersection_vertices(planar_graph)
    update_list_plausible_starting_point_of_roads(planar_graph)
    update_list_active_roads(planar_graph)
## COMPUTE RNG for NEWLY ADDED CENTERS: NEXT STEP -> EVOLVE STREET for (new,old) attracting vertices
def compute_rng_newly_added_centers(planar_graph):
    '''
        Description:
            1) update_delauney:
                1a) compute the Voronoi diagram for all vertices O(Nlog(N))
                1b) compute the delauney triangulation  O(N)
            2) For each attracting node (vi) and a node from the delauney triangulation (vj)
            3) Check for if for any node vk in the graph max(d_ix, d_xj) < d_ij
            4) If it is not the case, then vj is a relative neighbor of vi

    '''
    for vi in planar_graph.newly_added_attracting_vertices: # planar_graph.growing_graph.vertices()
        planar_graph.graph.vp['relative_neighbors'][vi] = []
        for vj in planar_graph.graph.vp['new_attracting_delauney_neighbors'][vi]: # planar_graph.growing_graph.vertices()
            try:
                d_ij = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][vi]][vj]
            except KeyError:
                d_ij = None
                continue
            for vx in planar_graph.graph.vertices(): 
                try:
                    d_ix = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][vi]][planar_graph.graph.vp['id'][vx]]
                except KeyError:
                    d_ix = None
                    continue
                try:
                    d_xj = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][planar_graph.graph.vp['id'][vx]]][vj]
                except KeyError:
                    d_xj = None
                    continue
            if max(d_ix, d_xj) < d_ij: break
            else:
                if d_ij != 0:
                    planar_graph.graph.vp['relative_neighbors'][vi].append(vj)
                    if vi not in planar_graph.graph.vp['relative_neighbors'][planar_graph.graph.vertex(vj)]:
                        planar_graph.graph.vp['relative_neighbors'][planar_graph.graph.vertex(vj)].append(planar_graph.graph.vp['id'][vi])
                        if planar_graph.graph.vp['end_point'][planar_graph.graph.vertex(vj)] == False and planar_graph.graph.vp['important_node'][planar_graph.graph.vertex(vj)] == False:
                            planar_graph.graph.vp['intersection'][planar_graph.graph.vertex(vj)] = True
                            # FIND THE STARTING VERTEX OF THE ROAD THIS POINTS BELONGS TO
                            # ADD ROAD 
    update_intersections(planar_graph)
    update_list_intersection_vertices(planar_graph)


def compute_rng_old_centers(planar_graph):
    '''
        Description:
            1) compute the delauney triangulation with just:
                1a) End points
                1b) old attracting vertices
            2) Compare them with just the nodes that are end points and are not in the road starting 
                from the vertex vi I am considering.
    '''
    for vi in planar_graph.old_attracting_vertices: # planar_graph.growing_graph.vertices()
        list_nodes_road_vi = get_list_nodes_in_roads_starting_from_v(planar_graph,vi)
        planar_graph.graph.vp['relative_neighbors'][vi] = []
        for vj in planar_graph.graph.vp['old_attracting_delauney_neighbors'][vi]: # planar_graph.growing_graph.vertices()
            try:
                d_ij = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][vi]][vj]
            except KeyError:
                d_ij = None
                continue
            for vx in planar_graph.end_points: 
                if vx not in list_nodes_road_vi:
                    try:
                        d_ix = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][vi]][planar_graph.graph.vp['id'][vx]]
                    except KeyError:
                        d_ix = None
                        continue
                    try:
                        d_xj = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][planar_graph.graph.vp['id'][vx]]][vj]
                    except KeyError:
                        d_xj = None
                        continue
                if max(d_ix, d_xj) < d_ij: break
            else:
                if d_ij != 0:
                    planar_graph.graph.vp['relative_neighbors'][vi].append(vj)
                    if vi not in planar_graph.graph.vp['relative_neighbors'][planar_graph.graph.vertex(vj)]:
                        planar_graph.graph.vp['relative_neighbors'][planar_graph.graph.vertex(vj)].append(planar_graph.graph.vp['id'][vi])
