from pyhull.delaunay import DelaunayTri
from scipy.spatial import Delaunay
from scipy.spatial import distance_matrix
import numpy as np
# FROM PROJECT
from vertices_functions import *
from output import *
## UPDATE DELAUNEY TRIANGULATION: NEXT STEP -> COMPUTE RNG (new added,old) attracting vertices      

def update_delauney_newly_attracting_vertices(planar_graph,debug = False):
    '''
        The dalueney graph will contain:
            1) The newly added attracting vertices
            2) All the in graph vertices in the graph
        NOTE: 
            "IN GRAPH" vertices can be attracted by new inserted vertices -> create new roads.
            I MAY CHANGE IT IF IN THE CENTER THE ROADS ARE TOO DENSE -> 
            [planar_graph.graph.vp['x'][v] for v in planar_graph.graph.vertices() if planar_graph.is_newly_added(v) or (planar_graph.is_in_graph(v) and not planar_graph.is_active(v))]

    '''
    ##NOTE: I take all the points as there may exist the case where a newly added is shielded by oldly added
    x = [planar_graph.graph.vp['x'][v] for v in planar_graph.graph.vertices()]# if is_newly_added(planar_graph,v) or (is_in_graph(planar_graph,v))]
    y = [planar_graph.graph.vp['y'][v] for v in planar_graph.graph.vertices()]# if is_newly_added(planar_graph,v) or (is_in_graph(planar_graph,v))]   
    idx_xy = [planar_graph.graph.vp['id'][v] for v in planar_graph.graph.vertices()]# if is_newly_added(planar_graph,v) or (is_in_graph(planar_graph,v))]  
    planar_graph.delauneyid2idx_new_vertices = {i:planar_graph.graph.vp['id'][planar_graph.graph.vertex(idx_xy[i])] for i in range(len(idx_xy))} 
    if len(x)==3:
        tri = Delaunay(np.array([x,y]).T)
        simplex = tri.simplices[0]
        for i, j in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]:
            vertex_i = planar_graph.graph.vertex(planar_graph.delauneyid2idx_new_vertices[i])   
            vertex_j_idx = planar_graph.delauneyid2idx_new_vertices[j]
            vertex_j = planar_graph.graph.vertex(planar_graph.delauneyid2idx_new_vertices[j])   
            vertex_i_idx = planar_graph.delauneyid2idx_new_vertices[i]
            if not j in planar_graph.graph.vp['new_attracting_delauney_neighbors'][vertex_i]:
                planar_graph.graph.vp['new_attracting_delauney_neighbors'][vertex_i].append(vertex_j_idx)
#                        print('vertex_i: ',vertex_i)
#                        print(planar_graph.graph.vp['new_attracting_delauney_neighbors'][vertex_i])
            if not i in planar_graph.graph.vp['new_attracting_delauney_neighbors'][vertex_j]:
                planar_graph.graph.vp['new_attracting_delauney_neighbors'][vertex_j].append(vertex_i_idx)
    elif len(x)>3:
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
#                        print('vertex_i: ',vertex_i)
#                        print(planar_graph.graph.vp['new_attracting_delauney_neighbors'][vertex_i])
                    if not i in planar_graph.graph.vp['new_attracting_delauney_neighbors'][vertex_j]:
                        planar_graph.graph.vp['new_attracting_delauney_neighbors'][vertex_j].append(vertex_i_idx)
#                        print('vertex_i: ',vertex_j)
#                        print(planar_graph.graph.vp['new_attracting_delauney_neighbors'][vertex_j])
    else:
        print('Do not have enough points for Delauney triangulation')
        raise ValueError
    if debug:
        print('\tCompute delauney for newly added vertices')
        print('\tconsidered subgraph:\n',idx_xy)                            
        print('\tx:\n',x)
        print('\ty:\n',y)
        print('\tidx_xy:\n',idx_xy)
        print('\tdictionary:\n',planar_graph.delauneyid2idx_new_vertices)
    
def update_delauney_old_attracting_vertices(planar_graph,debug = False):
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
    ##NOTE: I consider all end points and active vertices: (I may need to add that they are not in the graph)
    x = [planar_graph.graph.vp['x'][v] for v in planar_graph.graph.vertices() if (is_active(planar_graph,v)) or is_end_point(planar_graph,v)]#not is_newly_added(planar_graph,v) and is_active(planar_graph,v)
    y = [planar_graph.graph.vp['y'][v] for v in planar_graph.graph.vertices() if (is_active(planar_graph,v)) or is_end_point(planar_graph,v)]#not is_newly_added(planar_graph,v) and is_active(planar_graph,v)         
    ## Take their indices
    idx_xy = [planar_graph.graph.vp['id'][v] for v in planar_graph.graph.vertices() if (is_active(planar_graph,v)) or is_end_point(planar_graph,v)]#not is_newly_added(planar_graph,v) and is_active(planar_graph,v)
    ## Save the map of indices that I will use to retrieve the id of the vertex
    planar_graph.delauneyid2idx_old_attracting_vertices = {i:planar_graph.graph.vp['id'][planar_graph.graph.vertex(idx_xy[i])] for i in range(len(idx_xy))} 
    ## Compute delauney
    if len(x)==3:
        tri = Delaunay(np.array([x,y]).T)
        simplex = tri.simplices[0]
        for i, j in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]:
            vertex_i = planar_graph.graph.vertex(planar_graph.delauneyid2idx_old_attracting_vertices[i])   
            vertex_j = planar_graph.graph.vertex(planar_graph.delauneyid2idx_old_attracting_vertices[j])   
            vertex_j_idx = planar_graph.delauneyid2idx_old_attracting_vertices[j]
            vertex_i_idx = planar_graph.delauneyid2idx_old_attracting_vertices[i]
            if not j in planar_graph.graph.vp['old_attracting_delauney_neighbors'][vertex_i]:   
                planar_graph.graph.vp['old_attracting_delauney_neighbors'][vertex_i].append(vertex_j_idx)
            if not i in planar_graph.graph.vp['old_attracting_delauney_neighbors'][planar_graph.delauneyid2idx_old_attracting_vertices[j]]:
                planar_graph.graph.vp['old_attracting_delauney_neighbors'][vertex_j].append(vertex_i_idx)
    elif len(x)>3:
        tri = Delaunay(np.array([x,y]).T)
        # Iterate over all triangles in the Delaunay triangulation
        planar_graph.edges_delauney_for_old_attracting_vertices = []
        for simplex in tri.simplices:
            for i, j in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]:
                ## Initialize the neighborhood
                vertex_i = planar_graph.graph.vertex(planar_graph.delauneyid2idx_old_attracting_vertices[i])   
                vertex_j = planar_graph.graph.vertex(planar_graph.delauneyid2idx_old_attracting_vertices[j])   
                vertex_j_idx = planar_graph.delauneyid2idx_old_attracting_vertices[j]
                vertex_i_idx = planar_graph.delauneyid2idx_old_attracting_vertices[i]
                if not j in planar_graph.graph.vp['old_attracting_delauney_neighbors'][vertex_i]:   
                    planar_graph.graph.vp['old_attracting_delauney_neighbors'][vertex_i].append(vertex_j_idx)
#                    print('vertex_i: ',vertex_i)
#                    print(planar_graph.graph.vp['old_attracting_delauney_neighbors'][vertex_i])
                if not i in planar_graph.graph.vp['old_attracting_delauney_neighbors'][planar_graph.delauneyid2idx_old_attracting_vertices[j]]:
                    planar_graph.graph.vp['old_attracting_delauney_neighbors'][vertex_j].append(vertex_i_idx)
#                    print('vertex_j: ',vertex_j)
#                    print(planar_graph.graph.vp['old_attracting_delauney_neighbors'][vertex_j])
    if debug:
        print('\tCompute delauney for newly added vertices')
        print('\tconsidered subgraph:\n',idx_xy)                            
        print('\tx:\n',x)
        print('\ty:\n',y)
        print('\tidx_xy:\n',idx_xy)
        print('\tdictionary:\n',planar_graph.delauneyid2idx_old_attracting_vertices)
## CALLING ALL UPDATES: NEXT STEP -> COMPUTE RNG (new added,old) attracting vertices
def update_lists_next_rng(planar_graph,debug = False):
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
    if debug:
        print('\tUpdate list for next computation of rng')
    update_list_end_points(planar_graph,debug)
    update_list_newly_added_attracting_vertices(planar_graph,debug)
    update_list_important_vertices(planar_graph,debug)
    update_list_old_attracting_vertices(planar_graph,debug)
    update_list_in_graph_vertices(planar_graph,debug)
    update_list_intersection_vertices(planar_graph,debug)
    update_list_plausible_starting_point_of_roads(planar_graph,debug)
    update_list_roads(planar_graph,debug)
    update_list_active_roads(planar_graph,debug)
    update_list_active_vertices(planar_graph,debug)  
    update_list_vertices_starting_roads(planar_graph,debug) 

## COMPUTE RNG for NEWLY ADDED CENTERS: NEXT STEP -> EVOLVE STREET for (new,old) attracting vertices
def empty_relative_neighbors_vertices_in_graph_not_end_points(planar_graph): 
    '''
        Cleans the old relative neighbors for the points in the graph that are not end points. 
        Indeed if I have a new vertex that attracts a middle point, the middle point will update already calculated
        relative neighbors since it is updated from the call of the newly added not in graph centers.
        This function is not needed in the old update since I iterate on the end points, and the first thing that I do
        is to empty the relative neighbors of the end points.
        i.e. planar_graph.graph.vp['relative_neighbors'][vi] = []
    '''
    ingraphnotendpoints = [v for v in planar_graph.graph.vertices() if is_in_graph(planar_graph,v) and not is_end_point(planar_graph,v)]
    for v in ingraphnotendpoints:
        planar_graph.graph.vp['relative_neighbors'][v] = []


def compute_rng_newly_added_centers(planar_graph,debug = False):
    '''
        Description:
            1) update_delauney:
                1a) compute the Voronoi diagram for all vertices O(Nlog(N))
                1b) compute the delauney triangulation  O(N)
            2) For each attracting node (vi) and a node from the delauney triangulation (vj)
            3) Check for if for any node vk in the graph max(d_ix, d_xj) < d_ij
            4) If it is not the case, then vj is a relative neighbor of vi

    '''
    empty_relative_neighbors_vertices_in_graph_not_end_points(planar_graph)
    for vi in planar_graph.newly_added_attracting_vertices: # planar_graph.growing_graph.vertices()
#        print('RELATIVE NEIGHBORS: {}'.format(planar_graph.graph.vp['id'][vi]))
        planar_graph.graph.vp['relative_neighbors'][vi] = []
        for vj in planar_graph.graph.vp['new_attracting_delauney_neighbors'][vi]: # planar_graph.growing_graph.vertices()
            go2appending = True
#            print('vj:\t',vj)
            try:
                d_ij = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][vi]][vj]
            except KeyError:
                d_ij = None
                continue
            not2check = [vi,planar_graph.graph.vertex(vj)]
            for vx in planar_graph.graph.vertices(): 
                if vx not in not2check:
#                    print('vx:\t',vx)                
                    try:
                        d_ix = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][vi]][planar_graph.graph.vp['id'][vx]]                
                    except KeyError:
                        d_ix = None
                        continue
                    try:
                        d_xj = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][planar_graph.graph.vp['id'][vx]]][vj]
#                        print('d_ij: ',d_ij)
#                        print('d_ix: ',d_ix)                    
#                        print('d_xj: ',d_xj)              
                    except KeyError:
                        d_xj = None
                        continue
                    if max(d_ix, d_xj) < d_ij: 
#                        print(vj,' not relative neighbor')
                        go2appending = False
                        break
                    else:
                        pass
            if d_ij != 0 and go2appending:
                if vj not in planar_graph.graph.vp['relative_neighbors'][vi]:
                    planar_graph.graph.vp['relative_neighbors'][vi].append(vj)
    #                print('appended vj: ',vj)
                    if planar_graph.graph.vp['id'][vi] not in planar_graph.graph.vp['relative_neighbors'][planar_graph.graph.vertex(vj)]:
                        planar_graph.graph.vp['relative_neighbors'][planar_graph.graph.vertex(vj)].append(planar_graph.graph.vp['id'][vi])
                        if planar_graph.graph.vp['end_point'][planar_graph.graph.vertex(vj)] == False and planar_graph.graph.vp['important_node'][planar_graph.graph.vertex(vj)] == False:
                            planar_graph.graph.vp['intersection'][planar_graph.graph.vertex(vj)] = True
    if debug:
        print('\tCompute rng for newly added centers')

                            # FIND THE STARTING VERTEX OF THE ROAD THIS POINTS BELONGS TO
                        # ADD ROAD 
#    update_intersections(planar_graph)
#    update_list_intersection_vertices(planar_graph)

def new2old(planar_graph,debug=False):
    '''
        New vertices now become old
    '''
    for v in planar_graph.newly_added_attracting_vertices:
        planar_graph.graph.vp['newly_added_center'][v] = False
        planar_graph.old_attracting_vertices.append(v)
    planar_graph.list_nodes_road_vi = []
    if debug:
        print('\tNewly added vertices now are old')
        print('\t\tList nodes road vi: ',planar_graph.list_nodes_road_vi)

def compute_rng_old_centers(planar_graph,debug=False):
    '''
        Description:
            1) compute the delauney triangulation with just:
                1a) End points
                1b) old attracting vertices
            2) Compare them with just the nodes that are end points and are not in the road starting 
                from the vertex vi I am considering.
    '''
    if debug:
        print('RELATIVE NEIGHBORS OLD VERTICES: ')        

    for vi in planar_graph.end_points: # planar_graph.growing_graph.vertices()
        list_nodes_road_vi = get_list_nodes_in_roads_starting_from_v(planar_graph,vi,debug)
        planar_graph.graph.vp['relative_neighbors'][vi] = []
        if debug:
            print('List of nodes starting from {}: '.format(planar_graph.graph.vp['id'][vi]),list_nodes_road_vi)
        for vj in planar_graph.graph.vp['old_attracting_delauney_neighbors'][vi]: # planar_graph.growing_graph.vertices()
            go2append = True
#            print('vj:\t',vj)
            try:
                d_ij = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][vi]][vj]
            except KeyError:
                d_ij = None
                continue
            not2check = [vi,planar_graph.graph.vertex(vj)]
            for vx in planar_graph.old_attracting_vertices:#planar_graph.end_points: 
#                print('vx: ',vx)
                if vx not in list_nodes_road_vi:
                    if vx not in not2check:            
                        try:
                            d_ix = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][vi]][planar_graph.graph.vp['id'][vx]]
                        except KeyError:
                            d_ix = None
                            continue
                        try:
                            d_xj = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][planar_graph.graph.vp['id'][vx]]][vj]
#                            print('d_ij: ',d_ij)
#                            print('d_ix: ',d_ix)                    
#                            print('d_xj: ',d_xj)              
                        
                        except KeyError:
                            d_xj = None
                            continue
                        if max(d_ix, d_xj) < d_ij: 
                            go2append = False
#                            print(vj,' not relative neighbor')                        
                            break
                        else:
                            pass
            if d_ij != 0 and go2append:
#                print('appended vj: ',vj)   
                if vj not in planar_graph.graph.vp['relative_neighbors'][vi]:
                    planar_graph.graph.vp['relative_neighbors'][vi].append(vj)
                    if planar_graph.graph.vp['id'][vi] not in planar_graph.graph.vp['relative_neighbors'][planar_graph.graph.vertex(vj)]:
                        planar_graph.graph.vp['relative_neighbors'][planar_graph.graph.vertex(vj)].append(planar_graph.graph.vp['id'][vi])
    if debug:
        for v in planar_graph.end_points:
            print('\tRN End Point {}: '.format(planar_graph.graph.vp['id'][v]),planar_graph.graph.vp['relative_neighbors'][v])