import numpy as np
from scipy.spatial import Delaunay
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
        if debug:
            print('RELATIVE NEIGHBORS: {}'.format(planar_graph.graph.vp['id'][vi]))
        planar_graph.graph.vp['relative_neighbors'][vi] = []
        for vj in planar_graph.graph.vp['new_attracting_delauney_neighbors'][vi]: # planar_graph.growing_graph.vertices()
            go2appending = True
#            if debug:
#                print('vj:\t',vj)
            try:
                d_ij = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][vi]][vj]
            except KeyError:
                d_ij = None
                continue
            not2check = [vi,planar_graph.graph.vertex(vj)]
            for vx in planar_graph.graph.vertices(): 
                if vx not in not2check:
#                    if debug:
#                        print('vx:\t',vx)                
                    try:
                        d_ix = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][vi]][planar_graph.graph.vp['id'][vx]]                
                    except KeyError:
                        d_ix = None
                        continue
                    try:
                        d_xj = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][planar_graph.graph.vp['id'][vx]]][vj]
#                        if debug:
#                            print('d_ij: ',d_ij)
#                            print('d_ix: ',d_ix)                    
#                            print('d_xj: ',d_xj)              
                    except KeyError:
                        d_xj = None
                        continue
                    if max(d_ix, d_xj) < d_ij: 
#                        if debug:
#                            print(vj,' not relative neighbor')
                        go2appending = False
                        break
                    else:
                        pass
            if d_ij != 0 and go2appending:
                if vj not in planar_graph.graph.vp['relative_neighbors'][vi]:
                    planar_graph.graph.vp['relative_neighbors'][vi].append(vj)
#                   if debug:
#                        print('appended vj: ',vj)
                    if planar_graph.graph.vp['id'][vi] not in planar_graph.graph.vp['relative_neighbors'][planar_graph.graph.vertex(vj)]:
                        planar_graph.graph.vp['relative_neighbors'][planar_graph.graph.vertex(vj)].append(planar_graph.graph.vp['id'][vi])
                        if planar_graph.graph.vp['end_point'][planar_graph.graph.vertex(vj)] == False and planar_graph.graph.vp['important_node'][planar_graph.graph.vertex(vj)] == False:
                            planar_graph.graph.vp['intersection'][planar_graph.graph.vertex(vj)] = True
    if debug:
        print('\tCompute rng for newly added centers')
