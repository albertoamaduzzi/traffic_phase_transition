from planar_graph import planar_graph,generate_exponential_distribution_nodes_in_space_square,is_graph_connected
from pyhull.delaunay import DelaunayTri
from scipy.spatial import Delaunay
from scipy.spatial import distance_matrix


##----------------------------------------- ADDING POINTS, EDGES AND ROADS -------------------------------------------- <- SUBSTEPS OF EVOLVE STREETS                            
    ## STARTS WITH CONNECTED POINTS (DEPRECATED)


def add_initial_points2graph(planar_graph,r0,number_nodes,side_city):
    '''
        This functions constructs an initially connected graph -> I am leaving it, but it is deprecated
        NOTE: Once I add points I need to set new point to end_point, the previous point not anymore end_points,
        1) Check if important point is still active
        2) Add road
    '''
    x,y = generate_exponential_distribution_nodes_in_space_square(r0,number_nodes,side_city)
    planar_graph.initialize_distance_matrix(x,y)
    for point_idx in range(len(x)):
        planar_graph.graph.add_vertex()
        vertex = planar_graph.graph.vertex(planar_graph.graph.num_vertices()-1)
        id_ = planar_graph.graph.num_vertices()-1
        planar_graph.set_initialize_x_y_pos(planar_graph,vertex,x,y,point_idx)
        planar_graph.set_important_node(vertex,True)
        planar_graph.set_active_vertex(vertex,True)
        planar_graph.set_id(vertex,id_)
        planar_graph.set_empty_relative_neighbors(vertex)
        planar_graph.set_end_point(vertex,True)
        planar_graph.set_in_graph(vertex,True)
        planar_graph.set_is_intersection(vertex,False)
    for vertex in planar_graph.graph.vertices():
        for vertex1 in planar_graph.graph.vertices():
            if vertex != vertex1 and is_graph_connected(planar_graph.graph) == False:
                planar_graph.graph.add_edge(vertex,vertex1)
                edge = planar_graph.graph.edge(vertex,vertex1)
                planar_graph.set_length(edge)
                planar_graph.set_direction(edge)
                planar_graph.set_real_edge(planar_graph,edge,False)

## STARTS WITH DETACHED CENTERS
def add_centers2graph(planar_graph,r0,number_nodes,side_city):
    '''
        For each center generated:
            1) Add to graph
            2) Give it a name
            3) Initialize position
            4) Set it as important node (each important node attracts points in graph)
            5) Set it as active (It will deactivate just when all the attracted road reach it)

    '''
    planar_graph.print_geometrical_info()
    x,y = generate_exponential_distribution_nodes_in_space_square(r0,number_nodes,side_city)
#        x,y = planar_graph.city_box.contains_vector_points(np.array([x,y]).T)
    if planar_graph.distance_matrix_ is None:
        planar_graph.initialize_distance_matrix(x,y)
        for point_idx in range(len(x)):
            planar_graph.graph.add_vertex()
            vertex = planar_graph.graph.vertex(planar_graph.graph.num_vertices()-1)
            id_ = planar_graph.graph.num_vertices()-1
            planar_graph.set_id(vertex,id_)
            planar_graph.set_initialize_x_y_pos(planar_graph,vertex,x,y,point_idx)
            planar_graph.set_important_node(vertex,True)
            planar_graph.set_active_vertex(vertex,True)
            ## Will they be attracted? Yes As they are the first created centers
            planar_graph.set_in_graph(vertex,True)
            planar_graph.set_end_point(vertex,True)
            ## RELATIVE NEIGHBOR, ROADS starting from it.                
            planar_graph.set_empty_relative_neighbors(vertex)
            planar_graph.set_empty_road(vertex)
            ## Intersection
            planar_graph.set_is_intersection(vertex,False)
    else:
        for point_idx in range(len(x)):
            planar_graph.update_distance_matrix(np.array([planar_graph.graph.vp['x'].a,planar_graph.graph.vp['y'].a]).T,np.array([[x[point_idx],y[point_idx]]]))
            planar_graph.graph.add_vertex()
            vertex = planar_graph.graph.vertex(planar_graph.graph.num_vertices()-1)
            id_ = planar_graph.graph.num_vertices()-1
            planar_graph.set_id(vertex,id_)
            planar_graph.set_initialize_x_y_pos(planar_graph,vertex,x,y,point_idx)
            planar_graph.set_important_node(vertex,True)
            planar_graph.set_active_vertex(vertex,True)
            ## Will they be attracted? No
            planar_graph.set_in_graph(vertex,False)
            planar_graph.set_end_point(vertex,False)
            ## RELATIVE NEIGHBOR, ROADS starting from it.
            planar_graph.set_empty_relative_neighbors(vertex)
            planar_graph.set_empty_road(vertex)
            ## Intersection
            planar_graph.set_is_intersection(vertex,False)


def add_point2graph(planar_graph,source_vertex,dx,dy):
    '''
        Adds a point from the graph and initializes x,y with the vector dx,dy
        The added node is:
            1) Not important
            2) Not active
            3) End point
            4) With an empty set of relative neighbors
            5) In graph
            6) Added to a road
    '''
    x_new_node = [planar_graph.graph.vp['x'][source_vertex]+dx]
    y_new_node = [planar_graph.graph.vp['y'][source_vertex]+dy]
    ## INITIALIZE NEW VERTEX 
    planar_graph.graph.add_vertex()
    vertex = planar_graph.graph.vertex(planar_graph.graph.num_vertices()-1)
    id_ = planar_graph.graph.num_vertices()-1
    planar_graph.set_id(vertex,id_)
    planar_graph.set_initialize_x_y_pos(vertex,x_new_node,y_new_node,0)
    planar_graph.set_important_node(vertex,False)
    planar_graph.set_active_vertex(vertex,False)
    planar_graph.set_empty_relative_neighbors(vertex)
    planar_graph.set_end_point(vertex,True)
    planar_graph.set_in_graph(vertex,True)
    planar_graph.add_road(source_vertex,vertex)
    ## CHANGE INFO SOURCE VERTEX
    planar_graph.set_end_point(source_vertex,False)
    return planar_graph.graph.vertex(planar_graph.graph.num_vertices()-1)


##-------------------------------------- DISTANCE MATRIX ------------------------------------------

def initialize_distance_matrix(planar_graph,x,y):
    planar_graph.distance_matrix_ = distance_matrix(np.array([x,y]).T,np.array([x,y]).T) #distance_matrix(np.array([planar_graph.graph.vp['x'],planar_graph.graph.vp['y']]).T,np.array([planar_graph.graph.vp['x'],planar_graph.graph.vp['y']]).T)

def update_distance_matrix(planar_graph,old_points,new_point):
    '''
        Description:
            For each new point added in the graph, adds the distance of this point to all the other points.
            In this way I do have a reduction of calculation from N_nodes*N_nodes to N_nodes
        new_point is a vector of dimension (1,2) -> the bottleneck of this function resides in np.vstack as I am adding the 0 of the 
        concatenation of the column and the row one after the other
    '''
#        print('expected distance matrix shape {0},{1}'.format(len(old_points),len(old_points)),np.shape(planar_graph.distance_matrix_))        
    dm_row = distance_matrix(new_point,old_points)
#        print('expected row matrix shape {0},{1}'.format(1,len(old_points)),np.shape(dm_row))
    dm_col = distance_matrix(old_points,new_point) # dimension (1,starting_points)
    dm_col = np.vstack((dm_col,[0]))        
#        print('expected col matrix shape {0},{1}'.format(len(old_points)+1,1),np.shape(dm_col))
    dm_conc = np.concatenate((planar_graph.distance_matrix_,dm_row),axis = 0) # dimension (starting_points+1,starting_popints)
#        print('expected conc matrix shape {0},{1}'.format(len(old_points)+1,len(old_points)),np.shape(dm_conc))
    planar_graph.distance_matrix_ = np.concatenate((dm_conc,dm_col),axis = 1)





## UPDATE 

def close_road(planar_graph):
    '''
        Attach the edge if the growing points are close enough to their relative neighbors,
        in this way the relative neighbor becomes a growing point as well
    '''
    already_existing_edges = [[e.source(),e.target()] for e in planar_graph.graph.edges()]
    for v in planar_graph.important_vertices:
        for u_idx in planar_graph.graph.vp['relative_neighbors'][v]:
                if planar_graph.distance_matrix_[planar_graph.graph.vp['id'][v],u_idx] < planar_graph.rate_growth and [v,planar_graph.graph.vertex(u_idx)] not in already_existing_edges:
                    planar_graph.graph.add_edge(planar_graph.graph.vertex(u_idx),v)
                    if planar_graph.graph.vp['growing'][v] == False:
                        planar_graph.graph.vp['growing'][v] = True
                        print('closing road between: ',planar_graph.graph.vp['id'][v],planar_graph.graph.vp['id'][u_idx])
                    else:
                        pass


##--------------------------------------------- UPDATES ---------------------------------------------------- NEXT STEP -> COMPUTE DELAUNEY TRIANGULATION (new added,old) attracting vertices    
def update_list_important_vertices(planar_graph):
    planar_graph.important_vertices = [v for v in planar_graph.graph.vertices() if planar_graph.graph.vp['important_node'][v] == True]
def update_list_in_graph_vertices(planar_graph):
    planar_graph.is_in_graph_vertices = [v for v in planar_graph.graph.vertices() if planar_graph.graph.vp['is_in_graph'][v] == True]
def update_list_end_points(planar_graph):
    '''
        List of points to consider when deciding old attracting vertices relative neighbor
    '''
    planar_graph.end_points = [v for v in planar_graph.graph.vertices() if planar_graph.graph.vp['end_point'][v] == True]

## ACTIVE VERTICES  
def update_list_old_attracting_vertices(planar_graph):
    '''
        These are the vertices that attract just end points. 
            -Delauney triangulation just among these and the end points in graph
    '''
    planar_graph.old_attracting_vertices = [v for v in planar_graph.graph.vertices() if planar_graph.is_active(v) and not planar_graph.is_newly_added(v)]

def update_list_newly_added_attracting_vertices(planar_graph):
    planar_graph.newly_added_attracting_vertices = [v for v in planar_graph.graph.vertices() if planar_graph.is_active(v) and planar_graph.is_newly_added(v)]

def update_list_intersection_vertices(planar_graph):
    planar_graph.intersection_vertices = [v for v in planar_graph.graph.vertices() if planar_graph.graph.vp['intersection'][v] == True]

def update_list_plausible_starting_point_of_roads(planar_graph):
    planar_graph.plausible_starting_road_vertices = [v for v in planar_graph.graph.vertices() if planar_graph.is_important_node(v) or planar_graph.is_intersection(v)]

def update_list_active_vertices(planar_graph):
    '''
        For each important vertex check if it 
    '''
    planar_graph.active_vertices = []
    for attracting_vertex in planar_graph.important_vertices:
        for starting_road_vertex in planar_graph.important_vertices:
            for r in planar_graph.graph.vp['roads'][starting_road_vertex]:
                if r.is_closed == False and attracting_vertex in r.activated_by:
                    planar_graph.graph.vp['is_active'][attracting_vertex] = True
                    break

# This piece must be inserted to take update the roads, as I need each point that is evolving to have an attraction set
def update_attracted_by(planar_graph):
    '''
        For each vertex update the attraction:
            1) The end_points are attracted by the relative neighbors that are not in the graph
            2) The growing points are attracted by new added vertices (stop)
    '''
    print('updating attracted by: ')
    list_considered_vertices = []
    for v in planar_graph.graph.vertices():
        print('considering vertex: ',planar_graph.graph.vp['id'][v])
        planar_graph.graph.vp['attracted_by'][v] = []
        if planar_graph.is_end_point(v):
            print('is end point')
            list_considered_vertices.append(v)
            list_rn = planar_graph.graph.vp['relative_neighbors'][v]
            print('relative neighbors: ',list_rn)
            starting_vertex,local_idx,found = planar_graph.find_road_vertex(v)
            if found:
                for relative_neighbor in list_rn:
                    if relative_neighbor not in planar_graph.graph.vp['roads'][starting_vertex][local_idx].list_nodes:
                        planar_graph.graph.vp['attracted_by'][v].append(relative_neighbor)
            else:
                pass
        elif planar_graph.is_growing_and_not_attracting(v):
            list_considered_vertices.append(v)
            list_rn = planar_graph.graph.vp['relative_neighbors'][v]
            starting_vertex,local_idx,found = planar_graph.find_road_vertex(v)
            if found:
                for relative_neighbor in list_rn:
                    if relative_neighbor not in planar_graph.graph.vp['roads'][starting_vertex][local_idx].list_nodes:
                        planar_graph.graph.vp['attracted_by'][v].append(relative_neighbor)
            else:
                pass
        elif planar_graph.is_attracting_and_not_growing(v):
            list_considered_vertices.append(v)
            planar_graph.graph.vp['attracted_by'][v] = []
            pass
        elif planar_graph.is_growing_and_attracting(v):
            list_considered_vertices.append(v)
            planar_graph.graph.vp['attracted_by'][v] = []
            pass
    not_considered = [v for v in planar_graph.graph.vertices() if v not in list_considered_vertices]
    planar_graph.print_not_considered_vertices(not_considered)


def update_intersections(planar_graph):
    '''
        Description:
            newly attracted vertices can attract points that are growing:
            2 cases:
                1) Node attracted: growing, not end_point
                2) Node attracted: growing, end_point
        In case 1 the node gets attracted and becomes intersection
        NOTE:
            This function is called after:
                1) adding new attracting vertices
                2) compute_newly_added_centers_rng
    '''
    ## For each newly added attracting vertex
    for n_attracting_vertex in planar_graph.newly_added_attracting_vertices:
        for attracted_vertex_idx in planar_graph.graph.vp['relative_neighbors'][n_attracting_vertex]:
            attracted_vertex = planar_graph.graph.vertex(attracted_vertex_idx)
            ## If the attracted vertex (belonging to the relative neighbor) is not an end point, and is not an important node (then It can be just growing)
            if not planar_graph.is_end_point(attracted_vertex) and not planar_graph.is_important_node(attracted_vertex):
                starting_vertex,local_idx,found = planar_graph.find_road_vertex(attracted_vertex)
                if found:
                    planar_graph.global_counting_roads += 1
                    planar_graph.graph.vp['roads'][starting_vertex].append(road(starting_vertex,planar_graph.global_counting_roads,n_attracting_vertex))
                    planar_graph.graph.vp['roads'][starting_vertex][-1].copy_road_specifics(planar_graph,planar_graph.graph.vp['roads'][starting_vertex][local_idx])
                else:
                    pass

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
    x = [planar_graph.graph.vp['x'][v] for v in planar_graph.graph.vertices() if planar_graph.is_newly_added(v) or (planar_graph.is_in_graph(v))]
    y = [planar_graph.graph.vp['y'][v] for v in planar_graph.graph.vertices() if planar_graph.is_newly_added(v) or (planar_graph.is_in_graph(v))]   
    idx_xy = [planar_graph.graph.vp['id'][v] for v in planar_graph.graph.vertices() if planar_graph.is_newly_added(v) or (planar_graph.is_in_graph(v))]  
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
    x = [planar_graph.graph.vp['x'][v] for v in planar_graph.graph.vertices() if (not planar_graph.is_newly_added(v) and planar_graph.is_active(v)) or planar_graph.is_end_point(v)]
    y = [planar_graph.graph.vp['y'][v] for v in planar_graph.graph.vertices() if (not planar_graph.is_newly_added(v) and planar_graph.is_active(v)) or planar_graph.is_end_point(v)]         
    ## Take their indices
    idx_xy = [planar_graph.graph.vp['id'][v] for v in planar_graph.graph.vertices() if (planar_graph.graph.vp['attracting'][v] == True and planar_graph.graph.vp['newly_added_center'][v] == False) or (planar_graph.graph.vp['end_point'][v] == True)]
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
    planar_graph.update_list_end_points()
    planar_graph.update_list_newly_added_attracting_vertices()
    planar_graph.update_list_important_vertices()
    planar_graph.update_list_old_attracting_vertices()
    planar_graph.update_list_in_graph_vertices()
    planar_graph.update_list_intersection_vertices()
    planar_graph.update_list_plausible_starting_point_of_roads()

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
    planar_graph.update_lists_next_rng()
    planar_graph.update_delauney_newly_attracting_vertices()
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
    planar_graph.update_intersections()
    planar_graph.update_list_intersection_vertices()


def compute_rng_old_centers(planar_graph):
    '''
        Description:
            1) compute the delauney triangulation with just:
                1a) End points
                1b) old attracting vertices
            2) Compare them with just the nodes that are end points and are not in the road starting 
                from the vertex vi I am considering.
    '''
    planar_graph.update_lists_next_rng()
    planar_graph.update_delauney_old_attracting_vertices()
    for vi in planar_graph.old_attracting_vertices: # planar_graph.growing_graph.vertices()
        list_nodes_road_vi = planar_graph.get_list_nodes_in_roads_starting_from_v(vi)
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
