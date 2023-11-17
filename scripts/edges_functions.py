from planar_graph import planar_graph
from geometric_features import road

####-------------------------------------------- SET FUNCTIONS --------------------------------------------####
def set_length(planar_graph,edge):
    planar_graph.graph.ep['length'][edge] = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][edge.source()],planar_graph.graph.vp['id'][edge.target()]]

def set_direction(planar_graph,edge):
    planar_graph.graph.ep['direction'][edge] = planar_graph.graph.vp['pos'][edge.target()].a - planar_graph.graph.vp['pos'][edge.source()].a

def set_real_edge(planar_graph,edge,boolean):
    planar_graph.graph.ep['real_edge'][edge] = boolean


##---------------------------------------- ROAD OPERATIONS ---------------------------------------------

def add_edge2graph(planar_graph,source_vertex,target_vertex):
    '''
        source_idx: Vertex
        target_idx: Vertex
        Description:
            1) Add the edge
            2) Add the distance
            3) Add the direction
            4) Add the real_edge property
            5) Control that the source vertex is an important vertex:
                5a) Yes: Means that a new road is starting
                5b) No: Means that the road is already started and I need to find the road where the source vertex belongs to
                    5b1) I check if the source vertex is in the road and add the target vertex to it 
    '''
    source_idx = planar_graph.graph.vp['id'][source_vertex]
    target_idx = planar_graph.graph.vp['id'][target_vertex]
    planar_graph.graph.add_edge(source_idx,target_idx)
    planar_graph.graph.ep['distance'][planar_graph.graph.edge(source_idx,target_idx)] = planar_graph.distance_matrix_[source_idx,target_idx]
    planar_graph.graph.ep['direction'][planar_graph.graph.edge(source_idx,target_idx)] = planar_graph.graph.vp['pos'][target_vertex].a - planar_graph.graph.vp['pos'][source_vertex].a
    planar_graph.graph.ep['real_edge'][planar_graph.graph.edge(source_idx,target_idx)] = False
    if planar_graph.graph.vp['important_node'][source_vertex] == True:
        planar_graph.global_counting_roads += 1
        new_road = road(source_vertex,planar_graph.global_counting_roads)            
        planar_graph.graph.vp['roads'][source_vertex].append(new_road)
        distance_ = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][source_vertex],planar_graph.graph.vp['id'][target_vertex]]
        planar_graph.graph.vp['roads'][source_vertex][-1].add_node_in_road(source_vertex,target_vertex,distance_)
    else:
        ## Check that for each vertex that has got a road
        ## TODO: Add variable: starting road
        for initial_node in planar_graph.important_vertices:
            local_idx_road = 0
            for r in planar_graph.graph.vp['roads'][initial_node]:
                _,found = r.in_road(source_vertex)
                local_idx_road += 1
            if found:
                planar_graph.graph.vp['roads'][initial_node][local_idx_road].append(target_vertex)
                break

#TODO: Fix the generation of intersections, when a new center is added, and generates a new road, the point when the road starts, is 
#  Inersection, new kind of NODE, this node, is the beginning of a new road.
# I need a new variable in road() -> type_starting_point: ['important_node','intersection']] 


def create_road(planar_graph,source_vertex,activation_vertices):
    planar_graph.graph.vp['roads'][source_vertex].append(road(source_vertex,planar_graph.global_counting_roads,activation_vertices))
    planar_graph.global_counting_roads += 1

def add_road(planar_graph,source_vertex,vertex):
    '''
        Adds the vertex to the road
    '''
    if planar_graph.is_in_graph(source_vertex):
        if planar_graph.is_important_node(source_vertex):
            planar_graph.create_road(source_vertex,vertex)
        else:
            starting_vertex_road,local_idx_road,found = planar_graph.find_road_vertex(vertex)
            if found:
                planar_graph.graph.vp['roads'][starting_vertex_road][local_idx_road].append(vertex)
            else:
                pass
    else:
        print(planar_graph.print_properties_vertex(source_vertex))
        raise ValueError('The source vertex {} is not in the graph'.format(planar_graph.graph.vp['id'][source_vertex]))

def add_point2road(planar_graph,growing_node,added_vertex):
    '''
        Description:
            Adds added_vertex to the road of growing node
    '''
    starting_vertex_road,local_idx_road,found = planar_graph.find_road_vertex(growing_node)
    if found:
        distance_ = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][starting_vertex_road],planar_graph.graph.vp['id'][added_vertex]]
        planar_graph.graph.vp['roads'][starting_vertex_road][local_idx_road].add_node_in_road(growing_node,added_vertex,distance_)                            
    else:
        pass

def find_road_vertex(planar_graph,vertex):
    '''
        Description:
            Find the road that starts from vertex
        Output:
            starting_vertex of the road
            local_idx_road: index of the road in the list of roads starting from starting_vertex
        Complexity:
            O(number_vertices)
    '''
    local_idx_road = 0
    found = False
    for starting_vertex in planar_graph.important_vertices:
        for r in planar_graph.graph.vp['roads'][starting_vertex]:
            local_idx_road += 1
            if r.in_road(vertex):
                found = True
                return starting_vertex,local_idx_road,found
            else:
                print(planar_graph.graph.vp['id'][vertex],' not in road')
    return starting_vertex,0,found

def get_list_nodes_in_roads_starting_from_v(planar_graph,v):
    '''
        Output:
            List of vertices that are in the roads starting from v
            type: list vertex
    '''
    points_in_adjacent_roads_v = []
    for r in planar_graph.graph.vp['roads'][v]:
        for v_road in r.list_nodes:
            points_in_adjacent_roads_v.append(v_road)
    return points_in_adjacent_roads_v


