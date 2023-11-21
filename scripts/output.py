
##------------------------------------------------------------- PRINTING ----------------------------------------------------------

## ------------------------------------- GEOMETRY -------------------------------------
def print_geometrical_info(planar_graph):
    print('*****************************')
    print('Side bounding box: {} km'.format(planar_graph.side_city))
#        print('Number square grid: ',len(planar_graph.grid))
#        print('Side single square: {} km'.format(planar_graph.side_city/np.sqrt(len(planar_graph.grid))))
    print('----------------------')
    print('Initial number points: ',planar_graph.initial_number_points)
    print('Total number of POIs expected: ',planar_graph.total_number_nodes)
    print('*******************************')

## ------------------------------------- VERTICES -------------------------------------

def print_properties_vertex(planar_graph,vertex):
    '''
        Flushes all the properties of the vertex
    '''
    print('function, where planar_graph: ',id(planar_graph.graph))
    print('function, where vertex: ',id(vertex),'in graph: ',id(planar_graph.graph.vp['id'][vertex]))
    print('vertex: ',planar_graph.graph.vp['id'][vertex])
    print('is_active: ',planar_graph.graph.vp['is_active'][vertex])
    print('important_node: ',planar_graph.graph.vp['important_node'][vertex])
    print('end_point: ',planar_graph.graph.vp['end_point'][vertex])
    print('is_in_graph: ',planar_graph.graph.vp['is_in_graph'][vertex])
    print('newly_added_center: ',planar_graph.graph.vp['newly_added_center'][vertex])
    print('relative neighbors: ',planar_graph.graph.vp['relative_neighbors'][vertex])
    print('x: ',planar_graph.graph.vp['x'][vertex])
    print('y: ',planar_graph.graph.vp['y'][vertex])
    print('pos: ',planar_graph.graph.vp['pos'][vertex])
    print('out neighbor: ',[planar_graph.graph.vp['id'][v] for v in vertex.out_neighbours()])
    print('in neighbor: ',[planar_graph.graph.vp['id'][v] for v in vertex.in_neighbours()])
    for r in planar_graph.graph.vp['roads'][vertex]:
        print('road: ',r.id)
        print('number_iterations: ',r.number_iterations)
        print('length: ',r.length)
        print('list_nodes: ',[planar_graph.graph.vp['id'][v] for v in r.list_nodes])
        print('list_edges: ',[[planar_graph.graph.vp['id'][v1],planar_graph.graph.vp['id'][v2]] for v1,v2 in r.list_edges])
        print('end_node: ',planar_graph.graph.vp['id'][r.end_node])
        print('is_closed: ',r.is_closed)
        print('activated_by: ',[planar_graph.graph.vp['id'][v] for v in r.activated_by])

def print_not_considered_vertices(planar_graph,not_considered):
    '''
    Type: Debug
    '''
    print('XXXX   vertices whose attracted by is not updated   XXXX')
    for v in not_considered:
        print('id: ',planar_graph.graph.vp['id'][v])
        print('important by: ',planar_graph.graph.vp['important_node'][v])
        print('attracting: ',planar_graph.graph.vp['attracting'][v])
        print('growing: ',planar_graph.graph.vp['growing'][v])
        print('end_point: ',planar_graph.graph.vp['end_point'][v])

def print_delauney_neighbors(planar_graph,vi):
    print('old: ')
    old_dn = [v for v in planar_graph.graph.vertices() if len(planar_graph.graph.vp['old_attracting_delauney_neighbors'][v])!=0]
    for vj in old_dn:
        print(planar_graph.graph.vp['id'][vj])
        print('neighbors: ',planar_graph.graph.vp['old_attracting_delauney_neighbors'][vj])
    print('new: ')
    new_dn = [v for v in planar_graph.graph.vertices() if len(planar_graph.graph.vp['new_attracting_delauney_neighbors'][v])!=0]
    for vj in new_dn:
        print(planar_graph.graph.vp['id'][vj])
        print('neighbors: ',planar_graph.graph.vp['new_attracting_delauney_neighbors'][vj])

## ------------------------------------- EDGES -------------------------------------

def print_property_road(r):
    '''
        Print all the attributes of the road
    '''
    print('road: ',r.id)
    print('number_iterations: ',r.number_iterations)
    print('length: ',r.length)
    print('list_nodes: ',[v for v in r.list_nodes])
    print('list_edges: ',[[v1,v2] for v1,v2 in r.list_edges])
    print('end_node: ',r.end_node)
    print('is_closed: ',r.is_closed)
    print('activated_by: ',[v for v in r.activated_by])


##------------------------------------ ALL LISTS ------------------------------------
def print_all_lists(planar_graph):
    print('----------------------')
    print('List of all vertices: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.graph.vertices()])
    print('List of all active vertices: ',[r.id for r in planar_graph.active_vertices])
    print('List end points', [planar_graph.graph.vp['id'][v] for v in planar_graph.end_points])
    print('List of all important nodes: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.important_vertices])
    print('List in graph: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.is_in_graph_vertices])
    print('List old attracting: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.old_attracting_vertices])
    print('List new attracting: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.new_attracting_vertices])
    print('List of all roads: ',[r.id for r in planar_graph.list_roads])


##------------------------------------------------------------- ASSERTIONS ----------------------------------------------------------

def ASSERT_PROPERTIES_VERTICES(planar_graph,v):
    '''
        I here control that:
            1) If a vertex is an end node -> it must be in the graph
            2) If a vertex is not in the graph -> it must be active
    '''
    if planar_graph.graph.vp['end_point'] and not planar_graph.graph.vp['is_in_graph']:
        planar_graph.print_properties_vertex(v)
        raise ValueError('The END NODE vertex {} is not in the graph'.format(planar_graph.graph.vp['id'][v]))
    if not planar_graph.graph.vp['is_in_graph'] and not planar_graph.graph['is_active']:
        planar_graph.print_properties_vertex(v)
        raise ValueError('The vertex {} that is NOT in graph must be ACTIVE'.format(planar_graph.graph.vp['id'][v]))

