from termcolor import cprint

##------------------------------------------------------------- PRINTING ----------------------------------------------------------

## ------------------------------------- GEOMETRY -------------------------------------
def print_geometrical_info(planar_graph):
    cprint('*************** GEOMETRY **************','green')
    cprint('Side bounding box: '+ str(planar_graph.side_city) +' km','green')
    cprint('Area bounding box: '+ str(planar_graph.side_city**2) +' km^2','green')
    cprint('r0: ' + str(planar_graph.r0) + ' km','green')
    cprint('Growth road rate / size city:' + str(planar_graph.ratio_growth2size_city),'green')
    cprint('Rate growth: '+ str(planar_graph.rate_growth) +' km','green')
    cprint('*************** DYNAMICS **************','yellow')
    cprint('Initial number nodes: '+ str(planar_graph.initial_number_points),'green')
    cprint('Number nodes per tau_c: '+ str(planar_graph.number_nodes_per_tau_c),'green')    
    cprint('tau_c: '+ str(planar_graph.tau_c),'green')
#    cprint('Total number nodes: '+ str(planar_graph.total_number_nodes),'green')
    cprint('*************`*** GRID ****************','red')
    cprint('Number grids: '+ str(planar_graph.number_grids),'green')
#        cprint('Number square grid: ',len(planar_graph.grid))
#        cprint('Side single square: {} km'.format(planar_graph.side_city/np.sqrt(len(planar_graph.grid))))

## ------------------------------------- VERTICES -------------------------------------

def print_properties_vertex(planar_graph,vertex):
    '''
        Flushes all the properties of the vertex
    '''
    
    cprint('PRINT PROPERTIES VERTEX','magenta')
    cprint('vertex: ' + str(planar_graph.graph.vp['id'][vertex]),'magenta')
    cprint('is_active: ' + str(planar_graph.graph.vp['is_active'][vertex]),'magenta')
    cprint('important_node: ' + str(planar_graph.graph.vp['important_node'][vertex]),'magenta')
    cprint('end_point: ' + str(planar_graph.graph.vp['end_point'][vertex]),'magenta')
    cprint('is_in_graph: ' + str(planar_graph.graph.vp['is_in_graph'][vertex]),'magenta')
    cprint('newly_added_center: ' + str(planar_graph.graph.vp['newly_added_center'][vertex]),'magenta')
    cprint('relative neighbors: ' + str(planar_graph.graph.vp['relative_neighbors'][vertex]),'magenta')
    cprint('x: ' + str(planar_graph.graph.vp['x'][vertex]),'magenta')
    cprint('y: ' + str(planar_graph.graph.vp['y'][vertex]),'magenta')
    cprint('pos: ' + str(planar_graph.graph.vp['pos'][vertex]),'magenta')
    cprint('ROADS ASSOCIATED TO VERTEX','magenta')
    for r in planar_graph.graph.vp['roads_belonging_to'][vertex]:
        cprint('road: ' + str(r),'magenta')
    cprint('ROADS ACTIVATED BY VERTEX','magenta')
    for r in planar_graph.graph.vp['roads_activated'][vertex]:
        cprint('road: ' + str(r),'magenta')
def print_not_considered_vertices(planar_graph,not_considered):
    '''
    Type: Debug
    '''
    cprint('XXXX   vertices whose attracted by is not updated   XXXX')
    for v in not_considered:
        cprint('id: ',planar_graph.graph.vp['id'][v])
        cprint('important by: ',planar_graph.graph.vp['important_node'][v])
        cprint('attracting: ',planar_graph.graph.vp['attracting'][v])
        cprint('growing: ',planar_graph.graph.vp['growing'][v])
        cprint('end_point: ',planar_graph.graph.vp['end_point'][v])

def print_delauney_neighbors(planar_graph,vi):
    if vi in planar_graph.old_attracting_vertices:
        cprint('old: ')
        cprint('vertex: ',planar_graph.graph.vp['id'][vi])
        cprint('daluney',planar_graph.graph.vp['old_attracting_delauney_neighbors'][vi])
    elif vi in planar_graph.newly_added_attracting_vertices:
        cprint('new: ')
        cprint('vertex: ',planar_graph.graph.vp['id'][vi])
        cprint('daluney',planar_graph.graph.vp['new_attracting_delauney_neighbors'][vi])

## ------------------------------------- EDGES -------------------------------------

def print_property_road(planar_graph,r):
    '''
        Print all the attributes of the road
    '''

    cprint('PRINT PROPERTY ROAD','red')
    cprint('road: '+ str(r.id),'red')
    cprint('initial_node: '+ str(planar_graph.graph.vp['id'][r.initial_node]),'red')
    cprint('number_iterations: '+ str(r.number_iterations),'red')
    cprint('length: '+ str(r.length),'red')
    cprint('list_nodes: ','red')
    for v in r.list_nodes:
        cprint(str(planar_graph.graph.vp['id'][v])
                + ' intersection: ' + str(planar_graph.graph.vp['intersection'][v])
                + ' important node: ' + str(planar_graph.graph.vp['important_node'][v])
                + ' is active: ' + str(planar_graph.graph.vp['is_active'][v])
                + ' coordinates: ' + str(planar_graph.graph.vp['pos'][v])
                + ' is in graph: ' + str(planar_graph.graph.vp['is_in_graph'][v])
                ,'red')
    cprint('list_edges: ','red')
    for v1,v2 in r.list_edges:
        cprint('('+ str(planar_graph.graph.vp['id'][v1]) + ',' + str(planar_graph.graph.vp['id'][v2])+')','red')
    cprint('end_point: ' + str(planar_graph.graph.vp['id'][r.end_point]),'red')
    cprint('is_closed: ' + str(r.is_closed_),'red')
    cprint('activated_by: ','red')
    for ab in r.activated_by:
        cprint(str(planar_graph.graph.vp['id'][ab]) 
               + ' important node: ' + str(planar_graph.graph.vp['important_node'][ab])
               ,'red')
    cprint('type: '+ str(r.type_),'red')
    cprint('capacity_level: '+ str(r.capacity_level),'red')
    if r.closing_vertex != -1:
        cprint('Vertex causing the closure: '+ str(planar_graph.graph.vp['id'][r.closing_vertex]),'red')
    else:
        cprint('Vertex causing the closure: -1','red')
    print('----------------------------------------')



##------------------------------------ ALL LISTS ------------------------------------
def print_all_lists(planar_graph):
    cprint('PRINT ALL LISTS')
    cprint('List of all vertices: ')
    for v in planar_graph.graph.vertices():
        cprint(str(planar_graph.graph.vp['id'][v]),'blue')
    cprint('List of all active vertices: ')
    for v in planar_graph.active_vertices:
        cprint(planar_graph.graph.vp['id'][v],'blue')
    cprint('List end points')
    for v in planar_graph.end_points:
        cprint(planar_graph.graph.vp['id'][v],'blue')
    cprint('List of all important nodes: ')
    for v in planar_graph.important_vertices:
        cprint(planar_graph.graph.vp['id'][v],'blue')
    cprint('List in graph: ')
    for v in planar_graph.is_in_graph_vertices:
        cprint(planar_graph.graph.vp['id'][v],'blue')
    cprint('List old attracting: ')
    for v in planar_graph.old_attracting_vertices:
        cprint(planar_graph.graph.vp['id'][v],'blue')
    cprint('List new attracting: ')
    for v in planar_graph.newly_added_attracting_vertices:
        cprint(planar_graph.graph.vp['id'][v],'blue') 
    cprint('List of all roads: ')
    for r in planar_graph.list_roads:
        cprint(str(r.id ),'blue')

##------------------------------------------------------------- ASSERTIONS ----------------------------------------------------------

def ASSERT_PROPERTIES_VERTICES(planar_graph,v):
    '''
        I here control that:
            1) If a vertex is an end node -> it must be in the graph
            2) If a vertex is not in the graph -> it must be active
    '''
    if planar_graph.graph.vp['end_point'] and not planar_graph.graph.vp['is_in_graph']:
        print_properties_vertex(planar_graph,v)
        raise ValueError('The END NODE vertex {} is not in the graph'.format(planar_graph.graph.vp['id'][v]))
    if not planar_graph.graph.vp['is_in_graph'] and not planar_graph.graph['is_active']:
        print_properties_vertex(planar_graph,v)
        raise ValueError('The vertex {} that is NOT in graph must be ACTIVE'.format(planar_graph.graph.vp['id'][v]))


