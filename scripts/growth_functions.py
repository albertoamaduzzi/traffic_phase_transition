from global_functions import *
from pyhull.delaunay import DelaunayTri
from scipy.spatial import Delaunay
from scipy.spatial import distance_matrix
import numpy as np
import time
from termcolor import cprint
# FROM PROJECT
from vertices_functions import *
from geometric_features import road
from vector_operations import *
from output import *
from plots import *

##----------------------------------------- ADDING POINTS, EDGES AND ROADS -------------------------------------------- <- SUBSTEPS OF EVOLVE STREETS                            
    ## STARTS WITH CONNECTED POINTS (DEPRECATED)

def add_initial_points2graph(planar_graph,debug=False):
    '''
        This functions constructs an initially connected graph -> I am leaving it, but it is deprecated
        NOTE: Once I add points I need to set new point to end_point, the previous point not anymore end_points,
        1) Check if important point is still active
        2) Add road
    '''
    if debug:
        print('\t1a) Adding initial points:')
#    x,y =generate_uniform_distribution_nodes_in_space_square(planar_graph,initial_points = True,debug=debug)
    x,y = generate_exponential_distribution_nodes_in_space_square(planar_graph,initial_points = True,debug=debug)
    initialize_distance_matrix(planar_graph,x,y)
    for point_idx in range(len(x)):
        add_vertex(planar_graph)
        vertex = get_last_vertex(planar_graph)
        id_ = get_id_last_vertex(planar_graph)
        set_id(planar_graph,vertex,id_)
        set_newly_added(planar_graph,vertex,False)
        set_initialize_x_y_pos(planar_graph,vertex,x,y,point_idx)
        set_important_node(planar_graph,vertex,True)
        set_active_vertex(planar_graph,vertex,False)
        set_in_graph(planar_graph,vertex,True)
        set_end_point(planar_graph,vertex,True)
        set_relative_neighbors(planar_graph,vertex,[])
        set_road(planar_graph,vertex,[])
        set_is_intersection(planar_graph,vertex,False)
        set_roads_belonging_to_vertex(planar_graph,vertex,[],debug)
        set_roads_activated_by_vertex(planar_graph,vertex,[],debug)
        if debug:
            print_properties_vertex(planar_graph,vertex)
    if len(x)==1:
        pass
    elif len(x)==2:
        c_v0 = 0
        for source_vertex in planar_graph.graph.vertices():
            c_v1 = 0
            for target_vertex in planar_graph.graph.vertices(): 
                if c_v0!=c_v1:
                    add_edge2graph(planar_graph,source_vertex,target_vertex,[target_vertex],False,debug)
                c_v1 += 1
            c_v0 += 1
        for v in planar_graph.graph.vertices():
            for r in planar_graph.graph.vp['roads'][v]: 
                r.is_closed_ = True
                r.closing_vertex = r.end_point
    elif len(x) == 3:
        tri = Delaunay(np.array([x,y]).T)
        simplex = tri.simplices[0]
        for i,j in [(simplex[0],simplex[1]),(simplex[0],simplex[2]),(simplex[1],simplex[2])]:
            if i!=j:
                source_vertex = planar_graph.graph.vertex(i)
                target_vertex = planar_graph.graph.vertex(j)
                add_edge2graph(planar_graph,source_vertex,target_vertex,[target_vertex],False,debug)
                for v in planar_graph.graph.vertices():
                    for r in planar_graph.graph.vp['roads'][v]: 
                        r.is_closed_ = True
                        r.closing_vertex = r.end_point
    else:
        tri = Delaunay(np.array([x,y]).T)
        for simplex in tri.simplices:
            for i,j in [(simplex[0],simplex[1]),(simplex[0],simplex[2]),(simplex[1],simplex[2])]:
                if i!=j:
                    source_vertex = planar_graph.graph.vertex(i)
                    target_vertex = planar_graph.graph.vertex(j)
                    add_edge2graph(planar_graph,source_vertex,target_vertex,[target_vertex],False,debug)
                    for v in planar_graph.graph.vertices():
                        for r in planar_graph.graph.vp['roads'][v]: 
                            r.is_closed_ = True  
                            r.closing_vertex = r.end_point
    if debug: 
        for v in planar_graph.graph.vertices():
            for r in planar_graph.graph.vp['roads'][v]: 
                print_property_road(planar_graph,r)
    
## STARTS WITH DETACHED CENTERS
def add_centers2graph(planar_graph,debug=False):
    '''
        For each center generated:
            1) Add to graph
            2) Give it a name
            3) Initialize position
            4) Set it as important node (each important node attracts points in graph)
            5) Set it as active (It will deactivate just when all the attracted road reach it)

    '''
    if planar_graph.distance_matrix_ is None:
        add_initial_points2graph(planar_graph,debug)
    else:
        pass
    if debug:
        print('\t1a) Add centers to graph')
    x,y = generate_exponential_distribution_nodes_in_space_square(planar_graph,initial_points =False,debug=debug)
    for point_idx in range(len(x)):
        update_distance_matrix(planar_graph,np.array([planar_graph.graph.vp['x'].a,planar_graph.graph.vp['y'].a]).T,np.array([[x[point_idx],y[point_idx]]]))
        add_vertex(planar_graph)
        vertex = get_last_vertex(planar_graph)
        id_ = get_id_last_vertex(planar_graph)
        set_id(planar_graph,vertex,id_)
        set_newly_added(planar_graph,vertex,True)
        set_initialize_x_y_pos(planar_graph,vertex,x,y,point_idx)
        set_important_node(planar_graph,vertex,True)
        set_active_vertex(planar_graph,vertex,True)
        ## Will they be attracted? No
        set_in_graph(planar_graph,vertex,False)
        set_end_point(planar_graph,vertex,False)
        ## RELATIVE NEIGHBOR, ROADS starting from it.
        set_relative_neighbors(planar_graph,vertex,[])
        set_road(planar_graph,vertex,[])
        ## Intersection
        set_is_intersection(planar_graph,vertex,False)
#            print_properties_vertex(planar_graph,vertex)   
        set_roads_belonging_to_vertex(planar_graph,vertex,[],debug)
        set_roads_activated_by_vertex(planar_graph,vertex,[],debug)
        if debug:
            print_properties_vertex(planar_graph,vertex)
        

def add_point2graph(planar_graph,source_vertex,dx,dy,debug=False):
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
    if debug:
        print('\t\t ADD POINT TO GRAPH')
        print('\t\t\t source vertex: ' + str(planar_graph.graph.vp['id'][source_vertex]) + ' coords: ' + str(planar_graph.graph.vp['pos'][source_vertex]))
    x_new_node = [planar_graph.graph.vp['x'][source_vertex]+dx]
    y_new_node = [planar_graph.graph.vp['y'][source_vertex]+dy]
    ## INITIALIZE NEW VERTEX 
    add_vertex(planar_graph)
    id_ = get_id_last_vertex(planar_graph)
    vertex = get_last_vertex(planar_graph)
    set_id(planar_graph,vertex,id_)
    set_newly_added(planar_graph,vertex,False)
    set_initialize_x_y_pos(planar_graph,vertex,x_new_node,y_new_node,0)
    set_important_node(planar_graph,vertex,False)
    set_active_vertex(planar_graph,vertex,False)
    ## CHANGE INFO SOURCE VERTEX
    set_end_point(planar_graph,vertex,True)
    set_end_point(planar_graph,source_vertex,False)
    set_in_graph(planar_graph,vertex,True)
    ## RELATIVE NEIGHBOR, ROADS starting from it.
    set_relative_neighbors(planar_graph,vertex,[])
    set_road(planar_graph,vertex,[])
    set_is_intersection(planar_graph,vertex,False)
    set_roads_belonging_to_vertex(planar_graph,vertex,[],debug = False)
    set_roads_activated_by_vertex(planar_graph,vertex,[],debug = False)
    if debug:
        print_properties_vertex(planar_graph,vertex)
    return get_last_vertex(planar_graph)


##-------------------------------------- DISTANCE MATRIX ------------------------------------------

def initialize_distance_matrix(planar_graph,x,y):
    planar_graph.distance_matrix_ = distance_matrix(np.array([x,y]).T,np.array([x,y]).T) #distance_matrix(np.array([planar_graph.graph.vp['x'],planar_graph.graph.vp['y']]).T,np.array([planar_graph.graph.vp['x'],planar_graph.graph.vp['y']]).T)

def update_distance_matrix(planar_graph,old_points,new_point,debug=False):
    '''
        Description:
            For each new point added in the graph, adds the distance of this point to all the other points.
            In this way I do have a reduction of calculation from N_nodes*N_nodes to N_nodes
        new_point is a vector of dimension (1,2) -> the bottleneck of this function resides in np.vstack as I am adding the 0 of the 
        concatenation of the column and the row one after the other
    '''
    if debug:
        cprint('UPDATE DISTANCE MATRIX','red','on_black')
        cprint('expected distance matrix shape:' + str(len(old_points)) + ','+ str(len(old_points)) + ' shape: ' + str(np.shape(planar_graph.distance_matrix_)),'red','on_black')        
    dm_row = distance_matrix(new_point,old_points)
    if debug:    
        cprint('expected row matrix shape:' + str(1) + ','+ str(len(old_points)) + ' shape: ' + str(np.shape(dm_row)),'red','on_black')        
    dm_col = distance_matrix(old_points,new_point) # dimension (1,starting_points)
    dm_col = np.vstack((dm_col,[0]))  
    if debug:
        cprint('expected col matrix shape:' + str(len(old_points)+1) + ','+ str(1) + ' shape: ' + str(np.shape(dm_col)),'red','on_black')        
    dm_conc = np.concatenate((planar_graph.distance_matrix_,dm_row),axis = 0) # dimension (starting_points+1,starting_popints)
    if debug:
        cprint('expected conc matrix shape:' + str(len(old_points)+1) + ','+ str(len(old_points)) + ' shape: ' + str(np.shape(dm_conc)),'red','on_black')        
    planar_graph.distance_matrix_ = np.concatenate((dm_conc,dm_col),axis = 1)





        
##---------------------------------------- EVOLUTION ----------------------------------------------------##

## ---------------------------------------- EVOLVE VERTEX ----------------------------------------------------##

def evolve_uniquely_attracted_vertex(planar_graph,growing_node,available_vertices,intersection=False,debug=False):
    '''
        Evolve street from (in graph) points that have a unique attracting vertex 
    '''
#    print_delauney_neighbors(planar_graph,growing_node)
#    print('available attracting vertices: ',[planar_graph.graph.vp['id'][v] for v in planar_graph.graph.vp['relative_neighbors'][growing_node]])
    old_points = np.array([planar_graph.graph.vp['x'].a,planar_graph.graph.vp['y'].a]).T
    try:
        vertex_relative_neighbor = planar_graph.graph.vertex(available_vertices[0]) # planar_graph.graph.vertex(int_idx_relative_neighbor)
    except IndexError:
        print(available_vertices)
        raise ValueError('available vertices is empty')

    dx,dy = coordinate_difference(planar_graph.graph.vp['x'][vertex_relative_neighbor],planar_graph.graph.vp['y'][vertex_relative_neighbor],planar_graph.graph.vp['x'][growing_node],planar_graph.graph.vp['y'][growing_node])
    dx,dy = normalize_vector(dx,dy)
    dx,dy = scale(dx,dy,planar_graph.rate_growth)
    new_point = np.array([[planar_graph.graph.vp['x'][growing_node] +dx,planar_graph.graph.vp['y'][growing_node] +dy]])
    if debug:
        cprint('Evolving toward unique attractor','dark_grey','on_light_yellow')
        cprint('Evolving direction node: ' + str(vertex_relative_neighbor) + ' coords: ' + str(planar_graph.graph.vp['pos'][vertex_relative_neighbor]),'dark_grey','on_light_yellow')
        cprint(' dx: ' + str(dx)+ ' dy: ' +str(dy),'dark_grey','on_light_yellow')
        cprint('uniquely available vertex: ' +  str(available_vertices),'dark_grey','on_light_yellow')
        cprint('length segment: '+ str(np.sqrt(dx**2+dy**2)),'dark_grey','on_light_yellow')
        cprint('new added vertex: '+ str(new_point),'dark_grey','on_light_yellow')
    if np.shape(new_point)!=(1,2):
        raise ValueError('new point has wrong shape')
    update_distance_matrix(planar_graph,old_points,new_point)
    added_vertex = add_point2graph(planar_graph,growing_node,dx,dy,debug)
    if planar_graph.graph.vertex(growing_node)==added_vertex:
        raise ValueError('added vertex is the same as growing node')
    add_edge2graph(planar_graph,planar_graph.graph.vertex(growing_node),added_vertex,vertex_relative_neighbor,intersection,debug)
    ## Find the road where the points belongs to (looking at the starting vertex it is generated from)
    if planar_graph.iteration_count%planar_graph.iteration_plot==0:
        plot_relative_neighbors(planar_graph,growing_node,added_vertex,available_vertices,debug)

#    print('direction: ',planar_graph.graph.vp['id'][vertex_relative_neighbor])
#    print(' dx: ',dx,' dy: ',dy) 
#    print('added vertex: ',planar_graph.graph.vp['id'][added_vertex],' coords: ',planar_graph.graph.vp['pos'][added_vertex])

def evolve_multiply_attracted_vertices(planar_graph,growing_node,available_vertices,intersection=False,debug=False):
    '''
        NOTE:
            1) Relative neighbors are already calculated for all the attracting and attracted vertices.
                In particular: for each active node I have a set of relative neighbors, as relative neighborhood is a symmetric property
                also the attracted vertices have the set of relative neighbor containing the attracting vertices that are in_graph = False.
                See (compute_rng_newly_added_centers,compute_rng_old_attracting_centers) -> They differ in the Delauney triangle 
                one computes.
        Evolve street from (in graph) points that have a multiple attracting vertices
    '''
    if debug:
        cprint('Evolving toward multiple attractors','dark_grey','on_light_yellow')
        cprint('growing node: ' +str(growing_node),'dark_grey','on_light_yellow')
        cprint('Available vertices to grow into: ' + str(available_vertices),'dark_grey','on_light_yellow')
    epsilon = planar_graph.rate_growth
#    print_delauney_neighbors(planar_graph,growing_node)
#    print('available attracting vertices: ',[planar_graph.graph.vp['id'][v] for v in available_vertices])
    dx = 0
    dy = 0
    for neighbor_attracting_vertex_idx in available_vertices:
        vertex_relative_neighbor = planar_graph.graph.vertex(neighbor_attracting_vertex_idx)
        old_points = np.array([planar_graph.graph.vp['x'].a,planar_graph.graph.vp['y'].a]).T
        x,y = coordinate_difference(planar_graph.graph.vp['x'][vertex_relative_neighbor],planar_graph.graph.vp['y'][vertex_relative_neighbor],planar_graph.graph.vp['x'][growing_node],planar_graph.graph.vp['y'][growing_node])
        x,y = normalize_vector(x,y)
        if debug:
            cprint('\tv: ' + str(neighbor_attracting_vertex_idx) + ' coords: ' + str(planar_graph.graph.vp['pos'][vertex_relative_neighbor]),'dark_grey','on_light_yellow')
            cprint('\t dx: ' + str(x) + ' dy: '+str(y),'dark_grey','on_light_yellow')
        dx += x 
        dy += y
    if planar_graph.debug_sum_evolution:
        cprint('total sum (dx,dy): (' + str(dx)+','+str(dy) + ')','dark_grey','on_light_yellow')
        cprint('length segment: '+ str(np.sqrt(dx**2+dy**2)),'dark_grey','on_light_yellow')
    if np.sqrt(dx**2+dy**2)>epsilon: 
        dx,dy = normalize_vector(dx,dy)
        dx,dy = scale(dx,dy,planar_graph.rate_growth)
        new_point = np.array([[planar_graph.graph.vp['x'][growing_node] + dx,planar_graph.graph.vp['y'][growing_node] + dy]])
        if np.shape(new_point)!=(1,2):
            raise ValueError('new point has wrong shape')        
        update_distance_matrix(planar_graph,old_points,new_point)
        if planar_graph.debug_sum_evolution:
            cprint('\tEvolving sum attractor','dark_grey','on_light_yellow')
            cprint('growing node: ' +str(growing_node),'dark_grey','on_light_yellow')
            cprint('\tAverage direction (dx,dy) =  (' + str(dx)+','+str(dy) + ')','dark_grey','on_light_yellow')
            cprint('\tlength segment: ' + str(np.sqrt(dx**2+dy**2)),'dark_grey','on_light_yellow')
            cprint('\tnew point: ' + str(new_point),'dark_grey','on_light_yellow')
        if np.shape(new_point)!=(1,2):
            raise ValueError('new point has wrong shape')
        added_vertex = add_point2graph(planar_graph,growing_node,dx,dy,debug)
        if planar_graph.graph.vertex(growing_node)==added_vertex:
            raise ValueError('added vertex is the same as growing node')        
        add_edge2graph(planar_graph,planar_graph.graph.vertex(growing_node),added_vertex,available_vertices,intersection,debug)
        if planar_graph.debug_sum_evolution:
            cprint('Distance ' + str(planar_graph.graph.vp['id'][added_vertex]) + ' to attracting vertices:','dark_grey','on_light_yellow')
            for v in available_vertices:
                cprint('\t' + str(planar_graph.graph.vp['id'][v]) + ' distance: ' + str(planar_graph.distance_matrix_[planar_graph.graph.vp['id'][added_vertex],planar_graph.graph.vp['id'][v]]),'dark_grey','on_light_yellow')
                if planar_graph.distance_matrix_[planar_graph.graph.vp['id'][added_vertex],planar_graph.graph.vp['id'][v]]<epsilon:
                    cprint('The road with this end point must be closed:','yellow','on_light_red')
                    cprint('WAITING TO CLOSE ROADS','yellow','on_light_red')

        if planar_graph.iteration_count%planar_graph.iteration_plot==0:        
            plot_relative_neighbors(planar_graph,growing_node,added_vertex,available_vertices,debug)                            
#        print('direction sum of : ',[planar_graph.graph.vp['id'][vertex_relative_neighbor] for vertex_relative_neighbor in available_vertices])
#        print(' dx: ',dx,' dy: ',dy) 
#        print('added vertex: ',planar_graph.graph.vp['id'][added_vertex],' coords: ',planar_graph.graph.vp['pos'][added_vertex])
    else:
        for neighbor_attracting_vertex_idx in available_vertices:
            vertex_relative_neighbor = planar_graph.graph.vertex(neighbor_attracting_vertex_idx)
            old_points = np.array([planar_graph.graph.vp['x'].a,planar_graph.graph.vp['y'].a]).T
            intersection = True
#                            vertex_relative_neighbor = planar_graph.graph.vertex(int_idx_relative_neighbor)
            dx,dy = coordinate_difference(planar_graph.graph.vp['x'][vertex_relative_neighbor],planar_graph.graph.vp['y'][vertex_relative_neighbor],planar_graph.graph.vp['x'][growing_node],planar_graph.graph.vp['y'][growing_node])
            dx,dy = normalize_vector(dx,dy)
            dx,dy = scale(dx,dy,planar_graph.rate_growth)
            new_point = np.array([[planar_graph.graph.vp['x'][growing_node] +dx,planar_graph.graph.vp['y'][growing_node] +dy]])
            if planar_graph.debug_degenerate_evolution:
                cprint('Evolving in all directions due to degeneracy')
                cprint('growing node: ' +str(growing_node),'dark_grey','on_light_yellow')
                cprint('\tvertex: ' + str(vertex_relative_neighbor)+' coords: '+str(planar_graph.graph.vp['pos'][vertex_relative_neighbor]),'dark_grey','on_light_yellow')
                cprint('\tdirection (dx,dy) =  (' + str(dx)+','+str(dy) + ')','dark_grey','on_light_yellow')
                cprint('\tlength segment: ' + str(np.sqrt(dx**2+dy**2)),'dark_grey','on_light_yellow')
                cprint('\tnew point: ' + str(new_point),'dark_grey','on_light_yellow')
            if np.shape(new_point)!=(1,2):
                raise ValueError('new point has wrong shape')
            update_distance_matrix(planar_graph,old_points,new_point)
            added_vertex = add_point2graph(planar_graph,growing_node,dx,dy,debug)
            if planar_graph.graph.vertex(growing_node)==added_vertex:
                raise ValueError('added vertex is the same as growing node')
            add_edge2graph(planar_graph,planar_graph.graph.vertex(growing_node),added_vertex,vertex_relative_neighbor,intersection,debug)  
            if planar_graph.iteration_count%planar_graph.iteration_plot==0:
                plot_relative_neighbors(planar_graph,growing_node,added_vertex,available_vertices,debug)
#            print('direction sum of : ',planar_graph.graph.vp['id'][vertex_relative_neighbor])
#            print(' dx: ',dx,' dy: ',dy) 
#            print('added vertex: ',planar_graph.graph.vp['id'][added_vertex],' coords: ',planar_graph.graph.vp['pos'][added_vertex])

## ---------------------------------------- EVOLVE STREET ----------------------------------------------------##

def evolve_street_old_attractors(planar_graph,debug = False):
    '''
        Evolve street for the old attractors
    '''
    ## TODO: differentiate between intersections,important nodes and normal nodes
    already_grown_vertices = []
    if debug:
        cprint('Evolve streets toward old attractors','dark_grey','on_light_yellow')
        cprint('End points: ','dark_grey','on_light_yellow')
        for p in planar_graph.end_points:
            cprint(str(planar_graph.graph.vp['id'][p]) +
                   ' is active: ' + str(planar_graph.graph.vp['is_active'][p]) +
                   ' intersection: ' + str(planar_graph.graph.vp['intersection'][p]),'dark_grey','on_light_yellow')
        ## if the relative neighbor is just one
    for growing_node in planar_graph.end_points:
        if growing_node not in planar_graph.already_evolved_end_points:
            if debug:
                cprint('coords growing node: ' + str(planar_graph.graph.vp['id'][growing_node]) +
                       ' ' + str(planar_graph.graph.vp['pos'][growing_node]),'dark_grey','on_light_yellow')
            available_vertices = planar_graph.graph.vp['relative_neighbors'][growing_node]   
            if debug:
                cprint('\tavailable vertices before taking away nearest neighbors','dark_grey','on_light_yellow')
                for v in available_vertices:
                    cprint('\t'+str(v) +' type: ' + 
                   ' is active: ' + str(planar_graph.graph.vp['is_active'][v]) +
                   ' intersection: ' + str(planar_graph.graph.vp['intersection'][v]) +
                   ' end point: ' + str(planar_graph.graph.vp['end_point'][v]),'dark_grey','on_light_yellow')
            ## NOTE: There can be a moment in which
            available_vertices = [v for v in available_vertices if not planar_graph.graph.vp['is_in_graph'][planar_graph.graph.vertex(v)] and planar_graph.graph.vp['is_active'][planar_graph.graph.vertex(v)]] 
            number_relative_neighbors_growing_node = len(available_vertices)
    #        growing_vertex = planar_graph.graph.vertex(growing_node)
            if debug:
                cprint('\tavailable vertices node '+ str(growing_node),'dark_grey','on_light_yellow')
                for v in available_vertices:
                    cprint('\t'+str(v) +' type: ' + 
                   ' is active: ' + str(planar_graph.graph.vp['is_active'][v]) +
                   ' intersection: ' + str(planar_graph.graph.vp['intersection'][v]) +
                   ' end point: ' + str(planar_graph.graph.vp['end_point'][v]),'dark_grey','on_light_yellow')
            ## Take if relative neighbor of the growing node is just one  
            if number_relative_neighbors_growing_node==1:
                evolve_uniquely_attracted_vertex(planar_graph,growing_node,available_vertices,planar_graph.graph.vp['intersection'][growing_node],planar_graph.debug_unique_evolution)
            elif number_relative_neighbors_growing_node>1: # a relative neighbor is also a neighbor -> kill the growth
                evolve_multiply_attracted_vertices(planar_graph,growing_node,available_vertices,planar_graph.graph.vp['intersection'][growing_node],planar_graph.debug_multiple_evolution)
            else:
                pass
            if growing_node not in already_grown_vertices:
                already_grown_vertices.append(growing_node)
                vv = [planar_graph.graph.vp['id'][growing_node] for growing_node in already_grown_vertices]
                vv = np.sort(vv)
            planar_graph.already_evolved_end_points.append(growing_node)
        else:   
            pass
    if debug:
        cprint('\tAlready evolved end points: ','dark_grey','on_light_yellow' )
        for n in planar_graph.already_evolved_end_points:
            cprint(str(n),'dark_grey','on_light_yellow')

def evolve_street_newly_added_attractors(planar_graph,debug = False):
    '''
        Description:
            For each node that are active and newly added
                1) Check if the attracting node is still active (it must be true otherwhise raise Error)
                2) For each relative neighbor (growing node) of the attracting node
                    a) If the growing node is not already grown: -> evolve it
                    b) If the growing node is already grown -> do nothing
    '''
    already_grown_vertices = []
    if debug:
        cprint('Evolve streets toward new attractors: ' + str(planar_graph.newly_added_attracting_vertices),'dark_grey','on_light_yellow')
    for attracting_node in planar_graph.newly_added_attracting_vertices:
        if is_active(planar_graph,attracting_node):
            if debug:
                cprint('\tAttracting node: ' + str(attracting_node),'dark_grey','on_light_yellow')
            ## if the relative neighbor is just one and it is not already attached tothe graph (neigbor['growing'] == False)
            ## if the relative neighbor is just one
            for growing_node in planar_graph.graph.vp['relative_neighbors'][attracting_node]:
                if growing_node not in already_grown_vertices and planar_graph.graph.vp['is_in_graph'][growing_node] and growing_node not in planar_graph.newly_added_attracting_vertices:
                    if not is_end_point(planar_graph,growing_node) and not is_important_node(planar_graph,growing_node):
                        intersection = True
                    else:
                        intersection = False
                    if debug:
                        cprint(' coords growing node: ' + str(planar_graph.graph.vp['pos'][growing_node]),'dark_grey','on_light_yellow')
                    available_vertices = planar_graph.graph.vp['relative_neighbors'][growing_node]
                    if debug:
                        cprint('\tavailable vertices before filter ','dark_grey','on_light_yellow')
                        for v in available_vertices:
                            cprint(str(v) +' type: ' + str(type(v)),'dark_grey','on_light_yellow')
                    available_vertices = [v for v in available_vertices if not planar_graph.graph.vp['is_in_graph'][planar_graph.graph.vertex(v)]]
                    if debug:
                        cprint(' coords growing node: ' + str(planar_graph.graph.vp['pos'][growing_node]),'dark_grey','on_light_yellow')
                        cprint('\tavailable vertices node '+ str(growing_node),'dark_grey','on_light_yellow')
                        for v in available_vertices:
                            cprint(str(v) +' type: ' + str(type(v)),'dark_grey','on_light_yellow')
                        cprint('\tIntersection: ' + str(intersection),'dark_grey','on_light_yellow')
                    number_relative_neighbors_growing_node = len(available_vertices)
    #                print('growing node: ',planar_graph.graph.vp['id'][growing_node],' coords: ',planar_graph.graph.vp['pos'][growing_node])
                    ## Take if relative neighbor of the growing node is just one  
                    if number_relative_neighbors_growing_node==1:
                        evolve_uniquely_attracted_vertex(planar_graph,growing_node,available_vertices,intersection,planar_graph.debug_unique_evolution)
                    elif number_relative_neighbors_growing_node>1: # a relative neighbor is also a neighbor -> kill the growth
                        evolve_multiply_attracted_vertices(planar_graph,growing_node,available_vertices,intersection,planar_graph.debug_multiple_evolution)
                    else:
                        pass
                    if growing_node not in already_grown_vertices:
                        already_grown_vertices.append(growing_node)
                        vv = [planar_graph.graph.vp['id'][growing_node] for growing_node in already_grown_vertices]
                        vv = np.sort(vv)
                    planar_graph.already_evolved_end_points.append(growing_node)
                else: 
                    pass

        else:
            print('Attracting node: ',planar_graph.graph.vp['id'][attracting_node])
            raise ValueError('The attracting node is not active')


        intersection = False
        if debug:
            cprint('\tAlready evolved end points: ' )
            for n in planar_graph.already_evolved_end_points:
                cprint(str(n),'dark_grey','on_light_yellow')
    


## ------------------------- CLOSURE ROADS -------------------------------------------- 

def close_roads(planar_graph,debug=False):
    '''
        Attach the edge if the growing points are close enough to their relative neighbors,
        in this way the relative neighbor becomes a growing point as well.
        If the point is among the activated by of the road, then I close the road, otherwise I do not.
        TODO: Find a way to consider the ending points of the roads
        Description:
            For each active road:
                Control if the distance to the end point is less then rate of growth from an important node:
                    If yes:
                        1) Add edge to graph
                        2) Set the vertex in graph
                        NOTE:(Check if in the next iteration the end point is considered for relative neighbors)
                        3) Add node in road:
                         3a) update list of nodes
                         3b) edges
                         3c) distance
                         3d) end point
                         3e) iteration of the road
                        3) If the node is in the activated by: 
                            3a) Yes -> remove it
                            3b) No -> pass
                        4) If the activated by is empty: 
                            4a) Yes -> close the road (the road will not be in list_active_roads next iteration)
                            4b) pass
                        ) Set the road as belonging_to_vertex

    '''
    
    already_existing_edges = [[e.source(),e.target()] for e in planar_graph.graph.edges()]
    if debug:
        print('CLOSING ROADS')
        cprint('available active vertices:','green','on_white')        
        for v in planar_graph.graph.vertices():
            if planar_graph.graph.vp['is_active'][v]:
                cprint(str(planar_graph.graph.vp['id'][v]) +
                       ' roads activated: ' + str(planar_graph.graph.vp['roads_activated'][v])+
                       ' roads belonging to: ' + str(planar_graph.graph.vp['roads_belonging_to'][v])+
                        ' is active: ' + str(planar_graph.graph.vp['is_active'][v]) +
                        ' intersection: ' + str(planar_graph.graph.vp['intersection'][v]) +
                        ' end point: ' + str(planar_graph.graph.vp['end_point'][v]) +
                        ' is in graph: ' + str(planar_graph.graph.vp['is_in_graph'][v])
                         ,'green','on_white')

#        print('already existing edges: ',already_existing_edges)
    for r in planar_graph.list_active_roads:
        if debug:
            print_property_road(planar_graph,r)
        for active_v in planar_graph.active_vertices:
                ep = planar_graph.graph.vp['id'][r.end_point]
                if planar_graph.distance_matrix_[planar_graph.graph.vp['id'][active_v],ep] < planar_graph.rate_growth and [active_v,ep] not in already_existing_edges:
                    planar_graph.graph.add_edge(ep,active_v)
                    planar_graph.graph.vp['is_in_graph'][active_v] = True 
                    starting_vertex_road,local_idx_road,found,id_ = find_road_vertex(planar_graph,ep,debug)
                    if found:
                        planar_graph.graph.vp['roads'][starting_vertex_road][local_idx_road].add_node_in_road(ep,active_v,planar_graph.distance_matrix_[planar_graph.graph.vp['id'][ep],planar_graph.graph.vp['id'][active_v]])
                        set_roads_belonging_to_vertex(planar_graph,active_v,id_,debug)
                    if planar_graph.graph.vp['id'][active_v] in r.activated_by:
                        
                        a_activated_roads = np.array(r.activated_by)
                        indices = np.where(a_activated_roads != planar_graph.graph.vp['id'][active_v])
                        r.activated_by = [a_activated_roads[xi] for xi in range(len(a_activated_roads)) if xi in np.unique(indices[0])]
                        if r.id not in planar_graph.graph.vp['roads_activated'][active_v]:
                            print_properties_vertex(planar_graph,active_v)
                            raise ValueError('Road not in roads activated by vertex even though the vertex is in the activated by of the road')
                        else:
                            a_activated_roads = np.array(planar_graph.graph.vp['roads_activated'][active_v])
                            indices = np.where(a_activated_roads != r.id)
                            planar_graph.graph.vp['roads_activated'][active_v] = [a_activated_roads[xi] for xi in range(len(a_activated_roads)) if xi in np.unique(indices[0])]
                        # CLOSING ROAD
                        if len(r.activated_by)==0:
                            r.is_closed_ = True
                            if r.closing_vertex == -1:
                                r.closing_vertex = active_v
                            else:
                                print_property_road(planar_graph,r)
                                raise ValueError('Road already closed')
                            deactivate_vertex(planar_graph,active_v,False)
    #                    planar_graph.graph.vp['end_point'][active_v] = True
                        if debug:
                            cprint('Vertex '+ str(planar_graph.graph.vp['id'][active_v]) +' is in '+ str(r.id) + ' road {activated by}','yellow','on_light_red')                            
                            cprint('After removing vertex'+ str(planar_graph.graph.vp['id'][active_v]) +'is in '+ str(r.id) + 'road {activated by}','yellow','on_light_red')
                            cprint('Road: ' + str(r.id) + ' activated by: ' + str(r.activated_by),'yellow','on_light_red')
                            print('\tclosing road: ',r.id,' with end point: ',ep,' and starting point: ',r.initial_node)
                            print_property_road(planar_graph,r)
                            print('Closing road vertex: ')
                            print_properties_vertex(planar_graph,active_v)
                    else:
                        pass
                else:
                    pass



def deactivate_vertex(planar_graph,vertex,debug=False):
    '''
        If all the roads are that start are activated by vertex are closed, then deactivate vertex.
        It is called whenever a close a road with that node. If the 
        TODO: This is wrong as I am looking at roads_activated -> persepctive of the node, may be more convinient taking the perspective of the road
    '''
    if debug:
        cprint('DEACTIVATE VERTEX: ' + str(vertex),'red','on_white')
    if len(planar_graph.graph.vp['roads_activated'][vertex])==0:
        set_active_vertex(planar_graph,vertex,False)
        pass
    else:
        cprint('Still roads to grow vertex: '+ str(planar_graph.graph.vp['id'][vertex]),'red','on_white')
        for r_id in planar_graph.graph.vp['roads_activated'][vertex]:
            if debug:
                cprint(str(r_id),'red','on_white')
#                print_property_road(planar_graph,planar_graph.list_roads[r_id])

def get_road_from_id_(planar_graph,r_id,debug=False):
    if debug:
        print(planar_graph.list_roads)
        print('Get road id: ',r_id)
    for r in planar_graph.list_roads:
        if debug:
            print('r.id: ',r.id)
            print('r_id: ',r_id)
        if r.id == r_id:
            return r
