import graph_tool 
from shapely.geometry import Polygon,Point,LineString
import numpy as np
from shapely.prepared import prep
import geopandas as gpd
import matplotlib.pyplot as plt
from collections import defaultdict
from termcolor import cprint

## FROM PROJECT

'''
    For each t:
        Add a set of nodes
            1) They are attracting nodes:
                1a) 
'''


class road:
    '''
        Input:
            initial_node: vertex 
            global_counting_roads: int [integer that I am updating every time I create a new road]
            activation_vertex: [vertex] [list of vertices that starts the attraction of the road]
            type_initial_node: string ['important_node','intersection']
        NOTE: 
            list_nodes: list dtype = vertex
        Road will become useful when I need to update rng, indeed It is happening that if:
            Node i and Node j are attracted one another and they grow the first step of the road
            they stop, as now their relative neighbors are the points that have grown,
            for this reason I want to avoid the elements of the road in the rng calculation
        Need to add a condition not to growing element grow back to the starting point  
    '''
    # TODO: need to block the growing nodes to grow back to their starting point
    def __init__(self,initial_node,second_node,global_counting_roads,activation_vertex,type_ = 0,unit_length = 0.01,debug = False):
        if debug:
            cprint('CREATING ROAD: ' + str(global_counting_roads),'light_magenta')        
        self.id = global_counting_roads
        self.initial_node = initial_node
        self.number_iterations = 0
        self.length = unit_length
        self.list_nodes = [initial_node,second_node] # Vertex
#        self.linestring = LineString(self.list_nodes)
        self.list_edges = [[initial_node,second_node]] # [vertex,vertex
        self.evolution_attractors = defaultdict()#{t:[] for t in range()}
        self.end_point = second_node    
        self.is_closed_ = False
        self.closing_vertex = -1

        if isinstance(activation_vertex,(list,np.ndarray,graph_tool.libgraph_tool_core.Vector_int32_t)):
            if debug:
                cprint('ARRAY: ' + str(len(activation_vertex)) + ' NODES','magenta')
                for n in activation_vertex:
                    cprint(str(n),'light_magenta')
            self.activated_by = np.array(activation_vertex,dtype=int)
        else:
            self.activated_by = [int(activation_vertex)]
            if debug:
                cprint(str(type(activation_vertex)),'magenta')
                for n in self.activated_by:
                    cprint('ACTIVATED BY: ' + str(n),'light_magenta')
        ## TYPE ROAD
        self.type_ = type_
        self.capacity_level = 0
        ## 
    def add_node_in_road(self,source_node,new_vertex,distance_sn,debug = False):
        '''
            Input:
                new_vertex: vertex
            Description:
                Use to add point in the road
        '''
        if debug:
            print('ADDING NODE {0} IN ROAD {1}'.format(new_vertex,self.id))
        self.list_nodes.append(new_vertex)
        self.list_edges.append([source_node,new_vertex])
        self.length += distance_sn
        self.number_iterations += 1
        self.end_point = new_vertex
    
    def in_road(self,vertex):
        return vertex in self.list_nodes

    def activating_node(self):
        return self.activated_by
    
    def copy_road_specifics(self,road):
        self.number_iterations = road.number_iterations
        self.length = road.length
        self.list_nodes = road.list_nodes
#        self.linestring = LineString(self.list_nodes)
        self.list_edges = road.list_edges
    
## ----------------------------------- GET FUNCTIONS ----------------------------------- ##        
    def get_type_initial_node(self):
        return self.type_initial_node

##------------------------------------- IS FUNCTIONS -------------------------------------##
    def is_closed(self):
        return self.is_closed_




if __name__ == '__main__':
    SIDE_CITY = 2
    RESOLUTION_GRID = 0.1
    city_box = Grid(SIDE_CITY,RESOLUTION_GRID)
    grid = city_box.partition()
    fig, ax = plt.subplots(figsize=(15, 15))
    gpd.GeoSeries(grid).boundary.plot(ax=ax)
    gpd.GeoSeries([city_box.geom]).boundary.plot(ax=ax,color="red")
    plt.show()


