from shapely.geometry import Polygon,Point,LineString
import numpy as np
from shapely.prepared import prep
import geopandas as gpd
import matplotlib.pyplot as plt
from collections import defaultdict

'''
    For each t:
        Add a set of nodes
            1) They are attracting nodes:
                1a) 
'''

class node:
    def __init__(self,int_idx_vertex,type,global_counting_roads,initial_node=False):
        self.final_node = False
        self.list_roads_belong = []
        if initial_node:
            self.initial_node = initial_node
            self.list_roads_belong.append(global_counting_roads)
        self.type = type
        self.id_attractor: int_idx_vertex # s
        self.current_attracted_vertices: list # V(s)
        self.current_attracted_roads: list # [starting_vertex,...,ending_vertex]

    def get_index_roads(self):
        '''
            Returns the road.id the node belongs to
        '''
        return self.list_roads_belong

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
    def __init__(self,initial_node,global_counting_roads,activation_vertex,type_initial_node):
        self.id = global_counting_roads
        self.initial_node = initial_node
        self.number_iterations = 0
        self.length = 0
        self.list_nodes = [initial_node] # Vertex
        self.type_initial_node = type_initial_node
#        self.linestring = LineString(self.list_nodes)
        self.list_edges = []
        self.evolution_attractors = defaultdict()#{t:[] for t in range()}
        self.end_point = None    
        self.is_closed = False
        if type(activation_vertex) == list or type(activation_vertex) == np.array:
            self.activated_by = activation_vertex
        else:
            self.activated_by = [activation_vertex]

    def add_node_in_road(self,source_node,new_vertex,distance_sn):
        '''
            Input:
                new_vertex: vertex
            Description:
                Use to add point in the road
        '''

        self.list_nodes.append(new_vertex)
        self.list_edges.append([source_node,new_vertex])
        self.length += distance_sn
        self.number_iterations += 1
        self.end_node = new_vertex
    
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

class Grid:
    def __init__(self,side_city,resolution_grid):
        self.side_city = side_city
        self.resolution_grid = resolution_grid
        self.minx = -self.side_city/2
        self.miny = -self.side_city/2
        self.maxx = self.side_city/2
        self.maxy = self.side_city/2
        self.geom = Polygon([[self.minx,self.miny],[self.minx,self.maxy],[self.maxx,self.maxy],[self.maxx,self.miny],[self.minx,self.miny]])
        self.grid2point = defaultdict(list)

    def get_polygon(self):
        return self.geom        

    def grid_bounds(self):
        '''
            Return grid: np.array(),shape = nx,ny -> of polygons
        '''
        nx = int((self.maxx - self.minx)/self.resolution_grid)
        ny = int((self.maxx - self.minx)/self.resolution_grid)
        gx, gy = np.linspace(self.minx,self.maxx,nx), np.linspace(self.miny,self.maxy,ny)
#        self.grid = []
        self.grid = np.empty(shape = (nx,ny),dtype=object)
        for i in range(len(gx)-1):
            for j in range(len(gy)-1):
                poly_ij = Polygon([[gx[i],gy[j]],[gx[i],gy[j+1]],[gx[i+1],gy[j+1]],[gx[i+1],gy[j]]])
                self.grid[i][j] =poly_ij
#                self.grid.append(poly_ij)                 
        return self.grid 

    def partition(self):
        prepared_geom = prep(self.geom)
        self.grid = list(filter(prepared_geom.intersects, self.grid_bounds()))
        return self.grid
    
    def point2grid(self,point):
        '''
            Input: [x,y] coordinates
            Description: 
                Associate those coordinates to the grid in the dictionary
        '''
        found = False
        for i,grid in enumerate(self.grid):
            if grid.contains(Point(point)):
                self.grid2point[i].append(point)
                found = True
        return self.grid2point,found
    
    def contains_vector_points(self,vector_points):
        '''
            Input: np.array([[x1,y1],[x2,y2],...,[xn,yn]])
            Output:
                1) Insert points in the grid
                2) np.array([[x1,y1],[x2,y2],...,[xn,yn]]) with only the points inside the box
        '''
        points_inside_box = []
        for point in vector_points:
            _,found = self.point2grid(point)
            if found:
                points_inside_box.append(point)
            else:
                pass
        
        points_inside_box = np.array(points_inside_box)
        print(points_inside_box,np.shape(points_inside_box))
        return points_inside_box[:][0],points_inside_box[:][1]
    
if __name__ == '__main__':
    SIDE_CITY = 2
    RESOLUTION_GRID = 0.1
    city_box = Grid(SIDE_CITY,RESOLUTION_GRID)
    grid = city_box.partition()
    fig, ax = plt.subplots(figsize=(15, 15))
    gpd.GeoSeries(grid).boundary.plot(ax=ax)
    gpd.GeoSeries([city_box.geom]).boundary.plot(ax=ax,color="red")
    plt.show()


