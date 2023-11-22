from shapely.geometry import Polygon,Point,LineString
import numpy as np
from shapely.prepared import prep
import geopandas as gpd
import matplotlib.pyplot as plt
from collections import defaultdict

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



        
        
