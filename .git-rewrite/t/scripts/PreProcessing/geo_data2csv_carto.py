'''
    Requirement:
        1) Country.tiff -> population file (Taken from https://data.humdata.org/dataset) 
        2) edges.csv, nodes.csv -> graph of the city NOTE: This must be modified in future versions as it is misleading [mobility_planner.py] should take care of all the graph
        2a) GraphFromPhml -> graph of the city
        3) polygons.geojson -> polygons of the city
        4) config.json -> configuration file
    Structure Directory:
        1) data
        2) carto
        3) city
        3a) grid: (Contains Ngridsize folders. Each one a grid geojson and the lattice graphml file)
            3ai) {grid_size_i} -> ['grid.geojson','centroid_lattice.graphml','grid2origindest.json','origindest2grid.json']
        3b) ring: [rings_n_{0}.geojson]
            3bi) {number_rings} -> [rings.geojson,ring2origindest.json,origindest2ring.json]
        3c) hexagon: [hexagon_resolution_{0}.geojson]
            3ci) {resolution} -> [hexagon.geojson,hexagon2origindest.json,origindest2hexagon.json]
        3d) polygon: [city.geojson]
            polygon2origindest.json,origindest2polygon.json
        3e) graph: [edges.csv,nodes.csv,{city}_new_tertiary_simplified.graphml]
        3f) OD: 
            [{city}_oddemand_{start}_{end}_R_{number}.csv]
        3g) maps: [osmid2idx.json,idx2osmid.json]
            []

    OUTPUT COLORS:
        polygon: magenta
        grid: yellow
        ring: blue
        hexagon: green
        graph: red
        od: cyan

'''


import geopandas as gpd
import numpy as np
import os
import pandas as pd
import osmnx as ox
from collections import defaultdict
from shapely.geometry import Point,Polygon
import matplotlib.pyplot as plt
import json
import networkx as nx
import haversine as hs
import shapely as shp
import h3
import shapely.geometry as sg
import rasterio
from rasterio.mask import mask
from termcolor import cprint
import time
from multiprocessing import Pool
import sys
import socket

if socket.gethostname()=='artemis.ist.berkeley.edu':
    TRAFFIC_DIR ='/home/alberto/LPSim/traffic_phase_transition'
else:
    TRAFFIC_DIR = os.getenv('TRAFFIC_DIR')
sys.path.append(os.path.join(TRAFFIC_DIR,'scripts','ServerCommunication'))
sys.path.append(os.path.join(TRAFFIC_DIR,'scripts','GenerationNet'))
sys.path.append(os.path.join(TRAFFIC_DIR,'scripts','GeometrySphere'))
from global_functions import ifnotexistsmkdir
from HostConnection import *
from GeometrySphere import *
from PolygonSettings import *
from plot import *

RUNNING_ON_SERVER = False
SERVER_TRAFFIC_DIR = '/home/alberto/LPSim/LivingCity'
'''
TODO: Create all the files that are needed to handle the Origin and destination in such a way that traffic can be studied with different initial configurations.
Output:
    1) nodes.csv -> NEEDED IN SIMULATION
    2) edges.csv -> NEEDED IN SIMULATION
    3) od_demand_{0}_{1}_R_{2}.csv -> NEEDED IN SIMULATION
    4) polygon2origindest.json -> NEEDED IN OD (to build the OD for all the different tilings in particular for: getPolygonPopulation)
    5) osmid2idx.json -> NEEDED IN OD (to build the OD for all the different tilings in particular for: getPolygonPopulation)
    6) idx2osmid.json -> NEEDED IN OD (to build the OD for all the different tilings in particular for: getPolygonPopulation)
    7) grid.geojson
'''



## GLOBAL FUNCTIONS


def aggregate_population(hexagon, hexagon_transform, clipped_data):
    # Convert hexagon to pixel coordinates
    hexagon_coords = np.array(hexagon.exterior.xy).T
    hexagon_pixel_coords = np.array(rasterio.transform.rowcol(hexagon_transform, hexagon_coords[:, 0], hexagon_coords[:, 1]))
    # Ensure coordinates are within valid bounds
    valid_coords = (
        (0 <= np.array(hexagon_pixel_coords[0])) & (np.array(hexagon_pixel_coords[0]) < clipped_data.shape[1]) &
        (0 <= np.array(hexagon_pixel_coords[1])) & (np.array(hexagon_pixel_coords[1]) < clipped_data.shape[2])
    )

    # Filter valid coordinates
    valid_pixel_coords = hexagon_pixel_coords[:, valid_coords]

    # Extract values from clipped data
    values = clipped_data[0, valid_pixel_coords[0], valid_pixel_coords[1]]

    # Aggregate population data (example: sum)
    population_sum = np.sum(values)
    return population_sum


def GetTotalMovingPopulation(OD_vector):
    return np.sum(OD_vector)
    

class GeometricalSettingsSpatialPartition:
    def __init__(self,city,TRAFFIC_DIR):

        self.crs = 'epsg:4326'
        self.city = city
        # INPUT DIRS
        self.config_dir_local = os.path.join(TRAFFIC_DIR,'config')
        self.tiff_file_dir_local = os.path.join(TRAFFIC_DIR,'data','carto','tiff_files') 
        self.shape_file_dir_local = os.path.join(TRAFFIC_DIR,'data','carto',self.city,'shape_files')
        self.ODfma_dir = os.path.join(TRAFFIC_DIR,'data','carto',self.city,'ODfma')
        # OUTPUT DIRS
        self.save_dir_local = os.path.join(TRAFFIC_DIR,'data','carto',self.city) 
        self.save_dir_server = os.path.join(SERVER_TRAFFIC_DIR,self.city) # DIRECTORY WHERE TO SAVE THE FILES /home/alberto/LPSim/LivingCity/{city}

        if os.path.isfile(os.path.join(self.save_dir_local,self.city + '_new_tertiary_simplified.graphml')):
            self.GraphFromPhml = ox.load_graphml(filepath = os.path.join(self.save_dir_local,self.city + '_new_tertiary_simplified.graphml')) # GRAPHML FILE
        else:
            raise ValueError('Graph City not found')
        if os.path.isfile(os.path.join(self.shape_file_dir_local,self.city + '.shp')):
            self.gdf_polygons = gpd.read_file(os.path.join(self.shape_file_dir_local,self.city + '.shp')) # POLYGON FILE

        else:
            raise ValueError('Polygon City not found')
        self.nodes = None
        self.edges = None
        self.osmid2index = defaultdict()
        self.index2osmid = defaultdict()
        self.polygon2OD = defaultdict(list)
        self.start = 7
        self.end = 8
        self.R = 1
        self.print_info()
        self.Files2Upload = defaultdict(list)
        self.Files2Upload[os.path.join(self.save_dir_local,self.city + '_new_tertiary_simplified.graphml')] = os.path.join(self.save_dir_server,self.city + '_new_tertiary_simplified.graphml')
        self.Files2Upload[os.path.join(self.tiff_file_dir_local,'Country.tiff')] = os.path.join(self.save_dir_server,'Country.tiff')

##-------------------------------- PRINT --------------------------------##
    def print_info(self):
        cprint('city: ' + self.city,'red')


##------------------------------------------------ TILING ------------------------------------------------##
    
    def get_squared_grid(self,grid_size):
        '''
            centroid: Point -> centroid of the city
            bounding_box: tuple -> (minx,miny,maxx,maxy)
            grid: GeoDataFrame -> grid of points of size grid_size
            In this way grid is ready to be used as the matrix representation of the city and the gradient and the curl defined on it.
            From now on I will have that the lattice is associated to the centroid grid.
            Usage:
                grid and lattice are together containing spatial and network information
        '''
        
        if 1 == 1:
            cprint('Initialize Grid: ' + str(grid_size),'yellow')
            self.grid_size = grid_size
            ifnotexistsmkdir(os.path.join(self.save_dir_local,'grid'))
            ifnotexistsmkdir(os.path.join(self.save_dir_local,'grid',str(self.grid_size)))
            self.dir_grid = os.path.join(self.save_dir_local,'grid')
            t0 = time.time()
            if os.path.isfile(os.path.join(self.dir_grid,self.grid_size,"grid.geojson")):
                cprint('ALREADY COMPUTED'.format(os.path.join(self.dir_grid,self.grid_size,"grid.geojson")),'yellow')
                self.grid = gpd.read_file(os.path.join(self.dir_grid,self.grid_size,"grid.geojson"))
                self.Files2Upload[os.path.join(self.dir_grid,self.grid_size,"grid.geojson")] = os.path.join(self.save_dir_server,'grid',self.grid_size,"grid.geojson")
                self.centroid = self.gdf_polygons.geometry.unary_union.centroid
                self.bounding_box = self.gdf_polygons.geometry.unary_union.bounds
                self.grid_size = grid_size
                bbox = shp.geometry.box(*self.bounding_box)
                bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=self.crs)
                x = np.arange(self.bounding_box[0], self.bounding_box[2], grid_size)
                y = np.arange(self.bounding_box[1], self.bounding_box[3], grid_size)

            else:
                cprint('COMPUTING {}'.format(os.path.join(self.dir_grid,self.grid_size,"grid.geojson")),'green')
                self.centroid = self.gdf_polygons.geometry.unary_union.centroid
                self.bounding_box = self.gdf_polygons.geometry.unary_union.bounds
                self.grid_size = grid_size
                bbox = shp.geometry.box(*self.bounding_box)
                bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=self.crs)
                x = np.arange(self.bounding_box[0], self.bounding_box[2], grid_size)
                y = np.arange(self.bounding_box[1], self.bounding_box[3], grid_size)
                grid_points = gpd.GeoDataFrame(geometry=[shp.geometry.box(xi, yi,maxx = max(x),maxy = max(y)) for xi in x for yi in y], crs=self.crs)
                ij = [[i,j] for i in range(len(x)) for j in range(len(y))]
                grid_points['i'] = np.array(ij)[:,0]
                grid_points['j'] = np.array(ij)[:,1]
                # Clip the grid to the bounding box
                self.grid = gpd.overlay(grid_points, bbox_gdf, how='intersection')
                self.grid['centroidx'] = self.grid.geometry.centroid.x
                self.grid['centroidy'] = self.grid.geometry.centroid.y                
                self.grid['area'] = self.grid['geometry'].apply(ComputeAreaSquare)
                if self.resolution == 8:
                    self.getGridPopulation()
                self.grid['density_population'] = self.grid['population']/self.grid['area']
                self.grid.to_file(os.path.join(self.dir_grid,self.grid_size,"grid.geojson"), driver="GeoJSON")
                self.Files2Upload[os.path.join(self.dir_grid,self.grid_size,"grid.geojson")] = os.path.join(self.save_dir_server,'grid',self.grid_size,'grid_size.geojson')
            # LATTICE
            self.get_lattice()
            plot_grid_tiling(self.grid,self.gdf_polygons,self.save_dir_local,self.grid_size)
            t1 = time.time()
            cprint('time to compute the grid: ' + str(t1-t0),'yellow')
        else:
            pass    
    def get_lattice(self):
        '''
            Output:
                lattice: graph -> graph object of the lattice
            Description:
                This function is used to get the lattice of the city, it is a graph object that contains the nodes and the edges of the city.
                It is used to compute the gradient and the curl of the city.
        '''
        ## BUILD GRAPH OBJECT GRID
        cprint('get_lattice','yellow')
        x = np.arange(self.bounding_box[0], self.bounding_box[2], self.grid_size)
        y = np.arange(self.bounding_box[1], self.bounding_box[3], self.grid_size)

        t0 = time.time()
        if os.path.isfile(os.path.join(self.dir_grid,self.grid_size,"centroid_lattice.graphml")):
            cprint('{} ALREADY COMPUTED'.format(os.path.join(self.dir_grid,self.grid_size,"centroid_lattice.graphml")),'yellow')
            self.lattice = nx.read_graphml(os.path.join(self.dir_grid,self.grid_size,"centroid_lattice.graphml"))
        else:
            cprint('COMPUTING {}'.format(os.path.join(self.dir_grid,self.grid_size,"centroid_lattice.graphml")),'yellow')
            self.lattice = nx.grid_2d_graph(len(x),len(y))
            node_positions = {(row['i'],row['j']): {'x': row['centroidx'],'y':row['centroidy']} for idx, row in self.grid.iterrows()}
            # Add position attributes to nodes
            nx.set_node_attributes(self.lattice, node_positions)
            c = 0
            for node in self.lattice.nodes():
                c +=1
                if c ==2:
                    break
            for edge in self.lattice.edges():
                try:
                    dx,dy = ProjCoordsTangentSpace(self.lattice.nodes[edge[1]]['x'],self.lattice.nodes[edge[1]]['y'],self.lattice.nodes[edge[0]]['x'],self.lattice.nodes[edge[0]]['y'])
                    self.lattice[edge[0]][edge[1]]['dx'] = dx
                    self.lattice[edge[0]][edge[1]]['dy'] = dy  
                    self.lattice[edge[0]][edge[1]]['distance'] = hs.haversine((self.lattice.nodes[edge[0]]['y'],self.lattice.nodes[edge[0]]['x']),(self.lattice.nodes[edge[1]]['y'],self.lattice.nodes[edge[1]]['x']))
                    self.lattice[edge[0]][edge[1]]['angle'] = np.arctan2(self.lattice[edge[0]][edge[1]]['dy'],self.lattice[edge[0]][edge[1]]['dx'])
                    self.lattice[edge[0]][edge[1]]['d/dx'] = 1/dx  
                    self.lattice[edge[0]][edge[1]]['d/dy'] = 1/dy
                except KeyError:
                    pass
            ## SAVE GRID AND LATTICE
            nx.write_graphml(self.lattice, os.path.join(self.dir_grid,self.grid_size,"centroid_lattice.graphml"))  
            self.Files2Upload[os.path.join(self.dir_grid,self.grid_size,"centroid_lattice.graphml")] = os.path.join(self.save_dir_server,'grid',self.grid_size,"centroid_lattice.graphml".format(self.grid_size))  
        t1 = time.time()
        cprint('time to compute the lattice: ' + str(t1-t0),'yellow')


    def get_rings(self,number_of_rings):
        '''
            Compute the rings of the city and the intersection with polygons
            rings: dict -> {idx:ring}
        '''
        if 1 == 1:
            cprint('get_rings: ' + str(number_of_rings),'blue')
            t0 = time.time()
            self.rings = defaultdict(list)
            self.number_of_rings = number_of_rings
            gdf_original_crs = gpd.GeoDataFrame(geometry=[self.centroid], crs=self.crs)
            self.radius = max([abs(self.bounding_box[0] -self.bounding_box[2])/2,abs(self.bounding_box[1] - self.bounding_box[3])/2]) 
            self.radiuses = np.linspace(0,self.radius,self.number_of_rings)
            ifnotexistsmkdir(os.path.join(self.save_dir_local,'ring'))
            ifnotexistsmkdir(os.path.join(self.save_dir_local,'ring',str(self.number_of_rings)))
            self.dir_rings = os.path.join(self.save_dir_local,'ring')
            if os.path.isfile(os.path.join(self.save_dir_local,self.number_of_rings,'ring','rings.geojson')):
                cprint('{} ALREADY COMPUTED'.format(os.path.join(self.save_dir_local,self.number_of_rings,'ring','rings.geojson')),'blue')
                self.rings = gpd.read_file(os.path.join(self.save_dir_local,self.number_of_rings,'ring','rings.geojson'))
                self.Files2Upload[os.path.join(self.save_dir_local,self.number_of_rings,'ring','rings.geojson')] = os.path.join(self.save_dir_server,'ring',self.number_of_rings,'rings.geojson')
                if self.resolution == 8:
                    self.getRingPopulation()
            else:

                for i,r in enumerate(self.radiuses):
                    if i == 0:
                        intersection_ = gdf_original_crs.buffer(r)
                        self.rings[i] = intersection_
                    else:
                        intersection_ = gdf_original_crs.buffer(r).intersection(gdf_original_crs.buffer(self.radiuses[i-1]))
                        complement = gdf_original_crs.buffer(r).difference(intersection_)
                        self.rings[i] = complement
                    self.rings = gpd.GeoDataFrame(geometry=pd.concat(list(self.rings.values()), ignore_index=True),crs=self.crs)
                    ## SAVE RINGS
                    if self.resolution == 8:
                        self.getRingPopulation()
                    self.rings.to_file(os.path.join(self.save_dir_local,self.number_of_rings,'ring','rings.geojson'), driver="GeoJSON")  
                    self.Files2Upload[os.path.join(self.save_dir_local,self.number_of_rings,'ring','rings.geojson')] = os.path.join(self.save_dir_server,'ring',self.number_of_rings,'rings.geojson')
                    plot_ring_tiling(self.rings,self.save_dir_local,self.radiuses,r)
            t1 = time.time()
            cprint('time to compute the rings: ' + str(t1-t0),'red')
        else:
            pass    
    
    def get_hexagon_tiling(self,resolution=8):
        '''
            This function is used to get the hexagon tiling of the area, it can be used just for US right now as we have just
            tif population file just for that
        '''
        ## READ TIF FILE
        cprint('get_hexagon_tiling','green')
        self.resolution = resolution
        ifnotexistsmkdir(os.path.join(self.save_dir_local,'hexagon'))
        ifnotexistsmkdir(os.path.join(self.save_dir_local,'hexagon',str(self.resolution)))
        self.dir_hexagons = os.path.join(self.save_dir_local,'hexagon')
        if not os.path.exists(os.path.join(self.save_dir_local,'hexagon',self.resolution,'hexagon.geojson')):
            cprint('COMPUTING: {} '.format(os.path.join(self.save_dir_local,'hexagon',self.resolution,'hexagon.geojson')),'green')
            with rasterio.open(self.tif_file) as dataset:
                clipped_data, clipped_transform = mask(dataset, self.gdf_polygons.geometry, crop=True)
            ## CHANGE NULL ENTRANCIES (-99999) for US (may change for other Countries [written in United Nation page of Download])
            clipped_data = np.array(clipped_data)
#            print('resolution: ',resolution)
#            print('clipped_data: ',np.shape(clipped_data))
#            print('clipped_transform: ',np.shape(clipped_transform))
            condition = clipped_data<0
            clipped_data[condition] = 0
            # Define hexagon resolution
            bay_area_geometry = self.gdf_polygons.unary_union
#            print('bay_area_geometry: ',type(bay_area_geometry))
            # Convert MultiPolygon to a single Polygon
            bay_area_polygon = bay_area_geometry.convex_hull
            # Convert Polygon to a GeoJSON-like dictionary
            bay_area_geojson = sg.mapping(bay_area_polygon)
#            print('bay_area_geojson: ',type(bay_area_geojson))
            # Get hexagons within the bay area
            hexagons = h3.polyfill(bay_area_geojson, resolution, geo_json_conformant=True)
#            print('hexagons: ',type(hexagons))
            # Convert hexagons to Shapely geometries
            hexagon_geometries = [sg.Polygon(h3.h3_to_geo_boundary(h, geo_json=True)) for h in hexagons]
#            print('hexagon_geometries: ',np.shape(hexagon_geometries),' type: ',type(hexagon_geometries))

            # Aggregate population data for each hexagon
            population_data_hexagons = [aggregate_population(hexagon, clipped_transform, clipped_data) for hexagon in hexagon_geometries]
#            print('population_data_hexagons: ',np.shape(population_data_hexagons))
            centroid_hexagons = [h.centroid for h in hexagon_geometries]
            centroidx = [h.centroid.x for h in hexagon_geometries]
            centroidy = [h.centroid.y for h in hexagon_geometries]
            # Create GeoDataFrame
#            print('len hexagon_geometries: ',centroid_hexagons)
            self.gdf_hexagons = gpd.GeoDataFrame(geometry=hexagon_geometries, data={'population':population_data_hexagons,'centroid_x':centroidx,'centroid_y':centroidy},crs = self.crs)
            self.gdf_hexagons.reset_index(inplace=True)
        else:
            cprint('{} ALREADY COMPUTED'.format(os.path.join(self.save_dir_local,'hexagon',self.resolution,'hexagon.geojson')),'green')
            self.gdf_hexagons = gpd.read_file(os.path.join(self.save_dir_local,'hexagon',self.resolution,'hexagon.geojson'))
        if self.resolution == 8:
            self.getPolygonPopulation()
#        cprint('columns of the hexagons: ','green')
#        for col in self.gdf_hexagons.columns:
#            cprint(col,'green')
        self.gdf_hexagons['area'] = self.gdf_hexagons.to_crs({'proj':'cea'}).area / 10**6
        self.gdf_hexagons['density_population'] = self.gdf_hexagons['population']/self.gdf_hexagons['area']        
        self.gdf_hexagons.to_file(os.path.join(self.save_dir_local,'hexagon',self.resolution,'hexagon.geojson'), driver="GeoJSON")
        self.Files2Upload[os.path.join(self.save_dir_local,'hexagon',self.resolution,'hexagon.geojson')] = os.path.join(self.save_dir_server,'hexagon',self.resolution,'hexagon.geojson')
        plot_hexagon_tiling(self.gdf_hexagons,self.gdf_polygons,self.save_dir_local,self.resolution)

## ------------------------------------------------ MAP TILE TO POLYGONS ------------------------------------------------ ##
    def getPolygonPopulation(self):
        '''
            Consider just the hexagons of the tiling that have population > 0
            Any time we have that a polygon is intersected by the hexagon, we add to the population column
            of the polygon the population of the hexagon times the ratio of intersection area with respect to the hexagon area
        '''
        cprint('getPolygonPopulation {}'.format(self.city),'green')
        if self.gdf_hexagons is None:
            raise ValueError('grid is None')
        elif self.gdf_polygons is None:
            raise ValueError('gdf_polygons is None')
        else:
            polygon_sindex = self.gdf_polygons.sindex
            populationpolygon = np.zeros(len(self.gdf_polygons))
            for idxh,hex in self.gdf_hexagons.loc[self.gdf_hexagons['population'] > 0].iterrows():
                possible_matches_index = list(polygon_sindex.intersection(hex.geometry.bounds))
                possible_matches = self.gdf_polygons.iloc[possible_matches_index]    
                # Filter based on actual intersection
                if len(possible_matches) > 0:
                    intersecting = possible_matches[possible_matches.geometry.intersects(hex.geometry)]
                    for idxint,int_ in intersecting.iterrows():
                        populationpolygon[idxint] += hex.population*int_.geometry.intersection(hex.geometry).area/hex.geometry.area        
                        if int_.geometry.intersection(hex.geometry).area/hex.geometry.area>1.01:
                            raise ValueError('Error the area of the intersection is greater than 1: ',int_.geometry.intersection(hex.geometry).area/hex.geometry.area)
                else:
                    pass
                self.gdf_polygons['population'] = populationpolygon

    def getGridPopulation(self):
        '''
            Associates the mass to the grid 
        '''
        cprint('getGridPopulation {}'.format(self.city),'yellow')
        if self.grid is None:
            raise ValueError('grid is None')
        elif self.gdf_polygons is None:
            raise ValueError('gdf_polygons is None')
        else:
            t0 = time.time()
            grid_sindex = self.grid.sindex
            populationpolygon = np.zeros(len(self.grid))
            for idxh,hex in self.gdf_hexagons.loc[self.gdf_hexagons['population'] > 0].iterrows():
                possible_matches_index = list(grid_sindex.intersection(hex.geometry.bounds))
                possible_matches = self.grid.iloc[possible_matches_index]    
                # Filter based on actual intersection
                if len(possible_matches) > 0:
                    intersecting = possible_matches[possible_matches.geometry.intersects(hex.geometry)]
                    for idxint,int_ in intersecting.iterrows():
                        populationpolygon[idxint] += hex.population*int_.geometry.intersection(hex.geometry).area/hex.geometry.area        
                        if int_.geometry.intersection(hex.geometry).area/hex.geometry.area>1.01:
                            raise ValueError('Error the area of the intersection is greater than 1: ',int_.geometry.intersection(hex.geometry).area/hex.geometry.area)
                else:
                    pass
                self.grid['population'] = populationpolygon
            t1 = time.time()
            cprint('time to compute the intersection: ' + str(t1-t0),'green')

    def getRingPopulation(self):
        '''
            Gives the population to the rings
        '''
        if self.rings is None:
            raise ValueError('rings is None')
        elif self.gdf_polygons is None:
            raise ValueError('gdf_polygons is None')
        else:
            cprint('getRingPopulation {}'.format(self.city),'green')
            ring_sindex = self.rings.sindex
            populationpolygon = np.zeros(len(self.rings))
            for idxh,hex in self.gdf_hexagons.loc[self.gdf_hexagons['population'] > 0].iterrows():
                possible_matches_index = list(ring_sindex.intersection(hex.geometry.bounds))
                possible_matches = self.grid.iloc[possible_matches_index]    
                # Filter based on actual intersection
                if len(possible_matches) > 0:
                    intersecting = possible_matches[possible_matches.geometry.intersects(hex.geometry)]
                    for idxint,int_ in intersecting.iterrows():
                        populationpolygon[idxint] += hex.population*int_.geometry.intersection(hex.geometry).area/hex.geometry.area        
                        if int_.geometry.intersection(hex.geometry).area/hex.geometry.area>1.01:
                            raise ValueError('Error the area of the intersection is greater than 1: ',int_.geometry.intersection(hex.geometry).area/hex.geometry.area)
                else:
                    pass
                self.rings['population'] = populationpolygon

##-------------------------------------------- EDGE FILES --------------------------------------------##
    def adjust_edges(self):
        '''
            If The edges file has got columns u,v that are osmid, replaces them with the index
            If The edges file has got columns u,v that are index, creates osmid_u and osmid_v
            If both the columns are already there does nothing
        '''
        cprint('Adjust edges file','green')
        try:
            self.edges = pd.read_csv(os.path.join(self.save_dir_local,'edges.csv'))
            self.edges['u'] = self.edges['u'].apply(lambda x: self.osmid2index[x])
            self.edges['v'] = self.edges['v'].apply(lambda x: self.osmid2index[x])
            self.edges.to_csv(os.path.join(self.save_dir_local,'edges.csv'),index=False)
        except KeyError:
            cprint('edges.csv ALREADY COMPUTED','green')
            try:
                self.edges['osmid_u'] = self.edges['u'].apply(lambda x: self.index2osmid[x])
                self.edges['osmid_v'] = self.edges['v'].apply(lambda x: self.index2osmid[x])
                self.edges.to_csv(os.path.join(self.save_dir_local,'edges.csv'),index=False)
            except KeyError:
                cprint('edges.csv HAS GOT ALREADY, [u,v,osmid_u,osmid_v]','green')
                pass

##------------------------------------- OD -------------------------------------##
    def grid2origindest(self):
        '''
            Given a network taken from the cartography or ours:
                Build the set of origin and destinations from the grid that are coming from the 
                geodataframe.
            
        '''
        cprint('grid2origindest','yellow')
        t0 = time.time()
        if os.path.isfile(os.path.join(self.dir_grid,self.grid_size,'grid2origindest.json')) and os.path.isfile(os.path.join(self.dir_grid,self.grid_size,'origindest2grid.json')):
            cprint('{} ALREADY COMPUTED'.format(os.path.join(self.dir_grid,self.grid_size,'grid2origindest.json')),'green')
            with open(os.path.join(self.dir_grid,self.grid_size,'grid2origindest.json'),'r') as f:
                self.Grid2OD = json.load(f)
            self.Files2Upload[os.path.join(self.dir_grid,self.grid_size,'grid2origindest.json')] = os.path.join(self.save_dir_server,'grid',self.grid_size,'grid2origindest.json')
            with open(os.path.join(self.dir_grid,self.grid_size,'origindest2grid.json'),'r') as f:
                self.OD2Grid = json.load(f)
            self.Files2Upload[os.path.join(self.dir_grid,self.grid_size,'origindest2grid.json')] = os.path.join(self.save_dir_server,'grid',self.grid_size,'origindest2grid.json')
        else:
            n_nodes = np.zeros(len(self.grid))
            grid2origindest = defaultdict(list)
            origindest2grid = defaultdict(list)
            for node in self.GraphFromPhml.nodes():
                containing_grid = self.grid.geometry.apply(lambda x: Point(self.GraphFromPhml.nodes[node]['x'],self.GraphFromPhml.nodes[node]['y']).within(x))
                idx_containing_grid = self.grid[containing_grid].index
                tract_id_grid = self.grid.loc[idx_containing_grid]['index']
                if len(idx_containing_grid)==1: 
                    try:
                        tract_id_grid = int(tract_id_grid.tolist()[1])
                    except IndexError:
                        tract_id_grid = int(tract_id_grid.tolist()[0])
                    grid2origindest[int(tract_id_grid)].append(node)
                    origindest2grid[node] = int(tract_id_grid)
                    n_nodes[int(tract_id_grid)] = len(grid2origindest[int(tract_id_grid)])
                else:
                    pass
            self.grid.loc['number_nodes'] = n_nodes
            self.Grid2OD = grid2origindest
            self.OD2Grid = origindest2grid
            with open(os.path.join(self.dir_grid,self.grid_size,'grid2origindest.json'),'w') as f:
                json.dump(self.Grid2OD,f,indent=4)
            self.Files2Upload[os.path.join(self.dir_grid,self.grid_size,'grid2origindest.json')] = os.path.join(self.save_dir_server,'grid',self.grid_size,'grid2origindest.json')
            with open(os.path.join(self.dir_grid,self.grid_size,'origindest2grid.json'),'w') as f:
                json.dump(self.OD2Grid,f,indent=4)
            self.Files2Upload[os.path.join(self.dir_grid,self.grid_size,'origindest2grid.json')] = os.path.join(self.save_dir_server,'grid',self.grid_size,'origindest2grid.json')
        HistoPoint2Geometry(self.Grid2OD,'grid',self.dir_grid,resolution = self.grid_size)
        t1 = time.time()
        cprint('time to compute grid2origindest: ' + str(t1-t0),'yellow')

    def hexagon2origindest(self):
        '''
            Given a network taken from the cartography or ours:
                Build the set of origin and destinations from the hexagon that are coming from the 
                geodataframe.
            
        '''
        cprint('{} hexagon2origindest'.format(self.city),'red')
        t0 = time.time()
        if os.path.isfile(os.path.join(self.dir_hexagons,self.resolution,'hexagon2origindest.json')) and os.path.isfile(os.path.join(self.dir_hexagons,self.resolution,'origindest2hexagon.json')):
            cprint('{} ALREADY COMPUTED'.format(os.path.join(self.dir_hexagons,self.resolution,'hexagon2origindest.json')),'green')
            with open(os.path.join(self.dir_hexagons,self.resolution,'hexagon2origindest.json'),'r') as f:
                self.Hex2OD = json.load(f)
            self.Files2Upload[os.path.join(self.dir_hexagons,self.resolution,'hexagon2origindest.json')] = os.path.join(self.save_dir_server,'hexagon',self.resolution,'hexagon2origindest.json')
            with open(os.path.join(self.dir_hexagons,self.resolution,'origindest2hexagon.json'),'r') as f:
                self.OD2Hex = json.load(f)
            self.Files2Upload[os.path.join(self.dir_hexagons,self.resolution,'origindest2hexagon.json')] = os.path.join(self.save_dir_server,'hexagon',self.resolution,'origindest2hexagon.json')
        else:
            cprint('COMPUTING {} '.format(os.path.join(self.dir_hexagons,self.resolution,'hexagon2origindest.json')),'green')
            n_nodes = np.zeros(len(self.gdf_hexagons))
            hexagon2origindest = defaultdict(list)
            origindest2hexagon = defaultdict(list)
            for node in self.GraphFromPhml.nodes():
                containing_hexagon = self.gdf_hexagons.geometry.apply(lambda x: Point(self.GraphFromPhml.nodes[node]['x'],self.GraphFromPhml.nodes[node]['y']).within(x))
                idx_containing_hexagon = self.gdf_hexagons[containing_hexagon].index
                tract_id_hexagon = self.gdf_hexagons.loc[idx_containing_hexagon]['index']
                if len(idx_containing_hexagon)==1: 
                    try:
                        tract_id_hexagon = int(tract_id_hexagon.tolist()[1])
                    except IndexError:
                        tract_id_hexagon = int(tract_id_hexagon.tolist()[0])
                    hexagon2origindest[int(tract_id_hexagon)].append(node)
                    origindest2hexagon[node] = int(tract_id_hexagon)
                    n_nodes[int(tract_id_hexagon)] = len(hexagon2origindest[int(tract_id_hexagon)])
                else:
                    pass
            self.gdf_hexagons['number_nodes'] = list(n_nodes)
            self.Hex2OD = hexagon2origindest
            self.OD2Hex = origindest2hexagon
            with open(os.path.join(self.dir_hexagons,self.resolution,'hexagon2origindest.json'),'w') as f:
                json.dump(self.Hex2OD,f,indent=4)
            self.Files2Upload[os.path.join(self.dir_hexagons,self.resolution,'hexagon2origindest.json')] = os.path.join(self.save_dir_server,'hexagon',self.resolution,'hexagon2origindest.json')
            with open(os.path.join(self.dir_hexagons,self.resolution,'origindest2hexagon.json'),'w') as f:
                json.dump(self.OD2Hex,f,indent=4)
            self.Files2Upload[os.path.join(self.dir_hexagons,self.resolution,'origindest2hexagon.json')] = os.path.join(self.save_dir_server,'hexagon',self.resolution,'origindest2hexagon.json')
        HistoPoint2Geometry(self.Hex2OD,'hexagon',self.dir_hexagons,resolution = self.resolution)
        t1 = time.time()
        cprint('time to compute hexagon2origindest: ' + str(t1-t0),'red')
    
    def rings2origindest(self):
        '''
            Given a network taken from the cartography or ours:
                Build the set of origin and destinations from the rings that are coming from the 
                geodataframe.
            
        '''
        cprint('rings2origindest','blue')
        t0 = time.time()
        if os.path.isfile(os.path.join(self.save_dir_local,'grid',self.radius,'rings2origindest.json')) and os.path.isfile(os.path.join(self.save_dir_local,'grid',self.radius,'origindest2rings.json')):
            cprint('{} ALREADY COMPUTED'.format(os.path.join(self.save_dir_local,'grid',self.radius,'rings2origindest.json')),'blue')
            with open(os.path.join(self.save_dir_local,'grid',self.radius,'rings2origindest.json'),'r') as f:
                self.Rings2OD = json.load(f)
            self.Files2Upload[os.path.join(self.save_dir_local,'grid',self.radius,'rings2origindest.json')] = os.path.join(self.save_dir_server,'grid','rings2origindest_radius_{}.json'.format(self.radius))
            with open(os.path.join(self.save_dir_local,'grid',self.radius,'origindest2rings.json'),'r') as f:
                self.OD2Rings = json.load(f)
            self.Files2Upload[os.path.join(self.save_dir_local,'grid',self.radius,'origindest2rings.json')] = os.path.join(self.save_dir_server,'grid','origindest2rings_radius_{}.json'.format(self.radius))
        else:
            cprint('COMPUTING {}'.format(os.path.join(self.save_dir_local,'grid',self.radius,'rings2origindest.json')),'blue')
            n_nodes = np.zeros(len(self.rings))
            rings2origindest = defaultdict(list)
            origindest2rings = defaultdict(list)
            for node in self.GraphFromPhml.nodes():
                containing_ring = self.rings.geometry.apply(lambda x: Point(self.GraphFromPhml.nodes[node]['x'],self.GraphFromPhml.nodes[node]['y']).within(x))
                idx_containing_ring = self.rings[containing_ring].index
                tract_id_ring = self.rings.loc[idx_containing_ring]['index']
                if len(idx_containing_ring)==1: 
                    try:
                        tract_id_ring = int(tract_id_ring.tolist()[1])
                    except IndexError:
                        tract_id_ring = int(tract_id_ring.tolist()[0])
                    rings2origindest[int(tract_id_ring)].append(node)
                    origindest2rings[node] = int(tract_id_ring)
                    n_nodes[int(tract_id_ring)] = len(rings2origindest[int(tract_id_ring)])
                else:
                    pass
            self.rings['number_nodes'] = list(n_nodes)
            self.Rings2OD = rings2origindest
            self.OD2Rings = origindest2rings
            with open(os.path.join(self.save_dir_local,'grid',self.radius,'rings2origindest.json'),'w') as f:
                json.dump(self.Rings2OD,f,indent=4)
            self.Files2Upload[os.path.join(self.save_dir_local,'grid',self.radius,'rings2origindest.json')] = os.path.join(self.save_dir_server,'grid','rings2origindest_radius_{}.json'.format(self.radius))
            with open(os.path.join(self.save_dir_local,'grid',self.radius,'origindest2rings.json'),'w') as f:
                json.dump(self.OD2Rings,f,indent=4)
            self.Files2Upload[os.path.join(self.save_dir_local,'grid',self.radius,'origindest2rings.json')] = os.path.join(self.save_dir_server,'grid','origindest2rings_radius_{}.json'.format(self.radius))
        HistoPoint2Geometry(self.Rings2OD,'ring',self.dir_rings,resolution = self.radius)
        t1 = time.time()
        cprint('time to compute rings2origindest: ' + str(t1-t0),'blue')

    def polygon2origindest(self,debug = False):
        '''
            Given a network taken from the cartography or ours:
                Build the set of origin and destinations from the polygons that are coming from the 
                geodataframe.
            
        '''
        t0 = time.time()
        if os.path.isfile(os.path.join(self.save_dir_local,'polygon','polygon2origindest.json')) and os.path.isfile(os.path.join(self.save_dir_local,'polygon','origindest2polygon.json')):
            cprint('{} ALREADY COMPUTED'.format(os.path.join(self.save_dir_local,'polygon','polygon2origindest.json')),'magenta')
            with open(os.path.join(self.save_dir_local,'polygon','polygon2origindest.json'),'r') as f:
                self.polygon2OD = json.load(f)
            self.Files2Upload[os.path.join(self.save_dir_local,'polygon','polygon2origindest.json')] = os.path.join(self.save_dir_server,'polygon','polygon2origindest.json')
            with open(os.path.join(self.save_dir_local,'polygon','origindest2polygon.json'),'r') as f:
                self.OD2polygon = json.load(f)
            self.Files2Upload[os.path.join(self.save_dir_local,'polygon','origindest2polygon.json')] = os.path.join(self.save_dir_server,'polygon','origindest2polygon.json')            
            if self.city == 'SFO':
                key = 'TRACT'
                self.tif_file =  os.path.join(self.tiff_file_dir_local,'usa_ppp_2020_UNadj_constrained.tif')
            if self.city == 'LAX':
                key = 'external_i'
                self.tif_file =  os.path.join(self.tiff_file_dir_local,'usa_ppp_2020_UNadj_constrained.tif')
            if self.city == 'LIS':
                key = 'ID'
                self.tif_file =  os.path.join(self.tiff_file_dir_local,'prt_ppp_2020_UNadj_constrained.tif')
            if self.city == 'RIO':
                key = 'Zona'
                self.tif_file =  os.path.join(self.tiff_file_dir_local,'bra_ppp_2020_UNadj_constrained.tif')

            if self.city == 'BOS':
                key = 'tractid'
                self.tif_file =  os.path.join(self.tiff_file_dir_local,'usa_ppp_2020_UNadj_constrained.tif')
        else:
            cprint('COMPUTING {}'.format(os.path.join(self.save_dir_local,'polygon','polygon2origindest.json')),'magenta')
            polygon2origindest = defaultdict(list)
            origindest2polygon = defaultdict(list)
            for node in self.GraphFromPhml.nodes():
                containing_polygon = self.gdf_polygons.geometry.apply(lambda x: Point(self.GraphFromPhml.nodes[node]['x'],self.GraphFromPhml.nodes[node]['y']).within(x))
                idx_containing_polygon = self.gdf_polygons[containing_polygon].index
        #        print('idx_containing_polygon: ',idx_containing_polygon)
                if self.city == 'SFO':
                    key = 'TRACT'
                    self.tif_file =  os.path.join(self.tiff_file_dir_local,'usa_ppp_2020_UNadj_constrained.tif')
                if self.city == 'LAX':
                    key = 'external_i'
                    self.tif_file =  os.path.join(self.tiff_file_dir_local,'usa_ppp_2020_UNadj_constrained.tif')
                if self.city == 'LIS':
                    key = 'ID'
                    self.tif_file =  os.path.join(self.tiff_file_dir_local,'prt_ppp_2020_UNadj_constrained.tif')
                if self.city == 'RIO':
                    key = 'Zona'
                    self.tif_file =  os.path.join(self.tiff_file_dir_local,'bra_ppp_2020_UNadj_constrained.tif')

                if self.city == 'BOS':
                    key = 'tractid'
                    self.tif_file =  os.path.join(self.tiff_file_dir_local,'usa_ppp_2020_UNadj_constrained.tif')
                tract_id_polygon = self.gdf_polygons.loc[idx_containing_polygon][key]
                if len(tract_id_polygon)==1: 
                    try:
                        tract_id_polygon = int(tract_id_polygon.tolist()[1])
                    except IndexError:
                        tract_id_polygon = int(tract_id_polygon.tolist()[0])
        #            print('tract_id: ',tract_id)
                    polygon2origindest[int(tract_id_polygon)].append(node)
                    origindest2polygon[node] = int(tract_id_polygon)
                    if debug:
                        print('found polygon: ',idx_containing_polygon)
                        print('tract_id: ',tract_id_polygon)
                        print('type tract_id: ',type(tract_id_polygon))
                        print('values dict: ',polygon2origindest[tract_id_polygon])
                        print('dict: ',polygon2origindest)
                elif len(tract_id_polygon)>1:
                    print('tract_id: ',tract_id_polygon)
                    print('more than one tract id: THIS IS STRANGE')
                else:
                    pass

            self.polygon2OD = polygon2origindest
            self.OD2polygon = origindest2polygon
            with open(os.path.join(self.save_dir_local,'polygon','polygon2origindest.json'),'w') as f:
                json.dump(self.polygon2OD,f,indent=4)
            self.Files2Upload[os.path.join(self.save_dir_local,'polygon','polygon2origindest.json')] = os.path.join(self.save_dir_server,'polygon','polygon2origindest.json')
            with open(os.path.join(self.save_dir_local,'polygon','origindest2polygon.json'),'w') as f:
                json.dump(self.OD2polygon,f,indent=4)
            self.Files2Upload[os.path.join(self.save_dir_local,'polygon','origindest2polygon.json')] = os.path.join(self.save_dir_server,'polygon','origindest2polygon.json')
        HistoPoint2Geometry(self.polygon2OD,'polygon',self.save_dir_local)
        self.gdf_polygons.rename(columns = {key:'index'})
        t1 = time.time()
#        cprint('time to compute polygon2origindest: ' + str(t1-t0),'red')
#        print('columns of the gdf_polygons: ',self.gdf_polygons.columns)
#        print('columns of the gdf_polygons: ',self.gdf_polygons)
#        self.gdf_polygons['index'] = self.gdf_polygons[key]
#        self.gdf_polygons['index'] = self.gdf_polygons['index'].astype(int)
         



## ---------------------------------- OD FROM FILE---------------------------------- ##
    def OD_from_fma(self,
                    file,
                    number_of_rings,
                    grid_sizes,
                    resolutions,
                    offset = 6,
                    seconds_in_minute = 60,
                    ):
        '''
            Each fma file contains the origin and destinations with the rate of people entering the graph.
            This function, takes advantage of the polygon2origindest dictionary to build the origin and destination
            selecting at random one of the nodes that are contained in the polygon.
        '''
        t0 = time.time()
        PRINTING_INTERVAL = 10000000
        O_vector = []
        D_vector = []
        OD_vector = []
        self.total_number_people_not_considered = 0
        self.total_number_people_considered = 0    
        # THESE POINTS ARE NATURALLY ON THE POLYGONS        
        with open(file,'r') as infile:
            count_line = 0
            for line in infile:
                count_line += 1
                if count_line > offset:
                    tok = line.split(' ')
                    O_vector.append(int(tok[0]))
                    D_vector.append(int(tok[1]))
                    OD_vector.append(int(float(tok[2].split('\n')[0])))
        # START TAB
        cprint('OD_from_fma {} '.format(self.city) + file,'cyan')
        self.R = GetTotalMovingPopulation(OD_vector)/3600 # R is the number of people that move in one second (that is the time interval for the evolution )
        if self.city == 'SFO':
            Rmin = 145
            Rmax = 180 
        elif self.city == 'LAX':
            Rmin = 100
            Rmax = 200
        elif self.city == 'LIS':
            Rmin = 60
            Rmax = 80
        elif self.city == 'RIO':
            Rmin = 75
            Rmax = 100
        elif self.city == 'BOS':
            Rmin = 150
            Rmax = 200
        else:
            raise ValueError('City not found')
        spacing = (Rmax/self.R - Rmin/self.R)/20
        cprint('R: ' + str(self.R) + ' Rmin: ' + str(Rmin) + ' Rmax: ' + str(Rmax) + ' spacing: ' + str(spacing),'cyan')
        gridIdx2ij = {self.grid['index'][i]: (self.grid['i'].tolist()[i],self.grid['j'].tolist()[i]) for i in range(len(self.grid))}
        for multiplicative_factor in np.arange(Rmin/self.R,Rmax/self.R,spacing):
            # Not a repetition, is necessary to update the R
            self.R = GetTotalMovingPopulation(OD_vector)/3600 
            if os.path.isfile(os.path.join(self.save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(self.city,self.start,self.end,int(multiplicative_factor*self.R)))):
                cprint(os.path.join(self.save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(self.city,self.start,self.end,int(multiplicative_factor*self.R))),'cyan')
                continue
            else:
                gridIdx2dest = defaultdict(int)
                for o in self.grid['index'].tolist():
                    for d in self.grid['index'].tolist():
                        gridIdx2dest[(o,d)] = 0    
                cprint('COMPUTING {}'.format(os.path.join(self.save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(self.city,self.start,self.end,int(multiplicative_factor*self.R)))),'green')
                count_line = 0
                users_id = []
                time_ = []
                origins = []
                destinations = []
                for i in range(len(O_vector)):
                    origin = O_vector[i]
                    destination = D_vector[i]
                    number_people = OD_vector[i]
                    bin_width = 1                        
    #                    print('Number of people: ',number_people,' origin: ',origin,' destination: ',destination)
                    if number_people > 0:
                        iterations = multiplicative_factor*number_people/bin_width   
                        time_increment = 1/iterations
    #                        if iterations < 20:
    #                            print('iterations: ',iterations)
                        try:
                            for it in range(int(iterations)):
    #                                print('iteration: ',it)
    #                                print('polygon2origindest: ',self.polygon2OD[str(origin)],' ',self.polygon2OD[str(destination)])
                                if type(list(self.polygon2OD.keys())[0]) == str:
                                    origin = str(origin)
                                    destination = str(destination)
                                elif type(list(self.polygon2OD.keys())[0]) == int:
                                    origin = int(origin)
                                    destination = int(destination)
                                elif type(list(self.polygon2OD.keys())[0]) == float:
                                    origin = float(origin)
                                    destination = float(destination)
                                
                                if len(self.polygon2OD[destination])>0 and len(self.polygon2OD[origin])>0:
                                    if count_line%PRINTING_INTERVAL==0:
                                        cprint('iteration: ' + str(it) + ' number_people: ' + str(number_people) + ' origin: ' + str(origin) + ' #nodes('+ str(len(self.polygon2OD[origin])) + ') ' + ' destination: ' + str(destination) + ' #nodes('+ str(len(self.polygon2OD[destination])) + ') '+ ' R: ' + str(self.R),'green')        
                                        print('time insertion: ',self.start*seconds_in_minute**2 + it*time_increment*seconds_in_minute**2,' time min: ',self.start*seconds_in_minute**2,' time max: ',self.end*seconds_in_minute**2,' time max iteration: ',self.start*seconds_in_minute**2 + (iterations)*time_increment*seconds_in_minute**2)
                                    users_id.append(count_line)
    #                                    if iterations < 20:
    #                                        print('user id: ',count_line)
    #                                        print('iteration: ',it)
    #                                        print(' time insertion: ',self.start*seconds_in_minute**2 + it*time_increment*seconds_in_minute**2)
    #                                        print(' origin: ',origin,' destination: ',destination)
                                    t = self.start*(seconds_in_minute**2) + it*time_increment*seconds_in_minute**2

                                    time_.append(t) # TIME IN HOURS

                                    i = np.random.randint(0,len(self.polygon2OD[origin]))
                                    origins.append(self.osmid2index[self.polygon2OD[origin][i]])
                                    i = np.random.randint(0,len(self.polygon2OD[destination]))                        
                                    destinations.append(self.osmid2index[self.polygon2OD[destination][i]])
                                    ## FILLING ORIGIN DESTINATION GRID ACCORDING TO THE ORIGIN DESTINATION NODES
                                    ogrid = self.OD2Grid[origins[-1]]
                                    dgrid = self.OD2Grid[destinations[-1]]
                                    gridIdx2dest[(ogrid,dgrid)] += 1
                                    count_line += 1
                                    if count_line%PRINTING_INTERVAL==0:
                                        print('Origin: ',origin,' Osmid Origin: ',self.osmid2index[self.polygon2OD[str(origin)][i]],' Destination: ',destination,' Osmid Destination: ',self.osmid2index[self.polygon2OD[str(destination)][i]],' Number of people: ',number_people,' R: ',self.R)
                                        print(len(destinations),len(origins))

                                    self.total_number_people_considered += 1
                        except KeyError:
                            pass
    #                            list_types = np.unique(type(k) for k in self.polygon2OD.keys())
    #                            print('list_types: ',list_types)
    #                            print('key error: ',type(list(self.polygon2OD.keys())[0]))

                            self.total_number_people_not_considered += number_people
                    #print('Key not found at iteration: ',iterations,' number_people: ',number_people,' origin: ',origin,' destination: ',destination,' R: ',R)        
                if self.total_number_people_considered>0:
                    cprint('number_people considered: ' + str(self.total_number_people_considered),'cyan')
                    cprint('number_people not considered: ' + str(self.total_number_people_not_considered),'cyan')
                    cprint('Loss bigger 5%' + str(self.total_number_people_not_considered/self.total_number_people_considered>0.05),'cyan')
                else:
                    raise ValueError('Total number of people considered is zero')
            orig = []
            dest = []
            number_people = []
            idxorig = []
            idxdest = []
            for k in gridIdx2dest.keys():
                orig.append(k[0])
                dest.append(k[1])
                number_people.append(gridIdx2dest[k])
                idxorig.append(gridIdx2ij[k[0]])
                idxdest.append(gridIdx2ij[k[1]])
            df = pd.DataFrame({'origin':orig,'destination':dest,'number_people':number_people,'(i,j)O':idxorig,'(i,j)D':idxdest})
            df.to_csv(os.path.join(self.save_dir_local,'grid',self.grid_size,'ODgrid.csv'),sep=',',index=False)
            self.Files2Upload[os.path.join(self.save_dir_local,'grid',self.grid_size,'ODgrid.csv')] = os.path.join(self.save_dir_server,'grid',self.grid_size,'ODgrid.csv')
            self.df1 = pd.DataFrame({
                'SAMPN':users_id,
                'PERNO':users_id,
                'origin':origins,
                'destination':destinations,
                'dep_time':time_,
                'origin_osmid':origins,
                'destination_osmid':destinations
                })
            self.R = multiplicative_factor*self.R
            self.df1.to_csv(os.path.join(self.save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(self.city,self.start,self.end,int(self.R))),sep=',',index=False)
            self.Files2Upload[os.path.join(self.save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(self.city,self.start,self.end,int(self.R)))] = os.path.join(self.save_dir_server,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(self.city,self.start,self.end,int(self.R)))
            ## NEED TO UPLOAD THE FILES AND LAUNCH THE SIMULATIONS AUTOMATICALL
            cprint(str(self.Files2Upload),'green')
            self.configLaunch()
            plot_departure_times(self.df1,self.save_dir_local,self.start,self.end,int(self.R))
            self.configOD(number_of_rings,grid_sizes,resolutions)
            # END TAB
            t1 = time.time()
            cprint('time to compute OD_from_fma: ' + str(t1-t0),'cyan')
## ---------------------------------- CONFIG ---------------------------------- ##
    def configOD(self,
        number_of_rings,
        grid_sizes,
        resolutions,
        tif_file='usa_ppp_2020_UNadj_constrained.tif'):
        if os.path.isfile(os.path.join(self.save_dir_local,'{0}configOD_{1}_{2}_R{3}.json'.format(self.city,self.start,self.end,self.R))):
            cprint('{} config.json ALREADY COMPUTED'.format(self.city),'green')
        else:
            cprint('COMPUTING config.json','green')
            config = {
                'local_tiff_dir':self.tiff_file_dir_local, # Directory where the tiff file is stored
                'server_tiff_dir':self.Files2Upload[self.tiff_file_dir_local], # Directory where the tiff file is stored
                'local_city_dir':self.save_dir_local, # Directory where the cartographic data is stored
                'server_city_dir': self.Files2Upload[self.save_dir_server], # Directory where the shape files are stored
                'local_grid_dir':self.dir_grid, # Directory where the grid is stored
                'server_grid_dir':self.Files2Upload[self.dir_grid], # Directory where the grid is stored
                'local_hexagon_dir':self.dir_hexagons, # Directory where the hexagons are stored
                'server_hexagon_dir':self.Files2Upload[self.dir_hexagons], # Directory where the hexagons are stored
                'local_ring_dir':self.dir_rings, # Directory where the rings are stored
                'server_ring_dir':self.Files2Upload[self.dir_rings], # Directory where the rings are stored
                'local_polygon_dir':self.dir_polygon, # Directory where the polygons are stored
                'start': int(self.start), # Time where cumulation of trips starts
                'end': int(self.end), # If not specified, it is assumed that the OD is for 1 hour (time where cumlation of trips ends)
                "name":self.city,
                "number_trips":len(self.df1),
                "osmid2idx_path":os.path.join(self.save_dir_local,'osmid2idx.json'),
                "R":int(self.R),
                "grid_sizes":str(grid_sizes), # Grid size in km
                'number_of_rings':str(number_of_rings),
                'resolutions':str(resolutions),
                    }
            with open(os.path.join(self.save_dir_local,'{0}configOD_{1}_{2}_R{3}.json'.format(self.city,self.start,self.end,self.R)),'w') as f:
                json.dump(config,f,indent=4)
            if not RUNNING_ON_SERVER:
                print('Uploading in:')
                for file in self.Files2Upload.keys():
                    Upload2ServerPwd(file, self.Files2Upload[file])
                    print(self.Files2Upload[file])
            else:

                Upload2ServerPwd(os.path.join(self.save_dir_local,'{0}configOD_{1}_{2}_R{3}.json'.format(self.city,self.start,self.end,self.R)), os.path.join('/home/alberto/LPSim/LivingCity/berkeley_2018/',self.city,'{0}configOD_{1}_{2}_R{3}.json'.format(self.city,self.start,self.end,self.R)))

    def configLaunch(self):
        text2write = '[General]\nGUI=false\nUSE_CPU=false\nNETWORK_PATH=berkeley_2018/{0}/\nUSE_JOHNSON_ROUTING=false\nUSE_SP_ROUTING=true\nUSE_PREV_PATHS=false\nLIMIT_NUM_PEOPLE=256000\nADD_RANDOM_PEOPLE=false\nNUM_PASSES=1\nTIME_STEP=1\nSTART_HR={1}\nEND_HR=24\nOD_DEMAND_FILE=od_demand_{2}_{3}_R_{4}.csv\nSHOW_BENCHMARKS=false\nREROUTE_INCREMENT=0\nNUM_GPUS=1'.format(self.city,self.start,self.start,self.end,self.R)
        with open(os.path.join(self.save_dir_local,'command_line_options.ini'),'w') as f:
            f.write(text2write)
        self.Files2Upload[os.path.join(self.save_dir_local,'command_line_options.ini')] = os.path.join(self.save_dir_server,'command_line_options.ini')

##---------------------------------------- PRINT FUNCTIONS ----------------------------------------##

    def print_info_grid(self):
        cprint('PRINT GRID; columns: ','green')
        for col in self.grid.columns:
            cprint(col,'green')

def main(NameCity,TRAFFIC_DIR):        
    GeometricalInfo = GeometricalSettingsSpatialPartition(NameCity,TRAFFIC_DIR)
    ## FROM POLYGON TO ORIGIN DESTINATION -> OD FILE
    resolutions = [6,7,8]
    grid_sizes = list(np.arange(0.01,0.1,0.01))
    n_rings = list(np.arange(10,15,1))

    if map_already_computed[NameCity]:
        cprint('already computed origin destination; pass.','yellow')
        pass
    else:
        GeometricalInfo.polygon2origindest()
        for resolution in resolutions:
            GeometricalInfo.get_hexagon_tiling(resolution=resolution)
            GeometricalInfo.hexagon2origindest() 
        for grid_size in grid_sizes:
            GeometricalInfo.get_squared_grid(grid_size) 
            GeometricalInfo.grid2origindest()
        for n_ring in n_rings:
            GeometricalInfo.get_rings(n_ring)
            GeometricalInfo.rings2origindest()
        if 1 == 1:
            GeometricalInfo.gdf_polygons[['geometry','index']].to_file(os.path.join(GeometricalInfo.shape_file_dir_local,GeometricalInfo.city + 'new'+'.shp'))
            cprint('Setting the graph right','yellow')
        GeometricalInfo.nodes = pd.read_csv(os.path.join(GeometricalInfo.save_dir_local,'nodes.csv'))
        GeometricalInfo.osmid2index = dict(zip(GeometricalInfo.nodes['osmid'], GeometricalInfo.nodes['index']))
        cprint('osmid2index: ','yellow')
        with open(os.path.join(GeometricalInfo.save_dir_local,'osmid2idx.json'),'w') as f:
            json.dump(GeometricalInfo.osmid2index,f,indent=4)
        cprint('index2osmid: ','yellow')
        GeometricalInfo.index2osmid = dict(zip(GeometricalInfo.nodes['index'], GeometricalInfo.nodes['osmid']))
        with open(os.path.join(GeometricalInfo.save_dir_local,'idx2osmid.json'),'w') as f:
            json.dump(GeometricalInfo.index2osmid,f,indent=4)
        GeometricalInfo.adjust_edges()
        for file in os.listdir(os.path.join(GeometricalInfo.ODfma_dir,NameCity)):
            if file.endswith('.fma'):
                start = int(file.split('.')[0].split('D')[1])
                end = start + 1
                GeometricalInfo.start = start
                GeometricalInfo.end = end
                cprint('file.fma: ' + file,'yellow')
                file_name = os.path.join(GeometricalInfo.ODfma_dir,NameCity,file)
                if start == 7:
                    df_od = GeometricalInfo.OD_from_fma(file_name,n_rings,grid_sizes,resolutions)#,R
                    GeometricalInfo.configLaunch()


if __name__=='__main__':
## GLOBAL    
    dir2labels = {
        'BOS':"Boston, Massachusetts, USA",
        'SFO':"San Francisco, California, USA",
        'LAX':"Los Angeles, California, USA",
        'RIO':"Rio de Janeiro, Brazil",
        "LIS":"Lisbon, Portugal",
    }
    city2file_tif = {
        'BOS':"usa_ppp_2020_UNadj_constrained.tif",
        'SFO':"usa_ppp_2020_UNadj_constrained.tif",
        'LAX':"usa_ppp_2020_UNadj_constrained.tif",
        'RIO':"bra_ppp_2020_UNadj_constrained.tif",
        "LIS":"prt_ppp_2020_UNadj_constrained.tif",
    }
    R_city = {
        'BOS':np.linspace(120,200,80),
        'SFO':np.linspace(120,200,80),
        'LAX':np.linspace(120,200,80),
        'RIO':np.linspace(50,100,50),
        "LIS":np.linspace(20,70,50),
    }
    map_already_computed ={
        'BOS':False,
        'SFO':False,
        'LAX':False,
        'RIO':False,
        "LIS":False,
    }
    ## GLOBAL VARIABLES 

    list_cities = os.listdir(os.path.join(TRAFFIC_DIR,'data','carto'))
    arguments = [(list_cities[i],TRAFFIC_DIR) for i in range(len(list_cities))]
    print('arguments:\n',np.shape(arguments))
    with Pool() as pool:
    # Map the function to the arguments in parallel
        results = pool.starmap(main, arguments)

#    with open(os.path.join(root,'infocity.json'),'w') as f:
#        json.dump(infocity,f,indent=4)
