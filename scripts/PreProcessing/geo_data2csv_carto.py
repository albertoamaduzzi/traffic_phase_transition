
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
sys.path.append('~/Desktop/phd/ServerCommunication')
sys.path.append('~/Desktop/phd/berkeley/traffic_phase_transition/GenerationNet')
from global_functions import ifnotexistsmkdir
from HostConnection import Upload2ServerPwd
'''
TODO: Create all the files that are needed to handle the Origin and destination in such a way that traffic can be studied with different initial configurations.
Output:
    1) nodes.csv -> NEEDED IN SIMULATION
    2) edges.csv -> NEEDED IN SIMULATION
    3) od_demand_{0}to{1}_R_{2}.csv -> NEEDED IN SIMULATION
    4) polygon2origindest.json -> NEEDED IN OD (to build the OD for all the different tilings in particular for: hexagon2polygon)
    5) osmid2idx.json -> NEEDED IN OD (to build the OD for all the different tilings in particular for: hexagon2polygon)
    6) idx2osmid.json -> NEEDED IN OD (to build the OD for all the different tilings in particular for: hexagon2polygon)
    7) grid.geojson
'''


##------------------------------------- ENVIRONMENT -------------------------------------##
def set_environment():
    base_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
    data_dir = os.path.join(base_dir,'data')
    carto = os.path.join(data_dir,'carto')
    ifnotexistsmkdir(data_dir)
    ifnotexistsmkdir(carto)
    return data_dir,carto

##------------------------------------- READ FILE -------------------------------------##
def read_file_gpd(file_name):
    '''
        Input:
            file_name: string [name of the file to read]
        Output:
            df: dataframe [dataframe with the data of the file]
        Description:
            Read the file and return the dataframe
    '''
    df = gpd.read_file(file_name)
    return df

def plot_departure_times(df,save_dir,start,end,R):
    time_dep = df['dep_time'].to_numpy()/3600
    plt.hist(time_dep , bins = 24)
    plt.xlabel('Departure time')
    plt.ylabel('Number of trips')
    plt.savefig(os.path.join(save_dir,'histo_departure_time_{0}to{1}_R_{2}.png'.format(start,end,R)),dpi=200)

##------------------------------------- CONVERT TO CSV -------------------------------------##

def nodes_file_from_gpd(carto):
    '''
        Reads from shape and returns the nodes.csv
        Index is the index that will be used in too and origin and destination.
    '''
    osmid2id = defaultdict()
    i = 0
    for node in carto.nodes():
        osmid2id[node] = i
        i += 1  
    id_ = [i for i in range(len(carto.nodes()))]
    vertices = [node for node in carto.nodes()]
    df1 = pd.DataFrame({
        'osmid':vertices,
        'x':[carto.nodes[node]['x'] for node in vertices],
        'y':[carto.nodes[node]['y'] for node in vertices],
        'ref':['' for node in vertices],
        'index':id_
        })

    return df1,osmid2id
##------------------------------------- EDGES -------------------------------------##
def edges_file_from_gpd(carto,osmid2id):
    all_edges = carto.edges(data=True)
#    osmid = [BO[edge[0]][edge[1]]['osmid'] for edge in all_edges]    
#    name = [BO[edge[0]][edge[1]]['name'] for edge in all_edges]
#    df_osmid_u = pd.DataFrame([edge[0] for edge in all_edges],name='osmid_u')
#    df_osmid_v = pd.DataFrame([edge[1] for edge in all_edges],name='osmid_v')    
    try:
        lanes = [edge[2]['lanes'] if type(edge[2]['lanes']) != list else '' for edge in all_edges]
    except KeyError:
        lanes = ['' for edge in all_edges]
    try:
        speed_mph = [edge[2]['maxspeed'] for edge in all_edges]
    except KeyError:
        speed_mph = ['' for edge in all_edges]
    try:
        highway = [edge[2]['highway'] for edge in all_edges]
    except KeyError:
        highway = ['' for edge in all_edges]
    df1 = pd.DataFrame({
        'unique_id':[i for i in range(len(all_edges))],
        'u':[osmid2id[edge[1]] for edge in all_edges],
        'v':[osmid2id[edge[0]] for edge in all_edges],
        'length':[edge[2]['length'] for edge in all_edges],
        'lanes':lanes,
        'speed_mph':speed_mph,
        'highway':highway
        })
    
    return df1

##---------------------------------- GEOMETRIC FEATURES ----------------------------------##
def xy(lat,lon,lat0,lon0):
    '''
    Description:
    Projects in the tangent space of the earth in (lat0,lon0) 
    Return: 
    The projected coordinates of the lat,lon  '''
    PI = np.pi
    c_lat= 0.6*100000*(1.85533-0.006222*np.sin(lat0*PI/180))
    c_lon= c_lat*np.cos(lat0*PI/180)
    
    x = c_lon*(lon-lon0)
    y = c_lat*(lat-lat0)
    if isinstance(x,np.ndarray):
        pass
    else:        
        x = x.to_numpy()
    if isinstance(y,np.ndarray):
        pass
    else:
        y = y.to_numpy()
    return x,y

def calculate_area(geometry):
    x,y = geometry.centroid.xy
    lat0 = x[0]
    lon0 = y[0]
    # Extract the coordinates from the Polygon's exterior
    latlon = np.array([[p[0],p[1]] for p in geometry.exterior.coords]).T
    lat = latlon[0]
    lon = latlon[1]
    # Ensure the last coordinate is the same as the first to close the polygon
    x,y = xy(lat,lon,lat0,lon0)
    area = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)*np.sqrt((x[2] - x[1])**2 + (y[2] - y[1])**2)/1000000
    return area


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



def HistoPoint2Geometry(Geom2OD,GeomId,save_dir,resolution = None):
    '''
        Geom2OD: geopandas -> geometry contains (polygon,hexagon,ring,grid) Id {GeomID: [list of osmid]}
        GeomId: string -> name of the geometry
    '''
    t0 = time.time()
    if not resolution is None: 
        if os.path.join(save_dir,'histo_{0}.png'.format(GeomId)):
            cprint('histo_{0}.png ALREADY COMPUTED'.format(GeomId),'green')
        else:
            cprint('Number of point per geometry','yellow')
            bins = np.arange(len(Geom2OD.keys()))
            value = np.array([len(Geom2OD[polygon]) for polygon in Geom2OD.keys()])
            plt.bar(bins,value)
            plt.xlabel('number of {0}'.format(GeomId))
            plt.ylabel('number of points per {0}'.format(GeomId))
            plt.savefig(os.path.join(save_dir,'histo_{0}.png'.format(GeomId)),dpi=200)
    else:
        if os.path.join(save_dir,'histo_{0}_{1}.png'.format(GeomId,resolution)):
            cprint('histo_{0}.png ALREADY COMPUTED'.format(GeomId),'green')
        else:
            cprint('Number of point per geometry','yellow')
            t0 = time.time()
            bins = np.arange(len(Geom2OD.keys()))
            value = np.array([len(Geom2OD[polygon]) for polygon in Geom2OD.keys()])
            plt.bar(bins,value)
            plt.xlabel('number of {0}'.format(GeomId))
            plt.ylabel('number of points per {0}'.format(GeomId))
            plt.savefig(os.path.join(save_dir,'histo_{0}_{1}.png'.format(GeomId,resolution)),dpi=200)
    t1 = time.time()
    cprint('time to compute HistoPoint2Geometry: ' + str(t1-t0),'yellow')
def GetTotalMovingPopulation(OD_vector):
    return np.sum(OD_vector)
    
##------------------------------------- SAVE FILES -------------------------------------##

def save_nodes_edges(save_dir,df_nodes,df_edges):
    '''
        Input:
            carto: string [path to the folder where to save the files]
        Description:
            Save the files in the carto folder
    '''
    df_nodes.to_csv(os.path.join(save_dir,'nodes.csv'),sep=',',index=False)
    df_edges.to_csv(os.path.join(save_dir,'edges.csv'),sep=',',index=False)
    pass

def save_od(save_dir,df_od,R = 1,start=7,end=8):
    '''
        Input:
            carto: string [path to the folder where to save the files]
        Description:
            Save the files in the carto folder
    '''
    df_od.to_csv(os.path.join(save_dir,'od_demand_{0}to{1}_R_{2}.csv'.format(start,end,R)),sep=',',index=False)

    pass


class GeometricalSettingsSpatialPartition:
    def __init__(self,city,root,mother_dir_gdf_polygon,config_dir):
        self.crs = 'epsg:4326'
        self.city = city # NAME OF THE CITY
        self.root = root
        self.config_dir = config_dir
        self.save_dir = os.path.join(self.root,self.city) # DIRECTORY WHERE TO SAVE THE FILES ../data/carto/{city}
        self.GraphFromPhml = ox.load_graphml(filepath = os.path.join(self.save_dir,self.city + '_new_tertiary_simplified.graphml')) # GRAPHML FILE
        self.gdf_polygons = read_file_gpd(os.path.join(mother_dir_gdf_polygon,self.city,self.city + '.shp')) # POLYGON FILE
        self.gdf_polygons_dir = mother_dir_gdf_polygon
        self.nodes = None
        self.edges = None
        self.osmid2index = defaultdict()
        self.index2osmid = defaultdict()
        self.polygon2origindest = defaultdict(list)
        self.start = 7
        self.end = 8
        self.R = 1
        self.print_info()

##-------------------------------- PRINT --------------------------------##
    def print_info(self):
        cprint('city: ' + self.city,'red')
        cprint('config_dir: ' + self.config_dir,'red')
        cprint('save_dir: ' + self.save_dir,'red')
        cprint('direcory polygons: ' + self.gdf_polygons_dir,'red')


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
            cprint('get_squared_grid with grid size: ' + str(grid_size),'red')
            self.grid_size = grid_size
            ifnotexistsmkdir(os.path.join(self.save_dir,'grid'))
            self.dir_grid = os.path.join(self.save_dir,'grid')
            t0 = time.time()
            if os.path.isfile(os.path.join(self.dir_grid,"grid_size_{}.geojson".format(grid_size))):
                cprint('{0} grid_size_{1}.geojson ALREADY COMPUTED'.format(self.city,grid_size),'green')
                self.grid = gpd.read_file(os.path.join(self.dir_grid,"grid_size_{}.geojson".format(grid_size)))
                self.centroid = self.gdf_polygons.geometry.unary_union.centroid
                self.bounding_box = self.gdf_polygons.geometry.unary_union.bounds
                self.grid_size = grid_size
                bbox = shp.geometry.box(*self.bounding_box)
                bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=self.crs)
                x = np.arange(self.bounding_box[0], self.bounding_box[2], grid_size)
                y = np.arange(self.bounding_box[1], self.bounding_box[3], grid_size)

            else:
                cprint('COMPUTING {0} grid_size_{1}.geojson'.format(self.city,grid_size),'green')
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
                self.grid['area'] = self.grid['geometry'].apply(calculate_area)
                if self.resolution == 8:
                    self.get_intersection_hexagon2grid()
                self.grid['density_population'] = self.grid['population']/self.grid['area']
                cprint('save grid','green')
                self.grid.to_file(os.path.join(self.save_dir,'grid','grid_size_{}.geojson'.format(self.grid_size)), driver="GeoJSON")
            # LATTICE
            self.get_lattice()
            self.plot_grid_tiling()
            t1 = time.time()
            cprint('time to compute the grid: ' + str(t1-t0),'red')
        else:
            pass    
    def get_lattice(self):
        ## BUILD GRAPH OBJECT GRID
        cprint('get_lattice','red')
        x = np.arange(self.bounding_box[0], self.bounding_box[2], self.grid_size)
        y = np.arange(self.bounding_box[1], self.bounding_box[3], self.grid_size)

        t0 = time.time()
        if os.path.isfile(os.path.join(self.dir_grid,"centroid_lattice_size_{}.graphml".format(self.grid_size))):
            cprint('{0} centroid_lattice_size_{1}.graphml ALREADY COMPUTED'.format(self.city,self.grid_size),'green')
            self.lattice = nx.read_graphml(os.path.join(self.dir_grid,"centroid_lattice_size_{}.graphml".format(self.grid_size)))
        else:
            cprint('COMPUTING {0} centroid_lattice_size_{1}.graphml'.format(self.city,self.grid_size),'green')
            self.lattice = nx.grid_2d_graph(len(x),len(y))
            node_positions = {(row['i'],row['j']): {'x': [row['centroidx'],'y':row['centroidy']]} for idx, row in self.grid.iterrows()}
            # Add position attributes to nodes
            nx.set_node_attributes(self.lattice, node_positions)
            c = 0
            for node in self.lattice.nodes():
                print(self.lattice.nodes[node])
                c +=1
                if c ==2:
                    break
            for edge in self.lattice.edges():
                try:
                    self.lattice[edge[0]][edge[1]]['dx'] = self.lattice.nodes[edge[1]]['x'] - self.lattice.nodes[edge[0]]['x']
                    self.lattice[edge[0]][edge[1]]['dy'] = self.lattice.nodes[edge[1]]['y'] - self.lattice.nodes[edge[0]]['y']    
                except KeyError:
                    pass
            ## SAVE GRID AND LATTICE
            cprint('save lattice','green')
            nx.write_graphml(self.lattice, os.path.join(self.dir_grid,"centroid_lattice_size_{}.graphml".format(self.grid_size)))    
        t1 = time.time()
        cprint('time to compute the lattice: ' + str(t1-t0),'red')


    def get_rings(self,number_of_rings):
        '''
            Compute the rings of the city and the intersection with polygons
            rings: dict -> {idx:ring}
        '''
        if 1 == 1:
            cprint('get_rings','red')
            t0 = time.time()
            self.rings = defaultdict(list)
            self.number_of_rings = number_of_rings
            gdf_original_crs = gpd.GeoDataFrame(geometry=[self.centroid], crs=self.crs)
            self.radius = max([abs(self.bounding_box[0] -self.bounding_box[2])/2,abs(self.bounding_box[1] - self.bounding_box[3])/2]) 
            self.radiuses = np.linspace(0,self.radius,self.number_of_rings)
            ifnotexistsmkdir(os.path.join(self.save_dir,'ring'))
            self.dir_rings = os.path.join(self.save_dir,'ring')
            if os.path.isfile(os.path.join(self.dir_rings,'rings_n_{}.geojson'.format(self.number_of_rings))):
                cprint('rings_n_{}.geojson ALREADY COMPUTED'.format(self.number_of_rings),'green')
                self.rings = gpd.read_file(os.path.join(self.dir_rings,'rings_n_{}.geojson'.format(self.number_of_rings)))
                if self.resolution == 8:
                    self.get_intersection_hexagon2rings()
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
                        self.get_intersection_hexagon2rings()
                    cprint('columns of the rings: ','green')
                    for col in self.rings.columns:
                        cprint(col,'green')
                    self.rings.to_file(os.path.join(self.save_dir,'ring','rings_n_{}.geojson'.format(self.number_of_rings)), driver="GeoJSON")  
                    self.plot_ring_tiling()
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
        ifnotexistsmkdir(os.path.join(self.save_dir,'hexagon'))
        self.dir_hexagons = os.path.join(self.save_dir,'hexagon')
        if not os.path.exists(os.path.join(self.save_dir,'hexagon','hexagon_resolution_{}.geojson'.format(self.resolution))):
            cprint('COMPUTING {0} hexagon_resolution_{1}.geojson'.format(self.city,self.resolution),'green')
            with rasterio.open(self.tif_file) as dataset:
                clipped_data, clipped_transform = mask(dataset, self.gdf_polygons.geometry, crop=True)
            ## CHANGE NULL ENTRANCIES (-99999) for US (may change for other Countries [written in United Nation page of Download])
            clipped_data = np.array(clipped_data)
            print('resolution: ',resolution)
            print('clipped_data: ',np.shape(clipped_data))
            print('clipped_transform: ',np.shape(clipped_transform))
            condition = clipped_data<0
            clipped_data[condition] = 0
            # Define hexagon resolution
            bay_area_geometry = self.gdf_polygons.unary_union
            print('bay_area_geometry: ',type(bay_area_geometry))
            # Convert MultiPolygon to a single Polygon
            bay_area_polygon = bay_area_geometry.convex_hull
            # Convert Polygon to a GeoJSON-like dictionary
            bay_area_geojson = sg.mapping(bay_area_polygon)
            print('bay_area_geojson: ',type(bay_area_geojson))
            # Get hexagons within the bay area
            hexagons = h3.polyfill(bay_area_geojson, resolution, geo_json_conformant=True)
            print('hexagons: ',type(hexagons))
            # Convert hexagons to Shapely geometries
            hexagon_geometries = [sg.Polygon(h3.h3_to_geo_boundary(h, geo_json=True)) for h in hexagons]
            print('hexagon_geometries: ',np.shape(hexagon_geometries),' type: ',type(hexagon_geometries))

            # Aggregate population data for each hexagon
            population_data_hexagons = [aggregate_population(hexagon, clipped_transform, clipped_data) for hexagon in hexagon_geometries]
            print('population_data_hexagons: ',np.shape(population_data_hexagons))
            centroid_hexagons = [h.centroid for h in hexagon_geometries]
            centroidx = [h.centroid.x for h in hexagon_geometries]
            centroidy = [h.centroid.y for h in hexagon_geometries]
            # Create GeoDataFrame
#            print('len hexagon_geometries: ',centroid_hexagons)
            self.gdf_hexagons = gpd.GeoDataFrame(geometry=hexagon_geometries, data={'population':population_data_hexagons,'centroid_x':centroidx,'centroid_y':centroidy},crs = self.crs)
            self.gdf_hexagons.reset_index(inplace=True)
        else:
            cprint('{0} hexagon_resolution_{1}.geojson ALREADY COMPUTED'.format(self.city,self.resolution),'green')
            self.gdf_hexagons = gpd.read_file(os.path.join(self.save_dir,'hexagon','hexagon_resolution_{}.geojson'.format(self.resolution)))
        if self.resolution == 8:
            self.hexagon2polygon()
#        cprint('columns of the hexagons: ','green')
#        for col in self.gdf_hexagons.columns:
#            cprint(col,'green')
        self.gdf_hexagons['area'] = self.gdf_hexagons.to_crs({'proj':'cea'}).area / 10**6
        self.gdf_hexagons['density_population'] = self.gdf_hexagons['population']/self.gdf_hexagons['area']        
        self.gdf_hexagons.to_file(os.path.join(self.save_dir,'hexagon','hexagon_resolution_{}.geojson'.format(self.resolution)), driver="GeoJSON")
        self.plot_hexagon_tiling()

## ------------------------------------------------ MAP TILE TO POLYGONS ------------------------------------------------ ##
    def hexagon2polygon(self):
        '''
            Consider just the hexagons of the tiling that have population > 0
            Any time we have that a polygon is intersected by the hexagon, we add to the population column
            of the polygon the population of the hexagon times the ratio of intersection area with respect to the hexagon area
        '''
        cprint('hexagon2polygon {}'.format(self.city),'green')
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

    def get_intersection_hexagon2grid(self):
        '''
            Associates the mass to the grid 
        '''
        cprint('get_intersection_hexagon2grid {}'.format(self.city),'green')
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

    def get_intersection_hexagon2rings(self):
        '''
            Gives the population to the rings
        '''
        if self.rings is None:
            raise ValueError('rings is None')
        elif self.gdf_polygons is None:
            raise ValueError('gdf_polygons is None')
        else:
            cprint('get_intersection_polygon2rings {}'.format(self.city),'green')
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
            self.edges = pd.read_csv(os.path.join(self.save_dir,'edges.csv'))
            self.edges['u'] = self.edges['u'].apply(lambda x: self.osmid2index[x])
            self.edges['v'] = self.edges['v'].apply(lambda x: self.osmid2index[x])
            self.edges.to_csv(os.path.join(self.save_dir,'edges.csv'),index=False)
        except KeyError:
            cprint('edges.csv ALREADY COMPUTED','green')
            try:
                self.edges['osmid_u'] = self.edges['u'].apply(lambda x: self.index2osmid[x])
                self.edges['osmid_v'] = self.edges['v'].apply(lambda x: self.index2osmid[x])
                self.edges.to_csv(os.path.join(self.save_dir,'edges.csv'),index=False)
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
        if 1 == 1:
            cprint('grid2origindest','red')
            t0 = time.time()
            if os.path.isfile(os.path.join(self.dir_grid,'grid2origindest_grid_size_{}.json'.format(self.grid_size))):
                cprint('grid2origindest_grid_size_{}.json ALREADY COMPUTED'.format(self.grid_size),'green')
                with open(os.path.join(self.dir_grid,'grid2origindest_grid_size_{}.json'.format(self.grid_size)),'r') as f:
                    self.Grid2OD = json.load(f)
            else:
                n_nodes = np.zeros(len(self.grid))
                grid2origindest = defaultdict(list)
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
                        n_nodes[int(tract_id_grid)] = len(grid2origindest[int(tract_id_grid)])
                    else:
                        pass
                self.grid.loc['number_nodes'] = n_nodes
                self.Grid2OD = grid2origindest
                with open(os.path.join(self.save_dir,'grid2origindest_grid_size_{}.json'.format(self.grid_size)),'w') as f:
                    json.dump(self.Grid2OD,f,indent=4)
            HistoPoint2Geometry(self.Grid2OD,'square',self.dir_grid,resolution = self.grid_size)
            t1 = time.time()
            cprint('time to compute grid2origindest: ' + str(t1-t0),'red')
        else:
            pass
    def hexagon2origindest(self):
        '''
            Given a network taken from the cartography or ours:
                Build the set of origin and destinations from the hexagon that are coming from the 
                geodataframe.
            
        '''
        if 1 == 1:
            cprint('{} hexagon2origindest'.format(self.city),'red')
            t0 = time.time()
            if os.path.isfile(os.path.join(self.dir_hexagons,'hexagon2origindest_resolution_{}.json'.format(self.resolution))):
                cprint('{0} hexagon2origindest_resolution_{1}.json ALREADY COMPUTED'.format(self.city,self.resolution),'green')
                with open(os.path.join(self.dir_hexagons,'hexagon2origindest_resolution_{}.json'.format(self.resolution)),'r') as f:
                    self.Hex2OD = json.load(f)
            else:
                cprint('{0} COMPUTING hexagon2origindest_resolution_{1}.json'.format(self.city,self.resolution),'green')
                n_nodes = np.zeros(len(self.gdf_hexagons))
                hexagon2origindest = defaultdict(list)
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
                        n_nodes[int(tract_id_hexagon)] = len(hexagon2origindest[int(tract_id_hexagon)])
                    else:
                        pass
                self.gdf_hexagons['number_nodes'] = list(n_nodes)
                self.Hex2OD = hexagon2origindest
                with open(os.path.join(self.dir_hexagons,'hexagon2origindest_resolution_{}.json'.format(self.resolution)),'w') as f:
                    json.dump(self.Hex2OD,f,indent=4)
            HistoPoint2Geometry(self.Hex2OD,'hexagon',self.dir_hexagons,resolution = self.resolution)
            t1 = time.time()
            cprint('time to compute hexagon2origindest: ' + str(t1-t0),'red')
        else:
            pass    
    
    def rings2origindest(self):
        '''
            Given a network taken from the cartography or ours:
                Build the set of origin and destinations from the rings that are coming from the 
                geodataframe.
            
        '''
        if 1 == 1:
            cprint('rings2origindest','red')
            t0 = time.time()
            if os.path.isfile(os.path.join(self.dir_rings,'rings2origindest_radius_{}.json'.format(self.radius))):
                cprint('rings2origindest_radius_{}.json ALREADY COMPUTED'.format(self.radius),'green')
                with open(os.path.join(self.dir_rings,'rings2origindest_radius_{}.json'.format(self.radius)),'r') as f:
                    self.Rings2OD = json.load(f)
            else:
                cprint('COMPUTING rings2origindest_radius_{}.json'.format(self.radius),'green')
                n_nodes = np.zeros(len(self.rings))
                rings2origindest = defaultdict(list)
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
                        n_nodes[int(tract_id_ring)] = len(rings2origindest[int(tract_id_ring)])
                    else:
                        pass
                self.rings['number_nodes'] = list(n_nodes)
                self.Rings2OD = rings2origindest
                with open(os.path.join(self.dir_rings,'rings2origindest_radius_{}.json'.format(self.radius)),'w') as f:
                    json.dump(self.Rings2OD,f,indent=4)
            HistoPoint2Geometry(self.Rings2OD,'ring',self.dir_rings,resolution = self.radius)
            t1 = time.time()
            cprint('time to compute rings2origindest: ' + str(t1-t0),'red')
        else:
            pass        

    def polygon2origin_destination(self,debug = False):
        '''
            Given a network taken from the cartography or ours:
                Build the set of origin and destinations from the polygons that are coming from the 
                geodataframe.
            
        '''
        cprint('polygon2origin_destination','red')
        t0 = time.time()
        if os.path.isfile(os.path.join(self.save_dir,'polygon2origindest.json')):
            cprint('polygon2origindest.json ALREADY COMPUTED','green')
            with open(os.path.join(self.save_dir,'polygon2origindest.json'),'r') as f:
                self.polygon2origindest = json.load(f)
        else:
            cprint('COMPUTING polygon2origindest.json','green')
            polygon2origindest = defaultdict(list)
            for node in self.GraphFromPhml.nodes():
                containing_polygon = self.gdf_polygons.geometry.apply(lambda x: Point(self.GraphFromPhml.nodes[node]['x'],self.GraphFromPhml.nodes[node]['y']).within(x))
                idx_containing_polygon = self.gdf_polygons[containing_polygon].index
        #        print('idx_containing_polygon: ',idx_containing_polygon)
                if self.city == 'SFO':
                    key = 'TRACT'
                    self.tif_file =  os.path.join(self.gdf_polygons_dir,'usa_ppp_2020_UNadj_constrained.tif')
                if self.city == 'LAX':
                    key = 'external_i'
                    self.tif_file =  os.path.join(self.gdf_polygons_dir,'usa_ppp_2020_UNadj_constrained.tif')
                if self.city == 'LIS':
                    key = 'ID'
                    self.tif_file =  os.path.join(self.gdf_polygons_dir,'prt_ppp_2020_UNadj_constrained.tif')
                if self.city == 'RIO':
                    key = 'Zona'
                    self.tif_file =  os.path.join(self.gdf_polygons_dir,'bra_ppp_2020_UNadj_constrained.tif')

                if self.city == 'BOS':
                    key = 'tractid'
                    self.tif_file =  os.path.join(self.gdf_polygons_dir,'usa_ppp_2020_UNadj_constrained.tif')
                tract_id_polygon = self.gdf_polygons.loc[idx_containing_polygon][key]
                if len(tract_id_polygon)==1: 
                    try:
                        tract_id_polygon = int(tract_id_polygon.tolist()[1])
                    except IndexError:
                        tract_id_polygon = int(tract_id_polygon.tolist()[0])
        #            print('tract_id: ',tract_id)
                    polygon2origindest[int(tract_id_polygon)].append(node)
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

            self.polygon2origindest = polygon2origindest
        if self.city == 'SFO':
            key = 'TRACT'
            self.tif_file =  os.path.join(self.gdf_polygons_dir,'usa_ppp_2020_UNadj_constrained.tif')
        if self.city == 'LAX':
            key = 'external_i'
            self.tif_file =  os.path.join(self.gdf_polygons_dir,'usa_ppp_2020_UNadj_constrained.tif')
        if self.city == 'LIS':
            key = 'ID'
            self.tif_file =  os.path.join(self.gdf_polygons_dir,'prt_ppp_2020_UNadj_constrained.tif')
        if self.city == 'RIO':
            key = 'Zona'
            self.tif_file =  os.path.join(self.gdf_polygons_dir,'bra_ppp_2020_UNadj_constrained.tif')

        if self.city == 'BOS':
            key = 'tractid'
            self.tif_file =  os.path.join(self.gdf_polygons_dir,'usa_ppp_2020_UNadj_constrained.tif')
        
        HistoPoint2Geometry(self.polygon2origindest,'polygon',self.save_dir)
        self.gdf_polygons.rename(columns = {key:'index'})
        t1 = time.time()
        cprint('time to compute polygon2origindest: ' + str(t1-t0),'red')
#        print('columns of the gdf_polygons: ',self.gdf_polygons.columns)
#        print('columns of the gdf_polygons: ',self.gdf_polygons)
#        self.gdf_polygons['index'] = self.gdf_polygons[key]
#        self.gdf_polygons['index'] = self.gdf_polygons['index'].astype(int)
         



##------------------------------------------ PLOT ------------------------------------------##
    def plot_grid_tiling(self):
        # PLOT AREA
        cprint('plot_grid_tiling','blue')
        fig,ax = plt.subplots(1,1,figsize=(12,12))
        self.grid.plot(ax=ax,column='area',cmap='viridis', edgecolor='black')#color='white',
        self.gdf_polygons.plot(ax=ax, facecolor = 'none', edgecolor='black')
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Area $km^2$')
        ax.set_aspect('equal')
        plt.savefig(self.dir_grid + '/{}_area.png'.format(self.grid_size),dpi=200)
        # PLOT POPULATION
        fig,ax = plt.subplots(1,1,figsize=(12,12))
        self.grid.plot(ax=ax,column='population',cmap='viridis', edgecolor='black')#color='white',
        self.gdf_polygons.plot(ax=ax, facecolor = 'none', edgecolor='black')
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Population')
        ax.set_aspect('equal')
        plt.savefig(self.dir_grid + '/{}_population.png'.format(self.grid_size),dpi=200)
        # PLOT DENSITY
        fig,ax = plt.subplots(1,1,figsize=(12,12))
        self.grid.plot(ax=ax,column='density_population',cmap='viridis', edgecolor='black')#color='white',
        self.gdf_polygons.plot(ax=ax, facecolor = 'none', edgecolor='black')
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Population/Area($km^2$)')
        ax.set_aspect('equal')
        plt.savefig(self.dir_grid + '/{}_density.png'.format(self.grid_size),dpi=200)
        # AREA DISTRIBUTION
        ax.hist(self.grid['area'])
        ax.set_xlabel('Area $km^2$')
        ax.set_ylabel('Number of squares')
        plt.savefig(self.dir_grid + '/{}_histo_area.png'.format(self.grid_size),dpi=200)

    def plot_hexagon_tiling(self):
        if not os.path.isfile(self.dir_hexagons + '/{}_area.png'.format(self.resolution)):
            cprint('{} plot_hexagon_tiling'.format(self.city),'blue')
            fig,ax = plt.subplots(1,1,figsize=(12,12))
            self.gdf_hexagons.plot(ax=ax,column='area',cmap='viridis', edgecolor='black')#color='white',
            self.gdf_polygons.plot(ax=ax, facecolor = 'none', edgecolor='black')
            cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label('Area $km^2$')
            ax.set_aspect('equal')
            plt.savefig(self.dir_hexagons + '/{}_area.png'.format(self.resolution),dpi=200)
        else:
            cprint('{} plot_hexagon_tiling ALREADY COMPUTED'.format(self.city),'green')
        if not os.path.isfile(self.dir_hexagons + '/{}_population.png'.format(self.resolution)):
            fig,ax = plt.subplots(1,1,figsize=(12,12))
            self.gdf_hexagons.plot(ax=ax,column='population',cmap='viridis', edgecolor='black')#color='white',
            self.gdf_polygons.plot(ax=ax, facecolor = 'none', edgecolor='black')
            cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label('Population')
            ax.set_aspect('equal')
            plt.savefig(self.dir_hexagons + '/{}_population.png'.format(self.resolution),dpi=200)
        else:
            cprint('{} plot_hexagon_tiling ALREADY COMPUTED'.format(self.city),'green')
        if not os.path.isfile(self.dir_hexagons + '/{}_density.png'.format(self.resolution)):
            fig,ax = plt.subplots(1,1,figsize=(12,12))
            self.gdf_hexagons.plot(ax=ax,column='density_population',cmap='viridis', edgecolor='black')#color='white',
            self.gdf_polygons.plot(ax=ax, facecolor = 'none', edgecolor='black')
            cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label('Population/Area($km^2$)')
            ax.set_aspect('equal')
            plt.savefig(self.dir_hexagons + '/{}_density.png'.format(self.resolution),dpi=200)
        else:
            cprint('{} plot_hexagon_tiling ALREADY COMPUTED'.format(self.city),'green')
        if not os.path.isfile(self.dir_hexagons + '/{}_histo_area.png'.format(self.resolution)):
            fig,ax = plt.subplots(1,1,figsize=(12,12))
            ax.hist(self.gdf_hexagons['area'])
            ax.set_xlabel('Area $km^2$')
            ax.set_ylabel('Number of hexagons')
            plt.savefig(self.dir_hexagons + '/{}_histo_area.png'.format(self.resolution),dpi=200)
        else:
            cprint('{} plot_hexagon_tiling ALREADY COMPUTED'.format(self.city),'green')

    def plot_ring_tiling(self):
        if not os.path.isfile(self.dir_rings + '/{}_area.png'.format(self.radius)):
            cprint('{} plot_ring_tiling'.format(self.city),'blue')
            fig,ax = plt.subplots(1,1,figsize=(20,15))
            self.gdf_polygons.plot(ax = ax)
            centroid = self.gdf_polygons.geometry.unary_union.centroid
            for r in self.radiuses:
                circle = plt.Circle(np.array([centroid.x,centroid.y]), r, color='red',fill=False ,alpha = 0.5)
                ax.add_artist(circle)
                ax.set_in_layout(True)
                ax.grid(True)
                ax.get_shared_x_axes()
                ax.get_shared_y_axes()
                ax.set_aspect('equal')
                plt.savefig(self.dir_rings + '/{}_ring.png'.format(r),dpi=200)
        else:
            cprint('{} plot_ring_tiling ALREADY COMPUTED'.format(self.city),'green')
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
        with open(file,'r') as infile:
            count_line = 0
            for line in infile:
                count_line += 1
                if count_line > offset:
                    tok = line.split(' ')
                    O_vector.append(int(tok[0]))
                    D_vector.append(int(tok[1]))
                    OD_vector.append(int(float(tok[2].split('\n')[0])))

            cprint('OD_from_fma','green')
            cprint('{} OPENED FILE: '.format(self.city) + file,'green')
            cprint('{0}, R: '.format(self.city) + str(GetTotalMovingPopulation(OD_vector)/3600),'green')
            cprint('shape: '+ str(np.shape(OD_vector)),'green')
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
            print('R: ',self.R)
            print('Rmin: ',Rmin)
            print('Rmax: ',Rmax)
            print('spacing: ',spacing)
            for multiplicative_factor in np.arange(Rmin/self.R,Rmax/self.R,spacing):
                self.R = GetTotalMovingPopulation(OD_vector)/3600
#                if os.path.isfile(os.path.join(self.save_dir,'od_demand_{0}to{1}_R_{2}.csv'.format(self.start,self.end,int(multiplicative_factor*self.R)))):
#                    cprint('od_demand_{0}to{1}_R_{2}.csv ALREADY COMPUTED'.format(self.start,self.end,multiplicative_factor*self.R),'green')
#                    continue
#                else:
#                cprint('COMPUTING od_demand_{0}to{1}_R_{2}.csv {3}'.format(self.start,self.end,int(multiplicative_factor*self.R),self.city),'green')
#                print('Multiplicative Factor: ',multiplicative_factor)
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
#                                print('polygon2origindest: ',self.polygon2origindest[str(origin)],' ',self.polygon2origindest[str(destination)])
                                if type(list(self.polygon2origindest.keys())[0]) == str:
                                    origin = str(origin)
                                    destination = str(destination)
                                elif type(list(self.polygon2origindest.keys())[0]) == int:
                                    origin = int(origin)
                                    destination = int(destination)
                                elif type(list(self.polygon2origindest.keys())[0]) == float:
                                    origin = float(origin)
                                    destination = float(destination)
                                
                                if len(self.polygon2origindest[destination])>0 and len(self.polygon2origindest[origin])>0:
                                    if count_line%PRINTING_INTERVAL==0:
                                        cprint('iteration: ' + str(it) + ' number_people: ' + str(number_people) + ' origin: ' + str(origin) + ' #nodes('+ str(len(self.polygon2origindest[origin])) + ') ' + ' destination: ' + str(destination) + ' #nodes('+ str(len(self.polygon2origindest[destination])) + ') '+ ' R: ' + str(self.R),'green')        
                                        print('time insertion: ',self.start*seconds_in_minute**2 + it*time_increment*seconds_in_minute**2,' time min: ',self.start*seconds_in_minute**2,' time max: ',self.end*seconds_in_minute**2,' time max iteration: ',self.start*seconds_in_minute**2 + (iterations)*time_increment*seconds_in_minute**2)
                                    users_id.append(count_line)
#                                    if iterations < 20:
#                                        print('user id: ',count_line)
#                                        print('iteration: ',it)
#                                        print(' time insertion: ',self.start*seconds_in_minute**2 + it*time_increment*seconds_in_minute**2)
#                                        print(' origin: ',origin,' destination: ',destination)
                                    t = self.start*(seconds_in_minute**2) + it*time_increment*seconds_in_minute**2

                                    time_.append(t) # TIME IN HOURS

                                    i = np.random.randint(0,len(self.polygon2origindest[origin]))
                                    origins.append(self.osmid2index[self.polygon2origindest[origin][i]])
                                    i = np.random.randint(0,len(self.polygon2origindest[destination]))                        
                                    destinations.append(self.osmid2index[self.polygon2origindest[destination][i]])
                                    count_line += 1

                                    if count_line%PRINTING_INTERVAL==0:
                                        print('Origin: ',origin,' Osmid Origin: ',self.osmid2index[self.polygon2origindest[str(origin)][i]],' Destination: ',destination,' Osmid Destination: ',self.osmid2index[self.polygon2origindest[str(destination)][i]],' Number of people: ',number_people,' R: ',self.R)
                                        print(len(destinations),len(origins))

                                    self.total_number_people_considered += 1
                        except KeyError:
                            pass
#                            list_types = np.unique(type(k) for k in self.polygon2origindest.keys())
#                            print('list_types: ',list_types)
#                            print('key error: ',type(list(self.polygon2origindest.keys())[0]))

                            self.total_number_people_not_considered += number_people
                            #print('Key not found at iteration: ',iterations,' number_people: ',number_people,' origin: ',origin,' destination: ',destination,' R: ',R)        
                    
                if self.total_number_people_considered>0:
                    cprint('number_people considered: ' + str(self.total_number_people_considered),'green')
                    cprint('number_people not considered: ' + str(self.total_number_people_not_considered),'green')
                    cprint('Loss bigger 5%' + str(self.total_number_people_not_considered/self.total_number_people_considered>0.05),'green')
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
                print(self.city,' R: ',self.R,'Origins: ',len(origins),' Destinations: ',len(destinations),' Time: ',len(time_),' Users: ',len(users_id))
                self.df1.to_csv(os.path.join(self.save_dir,'od_demand_{0}to{1}_R_{2}.csv'.format(self.start,self.end,int(self.R))),sep=',',index=False)
                Upload2ServerPwd(os.path.join(self.save_dir,'od_demand_{0}to{1}_R_{2}.csv'.format(self.start,self.end,int(self.R))), os.path.join('/home/alberto/test/LPSim/LivingCity/berkeley_2018/',self.city,'od_demand_{0}to{1}_R_{2}.csv'.format(self.start,self.end,int(self.R))))
                plot_departure_times(self.df1,self.save_dir,self.start,self.end,int(self.R))
                self.configOD(number_of_rings,grid_sizes,resolutions)
            t1 = time.time()
            cprint('time to compute OD_from_fma: ' + str(t1-t0),'red')
## ---------------------------------- CONFIG ---------------------------------- ##
    def configOD(self,
        number_of_rings,
        grid_sizes,
        resolutions,
        tif_file='usa_ppp_2020_UNadj_constrained.tif'):
        if os.path.isfile(os.path.join(self.save_dir,'{0}configOD_{1}to{2}_R{3}.json'.format(self.city,self.start,self.end,self.R))):
            cprint('{} config.json ALREADY COMPUTED'.format(self.city),'green')
        else:
            cprint('COMPUTING config.json','green')
            config = {'carto_dir':self.save_dir, # Directory where the cartographic data is stored
                'shape_file_dir': self.save_dir, # Directory where the shape files are stored
                'dir_lattice': str([os.path.join(self.save_dir,'lattice','lattice_size_{}.graphml'.format(grid_size)) for grid_size in grid_sizes]), # Directory where the lattice is stored
                'dir_grid':str([os.path.join(self.save_dir,'grid','grid_size_{}.geojson'.format(grid_size)) for grid_size in grid_sizes]), # Directory where the grid is stored
                'dir_hexagons':str([os.path.join(self.save_dir,'hexagon','hexagon_resolution_{}.geojson'.format(resolution)) for resolution in resolutions]), # Directory where the hexagons are stored
                'dir_rings':str([os.path.join(self.save_dir,'ring','rings_n_{}.geojson'.format(n_ring)) for n_ring in number_of_rings]), # Directory where the rings are stored
                'dir_polygon':os.path.join(self.gdf_polygons_dir,self.city + 'new'+'.shp'), # Directory where the polygons are stored
                'dir_OD':self.save_dir, # Directory where the OD are stored
                'dir_edges':self.save_dir, # Directory where the edges are stored
                'dir_nodes':self.save_dir, # Directory where the nodes are stored
                'dir_graph':self.save_dir, # Directory where the graph is stored
                'dir_graphml':os.path.join(self.save_dir,self.city + '_new_tertiary_simplified.graphml'), # Directory where the graphml is stored
                'start': int(self.start), # Time where cumulation of trips starts
                'end': int(self.end), # If not specified, it is assumed that the OD is for 1 hour (time where cumlation of trips ends)
                "name":self.city,
                "number_trips":len(self.df1),
                "osmid2idx_path":os.path.join(self.save_dir,'osmid2idx.json'),
                "R":int(self.R),
                "grid_sizes":str(grid_sizes), # Grid size in km
                'number_of_rings':str(number_of_rings),
                'resolutions':str(resolutions),
                "tif_file":tif_file
                    }
            with open(os.path.join(self.save_dir,'{0}configOD_{1}to{2}_R{3}.json'.format(self.city,self.start,self.end,self.R)),'w') as f:
                json.dump(config,f,indent=4)
            Upload2ServerPwd(os.path.join(self.save_dir,'{0}configOD_{1}to{2}_R{3}.json'.format(self.city,self.start,self.end,self.R)), os.path.join('/home/alberto/test/LPSim/LivingCity/berkeley_2018/',self.city,'{0}configOD_{1}to{2}_R{3}.json'.format(self.city,self.start,self.end,self.R)))

##---------------------------------------- PRINT FUNCTIONS ----------------------------------------##

    def print_info_grid(self):
        cprint('PRINT GRID; columns: ','green')
        for col in self.grid.columns:
            cprint(col,'green')

def main(dir_,root,base_dir_shape,config_dir):        
    GeometricalInfo = GeometricalSettingsSpatialPartition(dir_,root,base_dir_shape,config_dir)
    ## FROM POLYGON TO ORIGIN DESTINATION -> OD FILE
    resolutions = [6,7,8]
    grid_sizes = list(np.arange(0.01,0.1,0.01))
    n_rings = list(np.arange(10,15,1))

    if map_already_computed[dir_]:
        cprint('already computed origin destination; pass.','yellow')
        pass
    else:
        GeometricalInfo.polygon2origin_destination()
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
            GeometricalInfo.gdf_polygons[['geometry','index']].to_file(os.path.join(GeometricalInfo.gdf_polygons_dir,GeometricalInfo.city + 'new'+'.shp'))
            cprint('Setting the graph right','yellow')
        GeometricalInfo.nodes = pd.read_csv(os.path.join(GeometricalInfo.save_dir,'nodes.csv'))
        GeometricalInfo.osmid2index = dict(zip(GeometricalInfo.nodes['osmid'], GeometricalInfo.nodes['index']))
        cprint('osmid2index: ','yellow')
        with open(os.path.join(GeometricalInfo.save_dir,'osmid2idx.json'),'w') as f:
            json.dump(GeometricalInfo.osmid2index,f,indent=4)
        cprint('index2osmid: ','yellow')
        GeometricalInfo.index2osmid = dict(zip(GeometricalInfo.nodes['index'], GeometricalInfo.nodes['osmid']))
        with open(os.path.join(GeometricalInfo.save_dir,'idx2osmid.json'),'w') as f:
            json.dump(GeometricalInfo.index2osmid,f,indent=4)
        GeometricalInfo.adjust_edges()
        for file in os.listdir(os.path.join(base_dir_shape,dir_)):
            if file.endswith('.fma'):
                start = int(file.split('.')[0].split('D')[1])
                end = start + 1
                GeometricalInfo.start = start
                GeometricalInfo.end = end
                cprint('file.fma: ' + file,'yellow')
                file_name = os.path.join(base_dir_shape,dir_,file)
                if start == 7:
                    df_od = GeometricalInfo.OD_from_fma(file_name,n_rings,grid_sizes,resolutions)#,R


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
    names_carto = {
        'BOS':"Boston",
        'SFO':"San_Francisco",
        'LAX':"Los_Angeles",
    }
    list_cities = ['BOS','SFO','LAX','RIO','LIS']
    populations = [4.5e6,7.5e6,13e6,12.6e6,2.8e6]
    config_dir = '/home/aamad/Desktop/phd/berkeley/traffic_phase_transition/config'
    root = '/home/aamad/Desktop/phd/berkeley/data/carto'
    base_dir_shape = '/home/aamad/Desktop/phd/berkeley/data'
    arguments = [(list_cities[i],root,base_dir_shape,config_dir) for i in range(len(list_cities))]
    print('arguments:\n',np.shape(arguments))
    with Pool() as pool:
    # Map the function to the arguments in parallel
        results = pool.starmap(main, arguments)

#    with open(os.path.join(root,'infocity.json'),'w') as f:
#        json.dump(infocity,f,indent=4)
'''
    Description:
        I take the shape file and convert it to nodes.csv, edges.csv 
        Take OD and convert it to origin_destination.csv
    Input:
        G: cartography that is polished in mobility_planner.py
        gdf: polygon file utilized in mobility planner to reconstruct a polished cartography
    Output:
        _routes.csv ->  each row is index of the person
        _people.csv -> time_arrival is time_departure+numb_steps


'''
