import os
from collections import defaultdict
import geopandas as gpd
import osmnx as ox
import socket

if socket.gethostname()=='artemis.ist.berkeley.edu':
    TRAFFIC_DIR = '/home/alberto/LPSim/traffic_phase_transition'
else:
    TRAFFIC_DIR = os.getenv('TRAFFIC_DIR')
SERVER_TRAFFIC_DIR = '/home/alberto/LPSim/LivingCity/berkeley_2018'

'''
    THIS OBJECT IS 1-1 WITH THE CITY
    Requirements FOR EACH CITY:
        - .graphml file
        - .shp file
        - .tiff file
    
'''
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
            self.bounding_box = self.gdf_polygons.geometry.unary_union.bounds
        else:
            raise ValueError('Polygon City not found')
        self.nodes = None
        self.edges = None
        self.osmid2index = defaultdict()
        self.index2osmid = defaultdict()
        self.start = 7
        self.end = 8
        self.R = 1
        self.Files2Upload = defaultdict(list)
        ## GEOMETRIES
        self.gdf_hexagons = None
        self.grid = None
        self.rings = None     
        self.lattice = None   
        ## MAPS INDICES ORIGIN DESTINATION WITH DIFFERENT GEOMETRIES
        self.polygon2OD = None
        self.OD2polygon = None
        self.hexagon2OD = None
        self.OD2hexagon = None
        self.grid2OD = None
        self.OD2grid = None
        self.ring2OD = None
        self.OD2ring = None
        self.ring2OD = None

    def UpdateFiles2Upload(self,local_file,server_file):
        self.Files2Upload[local_file] = server_file
