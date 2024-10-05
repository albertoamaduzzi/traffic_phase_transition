from termcolor import  cprint
import sys
import os
import json
import numpy as np
from PreprocessingObj import *
from PolygonSettings import *
import pandas as pd
from Polygon import *
from Hexagon import *
from Grid import *
from multiprocessing import Pool
import socket

if socket.gethostname()=='artemis.ist.berkeley.edu':
    sys.path.append(os.path.join('/home/alberto/LPSim','traffic_phase_transition','scripts','ServerCommunication'))
    sys.path.append(os.path.join('/home/alberto/LPSim','traffic_phase_transition','scripts','PreProcessing'))
else:
    sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','ServerCommunication'))
    sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','PreProcessing'))

from HostConnection import *
from plot import *
import warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger(__name__)

    ##  ------------------------- INITIALIZE HEXAGON 2 OD ------------------------- ##
def AllStepsHexagon(GeometricalInfo,resolution,NameCity):
    GeometricalInfo.gdf_hexagons = GetHexagon(GeometricalInfo.gdf_polygons,GeometricalInfo.tiff_file_dir_local,GeometricalInfo.save_dir_local,NameCity,resolution)
    SaveHexagon(GeometricalInfo.save_dir_local,resolution,GeometricalInfo.gdf_hexagons)
    GeometricalInfo.OD2hexagon,GeometricalInfo.hexagon2OD,GeometricalInfo.gdf_hexagons = Geometry2OD(gdf_geometry = GeometricalInfo.gdf_hexagons,
                                                                        GraphFromPhml = GeometricalInfo.GraphFromPhml,
                                                                        NameCity = GeometricalInfo.city,
                                                                        GeometryName ='hexagon',
                                                                        save_dir_local = GeometricalInfo.save_dir_local,
                                                                        resolution = resolution)
    
    
    if socket.gethostname()!='artemis.ist.berkeley.edu':
        # ADD FILES TO UPLOAD
        GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'hexagon',str(resolution),'hexagon.geojson'),os.path.join(GeometricalInfo.save_dir_server,'hexagon',str(resolution),'hexagon.geojson'))
        GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'hexagon',str(resolution),'hexagon2origindest.json'),os.path.join(GeometricalInfo.save_dir_server,'hexagon',str(resolution),'hexagon2origindest.json'))
        GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'hexagon',str(resolution),'origindest2hexagon.json'),os.path.join(GeometricalInfo.save_dir_server,'hexagon',str(resolution),'origindest2hexagon.json'))
    
    GeometricalInfo.gdf_polygons = getPolygonPopulation(GeometricalInfo.gdf_hexagons,GeometricalInfo.gdf_polygons,NameCity)
    SavePolygon(GeometricalInfo.save_dir_local,GeometricalInfo.gdf_polygons)

def ComputeHexagon(NameCity,TRAFFIC_DIR,resolutions):
    print('Computing Grid for city: ',NameCity)
    print('TRAFFIC_DIR: ',TRAFFIC_DIR)
    GeometricalInfo = GeometricalSettingsSpatialPartition(NameCity,TRAFFIC_DIR)
    for resolution in resolutions:
        AllStepsHexagon(GeometricalInfo,resolution,NameCity)

def ComputeHexagonParallelOnCity(NameCity,TRAFFIC_DIR,resolutions):
    print('Computing Grid for city: ',NameCity)
    print('TRAFFIC_DIR: ',TRAFFIC_DIR)
    GeometricalInfo = GeometricalSettingsSpatialPartition(NameCity,TRAFFIC_DIR)
    for resolution in resolutions:
        AllStepsHexagon(GeometricalInfo,resolution,NameCity)
    return GeometricalInfo

if __name__ == '__main__':
    if socket.gethostname()=='artemis.ist.berkeley.edu':
        TRAFFIC_DIR = '/home/alberto/LPSim/traffic_phase_transition'
    else:
        TRAFFIC_DIR = os.getenv('TRAFFIC_DIR')
    list_cities = os.listdir(os.path.join(TRAFFIC_DIR,'data','carto'))
    parallel = False
    resolutions = [8]
    if parallel:
        with Pool(processes = 5) as pool:
            pool.starmap(ComputeHexagonParallelOnCity,[(city,TRAFFIC_DIR,resolutions) for city in list_cities])
    else:
        arguments = [(list_cities[i],TRAFFIC_DIR,resolutions) for i in range(len(list_cities))]
        for arg in arguments:
            ComputeHexagon(*arg)
