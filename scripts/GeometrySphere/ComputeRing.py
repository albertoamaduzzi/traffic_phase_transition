from termcolor import  cprint
import sys
import os
import json
import numpy as np
from PreprocessingObj import *
from PolygonSettings import *
import pandas as pd
from Hexagon import *
from Grid import *
from Ring import *
from multiprocessing import Pool
import socket
if socket.gethostname()=='artemis.ist.berkeley.edu':
    sys.path.append(os.path.join('/home/alberto/LPSim','traffic_phase_transition','scripts','ServerCommunication'))
else:
    sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','ServerCommunication'))
from HostConnection import *
if socket.gethostname()=='artemis.ist.berkeley.edu':
    sys.path.append(os.path.join('/home/alberto/LPSim','traffic_phase_transition','scripts','PreProcessing'))
else:
    sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','PreProcessing'))
from plot import *
import logging
logger = logging.getLogger(__name__)

def AllStepsRing(GeometricalInfo,radius,NameCity):
    '''
        This function computes all the steps for the ring partition
    '''
    GeometricalInfo.ring = GetRing(radius,
                                   GeometricalInfo.bounding_box,
                                   GeometricalInfo.crs,
                                   GeometricalInfo.save_dir_local)
    GeometricalInfo.ring = GetGeometryPopulation(GeometricalInfo.gdf_hexagons,GeometricalInfo.ring,'ring',NameCity)
    SaveRing(GeometricalInfo.save_dir_local,radius,GeometricalInfo.ring)
    plot_ring_tiling(GeometricalInfo.rings,GeometricalInfo.save_dir_local,GeometricalInfo.number_of_rings,radius)
    GeometricalInfo.OD2ring,GeometricalInfo.ring2OD,GeometricalInfo.ring = Geometry2OD(gdf_geometry = GeometricalInfo.ring,
                                                                 GraphFromPhml = GeometricalInfo.GraphFromPhml,
                                                                 NameCity = GeometricalInfo.city,
                                                                 GeometryName ='ring',
                                                                 save_dir_local = GeometricalInfo.save_dir_local,
                                                                 resolution = radius)
    
    # ADD FILES TO UPLOAD
    GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'ring',str(radius),'ring.geojson'),os.path.join(GeometricalInfo.save_dir_server,'ring',str(radius),'ring.geojson'))
    GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'ring',str(radius),'ring2origindest.json'),os.path.join(GeometricalInfo.save_dir_server,'ring',str(radius),'ring2origindest.json'))
    GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'ring',str(radius),'origindest2ring.json'),os.path.join(GeometricalInfo.save_dir_server,'ring',str(radius),'origindest2ring.json')) 
    MakeDir(GeometricalInfo.save_dir_server,'ring')
    MakeDir(os.path.join(GeometricalInfo.save_dir_server,'ring'),str(radius))
    for file in GeometricalInfo.Files2Upload:
        Upload2ServerPwd(file,GeometricalInfo.Files2Upload[file],GeometricalInfo.config_dir_local)

def ComputeRing(NameCity,TRAFFIC_DIR):
    '''
        Compute Rings In parallel. For each city.
    '''
    GeometricalInfo = GeometricalSettingsSpatialPartition(NameCity,TRAFFIC_DIR)
    radiuses = list(np.arange(10,15,1))
    ParametersRingParallel = [(GeometricalInfo,radius,NameCity) for radius in radiuses]
    with Pool(processes = 2) as pool:
        pool.map(AllStepsRing,ParametersRingParallel)

if __name__=='__main__':
    if socket.gethostname()=='artemis.ist.berkeley.edu':
        TRAFFIC_DIR ='/home/alberto/LPSim/traffic_phase_transition'
    else:
        TRAFFIC_DIR = os.getenv('TRAFFIC_DIR')
    list_cities = os.listdir(os.path.join(TRAFFIC_DIR,'data','carto'))
    arguments = [(list_cities[i],TRAFFIC_DIR) for i in range(len(list_cities))]
    with Pool() as pool:
        pool.map(ComputeRing,arguments)