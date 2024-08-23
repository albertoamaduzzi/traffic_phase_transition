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

def AllStepsGrid(GeometricalInfo,grid_size,NameCity):
    GeometricalInfo.grid = GetGrid(grid_size,
                                    GeometricalInfo.bounding_box,
                                    GeometricalInfo.crs,
                                    GeometricalInfo.save_dir_local)
    GeometricalInfo.grid = GetGeometryPopulation(GeometricalInfo.gdf_hexagons,GeometricalInfo.grid,'grid',NameCity)
    GeometricalInfo.lattice = GetLattice(GeometricalInfo.grid,grid_size,GeometricalInfo.bounding_box,GeometricalInfo.save_dir_local)
    plot_grid_tiling(GeometricalInfo.grid,GeometricalInfo.gdf_polygons,GeometricalInfo.save_dir_local,grid_size)
    GeometricalInfo.OD2grid,GeometricalInfo.grid2OD,GeometricalInfo.grid = Geometry2OD(gdf_geometry = GeometricalInfo.grid,
                                                                    GraphFromPhml = GeometricalInfo.GraphFromPhml,
                                                                    NameCity = GeometricalInfo.city,
                                                                    GeometryName ='grid',
                                                                    save_dir_local = GeometricalInfo.save_dir_local,
                                                                    resolution = grid_size)
    SaveGrid(GeometricalInfo.save_dir_local,grid_size,GeometricalInfo.grid)
    SaveLattice(GeometricalInfo.save_dir_local,grid_size,GeometricalInfo.lattice)

    direction_matrix, bool_ = GetDirectionMatrix(GeometricalInfo.save_dir_local,grid_size)
    if bool_:
        pass
    else:
        direction_matrix,distance_matrix = ComputeDirectionMatrix(GeometricalInfo.grid)
        direction_distance_matrix = DirectionDistance2Df(direction_matrix,distance_matrix)
        SaveDirectionDistanceMatrix(GeometricalInfo.save_dir_local,grid_size,direction_distance_matrix)
    
    # ADD FILES TO UPLOAD
    GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'grid',str(grid_size),'grid.geojson'),os.path.join(GeometricalInfo.save_dir_server,'grid',str(grid_size),'grid.geojson'))
    GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'grid',str(grid_size),'grid2origindest.json'),os.path.join(GeometricalInfo.save_dir_server,'grid',str(grid_size),'grid2origindest.json'))
    GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'grid',str(grid_size),'origindest2grid.json'),os.path.join(GeometricalInfo.save_dir_server,'grid',str(grid_size),'origindest2grid.json')) 
    GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'grid',str(grid_size),'centroid_lattice.graphml'),os.path.join(GeometricalInfo.save_dir_server,'grid',str(grid_size),'centroid_lattice.graphml'))
    GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'grid',str(grid_size),'direction_distance_matrix.csv'),os.path.join(GeometricalInfo.save_dir_server,'grid',str(grid_size),'direction_distance_matrix.csv'))
    if not socket.gethostname()=='artemis.ist.berkeley.edu':
        MakeDir(GeometricalInfo.save_dir_server,'grid')
        MakeDir(os.path.join(GeometricalInfo.save_dir_server,'grid'),str(grid_size))
        for file in GeometricalInfo.Files2Upload:
            Upload2ServerPwd(file,GeometricalInfo.Files2Upload[file],GeometricalInfo.config_dir_local)


def ComputeGrid(GeometricalInfo,NameCity,grid_sizes):
    print('Computing Grid for city: ',NameCity)
    GeometricalInfo.gdf_hexagons = GetHexagon(GeometricalInfo.gdf_polygons,GeometricalInfo.tiff_file_dir_local,GeometricalInfo.save_dir_local,NameCity,8)
    for grid_size in grid_sizes:
        AllStepsGrid(GeometricalInfo,grid_size,NameCity)
#    with Pool(processes = 5) as pool:
#        pool.starmap(AllStepsGrid,ParametersGridParallel)

def ComputeGridParallelOnCity(NameCity,TRAFFIC_DIR,grid_sizes):
    print('Computing Grid in Parallel for city: ',NameCity)
    print('TRAFFIC_DIR: ',TRAFFIC_DIR)
    GeometricalInfo = GeometricalSettingsSpatialPartition(NameCity,TRAFFIC_DIR)
    GeometricalInfo.gdf_hexagons = GetHexagon(GeometricalInfo.gdf_polygons,GeometricalInfo.tiff_file_dir_local,GeometricalInfo.save_dir_local,NameCity,8)
    for grid_size in grid_sizes:
        print("Computing Grid for grid_size: ",grid_size)
        AllStepsGrid(GeometricalInfo,grid_size,NameCity)

if __name__=='__main__':
    '''
        Compute Grids In parallel. For each city.
    '''
    if socket.gethostname()=='artemis.ist.berkeley.edu':
        TRAFFIC_DIR = '/home/alberto/LPSim/traffic_phase_transition'
    else:
        TRAFFIC_DIR = os.getenv('TRAFFIC_DIR')
    list_cities = os.listdir(os.path.join(TRAFFIC_DIR,'data','carto'))
    parallel = False
    grid_sizes = [0.02]
    if parallel:
        arguments = [(list_cities[i],TRAFFIC_DIR,grid_sizes) for i in range(len(list_cities))]
        with Pool(processes=3) as pool:
            pool.starmap(ComputeGridParallelOnCity,arguments)
    else:
        arguments = [(list_cities[i],grid_sizes) for i in range(len(list_cities))]
        for arg in arguments:
            GeometricalInfo = GeometricalSettingsSpatialPartition(arg[0],TRAFFIC_DIR)
            ComputeGrid(GeometricalInfo,*arg)
