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
sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','ServerCommunication'))
from HostConnection import *
sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','PreProcessing'))
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
    SaveGrid(GeometricalInfo.save_dir_local,grid_size,GeometricalInfo.grid)
    SaveLattice(GeometricalInfo.save_dir_local,grid_size,GeometricalInfo.lattice)
    plot_grid_tiling(GeometricalInfo.grid,GeometricalInfo.gdf_polygons,GeometricalInfo.save_dir_local,grid_size)
    GeometricalInfo.OD2grid,GeometricalInfo.grid2OD = Geometry2OD(gdf_geometry = GeometricalInfo.grid,
                                                                    GraphFromPhml = GeometricalInfo.GraphFromPhml,
                                                                    NameCity = GeometricalInfo.city,
                                                                    GeometryName ='grid',
                                                                    save_dir_local = GeometricalInfo.save_dir_local,
                                                                    resolution = grid_size)
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
    MakeDir(GeometricalInfo.save_dir_server,'grid')
    MakeDir(os.path.join(GeometricalInfo.save_dir_server,'grid'),str(grid_size))
    for file in GeometricalInfo.Files2Upload:
        Upload2ServerPwd(file,GeometricalInfo.Files2Upload[file],GeometricalInfo.config_dir_local)


def ComputeGrid(NameCity,TRAFFIC_DIR):
    print('Computing Grid for city: ',NameCity)
    print('TRAFFIC_DIR: ',TRAFFIC_DIR)
    GeometricalInfo = GeometricalSettingsSpatialPartition(NameCity,TRAFFIC_DIR)
    GeometricalInfo.gdf_hexagons = GetHexagon(GeometricalInfo.gdf_polygons,GeometricalInfo.tiff_file_dir_local,GeometricalInfo.save_dir_local,NameCity,8)
    grid_sizes = list(np.arange(0.02,0.1,0.01))
    ParametersGridParallel = [(GeometricalInfo,grid_size,NameCity) for grid_size in grid_sizes]
    for grid_size in grid_sizes:
        AllStepsGrid(GeometricalInfo,grid_size,NameCity)
#    with Pool(processes = 5) as pool:
#        pool.starmap(AllStepsGrid,ParametersGridParallel)

if __name__=='__main__':
    '''
        Compute Grids In parallel. For each city.
    '''
    TRAFFIC_DIR = os.getenv('TRAFFIC_DIR')
    list_cities = ['BOS']#os.listdir(os.path.join(TRAFFIC_DIR,'data','carto'))
    arguments = [(list_cities[i],TRAFFIC_DIR) for i in range(len(list_cities))]
    for arg in arguments:
        ComputeGrid(*arg)
#    with Pool(processes=3) as pool:
#        pool.starmap(ComputeGrid,arguments)
