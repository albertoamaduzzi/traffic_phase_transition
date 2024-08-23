from termcolor import  cprint
import os
import json
import numpy as np
import multiprocessing as mp
import pandas as pd
import sys
import socket
if socket.gethostname()=='artemis.ist.berkeley.edu':
    sys.path.append(os.path.join('/home/alberto/LPSim','traffic_phase_transition','scripts','ServerCommunication'))
else:
    sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','ServerCommunication'))

from HostConnection import *
from PreprocessingObj import *
from PolygonSettings import *
from Hexagon import *
from Grid import *
from Ring import *
from Polygon import *
from ODfromfma import *
import warnings
warnings.filterwarnings("ignore")

'''
    This script is responsible for tiling city with different geometrical shapes.
    The geometrical shapes are:
        - Hexagon (Since the tiff files are read in hexagons using h3 and rasterio)
        - Grid (Since the definition of the grid allow the definition of the gradient and the curl, in the perspective of defining a potential)
        - Ring (Since the definition of the ring since for monocentric cities we have a clear definition of the ring)
        - Polygon (Since the shape files contains plygons, and from these shape files we download the osmnx graph)
    The main function is the following:
        - main(NameCity,TRAFFIC_DIR)
    The main function is responsible for the following:
        
'''


def main(NameCity,TRAFFIC_DIR):        
    GeometricalInfo = GeometricalSettingsSpatialPartition(NameCity,TRAFFIC_DIR)
    ## FROM POLYGON TO ORIGIN DESTINATION -> OD FILE
    resolutions = [8]#[6,7,8]
    grid_sizes = [0.02]#list(np.arange(0.02,0.1,0.01))
    n_rings = list(np.arange(10,15,1))
    ## ------------------------- INITIALIZE POLYGON 2 OD ------------------------- ##
    GeometricalInfo.OD2polygon,GeometricalInfo.polygon2OD,GeometricalInfo.gdf_polygons = Geometry2OD(gdf_geometry = GeometricalInfo.gdf_polygons,
                                                                        GraphFromPhml = GeometricalInfo.GraphFromPhml,
                                                                        NameCity = GeometricalInfo.city,
                                                                        GeometryName ='polygon',
                                                                        save_dir_local = GeometricalInfo.save_dir_local,
                                                                        resolution = None)
    # ADD FILES TO UPLOAD
    GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'polygon','polygon2origindest.json'),os.path.join(GeometricalInfo.save_dir_server,'polygon','polygon2origindest.json'))
    GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'polygon','origindest2polygon.json'),os.path.join(GeometricalInfo.save_dir_server,'polygon','origindest2polygon.json'))
    
    ##  ------------------------- INITIALIZE HEXAGON 2 OD ------------------------- ##
    for resolution in resolutions:
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
    ## ------------------------- INITIALIZE GRID 2 OD ------------------------- ##
    for grid_size in grid_sizes:
        # Grid Computations and Save
        GeometricalInfo.grid = GetGrid(grid_size,
                                    GeometricalInfo.bounding_box,
                                    GeometricalInfo.crs,
                                    GeometricalInfo.save_dir_local)
        GeometricalInfo.grid = GetGeometryPopulation(GeometricalInfo.gdf_hexagons,GeometricalInfo.grid,'grid',NameCity)
        GeometricalInfo.lattice = GetLattice(GeometricalInfo.grid,grid_size,GeometricalInfo.bounding_box,GeometricalInfo.save_dir_local)
        # Map Grid2OD and OD2Grid
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
        if socket.gethostname()!='artemis.ist.berkeley.edu':
            GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'grid',str(grid_size),'grid.geojson'),os.path.join(GeometricalInfo.save_dir_server,'grid',str(grid_size),'grid.geojson'))
            GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'grid',str(grid_size),'grid2origindest.json'),os.path.join(GeometricalInfo.save_dir_server,'grid',str(grid_size),'grid2origindest.json'))
            GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'grid',str(grid_size),'origindest2grid.json'),os.path.join(GeometricalInfo.save_dir_server,'grid',str(grid_size),'origindest2grid.json')) 
            GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'grid',str(grid_size),'centroid_lattice.graphml'),os.path.join(GeometricalInfo.save_dir_server,'grid',str(grid_size),'centroid_lattice.graphml'))
            GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'grid',str(grid_size),'direction_distance_matrix.csv'),os.path.join(GeometricalInfo.save_dir_server,'grid',str(grid_size),'direction_distance_matrix.csv'))
    ## ------------------------- INITIALIZE RING 2 OD ------------------------- ##
    if 0==1:
        for n_ring in n_rings:
            # Ring Computations and Save
            GeometricalInfo.rings = GetRing(n_ring,
                                            GeometricalInfo.gdf_polygons,
                                            GeometricalInfo.crs,
                                            GeometricalInfo.save_dir_local)
            GeometricalInfo.rings = GetGeometryPopulation(GeometricalInfo.gdf_hexagons,GeometricalInfo.rings,'ring',NameCity)
            SaveRing(GeometricalInfo.save_dir_local,n_ring,GeometricalInfo.rings)
            GeometricalInfo.OD2ring,GeometricalInfo.ring2OD,GeometricalInfo.rings = Geometry2OD(gdf_geometry = GeometricalInfo.rings,
                                                                        GraphFromPhml = GeometricalInfo.GraphFromPhml,
                                                                        NameCity = GeometricalInfo.city,
                                                                        GeometryName ='ring',
                                                                        save_dir_local = GeometricalInfo.save_dir_local,
                                                                        resolution = n_ring)
            if socket.gethostname()!='artemis.ist.berkeley.edu':
            # ADD FILES TO UPLOAD
                GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'ring',str(n_ring),'ring.geojson'),os.path.join(GeometricalInfo.save_dir_server,'ring',str(n_ring),'ring.geojson'))
                GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'ring',str(n_ring),'ring2origindest.json'),os.path.join(GeometricalInfo.save_dir_server,'ring',str(n_ring),'ring2origindest.json'))
                GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'ring',str(n_ring),'origindest2ring.json'),os.path.join(GeometricalInfo.save_dir_server,'ring',str(n_ring),'origindest2ring.json'))
    ## -------------------------- FMA FILES -------------------------- ##
    AdjustDetailsBeforeConvertingFma2Csv(GeometricalInfo)
    for file in os.listdir(os.path.join(GeometricalInfo.ODfma_dir)):
        if file.endswith('.fma'):
            start = int(file.split('.')[0].split('D')[1])
            print('start: ',start)  
            end = start + 1
            GeometricalInfo.start = start
            GeometricalInfo.end = end
            cprint('file.fma: ' + file,'yellow')
            ODfmaFile = os.path.join(GeometricalInfo.ODfma_dir,file)
            for grid_size in grid_sizes:
                if start == 7:
                    _,_,ROutput = OD_from_fma(GeometricalInfo.polygon2OD,
                                        GeometricalInfo.osmid2index,
                                        GeometricalInfo.grid,
                                        grid_size,
                                        GeometricalInfo.OD2grid,
                                        NameCity,
                                        ODfmaFile,
                                        start,
                                        end,
                                        GeometricalInfo.save_dir_local,
                                        n_rings,
                                        grid_sizes,
                                        resolutions,
                                        offset = 6,
                                        seconds_in_minute = 60,
                                        )
                    if socket.gethostname()!='artemis.ist.berkeley.edu':
                        GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'grid',str(round(grid_size,3)),'ODgrid.csv'),os.path.join(GeometricalInfo.save_dir_server,'grid',str(round(grid_size,3)),'ODgrid.csv'))
                        print('R Output: ',ROutput)
                        for R in ROutput:
                            GeometricalInfo.UpdateFiles2Upload(os.path.join(GeometricalInfo.save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(NameCity,start,end,int(R))),os.path.join(GeometricalInfo.save_dir_server,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(NameCity,start,end,int(R))) )
 #   for localfile in GeometricalInfo.Files2Upload.keys():
 #       serverfile = GeometricalInfo.Files2Upload[localfile]
 #       Upload2ServerPwd(localfile,serverfile)


if __name__=='__main__':
    '''
        Compute Grids, Rings and Hexagons In parallel. For each city.
        This is useful once the different geometries are computed, in ComputeGrid.py,ComputeRings.py ecc.
        as in those cases the different geometries are computed for each of 
        the geometrical parameters asked.

    '''
    if socket.gethostname()=='artemis.ist.berkeley.edu':
        TRAFFIC_DIR = '/home/alberto/LPSim/traffic_phase_transition'
    else:
        TRAFFIC_DIR = os.getenv('TRAFFIC_DIR')
    list_cities = os.listdir(os.path.join(TRAFFIC_DIR,'data','carto'))
    arguments = [(list_cities[i],TRAFFIC_DIR) for i in range(len(list_cities))]
    for argument in arguments:
        if argument[0] == 'BOS':
            main(*argument)
#    with mp.Pool(processes=2) as pool:
#        pool.starmap(main,arguments)