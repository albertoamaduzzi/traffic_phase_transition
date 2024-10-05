'''
    @file: PreprocessingObj.py
    @brief: This file contains the class GeometricalSettingsSpatialPartition
    GeometricalSettingsSpatialPartition:
        Properties:
            - 1 - 1 correspndence with the city
            Requires:
                - .graphml file
                - .shp file
                - .tiff file
    @params: crs -> Coordinate Reference System: str
    @params: city -> City Name: str
    @params: config_dir_local -> Configuration Directory Local: str 
    @params: tiff_file_dir_local -> Tiff File Directory Local: str 
    @params: shape_file_dir_local -> Shape File Directory Local: str
    @params: ODfma_dir -> Origin Destination File Matrix Directory: str
    @params: save_dir_local -> Save Directory Local: str
    @params: save_dir_server -> Save Directory Server: str
    @params: GraphFromPhml -> Graph from Phml: ox.graph
    @params: gdf_polygons -> GeoDataFrame Polygons: gpd.GeoDataFrame
    @params: bounding_box -> Bounding Box: tuple
    @params: nodes -> Nodes: None
    @params: edges -> Edges: None
    @params: osmid2index -> Osmid to Index: defaultdict
    @params: index2osmid -> Index to Osmid: defaultdict
    @params: start -> Start: int
    @params: end -> End: int
    @params: R -> Radius: int
    @params: Files2Upload -> Files to Upload: defaultdict
    @params: gdf_hexagons -> GeoDataFrame Hexagons: None
    @params: grid -> Grid: None
    @params: rings -> Rings: None
    @params: lattice -> Lattice: None
    @params: polygon2OD -> Polygon to Origin Destination: None
    @params: OD2polygon -> Origin Destination to Polygon: None
    @params: hexagon2OD -> Hexagon to Origin Destination: None
    @params: OD2hexagon -> Origin Destination to Hexagon: None
    @params: grid2OD -> Grid to Origin Destination: None
    @params: OD2grid -> Origin Destination to Grid: None
    @params: ring2OD -> Ring to Origin Destination: None
    @params: OD2ring -> Origin Destination to Ring: None
    @params: ring2OD -> Ring to Origin Destination: None
    @methods: UpdateFiles2Upload -> Update Files to Upload: None
    
    NOTE: It has got encoded all the structure for the file system we are going to create
'''
# A 
import ast
# C
from collections import defaultdict
# G
import gc
import geopandas as gpd
# J
import json
import osmnx as ox
# M
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
# N
from numba import prange
import numpy as np
# O
import os
# P
import pandas as pd
# S
from shapely.geometry import box,LineString,Point,MultiPoint,MultiLineString,MultiPolygon,Polygon
from shapely.ops import unary_union
import socket
import sys
# T
from termcolor import  cprint
import time

# Project specific
# A
from AlgorithmCheck import *
# C
from ComputeGrid import *
from ComputeHexagon import *
from Config import *

# F
current_dir = os.path.join(os.getcwd()) 
mother_path = os.path.abspath(os.path.join(current_dir, os.pardir))
print('mother_path:', mother_path)
sys.path.append(os.path.join(mother_path, 'PreProcessing'))
sys.path.append(os.path.join(mother_path))
from FittingProcedures import *
# G
from GeometrySphere import *
from GenerateModifiedFluxesSimulation import *
from GravitationalFluxes import *                                               # FIT section
from Grid import *
# H 
from Hexagon import *
if socket.gethostname()=='artemis.ist.berkeley.edu':
    sys.path.append(os.path.join('/home/alberto/LPSim','traffic_phase_transition','scripts','ServerCommunication'))
else:
    sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','ServerCommunication'))
from HostConnection import *
# M
from MainPolycentrism import *
from ModifyPotential import *
# O 
from ODfromfma import *
# P
from plot import *
from Polycentrism import *
from PolycentrismPlot import *
from PolygonSettings import *
from Potential import *
from PreprocessingObj import *


logger = logging.getLogger(__name__)

if socket.gethostname()=='artemis.ist.berkeley.edu':
    TRAFFIC_DIR = '/home/alberto/LPSim/traffic_phase_transition'
else:
    TRAFFIC_DIR = os.getenv('TRAFFIC_DIR')
SERVER_TRAFFIC_DIR = '/home/alberto/LPSim/LivingCity/berkeley_2018'


class GeometricalSettingsSpatialPartition:
    """
    @class: GeometricalSettingsSpatialPartition
    @brief: This class contains the properties of the GeometricalSettingsSpatial

    """
    def __init__(self,city,TRAFFIC_DIR):
        logging.basicConfig(filename=os.path.join(TRAFFIC_DIR,"log.log"), level=logging.INFO)
        self.config =GenerateConfigGeometricalSettingsSpatialPartition(city,TRAFFIC_DIR)
        self.crs = 'epsg:4326'
        self.city = city
        ## 
        self.grid_size = self.config['grid_size']
        self.hexagon_resolution = self.config['hexagon_resolution']
        # INPUT DIRS
        self.config_dir_local = self.config['config_dir_local']
        self.tiff_file_dir_local = self.config['tiff_file_dir_local'] 
        self.shape_file_dir_local = self.config['shape_file_dir_local']
        self.ODfma_dir = self.config['ODfma_dir']
        # OUTPUT DIRS
        self.save_dir_local = self.config['save_dir_local'] 
        self.save_dir_grid = os.path.join(self.save_dir_local,'grid',self.grid_size)
        self.save_dir_hexagon = os.path.join(self.save_dir_local,'hexagon',self.hexagon_resolution)
        self.save_dir_polygon = os.path.join(self.save_dir_local,'polygon')
        self.save_dir_OD = os.path.join(self.save_dir_local,'OD')
        self.save_dir_potential = os.path.join(self.save_dir_local,'potential')
        self.save_dir_plots = os.path.join(self.save_dir_local,'plots')
        self.save_dir_server = self.config['save_dir_server'] # DIRECTORY WHERE TO SAVE THE FILES /home/alberto/LPSim/LivingCity/{city}
        if os.path.isfile(os.path.join(self.save_dir_local,self.city + '_new_tertiary_simplified.graphml')):
            self.GraphFromPhml = ox.load_graphml(filepath = os.path.join(self.save_dir_local,self.city + '_new_tertiary_simplified.graphml')) # GRAPHML FILE
        else:
            raise ValueError('Graph City not found: ',os.path.join(self.save_dir_local,self.city + '_new_tertiary_simplified.graphml'))
        if os.path.isfile(os.path.join(self.shape_file_dir_local,self.city + '.shp')):
            self.gdf_polygons = gpd.read_file(os.path.join(self.shape_file_dir_local,self.city + '.shp')) # POLYGON FILE
            self.bounding_box = self.ComputeBoundingBox()
        else:
            raise ValueError('Polygon City not found')
        self.nodes = None
        self.edges = None
        self.osmid2index = defaultdict()
        self.index2osmid = defaultdict()
        self.start = self.config['start_group_control']
        self.end = self.config['end_group_control']
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
        # INFO FLUXES
        self.Rmin = CityName2RminRmax[self.city][0]
        self.Rmax = CityName2RminRmax[self.city][1]
        self.R = 1
        # INFO NUMBer SIMULATIONS
        self.number_simulation_per_UCI = self.config["number_simulation_per_UCI"]
        self.InfoConfigurationPolicentricity = None
        SaveJsonDict(self.config_dir_local,city + '_geometric_info.json')
        # INFO ALGORITHM
        self.StateAlgorithm = InitWholeProcessStateFunctions()
        logger.info('Geometrical Settings Spatial Partition Inititalized: {}'.format(self.city))

#### GEOMETRICAL SETTINGS ####
    def ComputeBoundingBox(self):
        """Compute the Bounding Box of the City"""
        return self.gdf_polygons.geometry.unary_union.bounds

    def GetLattice(self):
        """
            1- Gets Lattice from grid.

        """
        self.lattice = GetLattice(self.grid,self.grid_size,self.bounding_box,self.save_dir_local)
        self.StateAlgorithm["GetLattice"] = True

    def GetGrid(self):
        """
            1- Get the Grid.
            2- Compute the Boundaries and interior of the Grid.
        """
        if not self.StateAlgorithm["GetGrid"]:
            self.grid = GetGrid(self.grid_size,self.bounding_box,self.crs,self.save_dir_local)
            self.StateAlgorithm["GetGrid"] = True
        else:
            pass
        if not self.StateAlgorithm["GetBoundariesInterior"]:
            self.grid = GetBoundariesInterior(self.grid,self)
            self.StateAlgorithm["GetBoundariesInterior"] = True
        else:
            pass
        self.grid = GetGeometryPopulation(self.gdf_hexagons,self.grid,'grid',self.city)
        self.StateAlgorithm["GetGrid"] = True

    def GetGeometries(self):
        """
            Get the Geometries that are useful for the simulation.
        """
        self.OD2polygon,self.polygon2OD,self.gdf_polygons = Geometry2OD(gdf_geometry = self.gdf_polygons,
                                                                            GraphFromPhml = self.GraphFromPhml,
                                                                            NameCity = self.city,
                                                                            GeometryName ='polygon',
                                                                            save_dir_local = self.save_dir_local,
                                                                            resolution = None)
        # COMPUTE HEXAGON TILING
        resolution = self.hexagon_resolution
        self.gdf_hexagons = GetHexagon(self.gdf_polygons,self.tiff_file_dir_local,self.save_dir_local,self.city,resolution)
        SaveHexagon(self.save_dir_local,resolution,self.gdf_hexagons)
        self.OD2hexagon,self.hexagon2OD,self.gdf_hexagons = Geometry2OD(gdf_geometry = self.gdf_hexagons,
                                                                            GraphFromPhml = self.GraphFromPhml,
                                                                            NameCity = self.city,
                                                                            GeometryName ='hexagon',
                                                                            save_dir_local = self.save_dir_local,
                                                                            resolution = resolution)
        self.gdf_polygons = getPolygonPopulation(self.gdf_hexagons,self.gdf_polygons,self.city)
        SavePolygon(self.save_dir_local,self.gdf_polygons)
        ## ------------------------- INITIALIZE GRID 2 OD ------------------------- ##
        grid_size = self.grid_size
        # Grid Computations and Save
        self.GetGrid()
        self.GetLattice()
        # Map Grid2OD and OD2Grid
        self.OD2grid,self.grid2OD,self.grid = Geometry2OD(gdf_geometry = self.grid,
                                                            GraphFromPhml = self.GraphFromPhml,
                                                            NameCity = self.city,
                                                            GeometryName ='grid',
                                                            save_dir_local = self.save_dir_local,
                                                            resolution = grid_size)

    def ObtainDirectionMatrix(self):
        if not self.StateAlgorithm["GetDirectionMatrix"]:
            self.df_distance = ObtainDirectionMatrix(self.grid,self.save_dir_local,self.grid_size)
            self.StateAlgorithm["GetDirectionMatrix"] = True
        else:
            pass

    def ObtainODMatrixGrid(self):
        """
            Computes:
                - Distance Matrix in format 
                - OD Grid in format Tij
        """
        if not self.StateAlgorithm["GetODGrid"]:
            self.Tij = ObtainODMatrixGrid(self.save_dir_local,self.grid_size,self.grid)
            self.StateAlgorithm["GetODGrid"] = True
        else:
            pass

#### Vector Fields ####

    def RoutineVectorFieldAndPotential(self):
        """
            Computes:
                - Distance Matrix in format 
                - OD Grid in format Tij
        """
        grid_size = self.grid_size
        ## BASIC NEEDED OBJECTS
        self.ObtainDirectionMatrix()
        self.ObtainODMatrixGrid()
        self.GetLattice()
        if not self.StateAlgorithm["GetVectorField"]:
            self.VectorField = GetVectorField(self.Tij,self.df_distance)
            self.StateAlgorithm["GetVectorField"] = True
        self.VectorFieldDir = os.path.join(TRAFFIC_DIR,'data','carto',self.city,'grid',str(grid_size))
        if not self.StateAlgorithm["GetPotentialLattice"]:
            self.lattice = GetPotentialLattice(self.lattice,self.VectorField)
            self.StateAlgorithm["GetPotentialLattice"] = True
        else:
            pass
        if not self.StateAlgorithm["GetPotentialDataframe"]:
            self.lattice = SmoothPotential(self.lattice)
            self.StateAlgorithm["GetPotentialDataframe"] = True
        else:
            pass
        if not self.StateAlgorithm["ConvertLattice2PotentialDataframe"]:
            self.PotentialDataframe = ConvertLattice2PotentialDataframe(self.lattice)
            self.StateAlgorithm["ConvertLattice2PotentialDataframe"] = True
        else:
            pass
        if not self.StateAlgorithm["CompletePotentialDataFrame"]:
            self.PotentialDataframe = CompletePotentialDataFrame(self.VectorField,self.grid,self.PotentialDataframe)
            self.StateAlgorithm["CompletePotentialDataFrame"] = True
        else:
            pass
        if os.path.isfile(os.path.join(self.save_dir_grid,'PotentialDataframe.csv')):
            SavePotentialDataframe(self.PotentialDataframe,self.save_dir_grid)
        if os.path.isfile(os.path.join(self.VectorFieldDir,'VectorField.csv')):
            SaveVectorField(self.VectorField,self.VectorFieldDir)        



#### Prepare Input For Simulation ####

    def SetRmaxDivisibleByNSim(self):
        """
            Set Rmax divisible by the number of simulations per UCI
        """
        Delta = self.Rmax - self.Rmin
        self.Rmax = self.Rmin + Delta + Delta%self.number_simulation_per_UCI
        self.config["Rmax"] = self.Rmax
        self.config["number_simulation_per_UCI"] = self.number_simulation_per_UCI + 1
        self.ArrayRs = np.arange(self.Rmin,self.Rmax,self.number_simulation_per_UCI,dtype=int)
        self.config["ArrayRs"] = list(self.ArrayRs)
        SaveJsonDict(self.config_dir_local,self.city + '_geometric_info.json')        
        logger.info(f'New Rmax {self.Rmax}, New number of simulations {self.number_simulation_per_UCI}')

    def UpdateFiles2Upload(self,local_file,server_file):
        self.Files2Upload[local_file] = server_file
        logger.info('Files to Upload Updated: {}'.format(local_file))



    def OrderFilesFmaPerHour(self):
        Hour2Files = defaultdict()
        for file in os.listdir(os.path.join(self.ODfma_dir)):
            if file.endswith('.fma'):
                Hour2Files[int(file.split('.')[0].split('D')[1])] = file
        Hour2Files = sorted(Hour2Files)
        logger.info('Files Ordered in Hour2Files')
        return Hour2Files

    def ComputeInfoInputSimulation(self,Type):
        """
            Compute the File Input for Simulation.
        """            
        # Get File.fma each hour
        self.Hour2Files = self.OrderFilesFmaPerHour()
        # Iterate over all the files
        for grid_size in self.grid_sizes:
            # Generate The Brgin And End Files For Simulation Common to All Rs
            DfBegin = GenerateBeginDf(self.Hour2Files,
                                    self.ODfma_dir,
                                    self.start,
                                    self.polygon2OD,
                                    self.osmid2index,
                                    self.grid,
                                    grid_size,
                                    self.OD2grid,
                                    self.city,
                                    self.save_dir_local)
            # Update index Users
            NPeopleOffset = np.ones(len(self.ArrayRs))*len(DfBegin)
            R2DfSimulation = {R:None for R in self.ArrayRs}
            # Append The Control Group
            for time,ODfmaFile in self.Hour2Files.items():
                if time == self.start:
                    for R in self.ArrayRs:
                        if Type == "from_data":

                            Df_GivenR = GenerateDfFluxesFromData()
                        else:
                            Df_GivenR = GenerateDfFluxesFromGravityModel(self.InfoConfigurationPolicentricity[]['Tij'],self.osmid2index,self.grid2OD,self.start) 
                        R2DfSimulation[R] = pd.concat([DfBegin,Df_GivenR],ignore_index=True)
                    NPeopleOffset += self.ArrayRs*3600
                elif time > self.start:
                    for R in self.ArrayRs:
                        O_vector,D_vector,OD_vector = MapFile2Vectors(ODfmaFile)
                        R = np.sum(OD_vector)
                        # Do Not Change the OD and Concatenate the input for Simulation                            
                        df2 = ReturnFileSimulation(O_vector,
                                                D_vector,
                                                OD_vector,
                                                R,
                                                NPeopleOffset,
                                                self.polygon2OD,
                                                self.osmid2index,
                                                self.grid,
                                                grid_size,
                                                self.OD2grid,
                                                self.city,
                                                time,
                                                time + 1,
                                                self.save_dir_local)
                        OffsetNPeople += R*3600
                        R2DfSimulation[R] = pd.concat([R2DfSimulation[R],Df_GivenR],ignore_index=True)
                    NPeopleOffset += self.ArrayRs*3600
                else:
                    pass



                    # Case Read Fluxes From Data 
                    
                        # NOTE: ADD HERE THE POSSIBILITY OF HAVING OD FROM POTENTIAL CONSIDERATIONS
                        O_vector,D_vector,OD_vector = MapFile2Vectors(ODfmaFile)
                        for R in self.ArrayRs:
                            # Generate the Scaled OD
                            logger.info(f"Generating Scaled OD from fma, R: {R}")
                            ReturnFileSimulation(O_vector,
                                                D_vector,
                                                OD_vector,
                                                R,
                                                self.polygon2OD,
                                                self.osmid2index,
                                                self.grid,
                                                grid_size,
                                                self.OD2grid,
                                                self.city,
                                                start,
                                                end,
                                                self.save_dir_local,
                                                seconds_in_minute = 60)
                # Case Generate Fluxes
                    else:
                        GetODForSimulation(Tij_modified,
                        CityName2RminRmax,
                        NameCity,
                        osmid2index,
                        grid2OD,
                        p,
                        save_dir_local,
                        start = 7,
                        end = 8,
                        UCI = None,
                        df2 = None
                        )
                df1.sort_values(by=['dep_time'],ascending=True)            
                if socket.gethostname()!='artemis.ist.berkeley.edu':
                    self.UpdateFiles2Upload(os.path.join(self.save_dir_local,'grid',str(round(grid_size,3)),'ODgrid.csv'),os.path.join(self.save_dir_server,'grid',str(round(grid_size,3)),'ODgrid.csv'))
                    print('R Output: ',ROutput)
                    for R in ROutput:
                        self.UpdateFiles2Upload(os.path.join(self.save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(self.city,start,end,int(R))),os.path.join(self.save_dir_server,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(NameCity,start,end,int(R))) )
