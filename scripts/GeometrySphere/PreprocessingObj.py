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
        self.config = GenerateConfigGeometricalSettingsSpatialPartition(city,TRAFFIC_DIR)
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
        self.new_full_network_dir = '/home/alberto/LPSim/LivingCity/berkeley_2018/new_full_network' 
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
        self.DfBegin = None
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
        # INFO FIT
        self.k = None
        self.alpha = None
        self.beta = None
        self.d0 = None



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
                - Vector Field
            NOTE: In this case we are considering just the dataset and not produced any change in the potential.    
                
        """
        ## BASIC NEEDED OBJECTS
        self.ObtainDirectionMatrix()
        self.ObtainODMatrixGrid()
        self.GetLattice()
        PotentialDf, _,VectorField = GeneratePotentialFromFluxes(self.Tij,self.df_distance,self.lattice,self.grid,self.city)
        if os.path.isfile(os.path.join(self.save_dir_grid,'PotentialDataframe.csv')):
            SavePotentialDataframe(PotentialDf,self.save_dir_grid)        
        PI,LC,UCI,result_indices,_,cumulative,Fstar = ComputeUCI(self.grid,PotentialDf,self.df_distance)
        I = {'PI':PI,'LC':LC,'UCI':UCI,"Fstar":Fstar}           
        SaveJsonDict(I,os.path.join(self.save_dir_local,f'UCI_{UCI}.json'))         
        PlotFluxes(self.grid,self.Tij,self.gdf_polygons,self.save_dir_grid,80,UCI)
        PlotNewPopulation(self.grid, self.gdf_polygons,self.save_dir_grid,UCI)
        PlotVFPotMass(self.grid,self.gdf_polygons,PotentialDf,VectorField,self.save_dir_grid,'population','Ti',UCI)
        PotentialContour(self.grid,PotentialDf,self.gdf_polygons,self.save_dir_grid,UCI)
        PotentialSurface(self.grid,self.gdf_polygons,PotentialDf,self.save_dir_grid,UCI)
        PlotRotorDistribution(self.grid,PotentialDf,self.save_dir_grid,UCI)
        PlotLorenzCurve(cumulative,Fstar,result_indices,self.save_dir_grid, 0.1,UCI)
        PlotHarmonicComponentDistribution(self.grid,PotentialDf,self.save_dir_grid,UCI)
        PrintInfoFluxPop(self.grid,self.Tij)    
        return UCI  

    def InitializeDf4Sim(self):

        # Get File.fma each hour
        self.Hour2Files = self.OrderFilesFmaPerHour()
        # Generate The Brgin And End Files For Simulation Common to All Rs
        if not os.path.isfile(os.path.join(self.save_dir_local,'DfBegin.csv')):
            logger.info('Computing DfBegin')
            self.DfBegin = GenerateBeginDf(self.Hour2Files,
                                    self.ODfma_dir,
                                    self.start,
                                    self.polygon2OD,
                                    self.osmid2index,
                                    self.grid,
                                    self.grid_size,
                                    self.OD2grid,
                                    self.city,
                                    self.save_dir_local)
            self.DfBegin.to_csv(os.path.join(self.save_dir_local,'DfBegin.csv'),index=False)
        else:
            self.DfBegin = pd.read_csv(os.path.join(self.save_dir_local,'DfBegin.csv'))
            pass


    def ComputeFit(self):
        '''
            Gravitational model:
                T_ij = k * M_i^alpha * M_j^beta * exp(-d_ij/d0)
            M_i: Number of people living in the grid i
            M_j: Number of people living in the grid j
            d_ij: Distance between the centroids of the grid i and j
            k: constant
            alpha: exponent mass i
            beta: exponent mass j
            d0_1: -1/d0

        '''
        if not os.path.isfile(os.path.join(self.save_dir_grid,'PotentialDataframe.csv')):
            logk,alpha,gamma,d0_2min1 = VespignaniBlock(self.df_distance,self.grid,self.Tij,self.save_dir_potential)            
            self.k = np.exp(logk)
            self.alpha = alpha
            self.beta = gamma
            self.d0 = d0_2min1
        else:
            self.k,self.alpha,self.beta,self.d0 = UploadGravitationalFit(TRAFFIC_DIR,self.city)
    def TotalPopAndFluxes(self):
        """
            Compute the Total Population and Total Fluxes
        """
        logger.info('Computing Total Population and Total Fluxes')
        self.total_population = np.sum(self.grid['population'])
        self.total_flux = np.sum(self.Tij['number_people'])

    def ChangeMorpholgy(self,cov,distribution,num_peaks):
        """
            Change the Morphology of the City
        """
        # Number of People and Fluxes
        self.TotalPopAndFluxes()
        logger.info('Modify Morphology City: {}'.format(self.city))
        logger.info(f"cov: {cov}, num_peaks: {num_peaks}, city: {self.city}")    
        InfoCenters = {'center_settings': {"type":distribution},
                            'covariance_settings':{
                                "covariances":{"cvx":cov,"cvy":cov},
                            "Isotropic": True,
                            "Random": False}}
        logger.info(f'Generating Random Population {self.city}')
        new_population,index_centers = GenerateRandomPopulation(self.grid,num_peaks,self.total_population,InfoCenters,False)
        # From Population, using the gravitational model, generate the fluxes
        logger.info(f'Generating Modified Fluxes {self.city}')
        Modified_Fluxes = GenerateModifiedFluxes(new_population,self.df_distance,self.k,self.alpha,self.beta,self.d0,self.total_flux,False)
        logger.info(f'Computing New Vector Field {self.city}')
        New_Vector_Field = ComputeNewVectorField(Modified_Fluxes,self.df_distance)
        logger.info(f'Computing New Potential {self.city}')
        New_Potential_Dataframe = ComputeNewPotential(New_Vector_Field,self.lattice,new_population)
        PI,LC,UCI,result_indices,_,cumulative,Fstar = ComputeUCI(new_population,New_Potential_Dataframe,self.df_distance)
        I = {'PI':PI,'LC':LC,'UCI':UCI,"Fstar":Fstar}           
        SaveJsonDict(I,os.path.join(self.save_dir_local,f'UCI_{UCI}.json'))         
        PlotRoutineOD(new_population,
                    Modified_Fluxes,
                    self.gdf_polygons,
                    New_Potential_Dataframe,
                    New_Vector_Field,
                    self.save_dir_plots,
                    80,
                    UCI,
                    index_centers,
                    self.Tij,
                    cumulative,
                    Fstar,
                    result_indices)
        return Modified_Fluxes,UCI

## Simulation Output File ##
    def ComputeDf4SimChangedMorphology(self,UCI,R,Modified_Fluxes):
        NPeopleOffset = len(self.DfBegin)
        Df_GivenR = GenerateDfFluxesFromGravityModel(Modified_Fluxes,
                                                        self.osmid2index,
                                                        self.grid2OD,
                                                        self.start) 
        Df = pd.concat([self.DfBegin,Df_GivenR],ignore_index=True)
        assert len(Df) == NPeopleOffset + R*3600
        NPeopleOffset += R*3600
        DfEnd = self.AddEndFileInputSimulation(self,NPeopleOffset) 
        Df = pd.concat([DfEnd,Df_GivenR],ignore_index=True)
        end = self.start + 1
        Df.to_csv(os.path.join(self.new_full_network_dir ,'{0}_oddemand_{1}_{2}_R_{3}_{4}.csv'.format(self.city,self.start,end,str(int(R)),round(UCI,3))),sep=',',index=False)
        return os.path.join(self.new_full_network_dir ,'{0}_oddemand_{1}_{2}_R_{3}_{4}.csv'.format(self.city,self.start,end,str(int(R)),round(UCI,3)))


    def ComputeDf4SimNotChangedMorphology(self,UCI,R):
        """
            @params: UCI: float (Urban Centrality Index)
            Compute the Df for Simulation without changing the morphology of the city
        """
        NPeopleOffset = len(self.DfBegin)
        Df_GivenR = GenerateDfFluxesFromGravityModel(self.Tij,
                                                        self.osmid2index,
                                                        self.grid2OD,
                                                        self.start) 
        Df = pd.concat([self.DfBegin,Df_GivenR],ignore_index=True)
        assert len(Df) == NPeopleOffset + R*3600
        NPeopleOffset += R*3600
        DfEnd = self.AddEndFileInputSimulation(self,NPeopleOffset) 
        Df = pd.concat([DfEnd,Df_GivenR],ignore_index=True)
        end = self.start + 1
        Df.to_csv(os.path.join(self.new_full_network_dir ,'{0}_oddemand_{1}_{2}_R_{3}_{4}.csv'.format(self.city,self.start,end,str(int(R)),round(UCI,3))),sep=',',index=False)
        return os.path.join(self.new_full_network_dir ,'{0}_oddemand_{1}_{2}_R_{3}_{4}.csv'.format(self.city,self.start,end,str(int(R)),round(UCI,3)))



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

    def AddEndFileInputSimulation(self,NPeopleOffset):
        """
            @params: NPeopleOffset: int (Number of people that are inserted from the beginning of time
                                        to the end of the control group)
            @brief: Add the End File for Simulation
        """
        Count = 0
        for time,ODfmaFile in self.Hour2Files.items(): 
            if time > self.start:
                O_vector,D_vector,OD_vector = MapFile2Vectors(ODfmaFile)
                TotalFluxesTime = np.sum(OD_vector)
                # Do Not Change the OD and Concatenate the input for Simulation                            
                Df_GivenR = ReturnFileSimulation(O_vector,
                                            D_vector,
                                            OD_vector,
                                            TotalFluxesTime,
                                            NPeopleOffset,
                                            self.polygon2OD,
                                            self.osmid2index,
                                            self.grid,
                                            self.grid_size,
                                            self.OD2grid,
                                            self.city,
                                            time,
                                            time + 1,
                                            self.save_dir_local)
                NPeopleOffset += TotalFluxesTime
                if Count == 0:
                    DfEnd = Df_GivenR
                else:
                    DfEnd = pd.concat([DfEnd,Df_GivenR],ignore_index=True)
                Count += 1
        return DfEnd


    def ComputeInfoInputSimulation(self,Type):
        """
            Compute the File Input for Simulation.
        """            
        # Update index Users
        NPeopleOffset = np.ones(len(self.ArrayRs))*len(self.DfBegin)
        R2DfSimulation = {R:self.DfBegin for R in self.ArrayRs}
        # Append The Control Group
        for time,ODfmaFile in self.Hour2Files.items():
            # If I want to insert the number of people from the control group
            if time == self.start:
                # Compute the file for the different parameter values
                for R in self.ArrayRs:
                    if Type == "from_data":
                        logger.info("Compute Info Input Simulation from Data At ControlTime R: {}".format(R))
                        # Generate vector of simulation file.
                        O_vector,D_vector,OD_vector = MapFile2Vectors(ODfmaFile)
                        # Insert the control group
                        Df_GivenR = ReturnFileSimulation(O_vector,D_vector,OD_vector,R*3600,NPeopleOffset,
                                                        self.polygon2OD,
                                                        self.osmid2index,
                                                        self.grid,
                                                        self.grid_size,
                                                        self.OD2grid,
                                                        self.city,
                                                        time,
                                                        time + 1,
                                                        self.save_dir_local)
                    else:
                        logger.info("Compute Info Input Simulation from Gravity R: {}".format(R))                        
                        Df_GivenR = GenerateDfFluxesFromGravityModel(self.InfoConfigurationPolicentricity[]['Tij'],self.osmid2index,self.grid2OD,self.start) 
                    R2DfSimulation[R] = pd.concat([R2DfSimulation[R],Df_GivenR],ignore_index=True)
                NPeopleOffset += self.ArrayRs*3600
            elif time > self.start:
                for R in self.ArrayRs:
                    logger.info("Compute Info Input Simulation from Data R: {}".format(R))
                    O_vector,D_vector,OD_vector = MapFile2Vectors(ODfmaFile)
                    R = np.sum(OD_vector)
                    # Do Not Change the OD and Concatenate the input for Simulation                            
                    Df_GivenR = ReturnFileSimulation(O_vector,
                                                D_vector,
                                                OD_vector,
                                                R,
                                                NPeopleOffset,
                                                self.polygon2OD,
                                                self.osmid2index,
                                                self.grid,
                                                self.grid_size,
                                                self.OD2grid,
                                                self.city,
                                                time,
                                                time + 1,
                                                self.save_dir_local)
                    NPeopleOffset += R*3600
                    R2DfSimulation[R] = pd.concat([R2DfSimulation[R],Df_GivenR],ignore_index=True)
                    NPeopleOffset += self.ArrayRs*3600
                else:
                    pass
