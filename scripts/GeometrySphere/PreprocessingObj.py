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
sys.path.append(os.path.join(mother_path,'PreProcessing'))
from CreateCartoSimulator import *

logger = logging.getLogger(__name__)

def RetrieveUCI2UCIIntervalFromSavedGrid(dir_new_grids,dir_grid):
    """
        @param dir_new_grids: Directory where the grids are saved
        Extract the new Grids already computed.
        Generate the dictionary UCI2UCIInterval {0.0:[],0.1:[],...}
    """
    PlotDir = os.path.join(dir_new_grids,"plots")
    UCIS = np.linspace(0,1,11)
    UCI2UCIInterval = {UCI:[] for UCI in UCIS}
    for file in os.listdir(dir_new_grids):
        if "Grid_" in file:
            UCI = float(file.split("_")[1].split(".parquet")[0])
            for U in UCIS:
                if UCI >= U and UCI < U + 0.1:
                    UCI2UCIInterval[U].append(UCI)
            if os.path.exists(f"{PlotDir}/grid_{UCI}.png"):
                continue
            else:
                g = pd.read_parquet(f"{dir_new_grids}/{file}")
                grid = gpd.read_file(os.path.join(dir_grid,"grid.geojson"))
                grid["population"] = g["population"]
                fig, ax = plt.subplots(1,1, figsize=(10,10))
                grid.plot("population",ax=ax, legend=True)    
                ax.set_title(f"UCI: {UCI}")
                plt.savefig(f"{PlotDir}/grid_{UCI}.png")
    return UCI2UCIInterval

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
        self.save_dir_grid = os.path.join(self.save_dir_local,'grid',str(self.grid_size))
        self.save_dir_hexagon = os.path.join(self.save_dir_local,'hexagon',str(self.hexagon_resolution))
        self.save_dir_polygon = os.path.join(self.save_dir_local,'polygon')
        self.save_dir_OD = os.path.join(self.save_dir_local,'OD')
        self.save_dir_potential = os.path.join(self.save_dir_local,'potential')
        self.save_dir_plots = os.path.join(self.save_dir_local,'plots')
        self.save_dir_server = self.config['save_dir_server'] # DIRECTORY WHERE TO SAVE THE FILES /home/alberto/LPSim/LivingCity/{city}
        self.new_full_network_dir = '/home/alberto/LPSim/LivingCity/berkeley_2018/new_full_network' 
        self.berkeley_2018 = SERVER_TRAFFIC_DIR
        if os.path.isfile(os.path.join(self.save_dir_local,self.city + '_new_tertiary_simplified.graphml')):
            self.GraphFromPhml = ox.load_graphml(filepath = os.path.join(self.save_dir_local,self.city + '_new_tertiary_simplified.graphml')) # GRAPHML FILE
            self.mobility_planner = mobility_planner(self.city)

        else:
            raise ValueError('Graph City not found: ',os.path.join(self.save_dir_local,self.city + '_new_tertiary_simplified.graphml'))
        if os.path.isfile(os.path.join(self.shape_file_dir_local,self.city + '.shp')):
            self.gdf_polygons = gpd.read_file(os.path.join(self.shape_file_dir_local,self.city + '.shp')) # POLYGON FILE
            self.bounding_box = self.ComputeBoundingBox()
        else:
            raise ValueError('Polygon City not found')
        self.nodes = None
        self.edges = None
        if not os.path.isfile(os.path.join(self.save_dir_local,"osmid2idx.json")):
            self.osmid2index = {data['osmid']: data['index'] for _, _, data in self.GraphFromPhml.edges(data=True) if 'osmid' in data and 'index' in data}
            #            self.osmid2index = self.GraphFromPhml[['osmid','index']].set_index('osmid').to_dict()['index']
        else:
            with open(os.path.join(self.save_dir_local,"osmid2idx.json")) as f:
                self.osmid2index = json.load(f)
        if not os.path.isfile(os.path.join(self.save_dir_local,"idx2osmid.json")):
            self.index2osmid = {v:k for k,v in self.osmid2index.items()}
        else:
            with open(os.path.join(self.save_dir_local,"idx2osmid.json")) as f:
                self.index2osmid = json.load(f)
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
        # INFO UCI
        self.NumberUCIs = self.config["NumberUCIs"]
        self.IndicesEdge = None
        self.IndicesInside = None  
        self.Smaxi = None
        self.Smaxi_Mass = None
        self.Dmaxi = None
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
        self.SetRmaxDivisibleByNSim()
        # Control that I have a configuration for each interval of UCI accepted
        self.UCIInterval2UCIAccepted = {UCIInterval:None for UCIInterval in np.linspace(0,1,self.NumberUCIs)}
        logger.info("Bounding Box: {}".format(self.bounding_box))
#### GEOMETRICAL SETTINGS ####
    def ComputeBoundingBox(self):
        """Compute the Bounding Box of the City"""
        if self.city == 'LAX':
            return (-118.951721, 33.65004, -117.646374, 34.823301)
        else:
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
        if self.city != "":
            logger.info(f"Get Grid {self.city}")
            self.grid = GetGrid(self.grid_size,self.bounding_box,self.crs,self.save_dir_local)
            logger.info(f"Get Boundaries and Interior {self.city}")
            self.grid = GetBoundariesInterior(self.grid,self.gdf_polygons,self.city)
            logger.info('Grid and Boundaries and Interior Computed')
            self.grid = GetGeometryPopulation(self.gdf_hexagons,self.grid,'grid',self.city)
        else:
            self.grid = gpd.read_file(os.path.join(self.save_dir_grid,'grid.geojson'))




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
        self.grid["with_roads"] = self.grid['index'].apply(lambda x: str(x) in list(self.grid2OD.keys()))            
        SaveGrid(self.save_dir_local,self.grid_size,self.grid)
        self.ObtainDirectionMatrix()
        self.ObtainODMatrixGrid()
        logger.info(f"Compute grid Idx 2 Origin Destination {self.city}")
        self.gridIdx2dest = GridIdx2OD(self.grid)
        self.gridIdx2ij = {self.grid['index'][i]: (self.grid['i'].tolist()[i],self.grid['j'].tolist()[i]) for i in range(len(self.grid))}


### GENERATE MASS ###



    def GeneratePopulationAndSetUCIs(self):
        """
            @description:
                - Generate random population given covariance of the gaussian model and number of ceners.
                - Compute UCI of such a configuration and insert it in the UCIInterval2UCI
                - In this way UCIInterval2UCI contains all the UCIs that are going to be used for the simulations.
        """
        self.UCIInterval2UCI = RetrieveUCI2UCIIntervalFromSavedGrid(self.save_dir_local,self.save_dir_grid)
        # Either Fill From Zero or Complete the Missing UCIs (In case the simulations stop or some problem occurs)
        if self.UCIInterval2UCI is None:
            IsFull = True
            for UCIInterval,UCIs in self.UCIInterval2UCI.items():
                if len(UCIs) < 4:
                    IsFull = False
                    break
            if not IsFull:
                # Compute Parallely all the UCIs
                self.UCIInterval2UCI,_ = GenerateRandomPopulationAndComputeUCI(self.config['covariances'],
                                                                    ['gaussian'],
                                                                    self.config['list_peaks'],
                                                                    self.grid,
                                                                    np.sum(self.grid["population"]),
                                                                    self.df_distance,
                                                                    self.IndicesInside,
                                                                    self.Smaxi_Mass,
                                                                    self.Dmax_ij_Mass,
                                                                    self.UCIInterval2UCI,
                                                                    None,#self.AcceptedConfigurations,
                                                                    self.save_dir_local)

        with open(os.path.join(self.save_dir_local,'UCIInterval2UCI.json'),"w") as f:
            json.dump(self.UCIInterval2UCI,f)
        self.UCIInterval2UCI = PlotUCIsAvailable(self.UCIInterval2UCI,self.save_dir_local)


#### Vector Fields ####
    def ComputeSDMax(self): 
        """
            @description:
                - Computes IndicesInside: Sets of indices in Grid indices that are inside the boundary of the city.
                - Computes IndicesEdge: Sets of indices in Grid indices that are in the boundary of the city.
                - Computes Smaxi: Configuration of SMmax for the potential
                - Computes Dmaxi: Configuration of Dmax for the potential
                - Computes Smaxi_Mass: Configuration of SMmax for the mass
                - Computes Dmax_ij_Mass: Configuration of Dmax for the mass
                
        """       
        if self.IndicesInside is None:
            self.IndicesInside = GetIndicesInsideFromGrid(self.grid)
        if self.IndicesEdge is None:
            self.IndicesEdge = GetIndicesEdgeFromGrid(self.grid)
        if not os.path.isfile(os.path.join(self.save_dir_grid,f'Smax.parquet')):
            logger.info(f"RoutineVectorField: Extract Smax,Dmax: {self.city}")
            # POTENTIAL
            self.Smaxi,self.Dmaxi = ComputeVmaxUCI(self.df_distance,self.IndicesEdge,method = 'Pereira',case = 'Potential')
            pd.DataFrame({"Smax":self.Smaxi}).to_parquet(os.path.join(self.save_dir_grid,f'Smax.parquet'),engine = 'pyarrow',index=False)
            pd.DataFrame({"Dmax":self.Dmaxi}).to_parquet(os.path.join(self.save_dir_grid,f'Dmax.parquet'),engine = 'pyarrow',index=False)
            # MASS
            self.Smaxi_Mass,self.Dmax_ij_Mass = ComputeVmaxUCI(self.df_distance,self.IndicesEdge,method = 'Pereira',case = 'Mass')
            pd.DataFrame({"SmaxMass":self.Smaxi_Mass}).to_parquet(os.path.join(self.save_dir_grid,f'SmaxMass.parquet'),engine = 'pyarrow',index=False)
            pd.DataFrame({"DmaxMass":self.Dmax_ij_Mass}).to_parquet(os.path.join(self.save_dir_grid,f'DmaxMass.parquet'),engine = 'pyarrow',index=False)
        else:
            logger.info(f"RoutineVectorField: Upload Smax,Dmax: {self.city}")
            self.Smaxi = pd.read_parquet(os.path.join(self.save_dir_grid,f'Smax.parquet'))["Smax"].to_numpy(dtype=np.float64)
            self.Dmaxi = pd.read_parquet(os.path.join(self.save_dir_grid,f'Dmax.parquet'))["Dmax"].to_numpy(dtype=np.float64)            
            self.Smaxi_Mass = pd.read_parquet(os.path.join(self.save_dir_grid,f'SmaxMass.parquet'))["SmaxMass"].to_numpy(dtype=np.float64)
            self.Dmax_ij_Mass = pd.read_parquet(os.path.join(self.save_dir_grid,f'DmaxMass.parquet'))["DmaxMass"].to_numpy(dtype=np.float64)


    # this comment is not a bug
    def RoutineVectorFieldAndPotential(self):
        """
            Computes:
                - Distance Matrix in format 
                - OD Grid in format Tij
                - Vector Field
            NOTE: In this case we are considering just the dataset and not produced any change in the potential.    
                potential
            NOTE: The return value of this block is going to be the UCI of the city. In this case we chose the UCI relative to the population and 
            not the potential.
        """
        PotentialDf, _,VectorField = GeneratePotentialFromFluxes(self.Tij,self.df_distance,self.lattice,self.grid,self.city,self.save_dir_grid)
        self.ComputeSDMax()
        if not os.path.isfile(os.path.join(self.save_dir_grid,'PotentialDataframe.csv')):
            SavePotentialDataframe(PotentialDf,self.save_dir_grid)
        logger.info(f"Compute UCI {self.city}")        
        if not os.path.isfile(os.path.join(self.save_dir_grid,f'UCI_base.json')):
#            self.Vmax = ComputeJitV(self.Smaxi,self.Dmaxi)/2
            logger.info(f"Compute UCI {self.city}")
            PI,LC,UCI,result_indices,angle,cumulative,Fstar = ComputeUCICase(self.df_distance,self.grid,PotentialDf,self.IndicesInside,case = 'Potential')
            logger.info(f"Compute UCI Mass {self.city}")
            PIM,LCM,UCIM,result_indicesM,angleM,cumulativeM,FstarM = ComputeUCICase(self.df_distance,self.grid,PotentialDf,self.IndicesInside,case = 'Mass') 
            I = {'PI':PI,'LC':LC,'UCI':UCI,"Fstar":Fstar,"PI_M":PIM,'LC_M':LCM,'UCI_M':UCIM,"Fstar_M":FstarM}      
            UCIskip = {'UCI':UCI,"UCIM":UCIM}
            UCI_dir = os.path.join(self.save_dir_grid,f'UCI_{round(UCI,3)}')
            # Prepare Plot UCI
            Tij_Inside = self.Tij[self.Tij['origin'].isin(self.IndicesInside) & self.Tij['destination'].isin(self.IndicesInside)]
            VectorFieldInside = VectorField.iloc[self.IndicesInside]
            GridInside = self.grid.loc[self.grid["index"].isin(self.IndicesInside)]
            PotentialInside =  PotentialDf.loc[PotentialDf['index'].isin(self.IndicesInside)]
            # Plot UCI
            os.makedirs(UCI_dir,exist_ok=True)     
            SaveJsonDict(I,os.path.join(UCI_dir,f'UCI_{round(UCI,3)}.json'))
            SaveJsonDict(UCIskip,os.path.join(self.save_dir_grid,f'UCI_base.json'))


            PlotInsideOutside(self.gdf_polygons,GridInside,self.save_dir_grid)
            PlotEdges(self.gdf_polygons,GridInside,self.save_dir_grid)
            PlotRoads(self.gdf_polygons,GridInside,self.save_dir_grid)
            PlotRoutineOD(GridInside,
                        Tij_Inside,
                        self.gdf_polygons,
                        PotentialInside,
                        VectorFieldInside,
                        UCI_dir,
                        80,
                        UCI,
                        None,
                        Tij_Inside,
                        cumulative,
                        Fstar,
                        result_indices)
        else:
            with open(os.path.join(self.save_dir_grid,f'UCI_base.json')) as f:
                UCIskip = json.load(f)
            UCI = UCIskip['UCI']
            UCIM = UCIskip['UCIM']
        return UCIM  
## FIT ##

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
        if not os.path.isfile(os.path.join(TRAFFIC_DIR,'data','carto',self.city,'potential','FitVespignani.json')):
            logger.info(f'Computing Fit {self.city}')
            os.makedirs(os.path.join(TRAFFIC_DIR,'data','carto',self.city,'potential'),exist_ok=True)
            logk,alpha,gamma,d0_2min1 = VespignaniBlock(self.df_distance,self.grid,self.Tij,self.save_dir_potential)            
            self.k = np.exp(logk)
            self.alpha = alpha
            self.beta = gamma
            self.d0 = d0_2min1
        else:
            logger.info(f'Loading Fit {self.city}')
            self.k,self.alpha,self.beta,self.d0 = UploadGravitationalFit(TRAFFIC_DIR,self.city)
    def TotalPopAndFluxes(self):
        """
            Compute the Total Population and Total Fluxes
        """
        logger.info('Computing Total Population and Total Fluxes')
        self.total_population = np.sum(self.grid['population'])
        self.total_flux = np.sum(self.Tij['number_people'])


## NEW VECTOR FIELD AND POTENTIAL ## NOTE: There is redundancy and not logical perfection comparing this function to the non modified case.
    def ComputePotentialAndVectorFieldFromGrid(self,new_population,Tij):
        """
            @params Grid: Grid: gpd.GeoDataFrame
            @params Tij: Tij: pd.DataFrame
            Computes:
                - Potential
                - Vector Field
        """
        logger.info(f'Computing New Vector Field {self.city}')
        New_Vector_Field = ComputeNewVectorField(Tij,self.df_distance)
        logger.info(f'Computing New Potential {self.city}')
        New_Potential_Dataframe = ComputeNewPotential(New_Vector_Field,self.lattice,new_population)
        return New_Vector_Field,New_Potential_Dataframe

    def RoutineVectorFieldAndPotentialModified(self,GridNew,Tij):
        """
            @params GridNew: Grid: gpd.GeoDataFrame
            @params Tij: Tij: pd.DataFrame
            Computes:
                - Potential
                - Vector Field
                - UCI
                
        """
        New_Vector_Field,New_Potential_Dataframe = self.ComputePotentialAndVectorFieldFromGrid(GridNew["population"],Tij)
        PIM,LCM,UCIM,result_indicesM,angleM,cumulativeM,FstarM = ComputeUCICase(self.df_distance,GridNew,New_Potential_Dataframe,self.IndicesInside,case = 'Mass')
        PI,LC,UCI,result_indices,angle,cumulative,Fstar = ComputeUCICase(self.df_distance,GridNew,New_Potential_Dataframe,self.IndicesInside,case = 'Potential')
        Tij_InsideModified = Tij[Tij['origin'].isin(self.IndicesInside) & Tij['destination'].isin(self.IndicesInside)]
        Tij_Inside = self.Tij[self.Tij['origin'].isin(self.IndicesInside) & self.Tij['destination'].isin(self.IndicesInside)]
        I = {'PI':PI,'LC':LC,'UCI':UCI,"Fstar":Fstar,"PI_M":PIM,'LC_M':LCM,'UCI_M':UCIM,"Fstar_M":FstarM}           
        SaveJsonDict(I,os.path.join(self.save_dir_local,f'UCI_{round(UCI,3)}.json'))         
        UCI_dir = os.path.join(self.save_dir_grid,f'UCI_{round(UCI,3)}')
        os.makedirs(UCI_dir,exist_ok=True)
        # Mask For Plot
        VectorFieldInside = New_Vector_Field.iloc[self.IndicesInside]
        GridInside = GridNew.loc[GridNew["index"].isin(self.IndicesInside)]
        PotentialInside =  New_Potential_Dataframe.loc[New_Potential_Dataframe['index'].isin(self.IndicesInside)]
        # Plot UCI
        PlotLorenzCurveMassPot(cumulative,Fstar,result_indices,cumulativeM,FstarM,result_indicesM,UCI_dir,UCI,UCIM,shift = 0.1,verbose = False)        
        PlotRoutineOD(GridInside,
                    Tij_InsideModified,
                    self.gdf_polygons,
                    PotentialInside,
                    VectorFieldInside,
                    UCI_dir,
                    80,
                    UCI,
                    None,
                    Tij_Inside,
                    cumulative,
                    Fstar,
                    result_indices)


    def ComputeTijFromGrid(self,Grid_New):
        """
            @params cov: Covariance that sets the width of population
            @params distribution: [exponential,gaussian]
            @params num_peaks: Number of peaks in the population
            Change the Morphology of the City
        """
        # Number of People and Fluxes
        self.TotalPopAndFluxes()
        # From Population, using the gravitational model, generate the fluxes
        logger.info(f'Recovering Modified Fluxes {self.city}')
        Modified_Fluxes = GenerateModifiedFluxes(Grid_New['population'].to_numpy(),self.df_distance,self.k,self.alpha,self.beta,self.d0,self.total_flux,False)
        Tij_Modified = self.Tij.copy()
        Tij_Modified['number_people'] = Modified_Fluxes
        return Tij_Modified


##### FILE SIMULATIONS
    def InitializeDf4Sim(self):
        """
            @description: Creates DfBegin 
        """
        # Get File.fma each hour
        self.Hour2Files = self.OrderFilesFmaPerHour()
        # Generate The Brgin And End Files For Simulation Common to All Rs
        if not os.path.isfile(os.path.join(self.save_dir_local,'DfBegin.csv')):
            NPeopleOffset = 0
            Count = 0
            for time,ODfmaFile in self.Hour2Files.items(): 
                if time < self.start:
                    logger.info(f'Computing DfBegin {self.city}')
                    O_vector,D_vector,OD_vector = MapFile2Vectors(os.path.join(self.ODfma_dir,ODfmaFile))
                    logger.info(f'Number of Users {self.city}: {np.sum(OD_vector)}')
                    TotalFluxesTime = np.sum(OD_vector)
                    R = int(TotalFluxesTime/3600)
                    Df_GivenR = GetODForSimulationFromFmaPolygonInput(O_vector,
                                                                    D_vector,
                                                                    OD_vector,
                                                                    R,
                                                                    NPeopleOffset,
                                                                    self.polygon2OD,
                                                                    self.osmid2index,
                                                                    self.OD2grid,
                                                                    self.gridIdx2dest,
                                                                    time,
                                                                    60)
                    NPeopleOffset += len(Df_GivenR)
                    if Count == 0:
                        self.DfBegin = Df_GivenR
                    else:
                        self.DfBegin = pd.concat([self.DfBegin,Df_GivenR],ignore_index=True)
                    Count += 1
                else:
                    self.DfBegin.to_csv(os.path.join(self.save_dir_local,'DfBegin.csv'),index=False)
                    break
        else:
            self.DfBegin = pd.read_csv(os.path.join(self.save_dir_local,'DfBegin.csv'))
            pass

#            self.DfBegin = GenerateBeginDf(self.Hour2Files,
#                                    self.ODfma_dir,
#                                    self.start,
#                                    self.polygon2OD,
#                                    self.osmid2index,
#                                    self.OD2grid,
#                                    self.city,
#                                    self.gridIdx2dest)


    def ComputeEndFileInputSimulation(self):
        """
            @brief: Compute the End File for Simulation
        """
        if not os.path.isfile(os.path.join(self.save_dir_local,'DfEnd.csv')):
            logger.info(f'Computing DfEnd {self.city}')
            Count = 0
            NOffset = 0
            for time,ODfmaFile in self.Hour2Files.items(): 
                if time > self.start:
                    O_vector,D_vector,OD_vector = MapFile2Vectors(os.path.join(self.ODfma_dir,ODfmaFile))
                    TotalFluxesTime = np.sum(OD_vector)
                    R = int(TotalFluxesTime/3600)
                    # Do Not Change the OD and Concatenate the input for Simulation                            
                    Df_GivenR = GetODForSimulationFromFmaPolygonInput(O_vector,
                                                                    D_vector,
                                                                    OD_vector,
                                                                    R,
                                                                    NOffset,
                                                                    self.polygon2OD,
                                                                    self.osmid2index,
                                                                    self.OD2grid,
                                                                    self.gridIdx2dest,
                                                                    time,
                                                                    60)
                    NOffset += len(Df_GivenR)
                    if Count == 0:
                        self.DfEnd = Df_GivenR
                    else:
                        self.DfEnd = pd.concat([self.DfEnd,Df_GivenR],ignore_index=True)
                    Count += 1           
        
            self.DfEnd.to_csv(os.path.join(self.save_dir_local,'DfEnd.csv'),index=False)             
        else:
            logger.info(f'Loading DfEnd {self.city}')
            self.DfEnd = pd.read_csv(os.path.join(self.save_dir_local,'DfEnd.csv'))
            pass
    def ShiftFinalInputSimulation(self,NPeopleOffset):
        """
            @params NPeopleOffset: int
            @brief: Shift the Final Id for people on the simulation file Simulation
        """
        DfEnd = self.DfEnd
        DfEnd['SAMPN'] = [idx + NPeopleOffset for idx in DfEnd['SAMPN']]
        DfEnd['PERNO'] = [idx + NPeopleOffset for idx in DfEnd['PERNO']]
        return DfEnd



## Simulation Output File ##
    def ComputeDf4SimChangedMorphology(self,UCI,R,Modified_Fluxes):
        """
            @params UCI: float -> Urban Centrality Index
            @params R: int -> fraction of people per second
            @params Modified_Fluxes: DataFrame: [number_people,i,j] (Fluxes produced by gravitational model)
        """
        # If it has not been computed or it is already saved the simulation
        end = self.start + 1
        if not os.path.isfile(os.path.join(self.new_full_network_dir ,'{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(self.city,self.start,end,str(int(R)),round(UCI,3)))):
            # Computte the Df for Simulation
            NPeopleOffset = len(self.DfBegin)
            logger.info(f"Append Sim-File ControlGroup GENERATED {self.city}, R: {R}, UCI: {UCI}...")
            Df_GivenR = GenerateDfFluxesFromTij(Modified_Fluxes,
                                                            self.osmid2index,
                                                            self.grid2OD,
                                                            self.start,
                                                            NPeopleOffset,
                                                            R) 
            Df = pd.concat([self.DfBegin,Df_GivenR],ignore_index=True)
            NPeopleOffset += len(Df_GivenR)
            DfEnd = self.ShiftFinalInputSimulation(NPeopleOffset) 
            Df = pd.concat([Df,DfEnd],ignore_index=True)
            end = self.start + 1
            logger.info(f"Append Sim-File ControlGroup GENERATED {self.city}, R: {R}, UCI: {UCI}...")
#            PlotDepTime(Df, self.city, self.save_dir_local)
            Df.to_csv(os.path.join(self.new_full_network_dir ,'{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(self.city,self.start,end,str(int(R)),round(UCI,3))),sep=',',index=False)
        else:
            pass
        return os.path.join(self.new_full_network_dir ,'{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(self.city,self.start,end,str(int(R)),round(UCI,3)))
        

    def ComputeDf4SimNotChangedMorphology(self,UCI,R):
        """
            @params: UCI: float (Urban Centrality Index)
            Compute the Df for Simulation without changing the morphology of the city
        """
        NPeopleOffset = len(self.DfBegin)
        logger.info(f"Append Sim-File ControlGroup NOT Generated {self.city}, R: {R}, UCI: {round(UCI,3)}...")
        Df_GivenR = GenerateDfFluxesFromTij(self.Tij,
                                            self.osmid2index,
                                            self.grid2OD,
                                            self.start,
                                            NPeopleOffset,
                                            R) 
        Df = pd.concat([self.DfBegin,Df_GivenR],ignore_index=True)
        NPeopleOffset += len(Df_GivenR)
        DfEnd = self.ShiftFinalInputSimulation(NPeopleOffset) 
        Df = pd.concat([Df,DfEnd],ignore_index=True)
        end = self.start + 1
        logger.info(f"Save Sim-File ControlGroup NOT Generated {self.city}, R: {R}, UCI: {UCI}...")
        
        Df.to_csv(os.path.join(self.new_full_network_dir ,'{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(self.city,self.start,end,str(int(R)),round(UCI,3))),sep=',',index=False)
        return os.path.join(self.new_full_network_dir ,'{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(self.city,self.start,end,str(int(R)),round(UCI,3)))


#### Prepare Input For Simulation ####

    def SetRmaxDivisibleByNSim(self):
        """
            Set Rmax divisible by the number of simulations per UCI
        """
        Delta = self.Rmax - self.Rmin
        self.Rmax = self.Rmin + Delta + Delta%self.number_simulation_per_UCI
        self.Step = int((self.Rmax - self.Rmin)/self.number_simulation_per_UCI)        
        self.config["Rmax"] = self.Rmax
        if Delta%self.number_simulation_per_UCI == 0:
            self.config["number_simulation_per_UCI"] = self.number_simulation_per_UCI
            pass
        else:
            self.config["number_simulation_per_UCI"] = self.number_simulation_per_UCI + 1
        self.ArrayRs = np.arange(self.Rmin,self.Rmax,self.Step,dtype=int)
        self.config["ArrayRs"] = list(self.ArrayRs)
        SaveJsonDict(self.config_dir_local,self.city + '_geometric_info.json')        
        logger.info(f'New Rmax {self.Rmax}, New number of simulations {self.number_simulation_per_UCI}')

    def UpdateFiles2Upload(self,local_file,server_file):
        self.Files2Upload[local_file] = server_file
        logger.info('Files to Upload Updated: {}'.format(local_file))



    def OrderFilesFmaPerHour(self):
        from collections import OrderedDict
        Hour2Files = defaultdict()
        for file in os.listdir(os.path.join(self.ODfma_dir)):
            if file.endswith('.fma'):
                Hour2Files[int(file.split('.')[0].split('D')[1])] = file
        Hour2Files = OrderedDict(sorted(Hour2Files.items()))
        logger.info('Files Ordered in Hour2Files')
        return Hour2Files



