# O
import os
# F
# A 
import ast
# C
from collections import defaultdict
# G
import gc
# J
import json
# L
import logging
import logging.handlers
# M
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool,Queue,Barrier,cpu_count,Manager,Process,Value,log_to_stderr
import ctypes
# P
import psutil
import polars as pl
# S
import socket
import sys
# T
import time

TRAFFIC_DIR = os.environ["TRAFFIC_DIR"]
current_dir = os.path.join(os.getcwd()) 
mother_path = os.path.abspath(os.path.join(current_dir, os.pardir))
print('mother_path:', mother_path)
sys.path.append(os.path.join(mother_path, 'PreProcessing'))
sys.path.append(os.path.join(mother_path))
sys.path.append(os.path.join(TRAFFIC_DIR,'scripts','GeometrySphere'))
sys.path.append(os.path.join(TRAFFIC_DIR,'scripts','ServerCommunication'))
sys.path.append(os.path.join(TRAFFIC_DIR,'scripts','PostProcessingSimulations'))
sys.path.append(os.path.join(TRAFFIC_DIR,'scripts',"InitialConfigProcess"))

# Project specific
# A
from AlgorithmCheck import *
# C
from ComputeGrid import *
from ComputeHexagon import *
from ConcurrencyManager import *
# F
from FittingProcedures import *
# G
from GeometrySphere import *
from GenerateModifiedFluxesSimulation import *
from GravitationalFluxes import *                                               # FIT section
from Grid import *
# H 
from Hexagon import *
from HostConnection import *
# M
from MainPolycentrism import *
from ModifyPotential import *
from multiple_launches import *
# O 
from ODfromfma import *
# P
from plot import *
from Polycentrism import *
from PolycentrismPlot import *
from PolygonSettings import *
from Potential import *
from PreprocessingObj import *
# Post Processing
from TrajectoryAnalysis import *
from GenerateConfiguraiton import *

# Scripts
from Constants import *
from threading import Thread,Lock
import threading
import queue
import subprocess
## BASIC PARAMS
gc.set_threshold(10000,50,50)
plt.rcParams.update({
    "text.usetex": False,
})
logger = logging.getLogger(__name__)

StateAlgorithm = InitWholeProcessStateFunctions()


import matplotlib as mpl
def setup_mpl():
    mpl.rc('font', size=20)
    mpl.rcParams['legend.fontsize'] = 'small'
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['font.family']='DejaVu Math TeX Gyre'#'Helvetica 45 Light'
    mpl.rcParams['xtick.major.pad']='18'
    mpl.rcParams['ytick.major.pad']='18'
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['xtick.minor.width'] = 2
    mpl.rcParams['ytick.minor.width'] = 2
    mpl.rcParams['xtick.major.size'] = 6
    mpl.rcParams['ytick.major.size'] = 6
    mpl.rcParams['xtick.minor.size'] = 3
    mpl.rcParams['ytick.minor.size'] = 3
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['xtick.top']=True
    mpl.rcParams['ytick.right']=True
    mpl.rcParams['mathtext.default']='regular'
    mpl.rcParams['xtick.major.pad']='4'
    mpl.rcParams['ytick.major.pad']='4'
    mpl.rcParams['axes.labelpad']= 2
    mpl.rcParams['axes.labelsize'] = 20  # Set the font size of the axis labels
    alpha = 0.6
    to_rgba = mpl.colors.ColorConverter().to_rgba



def configure_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Create handlers
    log_queue = queue.Queue(-1)

    # Create a handler that writes log messages to the queue
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    # Create a handler that writes log messages to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    # Create formatters and add them to handlers
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Add handlers to the logger
    queue_listener = logging.handlers.QueueListener(log_queue, file_handler)
    queue_listener.start()

    return logger,queue_listener
logger,queue_listener = configure_logger(os.path.join(TRAFFIC_DIR,'main.log')) 

def CheckIndexUCI(UCIInterval2UCIAccepted,UCI):
    """
        @param UCIInterval2UCIAccepted: dict -> Dictionary with the interval of UCIs that are accepted.
        @param UCI: float -> UCI to check
        This function checks if the UCI is in the interval of UCIs that are accepted.
    """
    for k in range(len(list(UCIInterval2UCIAccepted.keys()))):
        index_UCI_represents = list(UCIInterval2UCIAccepted.keys())[k]
        index_UCI_represents_next = list(UCIInterval2UCIAccepted.keys())[k+1]
        # It must lie in the interval 
        if UCI >= index_UCI_represents and UCI <= index_UCI_represents_next:
            return index_UCI_represents
    assert index_UCI_represents is not None, "UCI is not in the interval of UCIs that are accepted."
    return index_UCI_represents

def CheckUCIIsFull(UCIInterval2UCIAccepted):
    """
        @param UCIInterval2UCIAccepted: dict -> Dictionary with the interval of UCIs that are accepted.
        This function checks if the UCI is in the interval of UCIs that are accepted.
    """
    for index_UCI_represents in UCIInterval2UCIAccepted.keys():
        if UCIInterval2UCIAccepted[index_UCI_represents] is None:
            return False
    return True

def RedefineRsWhenError(Rmax,Step,NumberRs):
    """
        @param Rmax: float -> Maximum Radius
        @param Step: float -> Step to reduce the Radius
        This function redefines the Rs that are going to be used in the simulation.
    """
    assert Rmax > 0, "Rmax must be positive."
    assert Step > 0, "Step must be positive."
    ArrayRs = np.array([Rmax - Step*(i+1) for i in range(NumberRs)])
    assert np.all(ArrayRs > 0), f"All Rs must be positive. Rmax: {Rmax}, Step: {Step}, NumberRs: {NumberRs}"
    return ArrayRs

def Init_Info_Simulations_Df(complete_path_info_simulations_df, list_columns_df_info_simulations):
    if os.path.isfile(complete_path_info_simulations_df):
        return pl.read_csv(complete_path_info_simulations_df)
    else:
        schema = {
            "dir_output_simulation_analysis": pl.Utf8,
            "complete_path_od_file_input_simulation": pl.Utf8,
            "complete_path_output_simulation_copied": pl.Utf8,
            "name_od_file_input_simulation": pl.Utf8,
            "name": pl.Utf8,
            "complete_path_graphml": pl.Utf8,
            "city": pl.Utf8,
            "UCI": pl.Float64,
            "R": pl.Int64,
            "iter": pl.Int64,
            "str_UCI": pl.Utf8,
            "dir_file_demand": pl.Utf8,
            "dir_new_full_network": pl.Utf8,
            "dir_output_simulation_gpu": pl.Utf8,
            "number_realizations_OD_ensemble": pl.Int64,
            "delta_t": pl.Int64,
            "complete_path_unloading_curve_base": pl.Utf8,
            "complete_path_unloading_curve_iter": pl.Utf8,
            "complete_path_route_file": pl.Utf8,
            "complete_path_people_file": pl.Utf8
        }
        return pl.DataFrame(schema=schema)

        
def Update_Info_Simulations_Df(
        Df_Info_Simulations,
        updating_dict):
    """
        Returns the Polar datafrane containing all the different informations about set of simulations
        that are run for the 4-plet (city,R,UCI,iter)
    """
    assert False not in [col in Df_Info_Simulations.columns for col in updating_dict.keys()], "Some columns are not in the DataFrame."
    # Convert updating_dict to DataFrame
    new_data = pl.DataFrame(updating_dict)
    
    # Ensure column types are compatible
    for col in new_data.columns:
        if col in Df_Info_Simulations.columns:
            new_data = new_data.with_columns(new_data[col].cast(Df_Info_Simulations[col].dtype))
    
    # Concatenate DataFrames
    return pl.concat([Df_Info_Simulations, new_data])
        

def Update_update_dict_df(update_dict_df,
                          complete_path_od_file_input_simulation,
                          complete_path_output_simulation_copied,
                          NameCity,
                          dir_output_simulation_analysis,
                          complete_path_graphml,
                          rounded_UCI,
                          str_UCI,
                          R,
                          ensemble_realization,
                          complete_path_route_file,
                          complete_path_people_file,
                          complete_path_unloading_curve_base,
                          complete_path_unloading_curve_iter,
                          dir_file_demand = os.path.join(BERKELEY_2018_DIR,'new_full_network'),
                          dir_new_full_network = os.path.join(BERKELEY_2018_DIR,'new_full_network'),
                          dir_output_simulation_gpu = os.environ["LPSim"],
                          number_realizations_OD_ensemble = 4,
                          delta_t = 10,
                        name_od_file_input_simulation = None
                          ):
    """
        @param update_dict_df: dict -> Dictionary with the information about the simulation

    """
    # Main() Variables
    update_dict_df["dir_output_simulation_analysis"] = dir_output_simulation_analysis                       # NOTE: We copy simulator output here.
    update_dict_df["complete_path_od_file_input_simulation"] = complete_path_od_file_input_simulation       #
    update_dict_df["complete_path_output_simulation_copied"] = complete_path_output_simulation_copied       #
    update_dict_df["name_od_file_input_simulation"] = name_od_file_input_simulation
    update_dict_df["name"] = NameCity                                                                       #
    update_dict_df["complete_path_graphml"] = complete_path_graphml                                                      #
    update_dict_df["city"] = NameCity
    update_dict_df["UCI"] = rounded_UCI                                                                    #
    update_dict_df["R"] = R
    update_dict_df["iter"] = ensemble_realization
    update_dict_df["str_UCI"] = str_UCI
    #### CONSTANTS ####
        # ProcessMain() Variables
    update_dict_df["dir_file_demand"] = dir_file_demand                                                     # NOTE: Simulator checks by default here for demand
    update_dict_df["dir_new_full_network"] = dir_new_full_network                                           # NOTE: Simulator checks by default here for cartography
    update_dict_df["dir_output_simulation_gpu"] = dir_output_simulation_gpu                                 # NOTE: Simulator Puts the output of the simulation here
    # PCTA complete analysis
    update_dict_df["number_realizations_OD_ensemble"] = number_realizations_OD_ensemble                     # NOTE: Number of different simulations launched for a couple UCI, R (what changes are the realization of nodes where people are inserted.)
    update_dict_df["delta_t"] = delta_t
    update_dict_df["complete_path_unloading_curve_base"] = complete_path_unloading_curve_base
    update_dict_df["complete_path_unloading_curve_iter"] = complete_path_unloading_curve_iter  
    update_dict_df["complete_path_route_file"] = complete_path_route_file
    update_dict_df["complete_path_people_file"] = complete_path_people_file                                                                   #
    return update_dict_df


def Set_Paths_Simulation_Analysis_Beta(city,
                                       str_UCI,
                                       rounded_UCI,
                                       R,
                                       ensemble_realization,
                                       Name_parameters_output,
                                       start,
                                       end,
                                       dir_output_simulation_analysis_base,
                                       number_realizations_OD_ensemble):
    # 
    suffix_parquet_people_file = f'0_people{start}to24.parquet'                                                                                                     # NOTE:   
    suffix_parquet_route_file = f'0_route{start}to24.parquet'                                                                                           # Files Simulation                 
    #
    name_od_file_input_simulation = f"{city}_oddemand_{start}_{end}_{Name_parameters_output}.csv"                                     # NOTE: OD file name -> this is the one the simulation looks for as input: MUST be in ../new_full_netwtork/
    complete_path_od_file_input_simulation = os.path.join(dir_file_demand,name_od_file_input_simulation)
    #
    dir_output_simulation_analysis = os.path.join(dir_output_simulation_analysis_base,str_UCI)                                                                      # NOTE: directory where I copy the output of the simulation
    os.makedirs(os.path.join(dir_output_simulation_analysis), exist_ok=True)                                                                                        # NOTE: Create the directory if it does not exist
    name_people_simulation_copied = f"{Name_parameters_output}_{suffix_parquet_people_file}"                                                                        # NOTE: name of the file that will be checked to be sure the output of the simulation is already there and we do not need to recompute it
    name_route_simulation_copied = f"{Name_parameters_output}_{suffix_parquet_route_file}"                                                                          # NOTE: name of the file that will be checked to be sure the output of the simulation is already there and we do not need to recompute it
    complete_path_output_simulation_copied = os.path.join(dir_output_simulation_analysis,name_people_simulation_copied)                                             # NOTE: complete path of the file that will be checked to be sure the output of the simulation is already there and we do not need to recompute it
    # beta routine    
    dir_output_simulation_plots_unload =os.path.join(dir_output_simulation_analysis_base,"Plots",str_UCI)
    dir_output_simulation_plots_beta =  os.path.join(dir_output_simulation_analysis_base,"Plots")
    os.makedirs(os.path.join(dir_output_simulation_plots_unload), exist_ok=True)                                                                                  # NOTE: Create the directory if it does not exist
    os.makedirs(os.path.join(dir_output_simulation_plots_beta), exist_ok=True)                                                                                  # NOTE: Create the directory if it does not exist
    # gaphml
    complete_path_graphml = os.path.join(TRAFFIC_DIR,'data','carto',city,city + '_new_tertiary_simplified.graphml')    
    # Unload Curve
    name_unloading_curve_base = f"UnloadCurve_R_{R}_UCI_{str_UCI}.parquet"
    name_unloading_curve_iter = f"UnloadCurve_R_{R}_UCI_{str_UCI}_iter_{ensemble_realization}.parquet"
    complete_path_unloading_curve_base = os.path.join(dir_output_simulation_plots_unload,name_unloading_curve_base)
    complete_path_unloading_curve_iter = os.path.join(dir_output_simulation_plots_unload,str(R),name_unloading_curve_iter)
    complete_path_people_file = os.path.join(dir_output_simulation_analysis,name_people_simulation_copied)                                                                 # NOTE: complete path of the file that will be checked to be sure the output of the simulation is already there and we do not need to recompute it
    complete_path_route_file = os.path.join(dir_output_simulation_analysis,name_route_simulation_copied)                                                                 # NOTE: complete path of the file that will be checked to be sure the output of the simulation is already there and we do not need to recompute it
    update_dict_df = Update_update_dict_df(defaultdict(),
                          complete_path_od_file_input_simulation,
                          complete_path_output_simulation_copied,
                          city,
                          dir_output_simulation_analysis,
                          complete_path_graphml,
                          rounded_UCI,
                          str_UCI,
                          R,
                          ensemble_realization,
                          complete_path_route_file,
                          complete_path_people_file,
                          complete_path_unloading_curve_base,
                          complete_path_unloading_curve_iter,
                          dir_file_demand = os.path.join(BERKELEY_2018_DIR,'new_full_network'),
                          dir_new_full_network = os.path.join(BERKELEY_2018_DIR,'new_full_network'),
                          dir_output_simulation_gpu = os.environ["LPSim"],
                          number_realizations_OD_ensemble = number_realizations_OD_ensemble,
                          delta_t = 10,
                          name_od_file_input_simulation = name_od_file_input_simulation)  
    return update_dict_df            



def Main(CityName,
         TRAFFIC_DIR,
         dir_data_2,
         list_columns_df_info_simulations,
         complete_path_info_simulations_df = os.path.join(TRAFFIC_DIR,"info_simulations.csv"),
         round_UCI = 3,
         number_realizations_OD_ensemble = 4):
    """
        @param NameCities: list -> List of the names of the cities
        @param list_columns_df_info_simulations: list -> 
            List of the columns of the DataFrame that contains the information about the simulations
    """
    # SCOPE CONSTANTS
    dir_output_simulation_analysis_base = os.path.join(dir_data_2,CityName,"Output")               # NOTE: Where I copy the output of the simulation
    Df_Info_Simulations = Init_Info_Simulations_Df(complete_path_info_simulations_df,list_columns_df_info_simulations)      # NOTE: DataFrame with the information about the simulations
    # 1) NOTE: Create Geometry Object
    GeoInfo = GeometricalSettingsSpatialPartition(CityName,TRAFFIC_DIR)                                                     # NOTE: Create the object that contains all the information about the geometry of the city
    # 2) NOTE: Compute Geometries
    GeoInfo.GetGeometries()                                                                                                 # NOTE: Compute the geometries of the city (grid, gdf_hexagons, gdf_polygons, lattice)
    # 3) NOTE: Compute the Potential and Vector field for non modified fluxes
    UCI = GeoInfo.RoutineVectorFieldAndPotential()                                                                          # NOTE: Compute the potential and vector field for non modified fluxes
    # 4) NOTE: Compute the Fit for the gravity model
    GeoInfo.ComputeFit()                                                                                                    # NOTE: Compute the fit for the gravity model
    # 5) NOTE: Initialize the Concatenated Df for Simulation [It is common for all different R]
    GeoInfo.InitializeDf4Sim()                                                                                              # NOTE: Initialize the concatenated DataFrame for simulation
    GeoInfo.ComputeEndFileInputSimulation()         
    # 6) NOTE: Launch Simulations Non Modified
    str_UCI = str(round(UCI,round_UCI))
    float_rounded_UCI = round(UCI,round_UCI)
    GeoInfo.Reset_ArrayRs_Given_UCI(UCI)
    for R in GeoInfo.ArrayRs:
        for ensemble_realization in range(number_realizations_OD_ensemble):
            Name_parameters_output = f"R_{R}_UCI_{str_UCI}_realization_{ensemble_realization}"
            # Prepare all the paths to run smoothly everything
            update_dict_df = Set_Paths_Simulation_Analysis_Beta(GeoInfo.city,
                                       str_UCI,
                                       float_rounded_UCI,
                                       R,
                                       ensemble_realization,
                                       Name_parameters_output,
                                       GeoInfo.start,
                                       GeoInfo.start + 1,
                                       dir_output_simulation_analysis_base,
                                       number_realizations_OD_ensemble)            
            # Simulate just if the file are not already simulated: NOTE: the simulation happens iff it is in Df_Info_Simulations
            is_success_simulation  = update_dict_df["complete_path_output_simulation_copied"] in Df_Info_Simulations["complete_path_output_simulation_copied"].to_list()                                                                             # NOTE: 1) Check if the computation is already done, aovid to recompute it
            is_computed_od_simulation_input = update_dict_df["complete_path_od_file_input_simulation"] in Df_Info_Simulations["complete_path_output_simulation_copied"].to_list()                                                                        # NOTE: 2) Check that the input OD file is already there and avoid to recompute it.
            #
            if not is_success_simulation:                                                                                                                                   # NOTE: Checking 1) 
                if not is_computed_od_simulation_input:                                                                                                                     # NOTE: Checking 2) 
                    # Not Modified Mobility
                    logger.info(f"ID-Non-Mod R: {R}, {GeoInfo.city}, UCI: {str_UCI}")
                    # 7) NOTE: Compute the realization of the OD
                    GeoInfo.ComputeDf4SimNotChangedMorphology(UCI,
                                                            R,
                                                            update_dict_df["complete_path_od_file_input_simulation"])
                else:
                    pass
                # 8) NOTE: Launch the Simulation
                SuccessSimulation = LaunchSimulation(GeoInfo,
                                                    R,
                                                    str_UCI,
                                                    Name_parameters_output,
                                                    update_dict_df)
                if SuccessSimulation:
                    # 9) NOTE: Update the DataFrame of Simulations
                    logger.info(f"Updating the DataFrame of Simulations")
                    Df_Info_Simulations = Update_Info_Simulations_Df(Df_Info_Simulations,
                                                                    update_dict_df)
                    Df_Info_Simulations.write_csv(complete_path_info_simulations_df)                                                                 # NOTE: Write the DataFrame to the csv file        

            else:
                pass

    Df_City2Config = Df_Info_Simulations.filter(pl.col("city") == CityName,
                                            pl.col("R") == R,
                                            pl.col("UCI") == float_rounded_UCI)
    # 10) NOTE: Analyze the Output
#    PCTA = Polycentrism2TrafficAnalyzer(Df_City2Config,
#                                        GeoInfo.ArrayRs,
#                                        [float_rounded_UCI])  
#    PCTA.CompleteAnalysis()
    logger.info(f"Finished the simulations not changed for {CityName}")
    # 11) NOTE: Generate New Population 
    GeoInfo.GeneratePopulationAndSetUCIs()                                                                      # Compute New Grid
    for UCI_Interval in GeoInfo.UCIInterval2UCI.keys():
        for UCIM in GeoInfo.UCIInterval2UCI[UCI_Interval]:
            from pandas import read_parquet
            from geopandas import GeoDataFrame
            # New Population
            #
            UCI = float(UCIM)
            str_UCI = str(round(UCI,round_UCI))
            float_rounded_UCI = round(UCI,round_UCI)
            GeoInfo.Reset_ArrayRs_Given_UCI(UCI)
            GridNew = read_parquet(os.path.join(GeoInfo.save_dir_local,f'Grid_{str_UCI}.parquet'))        # New Grid (read)
            GridNew[["geometry","centroidx","centroidy"]] = GeoInfo.grid[["geometry","centroidx","centroidy"]]  # New Grid (add centroid, geometry)
            GridNew = GeoDataFrame(GridNew)                                                                     # New Grid
            Modified_Fluxes = GeoInfo.ComputeTijFromGrid(GridNew)                                               # New Fluxes
            # 12) NOTE: Launch Simulations Modified
            for R in GeoInfo.ArrayRs:
                for ensemble_realization in range(number_realizations_OD_ensemble):
                    logger.info(f"{GeoInfo.city}: ID-Mod R: {R}, UCI: {str_UCI}, iter {ensemble_realization}")
                    Name_parameters_output = f"R_{R}_UCI_{str_UCI}_realization_{ensemble_realization}"
                    # Prepare all the paths to run smoothly everything
                    update_dict_df = Set_Paths_Simulation_Analysis_Beta(GeoInfo.city,
                                            str_UCI,
                                            float_rounded_UCI,
                                            R,
                                            ensemble_realization,
                                            Name_parameters_output,
                                            GeoInfo.start,
                                            GeoInfo.start + 1,
                                            dir_output_simulation_analysis_base,
                                            number_realizations_OD_ensemble)            
                    
                    # Simulate just if the file are not already simulated: NOTE: the simulation happens iff it is in Df_Info_Simulations
                    is_success_simulation  = update_dict_df["complete_path_output_simulation_copied"] in Df_Info_Simulations["complete_path_output_simulation_copied"].to_list()                                                                             # NOTE: 1) Check if the computation is already done, aovid to recompute it
                    is_computed_od_simulation_input = update_dict_df["complete_path_od_file_input_simulation"] in Df_Info_Simulations["complete_path_output_simulation_copied"].to_list()                                                                        # NOTE: 2) Check that the input OD file is already there and avoid to recompute it.
                    #
                    if not is_success_simulation:                                                       # Check if the simulation is already computed
                        if not is_computed_od_simulation_input:                                         # Check if the OD file is already computed
                            # 13) NOTE: Compute the realization of the OD
                            GeoInfo.ComputeDf4SimChangedMorphology(UCI,
                                                                    R,
                                                                    Modified_Fluxes,
                                                                    update_dict_df["complete_path_od_file_input_simulation"])

                        else: 
                            pass

                        # 14) NOTE: Launch the Simulation
                        Success_Simulation = LaunchSimulation(GeoInfo,
                                                                R,
                                                                str_UCI,
                                                                Name_parameters_output,
                                                                update_dict_df)

                        if Success_Simulation:  
                            # 15) NOTE: Update the DataFrame of Simulations
                            Df_Info_Simulations = Update_Info_Simulations_Df(Df_Info_Simulations,
                                                                        update_dict_df)
                            Df_Info_Simulations.write_csv(complete_path_info_simulations_df)      

                    else:
                        pass
                
            Df_City2Config = Df_Info_Simulations.filter(pl.col("city") == CityName,
                                                    pl.col("R") == R,
                                                    pl.col("UCI") == float_rounded_UCI)
            # 10) NOTE: Analyze the Output
#            PCTA = Polycentrism2TrafficAnalyzer(Df_City2Config,GeoInfo.ArrayRs,[float_rounded_UCI])  
#            PCTA.CompleteAnalysis()



    

if __name__ == '__main__':
    setup_mpl()
    # CONSTANTS
    NameCities = ["SFO","LAX","LIS","BOS"] # ["RIO","SFO","LAX","LIS","BOS"]
    container_name = "xuanjiang1998/lpsim:v1"    
    
    dir_file_demand = os.path.join(BERKELEY_2018_DIR,'new_full_network')
    dir_new_full_network = os.path.join(BERKELEY_2018_DIR,'new_full_network')
    dir_output_simulation_gpu = os.environ["LPSim"]
    dir_data_2 = "/data_2/alberto"
    complete_path_info_simulations_df = os.path.join(TRAFFIC_DIR,"info_simulations.csv")
    # NOTE: every time you change here, change also the corrispective update_dict_df
    list_columns_df_info_simulations = [# Main() Variables
                                        "dir_output_simulation_analysis",                   # NOTE: i.e. /data_2/alberto/CityName/Output
                                        "complete_path_od_file_input_simulation",           # NOTE: i.e. 
                                        "complete_path_output_simulation_copied",           # NOTE:                                                                    
                                        "name_od_file_input_simulation",
                                        "name",                                             # NOTE:
                                        "complete_path_graphml",                            # NOTE:
                                        "city",                                             # NOTE:
                                        "UCI",                                              # NOTE:
                                        "R",                                                # NOTE:
                                        "iter",                                             # NOTE:
                                        "str_UCI",                                          # NOTE:                     # NOTE: Name of the city, graphml file, UCI, R, iter
                                        # ProcessMain() Variables
                                        "dir_file_demand",                                  # NOTE:
                                        "dir_new_full_network",                             # NOTE:
                                        "dir_output_simulation_gpu",                        # NOTE:
                                        # PCTA complete analysis
                                        "number_realizations_OD_ensemble",                  # NOTE:
                                        "delta_t",                                          # NOTE:
                                        "complete_path_unloading_curve_base",               # NOTE:
                                        "complete_path_unloading_curve_iter",               # NOTE:                                 # NOTE: Path to the unloading curve
                                        "complete_path_route_file",                         # NOTE:
                                        "complete_path_people_file"                         # NOTE:
                                        ]                                                                                                                                # NOTE: list of informations stored from simulation for triplets (R,UCI,City)
    # Post Processing
    BaseConfig = os.path.join(os.environ["TRAFFIC_DIR"],"config")
    for CityName in NameCities:
        Main(CityName,
        TRAFFIC_DIR,
        dir_data_2,
        list_columns_df_info_simulations,
        complete_path_info_simulations_df,
        round_UCI = 3,
        number_realizations_OD_ensemble = 8)
