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
if socket.gethostname()=='artemis.ist.berkeley.edu':
    sys.path.append(os.path.join('/home/alberto/LPSim','traffic_phase_transition','scripts','ServerCommunication'))
else:
    sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','ServerCommunication'))
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

def Init_Info_Simulations_Df(
        complete_path_info_simulations_df,
        list_columns_df_info_simulations
        ):
    if os.paht.isfile(complete_path_info_simulations_df):
        from polars import read_csv
        return read_csv(complete_path_info_simulations_df)
    else:
        from polars import DataFrame
        return DataFrame({k:[] for k in list_columns_df_info_simulations})
        
def Update_Info_Simulations_Df(
        Df_Info_Simulations,
        updating_dict):
    """
        Returns the Polar datafrane containing all the different informations about set of simulations
        that are run for the 4-plet (city,R,UCI,iter)
    """
    assert False not in [col in Df_Info_Simulations.columns for col in updating_dict.keys()], "Some columns are not in the DataFrame."
    from polars import DataFrame,concat
    return concat([Df_Info_Simulations, DataFrame(updating_dict)]) 
        

def Update_update_dict_df(update_dict_df,
                          complete_path_od_file_input_simulation,
                          complete_path_output_simulation_copied,
                          NameCity,
                          dir_output_simulation_analysis,
                          name_file_graphml,
                          dir_file_demand = os.path.join(BERKELEY_2018_DIR,'new_full_network'),
                          dir_new_full_network = os.path.join(BERKELEY_2018_DIR,'new_full_network'),
                          dir_output_simulation_gpu = os.environ["LPSIM"],
                          number_realizations_OD_ensemble = 4,
                          delta_t = 10,
                          ):
    """
        @param update_dict_df: dict -> Dictionary with the information about the simulation

    """
    update_dict_df["dir_output_simulation_analysis"] = dir_output_simulation_analysis                       # NOTE: We copy simulator output here.
    update_dict_df["complete_path_od_file_input_simulation"] = complete_path_od_file_input_simulation       #
    update_dict_df["complete_path_output_simulation_copied"] = complete_path_output_simulation_copied       #
    update_dict_df["name"] = NameCity                                                                       #
    update_dict_df["graphml_file"] = name_file_graphml                                                      #
    #### CONSTANTS ####
        # ProcessMain() Variables
    update_dict_df["dir_file_demand"] = dir_file_demand                                                     # NOTE: Simulator checks by default here for demand
    update_dict_df["dir_new_full_network"] = dir_new_full_network                                           # NOTE: Simulator checks by default here for cartography
    update_dict_df["dir_output_simulation_gpu"] = dir_output_simulation_gpu                                 # NOTE: Simulator Puts the output of the simulation here
    update_dict_df["number_realizations_OD_ensemble"] = number_realizations_OD_ensemble                     # NOTE: Number of different simulations launched for a couple UCI, R (what changes are the realization of nodes where people are inserted.)
    update_dict_df["delta_t"] = delta_t                                                                 #
    return update_dict_df



def Main(CityName,
         TRAFFIC_DIR,
         dir_file_demand,
         dir_new_full_network,
         dir_output_simulation_gpu,
         dir_base_output_simulation_analysis,
         list_columns_df_info_simulations,
         round_UCI = 3,
         number_realizations_OD_ensemble = 4):
    """
        @param NameCities: list -> List of the names of the cities
        @param list_columns_df_info_simulations: list -> 
            List of the columns of the DataFrame that contains the information about the simulations
    """
    # Where we hold informations about the simulations
    update_dict_df = {
        # ProcessMain() Variables
        "dir_file_demand":[],                               # NOTE: Simulator checks by default here for demand
        "dir_new_full_network":[],                          # NOTE: Simulator checks by default here for cartography
        "dir_output_simulation_gpu": [],                    # NOTE: Simulator Puts the output of the simulation here
        "dir_output_simulation_analysis": [],               # NOTE: We copy simulator output here.
        "complete_path_od_file_input_simulation": [],       # NOTE: Complete path of the OD file input simulation
        "complete_path_output_simulation_copied": [],       # NOTE: Complete path of the output simulation copied
        ## numbers
        "number_realizations_OD_ensemble": [],              # NOTE: Number of different simulations launched for a couple UCI, R (what changes are the realization of nodes where people are inserted.)
        # CompleteAnalysis() Variables          
        "delta_t": [],                                      #    
        "name": [],                                         #
        "graphml_file": [],                                 #
        # Shared CompleteAnalysis()/ProcessMain() Variables
        }
    # SCOPE CONSTANTS
    dir_output_simulation_analysis_base = os.path.join(dir_base_output_simulation_analysis,CityName,"Output")
    list_columns_df_info_simulations = list(update_dict_df.keys())
    complete_path_info_simulations_df = os.path.join(TRAFFIC_DIR,"info_simulations.csv")
    Df_Info_Simulations = Init_Info_Simulations_Df(complete_path_info_simulations_df,list_columns_df_info_simulations)
    # 1) NOTE: Create Geometry Object
    GeoInfo = GeometricalSettingsSpatialPartition(CityName,TRAFFIC_DIR)
    # 2) NOTE: Compute Geometries
    GeoInfo.GetGeometries()
    # 3) NOTE: Compute the Potential and Vector field for non modified fluxes
    UCI = GeoInfo.RoutineVectorFieldAndPotential()
    # 4) NOTE: Compute the Fit for the gravity model
    GeoInfo.ComputeFit()
    # 5) NOTE: Initialize the Concatenated Df for Simulation [It is common for all different R]
    GeoInfo.InitializeDf4Sim()
    GeoInfo.ComputeEndFileInputSimulation()         
    # 6) NOTE: Launch Simulations Non Modified
    str_UCI = str(round(UCI,round_UCI))
    for R in GeoInfo.ArrayRs:
        for ensemble_realization in range(number_realizations_OD_ensemble):
            # update Dictionary
            suffix_parquet_people_file = f'0_people{GeoInfo.start}to24.parquet'                                                                             # Files Simulation                 
            Name_parameters_output = f"R_{R}_UCI_{round(UCI,3)}_realization_{ensemble_realization}"                                               # Ensemble Choice                          
            #
            name_od_file_input_simulation = f"{GeoInfo.city}_oddemand_{GeoInfo.start}_{GeoInfo.start + 1}_{Name_parameters_output}.csv"     # OD
            complete_path_od_file_input_simulation = os.path.join(dir_file_demand,name_od_file_input_simulation)
            #
            name_output_simulation_copied = f"{Name_parameters_output}_{suffix_parquet_people_file}"
            dir_output_simulation_analysis = os.path.join(dir_output_simulation_analysis_base,str_UCI)
            complete_path_output_simulation_copied = os.path.join(dir_output_simulation_analysis,name_output_simulation_copied) # Output dir
            os.makedirs(os.path.join(complete_path_output_simulation_copied), exist_ok=True)
            
            #
            update_dict_df = Update_update_dict_df(update_dict_df,
                                                complete_path_od_file_input_simulation = complete_path_od_file_input_simulation,
                                                complete_path_output_simulation_copied = complete_path_output_simulation_copied,
                                                NameCity = CityName,
                                                dir_output_simulation_analysis = dir_output_simulation_analysis,
                                                name_file_graphml = os.path.join(TRAFFIC_DIR,'data','carto',CityName,CityName + '_new_tertiary_simplified.graphml'),
                                                dir_file_demand = dir_file_demand,
                                                dir_new_full_network = dir_new_full_network,
                                                dir_output_simulation_gpu = dir_output_simulation_gpu,
                                                number_realizations_OD_ensemble = number_realizations_OD_ensemble,
                                                delta_t = 10,
                                                )                
            #
            is_success_simulation  = os.path.isfile(complete_path_output_simulation_copied)                               # Output file exists ?
            is_computed_od_simulation_input = os.path.isfile(complete_path_od_file_input_simulation)                      # OD file exists ?
            #
            if not is_success_simulation:                                                       # Check if the simulation is already computed
                if not is_computed_od_simulation_input:                                         # Check if the OD file is already computed
                    # Not Modified Mobility
                    logger.info(f"ID-Non-Mod R: {R}, {GeoInfo.city}, UCI: {round(UCI,3)}")
                    # 7) NOTE: Compute the realization of the OD
                    InputDocker = GeoInfo.ComputeDf4SimNotChangedMorphology(UCI,
                                                                            R,
                                                                            update_dict_df["complete_path_od_file_input_simulation"])

                    # 8) NOTE: Launch the Simulation
                    LaunchSimulations("NonModified",
                                GeoInfo,
                                None,
                                R,
                                UCI,
                                Number_Realizations_OD_Ensemble,
                                None)

                else: 
                    pass
            else:
                pass
                
    # 9) NOTE: Update the DataFrame of Simulations
    Df_Info_Simulations = Update_Info_Simulations_Df(Df_Info_Simulations,
                                                    update_dict_df)        
    City2Config = Df_Info_Simulations.filter(pl.col("city") == CityName,
                                            pl.col("R") == R,
                                            pl.col("UCI") == UCI).to_dict()
    # 10) NOTE: Analyze the Output
    PCTA = Polycentrism2TrafficAnalyzer(City2Config,GeoInfo.ArrayRs,[UCI])  
    PCTA.CompleteAnalysis()
    logger.info(f"Finished the simulations not changed for {CityName}")
    # 11) NOTE: Generate New Population 
    GeoInfo.GeneratePopulationAndSetUCIs()                                                                      # Compute New Grid
    for UCI_Interval in GeoInfo.UCIInterval2UCI.keys():
        for UCIM in GeoInfo.UCIInterval2UCI[UCI_Interval]:
            from pandas import read_parquet
            from geopandas import GeoDataFrame
            UCI = float(UCIM)
            # New Population
            GridNew = read_parquet(os.path.join(GeoInfo.save_dir_local,f'Grid_{round(UCIM,3)}.parquet'))        # New Grid (read)
            GridNew[["geometry","centroidx","centroidy"]] = GeoInfo.grid[["geometry","centroidx","centroidy"]]  # New Grid (add centroid, geometry)
            GridNew = GeoDataFrame(GridNew)                                                                     # New Grid
            Modified_Fluxes = GeoInfo.ComputeTijFromGrid(GridNew)                                               # New Fluxes
            # 12) NOTE: Launch Simulations Modified
            for R in GeoInfo.ArrayRs:
                for ensemble_realization in range(Number_Realizations_OD_Ensemble):
                    Name_parameters_output = f"R_{R}_UCI_{round(UCI,3)}_iter_{ensemble_realization}"                        
                    logger.info(f"ID-Mod R: {R}, {GeoInfo.city}, UCI: {round(UCI,3)}")
                    #
                    name_od_file_input_simulation = f"{GeoInfo.city}_oddemand_{GeoInfo.start}_{GeoInfo.start + 1}_{Name_parameters_output}.csv"     # OD
                    complete_path_od_file_input_simulation = os.path.join(dir_file_demand,
                                                                            name_od_file_input_simulation)
                    #
                    name_output_simulation_copied = f"{Name_parameters_output}_{suffix_parquet_people_file}"
                    complete_path_output_simulation_copied = os.path.join(dir_output_simulation_analysis,       # Output dir
                                                                        name_output_simulation_copied)

                    #
                    update_dict_df = Update_update_dict_df(update_dict_df,
                                                        complete_path_od_file_input_simulation = complete_path_od_file_input_simulation,
                                                        complete_path_output_simulation_copied = complete_path_output_simulation_copied,
                                                        NameCity = CityName,
                                                        dir_output_simulation_analysis = f"/data_2/alberto/{CityName}/Output",
                                                        name_file_graphml = os.path.join(TRAFFIC_DIR,'data','carto',CityName,CityName + '_new_tertiary_simplified.graphml'),
                                                        dir_file_demand = os.path.join(BERKELEY_2018_DIR,'new_full_network'),
                                                        dir_new_full_network = os.path.join(BERKELEY_2018_DIR,'new_full_network'),
                                                        dir_output_simulation_gpu = os.environ["LPSIM"],
                                                        number_realizations_OD_ensemble = 4,
                                                        delta_t = 10,
                                                        )                
                    #
                    is_success_simulation  = os.path.isfile(complete_path_output_simulation_copied)                               # Output file exists ?
                    is_computed_od_simulation_input = os.path.isfile(complete_path_od_file_input_simulation)                      # OD file exists ?
                    #
                    if not is_success_simulation:                                                       # Check if the simulation is already computed
                        if not is_computed_od_simulation_input:                                         # Check if the OD file is already computed
                            # 13) NOTE: Compute the realization of the OD
                            InputDocker = GeoInfo.ComputeDf4SimChangedMorphology(UCI,
                                                                                R,
                                                                                Modified_Fluxes,
                                                                                complete_path_od_demand_file)

                
                            # 14) NOTE: Launch the Simulation
                            LaunchSimulations("NonModified",
                                        GeoInfo,
                                        None,
                                        R,
                                        UCI,
                                        Number_Realizations_OD_Ensemble,
                                        None)

                        else: 
                            pass
                    else:
                        pass
                
            # 9) NOTE: Update the DataFrame of Simulations
            Df_Info_Simulations = Update_Info_Simulations_Df(Df_Info_Simulations,
                                                            update_dict_df)        
            City2Config = Df_Info_Simulations.filter(pl.col("city") == CityName,
                                                    pl.col("R") == R,
                                                    pl.col("UCI") == UCI).to_dict()
            # 10) NOTE: Analyze the Output
            PCTA = Polycentrism2TrafficAnalyzer(City2Config,GeoInfo.ArrayRs,[UCI])  
            PCTA.CompleteAnalysis()


    

if __name__ == '__main__':
    # CONSTANTS
    NameCities = ["BOS","SFO","RIO"]#["RIO","SFO","LAX","LIS","BOS"] # ["LAX"]
    container_name = "xuanjiang1998/lpsim:v1"    
    
    dir_file_demand = os.path.join(BERKELEY_2018_DIR,'new_full_network')
    dir_new_full_network = os.path.join(BERKELEY_2018_DIR,'new_full_network')
    dir_output_simulation_gpu = os.environ["LPSIM"]
    dir_base_output_simulation_analysis = "/data_2/alberto"
    #
    list_columns_df_info_simulations = ["dir_file_demand",                  # NOTE: Simulator checks by default here for demand
                                        "dir_new_full_network",             # NOTE: Simulator checks by default here for cartography
                                        "dir_output_simulation_gpu",        # NOTE: Simulator Puts the output of the simulation here
                                        "dir_output_simulation_analysis",   # NOTE: We copy simulator output here.    
                                        # 
                                        "delta_t",
                                        "city",
                                        "graphml_file",
                                        
                                        ]
    # Post Processing
    BaseConfig = os.path.join(os.environ["TRAFFIC_DIR"],"config")
    BThread = False
    if BThread:
        num_threads = len(NameCities)
        # Create a lock for synchronization
        lock = Lock()
        # Create and start threads
        threads = []
        for rank in range(num_threads):
            thread = Thread(target=Threaded_main, args=(rank, num_threads, lock))
            threads.append(thread)
            thread.start()
        # Wait for all threads to finish
        for thread in threads:
            thread.join()
    
    else:
        # Create a shared dictionary
#        manager = Manager()
#        shared_dict = manager.dict()
        # Create a monitor thread
#        monitor_thread = threading.Thread(target=monitor_processes,args=(shared_dict,))
#        monitor_thread.daemon = True
#        monitor_thread.start()
        # Queue
#        queue = Queue()
#        error_flag = Value(ctypes.c_bool, False)
#        R_error = Value(ctypes.c_int, 0)
        # Start the main process
        for CityName in NameCities:

            City2RUCI = {CityName: {"UCI": [], "R": []}}
            # Everything is handled inside the object
            GeoInfo = GeometricalSettingsSpatialPartition(CityName,TRAFFIC_DIR)
            # Compute the Geometries
            GeoInfo.GetGeometries()
            # Compute the Potential and Vector field for non modified fluxes
            UCI = GeoInfo.RoutineVectorFieldAndPotential()
            # Compute the Fit for the gravity model
            GeoInfo.ComputeFit()
            # Initialize the Concatenated Df for Simulation [It is common for all different R]
            GeoInfo.InitializeDf4Sim()
            GeoInfo.ComputeEndFileInputSimulation()
            # NOTE: Can Parallelize this part and launch the simulations in parallel.
            if len(GeoInfo.ArrayRs) < cpu_count():
                N = len(GeoInfo.ArrayRs)
            else:
                N = cpu_count() - 2
            concurrency_manager = ConcurrencyManager(N,GeoInfo.save_dir_local)
#            barrier = Barrier(N)
#            lock = Lock()
            # Do not Compute RArray if not necessary, stick with the given by CityRminRmax
            FirstTry = True
            # NOTE: Eliminate Concurrency since the Place we save stuff is the same and not conditioned
            # So there is race condition that cannot be handled
            Number_Realizations_OD_Ensemble = 4
            City2Config = {CityName:{
                # ProcessMain() Variables
                    ## directories
                    "dir_file_demand":os.path.join(BERKELEY_2018_DIR,'new_full_network'),           # NOTE: Simulator checks by default here for demand
                    "dir_new_full_network":os.path.join(BERKELEY_2018_DIR,'new_full_network'),      # NOTE: Simulator checks by default here for cartography
                    "dir_output_simulation_gpu": os.environ["LPSIM"], 
                    "dir_output_simulation_analysis": f"/data_2/alberto/{CityName}/Output", 
                ## numbers
                "number_realizations_OD_ensemble": Number_Realizations_OD_Ensemble,             # Number of different simulations launched for a couple UCI, R (what changes are the realization of nodes where people are inserted.)
                # CompleteAnalysis() Variables
                "delta_t": 10,
                "name": CityName,
                "graphml_file": os.path.join(TRAFFIC_DIR,'data','carto',CityName,CityName + '_new_tertiary_simplified.graphml'),
                # Shared CompleteAnalysis()/ProcessMain() Variables
                }
            }
            City2UCI2R2Infos = {CityName: {UCI: {R: {"people_file": None, "route_file": None} for R in GeoInfo.ArrayRs} for UCI in GeoInfo.UCIInterval2UCIAccepted.keys()}} 
            for R in GeoInfo.ArrayRs:
                for ensemble_realization in range(City2Config[CityName]["number_realizations_OD_ensemble"]):
                    if ensemble_realization == 0:
                        Name_parameters_output = f"R_{R}_UCI_{round(UCI1,3)}"
                        City2Config[CityName]["name_od_file_input_simulation"] = f"{GeoInfo.city}_oddemand_{GeoInfo.start}_{GeoInfo.start + 1}_{Name_parameters_output}.csv"
                        City2Config[CityName][""]
                    else:
                        Name_parameters_output = f"R_{R}_UCI_{round(UCI1,3)}_iter_{ensemble_realization}"                        
                        City2Config[CityName]["name_od_file_input_simulation"] = f"{GeoInfo.city}_oddemand_{GeoInfo.start}_{GeoInfo.start + 1}_{Name_parameters_output}.csv"                    
                        ProcessMain("NonModified",
                                    GeoInfo,
                                    None,
                                    R,
                                    UCI,
                                    Number_Realizations_OD_Ensemble,
                                    None)
            # Analyze the Output
            PCTA = Polycentrism2TrafficAnalyzer(City2Config[CityName],GeoInfo.ArrayRs,[UCI])  
            PCTA.CompleteAnalysis()

            while(False):
                # Reset The Processes
                processes = []
#                concurrency_manager.Reset()
                if FirstTry:
                    ArrayRs = GeoInfo.ArrayRs
                    logger.info("FirstTry: {}".format(FirstTry))
                    FirstTry = False
                else:
                    logger.info("Somehow Error: Rmax = {}".format(concurrency_manager.R_errors[concurrency_manager.GPU_error_index.value].value))
                    # Keep the Number Of Rs Tried in the Simulation
                    ArrayRs = RedefineRsWhenError(concurrency_manager.R_errors[concurrency_manager.GPU_error_index.value].value,GeoInfo.Step,len(GeoInfo.ArrayRs))
                for R in ArrayRs:
                    p = Process(target=ProcessMain, args=("NonModified",GeoInfo,None,R,UCI,concurrency_manager))
                    # NOTE: error_flag and R error are shared to find the Array for which we have no crash of the simulation
#                    p = Process(target=ProcessLauncherNonModified, args=(shared_dict,queue,barrier,lock,GeoInfo, UCI, R,error_flag,R_error))
                    processes.append(p)
                    p.start()
                # Every Time A Process Join
                for p in processes:
                    p.join()
                    # If Error From Docker or Container
                    if concurrency_manager.Docker_error.value:
                        for p in processes:
                            if p.is_alive():
                                p.terminate()
                    # If No Docker Errors
                    else:
                        # Error From GPU index
                        if concurrency_manager.GPU_error_index.value >= 0: 
                            # Extract The 
                            if concurrency_manager.GPU_errors[concurrency_manager.GPU_error_index.value].value:
                                for p in processes:
                                    if p.is_alive():
                                        p.terminate()
                    
                        # No Error From GPU
                        else:
                            # This Is The Only Case I Break The Loop
                            # NOTE: No Docker Problem, No GPU Problem 
                            break
                # If No Error Occurred Free The Processes
                if not concurrency_manager.Docker_error.value and not concurrency_manager.GPU_errors[concurrency_manager.GPU_error_index.value].value:
                    break 


#            barrier.wait()
            logger.info(f"Finished the simulations not changed for {CityName}")
            # Generate New Population
            GeoInfo.GeneratePopulationAndSetUCIs()
                    # Generate modified Fluxes
            print(GeoInfo.UCIInterval2UCI)
            for UCI_Interval in GeoInfo.UCIInterval2UCI.keys():
                for UCIM in GeoInfo.UCIInterval2UCI[UCI_Interval]:
                    if len(GeoInfo.ArrayRs) < cpu_count():
                        N = len(GeoInfo.ArrayRs)
                    else:
                        N = cpu_count() - 2
                    barrier = Barrier(N)
                    processes = []
                    from pandas import read_parquet
                    from geopandas import GeoDataFrame
                    UCI1 = float(UCIM)
                    # New Population
                    GridNew = read_parquet(os.path.join(GeoInfo.save_dir_local,f'Grid_{round(UCIM,3)}.parquet'))
                    GridNew[["geometry","centroidx","centroidy"]] = GeoInfo.grid[["geometry","centroidx","centroidy"]]
                    GridNew = GeoDataFrame(GridNew)
                    # New Fluxes
                    Modified_Fluxes = GeoInfo.ComputeTijFromGrid(GridNew)
                    # New UCI
#                    UCI1 = GeoInfo.RoutineVectorFieldAndPotentialModified(GridNew,Modified_Fluxes)
                    # Check that UCI is in a valid interval
                    concurrency_manager = ConcurrencyManager(N,GeoInfo.save_dir_local)
                    log_to_stderr()
                    FirstTry = True
                    for R in GeoInfo.ArrayRs:
                        ProcessMain("Modified",
                                    GeoInfo,
                                    Modified_Fluxes,
                                    R,
                                    UCI1,
                                    Number_Realizations_OD_Ensemble,
                                    None)
                    # Analyze the Output
#                    City2Config,Rs,UCIs = InitConfigPolycentrismAnalysis(CityName)  
                    PCTA = Polycentrism2TrafficAnalyzer(City2Config[CityName],GeoInfo.ArrayRs,[UCI1])  
                    PCTA.CompleteAnalysis()
                        
                    while(False):
                        # Reset The Processes
                        processes = []
    #                            concurrency_manager.Reset()
                        if FirstTry:
                            ArrayRs = GeoInfo.ArrayRs
                            FirstTry = False
                            logger.info("FirstTry: {}".format(FirstTry))
                        else:
                            logger.info("Somehow Error: Rmax = {}".format(concurrency_manager.R_errors[concurrency_manager.GPU_error_index.value].value))
                            # Keep the Number Of Rs Tried in the Simulation
                            ArrayRs = RedefineRsWhenError(concurrency_manager.R_errors[concurrency_manager.GPU_error_index.value].value,GeoInfo.Step,len(GeoInfo.ArrayRs))
                        for R in ArrayRs:
                            p = Process(target=ProcessMain, args=("Modified",GeoInfo,Modified_Fluxes,R,UCI1,concurrency_manager))
                            # NOTE: error_flag and R error are shared to find the Array for which we have no crash of the simulation
        #                    p = Process(target=ProcessLauncherNonModified, args=(shared_dict,queue,barrier,lock,GeoInfo, UCI, R,error_flag,R_error))
                            processes.append(p)
                            p.start()
                        # Every Time A Process Join
                        for p in processes:
                            p.join()
                            # If Error From Docker or Container
                            if concurrency_manager.Docker_error.value:
                                for p in processes:
                                    if p.is_alive():
                                        p.terminate()
                            # If No Docker Errors
                            else:
                                # Error From GPU index
                                if concurrency_manager.GPU_error_index.value >= 0: 
                                    # Extract The 
                                    if concurrency_manager.GPU_errors[concurrency_manager.GPU_error_index.value].value:
                                        for p in processes:
                                            if p.is_alive():
                                                p.terminate()
                                # No Error From GPU
                                else:
                                    # This Is The Only Case I Break The Loop
                                    # NOTE: No Docker Problem, No GPU Problem 
                                    break
                        if not concurrency_manager.Docker_error.value and not concurrency_manager.GPU_errors[concurrency_manager.GPU_error_index.value].value:
                            break 
        