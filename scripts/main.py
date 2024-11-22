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

if __name__ == '__main__':
    #NameCities = os.listdir(os.path.join(TRAFFIC_DIR,'data','carto'))
#    NameCities = ["BOS","LAX","SFO","LIS","RIO"]
    NameCities = ["BOS"]
    container_name = "xuanjiang1998/lpsim:v1"    
    OD_dir = os.path.join(TRAFFIC_DIR,'berkeley_2018',"new_full_network")
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
        manager = Manager()
        shared_dict = manager.dict()
        # Create a monitor thread
        monitor_thread = threading.Thread(target=monitor_processes,args=(shared_dict,))
        monitor_thread.daemon = True
        monitor_thread.start()
        # Queue
        queue = Queue()
        error_flag = Value(ctypes.c_bool, False)
        R_error = Value(ctypes.c_int, 0)
        # Start the main process
        for CityName in NameCities:
            City2RUCI = {CityName: {"UCI": [], "R": []}}
            # Everything is handled inside the object
            GeoInfo = GeometricalSettingsSpatialPartition(CityName,TRAFFIC_DIR)
            # Compute the Geometries
            GeoInfo.GetGeometries()
            # Compute the Potential and Vector field for non modified fluxes
            UCI = GeoInfo.RoutineVectorFieldAndPotential()
            # Check that UCI is in a valid interval
            index_UCI_represents = CheckIndexUCI(GeoInfo.UCIInterval2UCIAccepted,UCI)
            if GeoInfo.UCIInterval2UCIAccepted[index_UCI_represents] is None:
                 GeoInfo.UCIInterval2UCIAccepted[index_UCI_represents] = UCI
            else:
                pass
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
            # Run The Processes Until for All The Rs I do not Have A Docker Error
            # NOTE: GPU related Problems Act on ArrayRs
            while(not concurrency_manager.Docker_error.value):
                # Reset The Processes
                processes = []
                concurrency_manager.Reset()
                if FirstTry:
                    ArrayRs = GeoInfo.ArrayRs
                    FirstTry = False
                else:
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


#            barrier.wait()
            logger.info(f"Finished the simulations not changed for {CityName}")

            # Synchronize in the main process
#            City2Config = InitConfigPolycentrismAnalysis(CityName)                        
#            PCTA = Polycentrism2TrafficAnalyzer(City2Config[CityName])  
#            PCTA.CompleteAnalysis()
#            with open(os.path.join(BaseConfig,'post_processing_' + CityName +'.json'),'w') as f:
#                json.dump(City2Config,f,indent=4)    


            # Generate modified Fluxes
            for cov in GeoInfo.config['covariances']:
                for distribution in ['exponential']:
                    for num_peaks in GeoInfo.config['list_peaks']:
                        if len(GeoInfo.ArrayRs) < cpu_count():
                            N = len(GeoInfo.ArrayRs)
                        else:
                            N = cpu_count() - 2
                        barrier = Barrier(N)
                        processes = []
                        # NOTE: We are going to compute 20 UCIs, fixing an interval in the UCI space.
                        # All these if else are just to ensure that this will happen
                        # Check That The Grid Is Not Already Computed
                        GridNew,UCI1 = CheckAlreadyComputedGrids(GeoInfo.save_dir_local,cov,distribution,num_peaks)
                        if GridNew is not None:
                            Modified_Fluxes = GeoInfo.RecoverChangedMorpholgyFromGridNew(GridNew)
                        else:
                            # Compute The Modified Fluxes Otherwhise keep the old Computed ones. NOTE: To save computation
                            Modified_Fluxes,UCI1,GridNew = GeoInfo.ChangeMorpholgy(cov,distribution,num_peaks)
                        # Check that UCI is in a valid interval
                        index_UCI_represents = CheckIndexUCI(GeoInfo.UCIInterval2UCIAccepted,UCI1)
                        if GeoInfo.UCIInterval2UCIAccepted[index_UCI_represents] is None:
                            # Add it if it is not already in an valid interval
                            GeoInfo.UCIInterval2UCIAccepted[index_UCI_represents] = UCI1
                            GridNew.to_csv(os.path.join(GeoInfo.save_dir_local,f'GridNew_{cov}_{distribution}_{num_peaks}_{UCI1}.csv'),index=False)                    
                        else:
                            pass
                        log_to_stderr()
                        while(not concurrency_manager.Docker_error.value):
                            # Reset The Processes
                            processes = []
                            concurrency_manager.Reset()
                            if FirstTry:
                                ArrayRs = GeoInfo.ArrayRs
                                FirstTry = False
                            else:
                                # Keep the Number Of Rs Tried in the Simulation
                                ArrayRs = RedefineRsWhenError(concurrency_manager.R_errors[concurrency_manager.GPU_error_index.value].value,GeoInfo.Step,len(GeoInfo.ArrayRs))
                            for R in ArrayRs:
                                p = Process(target=ProcessMain, args=("Modified",GeoInfo,Modified_Fluxes,R,UCI,concurrency_manager))
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
'''                        
                        while(True):
                            if FirstTry:
                                ArrayRs = GeoInfo.ArrayRs
                                FirstTry = False
                            else:
                                # Keep the Number Of Rs Tried in the Simulation
                                ArrayRs = RedefineRsWhenError(R_error,GeoInfo.Step,len(GeoInfo.ArrayRs))
                            for R in ArrayRs:
                                p = Process(target = ProcessLauncherModified, args=(shared_dict,queue,barrier,lock,GeoInfo, Modified_Fluxes,R,UCI1,error_flag,R_error))
                                processes.append(p)
                                p.start()

                            # Every Time A Process Join
                            for p in processes:
                                p.join()
                                # Check if It Executed Or Gave an Error
                                # If I have encountered an Erro in the GPU out-of-memory I recompute the Rs that are best for the simulation, that will cause no memory error.
                                if error_flag.value:
                                    print("Error detected, terminating all processes.")
                                    for p in processes:
                                        if p.is_alive():
                                            p.terminate()
                            # If All The Processes Are Completed Without Error: Break
                            if not error_flag.value: 
                                break                

#                        barrier.wait()
                        logger.info(f"All simulations for {CityName} with cov={cov}, distribution={distribution}, num_peaks={num_peaks} completed successfully.")
                        # Check If I collected all UCIs
                        Terminate = CheckUCIIsFull(GeoInfo.UCIInterval2UCIAccepted)
                        if Terminate:
                            with open(os.path.join(GeoInfo.save_dir_local,'UCIInterval2UCI.json'),'w') as f:
                                json.dump(GeoInfo.UCIInterval2UCIAccepted,f,indent=4)
                            break
'''
#                        City2Config = InitConfigPolycentrismAnalysis(CityName)                        
#                        PCTA = Polycentrism2TrafficAnalyzer(City2Config[CityName])  
#                        PCTA.CompleteAnalysis()
#                        with open(os.path.join(BaseConfig,'post_processing_' + CityName +'.json'),'w') as f:
#                            json.dump(City2Config,f,indent=4)    

        