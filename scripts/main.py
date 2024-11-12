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
from multiprocessing import Pool,cpu_count
import socket
import sys
# 
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
import queue
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


def Threaded_main(rank, num_threads, lock):
    CityName = NameCities[rank]
    City2RUCI = {CityName:{"UCI":[],"R":[]}}
    # Everything is handled inside the object
    GeoInfo = GeometricalSettingsSpatialPartition(NameCities[rank],TRAFFIC_DIR)
    GeoInfo.GetGeometries()
    # Compute the Potential and Vector field for non modified fluxes
    UCI = GeoInfo.RoutineVectorFieldAndPotential()
    # Compute the Fit for the gravity model
    GeoInfo.ComputeFit()
    # Initialize the Concatenated Df for Simulation [It is common for all different R]
    GeoInfo.InitializeDf4Sim()
    GeoInfo.ComputeEndFileInputSimulation()    
    # NOTE: Can Parallelize this part and launch the simulations in parallel.
    City2RUCI[CityName]["R"] = list(GeoInfo.ArrayRs)
    for R in GeoInfo.ArrayRs:
        # Simulation for the monocentric case.
        NotModifiedInputFile = GeoInfo.ComputeDf4SimNotChangedMorphology(UCI,R)
        City2RUCI[CityName]["UCI"].append(UCI)

        while(True):
            lock.acquire()
            # Tells that the process is occupied
            LaunchDockerFromServer(container_name,CityName,GeoInfo.start,GeoInfo.start + 1,R,UCI)
            DeleteInputSimulation(NotModifiedInputFile)
            lock.release()
            break
        # Generate modified Fluxes
    for cov in GeoInfo.config['covariances']:
        for distribution in ['exponential']:
            for num_peaks in GeoInfo.config['list_peaks']:
                for R in GeoInfo.ArrayRs:
                    Modified_Fluxes,UCI1 = GeoInfo.ChangeMorpholgy(cov,distribution,num_peaks)
                    City2RUCI[CityName]["UCI"].append(UCI1)
                    ModifiedInputFile = GeoInfo.ComputeDf4SimChangedMorphology(UCI1,R,Modified_Fluxes)
                    start = GeoInfo.start
                    end = start + 1
                    InputSimulation = (container_name,CityName,start,end,R,UCI1)
                    # Launch the processes one after the other since you may have race conditions on GPUs (we have just two of them)
                    while(True):
                        # Start a lock into the window (ensures that each process have syncronous access to the window from a queue)
                        lock.acquire()
                        # Tells that the process is occupied
                        if os.path.isfile(os.path.join(OD_dir,f"{CityName}_oddemand_{start}_{end}_R_{R}_UCI_{round(UCI1,3)}.csv")):
                            logger.info(f"Launching docker, {CityName}, R: {R}, UCI: {round(UCI1,3)}")
                            LaunchDockerFromServer(InputSimulation)
                            DeleteInputSimulation(ModifiedInputFile)
                        lock.release()
                        break
                    # Post Process
                    City2Config = InitConfigPolycentrismAnalysis([CityName])                        
                    PCTA = Polycentrism2TrafficAnalyzer(City2Config[CityName])  
                    PCTA.CompleteAnalysis()
                    with open(os.path.join(BaseConfig,'post_processing_' + CityName +'.json'),'w') as f:
                        json.dump(City2Config,f,indent=4)    

def ComputeSimulationFileAndLaunchDocker(GeoInfo,UCI,R):
    NotModifiedInputFile = GeoInfo.ComputeDf4SimNotChangedMorphology(UCI,R)
    while(True):
        lock.acquire()
        # Tells that the process is occupied
        LaunchDockerFromServer(container_name,CityName,GeoInfo.start,GeoInfo.start + 1,R,UCI)
        DeleteInputSimulation(NotModifiedInputFile)
        lock.release()
        break

def ComputeSimulationFileAndLaunchDockerFromModifiedMob(GeoInfo,R):
    Modified_Fluxes,UCI1 = GeoInfo.ChangeMorpholgy(cov,distribution,num_peaks)
    ModifiedInputFile = GeoInfo.ComputeDf4SimChangedMorphology(UCI1,R,Modified_Fluxes)
    start = GeoInfo.start
    end = start + 1
    if os.path.isfile(ModifiedInputFile):
        logger.info(f"Launching docker, {CityName}, R: {R}, UCI: {round(UCI1,3)}")
        LaunchDockerFromServer(container_name,CityName,start,end,R,UCI1)
        DeleteInputSimulation(ModifiedInputFile)
    # Launch the processes one after the other since you


if __name__ == '__main__':
    #NameCities = os.listdir(os.path.join(TRAFFIC_DIR,'data','carto'))
#    NameCities = ["BOS","LAX","SFO","LIS","RIO"]
    NameCities = ["SFO"]
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
            with Pool(len(GeoInfo.ArrayRs)) as p:
                p.map(ComputeSimulationFileAndLaunchDocker,[(GeoInfo,UCI,R) for R in GeoInfo.ArrayRs])
#            for R in GeoInfo.ArrayRs:
                # Simulation for the monocentric case.
#                NotModifiedInputFile = GeoInfo.ComputeDf4SimNotChangedMorphology(UCI,R)
#                end = GeoInfo.start + 1
#                if os.path.isfile(NotModifiedInputFile):
#                    LaunchDockerFromServer(container_name,CityName,GeoInfo.start,GeoInfo.start + 1,R,UCI)
#                    DeleteInputSimulation(NotModifiedInputFile)
            # Generate modified Fluxes
            for cov in GeoInfo.config['covariances']:
                for distribution in ['exponential']:
                    for num_peaks in GeoInfo.config['list_peaks']:
                        if len(GeoInfo.ArrayRs) < cpu_count():
                            N = len(GeoInfo.ArrayRs)
                        else:
                            N = cpu_count() - 2
                        with Pool(len(GeoInfo.ArrayRs)) as p:
                            p.map(ComputeSimulationFileAndLaunchDockerFromModifiedMob,[(GeoInfo,R) for R in GeoInfo.ArrayRs])
#                        for R in GeoInfo.ArrayRs:
#                            Modified_Fluxes,UCI1 = GeoInfo.ChangeMorpholgy(cov,distribution,num_peaks)
#                            ModifiedInputFile = GeoInfo.ComputeDf4SimChangedMorphology(UCI1,R,Modified_Fluxes)
#                            start = GeoInfo.start
#                            end = start + 1
#                            if os.path.isfile(ModifiedInputFile):
#                                logger.info(f"Launching docker, {CityName}, R: {R}, UCI: {round(UCI1,3)}")
#                               LaunchDockerFromServer(container_name,CityName,start,end,R,UCI1)
#                                DeleteInputSimulation(ModifiedInputFile)

        #NOTE: TODO Change the code in such a way that I compute the UCI,R  -> construct the file.
        # Launch the simulations, delete the input files and save the output in parallel.
        # Then I can run the simulation for the different R in parallel.
        