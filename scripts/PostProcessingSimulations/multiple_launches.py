import subprocess
import os
import argparse
import json
import time
import shutil
import psutil
import platform
import numpy as np
from multiprocessing import Pool
import sys
import logging
from pynvml import *
from GPUHandler import *
logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.environ["TRAFFIC_DIR"],"scripts","GeometrySphere"))
#from ODfromfma import NUMBER_SIMULATIONS,CityName2RminRmax

LPSIM_DIR = '/home/alberto/LPSim' 
HOME_DIR = '/home/alberto/LPSim/LivingCity'
TRAFFIC_DIR = '/home/alberto/LPSim/LivingCity'#'/lpsim/LivingCity'

def CheckPlatform(verbose = False):
    if verbose:
        print("***** CHECK PLATFORM DETAILS ******")
        print('System: ',platform.system())
        print('Platform: ',platform.platform())
        print('Release: ',platform.release())
        print('Version: ',platform.version())
        print('Machine: ',platform.machine())
        print('Processor: ',platform.processor())
        print('architecture: ',platform.architecture())

def ModifyConfigIni(CityName,start,end,R,UCI,verbose = False):
    '''
        Input:
            CityName: str -> in ['BOS','LAX','SFO','RIO','LIS']
            start: Start time analysis (default = 7) since we want to study traffic at its peak
            end: End time analysis (default = 24) since we want all day
            R: int -> People per second (Is the leading parameter to control how many people are inserted per unit time)
            UCI: Measures the plycentricity of city
        Description:
            Write the configuration file that must be given in input to the LivingCity program
            NOTE: Do not change position in folder: ../LPSim/
    '''
    logger.info("Input File: {0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv".format(CityName,start,end,R,round(UCI,3)))
    file_txt = '[General]\nGUI=false\nUSE_CPU=false\nNETWORK_PATH=LivingCity/berkeley_2018/new_full_network/\nUSE_JOHNSON_ROUTING=false\nUSE_SP_ROUTING=true\nUSE_PREV_PATHS=false\nLIMIT_NUM_PEOPLE=256000\nADD_RANDOM_PEOPLE=false\nNUM_PASSES=1\nTIME_STEP=1\nSTART_HR={0}\nEND_HR=24\nOD_DEMAND_FILENAME={1}_oddemand_{2}_{3}_R_{4}_UCI_{5}.csv\nSHOW_BENCHMARKS=false\nREROUTE_INCREMENT=0\nPARTITION_FILENAME=ciccio.txt\nNUM_GPUS=1\n'.format(start,CityName,start,end,R,round(UCI,3))
    with open(os.path.join(TRAFFIC_DIR,'command_line_options.ini'),'w') as file:
        file.write(file_txt)
    file.close()
#        print(file_txt)    


def RenameMoveOutput(output_file,CityName,R,UCI,verbose = False):
    """
        @params output_file: File to move
        @params CityName: Name of the city
        @params R: Number of people per unit time
        @params UCI: Urban Centrality Index
        @description: Move the output file in the correct
    """
    saving_dir = os.path.join(HOME_DIR,'berkeley_2018',CityName,'Output')
    source_file = os.path.join(LPSIM_DIR, output_file)
    destination_file = os.path.join(saving_dir, f"R_{R}_UCI_{round(UCI,3)}_{output_file}")
    logger.info(f"Transfer: {output_file} -> {destination_file}")
    if not os.path.exists(saving_dir):
        # Make Sure saving_dir exists
        os.mkdir(saving_dir)
    if os.path.exists(source_file):
        os.rename(source_file, destination_file)
    else:
        logger.info(f"File {source_file} does not exist")
#     shutil.move(os.path.join(LPSIM_DIR,f"R_{R}_UCI_{round(UCI,3)}_{output_file}"),saving_dir)  

  
def CheckUseDocker(verbose = False):
    """
        Check if the Docker is installed in the system
    """
    docker_version = ['docker', '--version']
    try:
        subprocess.run(docker_version, check=True)
        if verbose:
            logger.info("execution Docker Granted")
    except subprocess.CalledProcessError:
        if verbose:
            logger.info("execution Docker DENIED")


def compare_environment_variables(verbose = True):
    shell_env = dict(os.environ)
    python_env = dict()
    # Get environment variables in Python subprocess
    process = subprocess.Popen(['printenv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        python_env = {line.split('=', 1)[0]: line.split('=', 1)[1].strip() for line in stdout.decode().split('\n') if line}
    if verbose:
        print("***** COMPARE ENVIRONMENT/PROCESS VARIABLES ******")
        print("Environment variables in shell:")
        for key, value in shell_env.items():
            print(f"{key}: {value}")
    
        print("\nEnvironment variables in Python subprocess:")
        for key, value in python_env.items():
            print(f"{key}: {value}")
    
    # Compare environment variables
    diff_keys = set(shell_env.keys()) - set(python_env.keys())
    if diff_keys:
        if verbose:
            print("\nEnvironment variables differ between shell and Python subprocess:")
            for key in diff_keys:
                print(f"{key}: shell = {shell_env[key]}, Python subprocess = not set")
    else:
        if verbose:
            print("\nNo differences found in environment variables between shell and Python subprocess.")




def check_gpu_availability():
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            processes = nvmlDeviceGetComputeRunningProcesses(handle)
            if len(processes) == 0:
                nvmlShutdown()
                return True, i
        nvmlShutdown()
        return False,None
    except NVMLError as e:
        logger.error("An error occurred while querying GPU status: %s", str(e))
        return False,None    

def DockerCommand(shared_dict,save_dir,pid,R,UCI,CityName):
    """
        @description:
            Once the process arrives here starts to check if the GPUis available.
            If it is available, it launches the docker command to run the simulation.
            Once run the command, checks if there is an error.
            Return False if an error occurred in GPU or in the docker command
    """
    os.environ['PATH'] += os.pathsep + '/usr/bin/'
    container_name = "xuanjiang1998/lpsim:v1"
    PWD = '/home/alberto/LPSim'
    env = os.environ.copy()
    docker_cmd = ['/usr/bin/docker', 'run', '-it', '--rm', '--gpus', 'all', '-v', f'{PWD}:/lpsim', '-w', '/lpsim', f'{container_name}', 'bash', '-c', './LivingCity/LivingCity']
    NumberTrials = 0
    while True:
        # Execute the Docker command
#        logger.info(f"Executing Simulation Trial {NumberTrials}...")
        GpuAvailable,gpu_id = check_gpu_availability()
        if GpuAvailable:
            logger.info(f"{pid} Free GPU {gpu_id}")
            free_gpu_memory(gpu_id)            
            logger.info(f"{pid} Launch Simulation in GPU {gpu_id}")
            process = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            stdout, stderr = process.communicate()
            
            # Log the output
            logger.info(stdout.decode())
            logger.error(stderr.decode())
            
            # Check if the command was successful
            if process.returncode == 0:
                if check_gpu_errors():
                    logger.error("GPU error detected. Breaking the loop.")
                    successful = False
                    InfoAboutFailureInSharedDict(shared_dict,save_dir,UCI,R,CityName,successful)
                    return successful
                else:
                    logger.info("Docker command executed successfully.")
                    successful = True
                    InfoAboutFailureInSharedDict(shared_dict,save_dir,UCI,R,CityName,successful)
                    return successful
            else:
                logger.error(f"Docker command Number {NumberTrials} failed. Retrying...")
                NumberTrials += 1
                if check_gpu_errors():
                    logger.error("GPU error detected. Breaking the loop.")
                    successful = False
                    InfoAboutFailureInSharedDict(shared_dict,save_dir,UCI,R,CityName,successful)
                    return successful
                else:
                    successful = False
                    InfoAboutFailureInSharedDict(shared_dict,save_dir,UCI,R,CityName,successful)

                    return successful
        else:
            pass 
        NumberTrials += 1


def LaunchDockerFromServer(CityName,start,end,R,UCI,lock,shared_dict,save_dir):
    """
        @params container_name: Name container from which launch the GPU simulations
        @params CityName: Name simulation city
        @params start,end: start and end simulation (int:hour)
        @params R: Number of people per unit time
        @params UCI: Urban Centrality Index
    """
    # Run Simulation Just If it does not exist the Output
    saving_dir = os.path.join(HOME_DIR,'berkeley_2018',CityName,'Output')
    output_file = '0_people{0}to24.csv'.format(start)
    destination_file = os.path.join(saving_dir, f"R_{R}_UCI_{round(UCI,3)}_{output_file}")
    if os.path.exists(destination_file):
        print(f"File {destination_file} already exists")
        return True
    else:
        logger.info(f"Launch Simulation")
        # NOTE: No Further Locks in the Continuing of this Tree of Computation since we lock here.
        # In particular not in DockerCommand and SaveSharedDict
        lock.acquire()
        ModifyConfigIni(CityName,start,end,R,UCI, True)
        pid = os.getpid()
        succesful = DockerCommand(shared_dict,save_dir,pid,R,UCI,CityName)   
        lock.release(block=True)
        if succesful:
            output_files = ['0_allPathsinEdgesCUDAFormat{0}to24.csv'.format(start),
                            '0_indexPathInit{0}to24.csv'.format(start),
                            '0_people{0}to24.csv'.format(start),
                            '0_route{0}to24.csv'.format(start)
                            ]
            for output_file in output_files:
                RenameMoveOutput(output_file,CityName,R,UCI,True)
            return True
        else:
            return False
def RsUCIsFromDir(OD_dir):
    Rs = []
    UCIs = []
    for file in os.listdir(OD_dir):
        if 'od' in file:
            if len(file.split('_')) == 8:
                start = file.split('_')[2]
                end = file.split('_')[3]
                R = file.split('_')[5]
                UCI = file.split('_')[7].split('.csv')[0]
                if R not in Rs:
                    Rs.append(R)
                if float(UCI) not in UCIs:
                    UCIs.append(float(UCI))
    Rs = sorted(Rs)
    UCIs = sorted(UCIs)
    UCisJump = []
    for i in range(len(UCIs)):
        if i == 0:
            UCisJump.append(str(UCIs[i]))
        elif (UCIs[i] - float(UCisJump[-1])) > 0.009:
            UCisJump.append(str(UCIs[i]))
    return Rs,UCisJump
    
def UCIsFromDirRsFromMetaData(OD_dir,CityName,NUMBER_SIMULATIONS,CityName2RminRmax):
    Rs = np.arange(CityName2RminRmax[CityName][0],CityName2RminRmax[CityName][1],(CityName2RminRmax[CityName][1]-CityName2RminRmax[CityName][0])/NUMBER_SIMULATIONS,dtype=int)
    Rs = [str(R) for R in Rs]
    UCIs = []
    for file in os.listdir(OD_dir):
        if 'od' in file:
            if len(file.split('_')) == 8:
                UCI = file.split('_')[7].split('.csv')[0]
                if UCI not in UCIs:
                    UCIs.append(UCI)
    
    UCIs = sorted(UCIs)
    return Rs,UCIs
    
def DeleteInputSimulation(InputFile):
    """
        @params InputFile: File to delete (is the input of simulation that is very big and I do not want to use up all the memory of the PC, since it is retrievable)
        @description : Free space deleting the input file 
    """

    if os.path.exists(InputFile):
        logger.info(f"Deleting {InputFile}")
        os.remove(InputFile)
    else:
        logger.info(f"Cannot delete: {InputFile} : It does not exist")


def monitor_processes(shared_dict,interval=600):
    """
        @param interval: int -> time to check the processes
        @param shared_dict: Manager.dict() -> shared dictionary
        @description: This function monitors the active processes and logs the memory and CPU usage.
    """
    while True:
        logger.info("Monitoring active processes...")
        pids2remove = []
        for pid, info in shared_dict.items():
            try:
                proc = psutil.Process(pid)
                memory_info = proc.memory_info()
                cpu_percent = proc.cpu_percent(interval=1)
                if "UCI" in info:
                    logger.info(f"Process {info['name']} (PID: {pid}): Memory Usage: {memory_info.rss / (1024 * 1024):.2f} MB, CPU Usage: {cpu_percent}%, R: {info['R']}, UCI: {info['UCI']}, City: {info['city']}, Modified Mobility: {info['modified_mobility']}")
                else:
                    logger.info(f"Process {info['name']} (PID: {pid}): Memory Usage: {memory_info.rss / (1024 * 1024):.2f} MB, CPU Usage: {cpu_percent}%, R: {info['R']}, City: {info['city']}, Modified Mobility: {info['modified_mobility']}")
            except psutil.NoSuchProcess:
                logger.warning(f"Process with PID {pid} does not exist (NoSuchProcess).")
                pids2remove.append(pid)
            except psutil.AccessDenied:
                logger.warning(f"Access denied to process with PID {pid} (AccessDenied).")
            except psutil.ZombieProcess:
                logger.warning(f"Process with PID {pid} is a zombie process (ZombieProcess).")
        for pid in pids2remove:
            del shared_dict[pid]        
        time.sleep(interval)




def convert_numpy_int64(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()    
    elif isinstance(obj, np.int32):
        return int(obj)
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_shared_dict(shared_dict, filename):
    with open(filename, 'w') as f:
        json.dump(dict(shared_dict), f,default=convert_numpy_int64,indent=4)

def free_gpu_memory(gpu_id):
    try:
        result = subprocess.run(['nvidia-smi', '--gpu-reset', '-i', str(gpu_id)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            logger.info(f"GPU memory on GPU {gpu_id} has been freed.")
        else:
            logger.warning(f"Failed to free GPU memory on GPU {gpu_id}: {result.stderr.decode()}")
    except FileNotFoundError:
        logger.warning("nvidia-smi command not found. Ensure that NVIDIA drivers are installed.")

########### Launch Simulation Non-Modified OD ############
def ProcessLauncherNonModified(shared_dict,queue,barrier,lock,GeoInfo,UCI,R,error_flag,R_error):
    """
        @param shared_dict: Manager.dict()
        @param queue: Queue
        @param barrier: Barrier
        @param GeoInfo: GeometricalSettingsSpatialPartition object
        @param R: float
        @description: This function launches:
         - SharedDict2Dict
         - Synchronize
         - ComputeSimulationFileAndLaunchDocker
        the simulation for the non-modified OD.
        It does this by queuing writing of the shared dict: 
        (the information about the processes running that then I check via)
    """
    SharedDict2DictNon(shared_dict,queue,barrier,GeoInfo,R,UCI)
    error_flag,R_error = ComputeSimulationFileAndLaunchDocker(GeoInfo,UCI,R,queue,lock,shared_dict)
    return Succesfull,R
def SharedDict2DictNon(shared_dict,queue,barrier,GeoInfo,R,UCI):
    """
        @param shared_dict: Manager.dict()
        @param queue: Queue
        @param barrier: Barrier
        @param GeoInfo: GeometricalSettingsSpatialPartition object
        @param R: float
        @description: This function writes the shared dictionary to a file
                     avoiding concurrency among different processes
    """
    pid = os.getpid()
    queue.put(pid)
    while queue.get() != pid:
        time.sleep(0.1)
    shared_dict[pid] = {
        'name': 'NonModifiedSimulation',
        'R': R,
        'UCI': UCI,
        'city': GeoInfo.city,
        'modified_mobility': True,
        "failed_simulation": False
    }
    saving_dir = os.path.join(GeoInfo.berkeley_2018,GeoInfo.city,'Output')
    save_shared_dict(shared_dict, os.path.join(saving_dir, 'shared_dict.json'))
    # Synchronize the processes
    barrier.wait()



def ComputeSimulationFileAndLaunchDocker(GeoInfo,UCI,R,queue,lock,shared_dict):
    """
        @param GeoInfo: GeometricalSettingsSpatialPartition object
        @param UCI: float
        @param R: float
        @param shared_dict: Manager.dict()
        @param barrier: Barrier
        @return: bool
        @description: This function computes the simulation file and launches the docker container.
    """
    logger.info(f"Running simulation for R: {R}")
    # Simulate some work
    saving_dir = os.path.join(GeoInfo.berkeley_2018,GeoInfo.city,'Output')
    output_file = '0_people{0}to24.csv'.format(GeoInfo.start)
    check_file = f"R_{R}_UCI_{round(UCI,3)}_{output_file}"
    if not os.path.isfile(os.path.join(saving_dir,check_file)):
        NotModifiedInputFile = GeoInfo.ComputeDf4SimNotChangedMorphology(UCI,R)
        pid = os.getpid()
        queue.put(pid)
        while queue.get() != pid:
            time.sleep(0.1)

        while(True):
            # Tells that the process is occupied
            success = LaunchDockerFromServer(GeoInfo.city,GeoInfo.start,GeoInfo.start + 1,R,UCI,lock,shared_dict,saving_dir)
            # Delete in any case the input file
            if success:
                DeleteInputSimulation(NotModifiedInputFile)
                return True,None
            else:
                return True,R
    else:
        return True



########### Launch Simulation Non-Modified OD ############
    
########### Launch Simulation Modified OD ############

def ProcessLauncherModified(shared_dict,queue,barrier,GeoInfo,Modified_Fluxes,R,UCI1,lock,error_flag,R_error):
    """
        @param shared_dict: Manager.dict()
        @param queue: Queue
        @param barrier: Barrier
        @param GeoInfo: GeometricalSettingsSpatialPartition object
        @param R: float
        @description: This function launches:
         - SharedDict2Dict
         - Synchronize
         - ComputeSimulationFileAndLaunchDocker
        the simulation for the non-modified OD.
        It does this by queuing writing of the shared dict: 
        (the information about the processes running that then I check via)
    """
    SharedDict2Dict(shared_dict,queue,barrier,GeoInfo,R)
    error_flag.value,R_error.value = ComputeSimulationFileAndLaunchDockerFromModifiedMob(GeoInfo,Modified_Fluxes,R,UCI1,queue,lock,shared_dict)

def SharedDict2Dict(shared_dict,queue,barrier,GeoInfo,R):
    """
        @param shared_dict: Manager.dict()
        @param queue: Queue
        @param barrier: Barrier
        @param GeoInfo: GeometricalSettingsSpatialPartition object
        @param R: float
        @description: This function writes the shared dictionary to a file
                     avoiding concurrency among different processes
    """
    pid = os.getpid()
    queue.put(pid)
    while queue.get() != pid:
        time.sleep(0.1)
    shared_dict[pid] = {
        'name': 'ComputeSimulationFileAndLaunchDockerFromModifiedMob',
        'R': R,
        'city': GeoInfo.city,
        'modified_mobility': True,
        "failed_simulation": False
    }
    saving_dir = os.path.join(GeoInfo.berkeley_2018,GeoInfo.city,'Output')
    save_shared_dict(shared_dict, os.path.join(saving_dir, 'shared_dict.json'))
    # Synchronize the processes
    barrier.wait()



def ComputeSimulationFileAndLaunchDockerFromModifiedMob(GeoInfo,Modified_Fluxes,R,UCI1,queue,lock,shared_dict):
    """
        @param GeoInfo: GeometricalSettingsSpatialPartition object
        @param R: float
        @param shared_dict: Manager.dict()
        @param barrier: Barrier
        @return: bool
        @description: This function computes the simulation file and launches the docker container.
    """
    logger.info(f"Running modified simulation for R: {R}")
    # Simulate some work
#    barrier.wait()  # Wait for all processes to reach this point
#    barrier.wait()
    saving_dir = os.path.join(GeoInfo.berkeley_2018,GeoInfo.city,'Output')
    output_file = '0_people{0}to24.csv'.format(GeoInfo.start)
    check_file = f"R_{R}_UCI_{round(UCI1,3)}_{output_file}"
#    save_shared_dict(shared_dict, os.path.join(saving_dir, 'shared_dict.json'))
    if not os.path.isfile(os.path.join(saving_dir,check_file)):

        ModifiedInputFile = GeoInfo.ComputeDf4SimChangedMorphology(UCI1,R,Modified_Fluxes)
        start = GeoInfo.start
        end = start + 1
        if os.path.isfile(ModifiedInputFile):
            logger.info(f"Launching docker, {GeoInfo.city}, R: {R}, UCI: {round(UCI1,3)}")
            pid = os.getpid()
            queue.put(pid)
            while queue.get() != pid:
                time.sleep(0.1)
            while(True):                
                succesful = LaunchDockerFromServer(GeoInfo.city,start,end,R,UCI1,lock,shared_dict,saving_dir)
                if succesful:
                    DeleteInputSimulation(ModifiedInputFile)
                    return True,None
                else:
                    return True,R
    else:
        logger.info(f"Simulation {check_file} already exists.")
        return True
    # Launch the processes one after the other since you
def get_pid(x):
    return os.getpid()

def queue_listener(queue,shared_dict):
    while True:
        result = queue.get()
        if result is None:
            break
        shared_dict[result['pid']] = result
        


# ERROR GPU
def InfoAboutFailureInSharedDict(shared_dict,save_dir,UCI,R,CityName,successful):
    """
        @param shared_dict: Manager.dict()
        @param queue: Queue
        @param barrier: Barrier
        @param GeoInfo: GeometricalSettingsSpatialPartition object
    """
    pid = os.getpid()
    shared_dict[pid] = {
        'name': 'NonModifiedSimulation',
        'R': R,
        'UCI': UCI,
        'city': CityName,
        'modified_mobility': True,
        "failed_simulation": successful
    }
    saving_dir = os.path.join(save_dir,CityName,'Output')
    save_shared_dict(shared_dict, os.path.join(saving_dir, 'shared_dict.json'))
    # Synchronize the processes



def ProcessMain(case,
                GeoInfo,
                Modified_Fluxes,
                R,
                UCI1,
                concurrent_manager):
    """
        @param case: str -> 'NonModified' or 'Modified'
        @param GeoInfo: GeometricalSettingsSpatialPartition object
        @param Modified_Fluxes: pd.DataFrame
        @param R: float
        @param UCI1: float
        ---- SYNCRONIZATION ----
        @param shared_dict: Manager.dict()
        @param queue: Queue
        @param barrier: Barrier
        @param lock: Lock
        @param error_flag: Value
        @param R_error: Value
    """
    os.chdir(os.getcwd())
    ### ENVIRONMENT VARIABLES ###
    os.environ['PATH'] += os.pathsep + '/usr/bin/'
    container_name = "xuanjiang1998/lpsim:v1"
    PWD = '/home/alberto/LPSim'
    env = os.environ.copy()
    docker_cmd = ['/usr/bin/docker', 'run', '-it', '--rm', '--gpus', 'all', '-v', f'{PWD}:/lpsim', '-w', '/lpsim', f'{container_name}', 'bash', '-c', './LivingCity/LivingCity']

    ##### FILE NAMES ##### 
    # /home/alberto/LPSim/LivingCity/berkeley_2018/boston/Output
    dir_new_full_network = os.path.join(GeoInfo.berkeley_2018,'new_full_network')
    saving_dir = os.path.join(GeoInfo.berkeley_2018,GeoInfo.city,'Output')
    output_file = '0_people{0}to24.csv'.format(GeoInfo.start)
    check_file = f"R_{R}_UCI_{round(UCI1,3)}_{output_file}"
    destination_file = os.path.join(saving_dir, f"R_{R}_UCI_{round(UCI1,3)}_{output_file}")
    InputFile = f"{GeoInfo.city}_oddemand_{GeoInfo.start}_{GeoInfo.start + 1}_R_{R}_UCI_{round(UCI1,3)}.csv"
    # Get the Information about the Process 
    pid = os.getpid()
    # Put in the Queue the Process
    concurrent_manager.queue.put(pid)
    # Align the Processes to Write in the Shared Dictionary
    while concurrent_manager.queue.get() != pid:
        time.sleep(0.1)
    concurrent_manager.shared_dict[pid] = {
        'name': case,
        'R': R,
        'UCI': round(UCI1,3),
        'city': GeoInfo.city,
        'GPU_occupied': False,
        "GPU_error": False,
        "docker_error": False,
        "failed_simulation": False
    }
    save_shared_dict(concurrent_manager.shared_dict, os.path.join(saving_dir, 'shared_dict.json'))
    # Synchronize the processes
    concurrent_manager.barrier.wait()
    InputDocker = os.path.join(dir_new_full_network,InputFile)
    # Compute The Docker Only If I Do Not Have Already Computed It and The OutputFile does not Exist
    logger.info(f"InputDocker: {InputDocker} {os.path.isfile(InputDocker)}")
    logger.info(f"OutputFile: {destination_file} {os.path.isfile(destination_file)}")
    ComputedDockerInput = os.path.isfile(InputDocker) # True If The Input is There
    ComputedOutput = os.path.isfile(destination_file) # True If The Output is There
    # Check If The Simulation is Already There (DO NOT REPEAT IT!)
    if ComputedOutput:
        logger.info(f"Simulation {check_file} already exists.")
        SuccessSimulation = True
        return SuccessSimulation
    # The Simulation Output is Not There
    else:
        # Check InputDocker (DO NOT REPEAT IT!)
        if not ComputedDockerInput:
            # Not Modified Mobility
            if case == 'NonModified':
                logger.info(f"ID-Non-Mod R: {R}, {GeoInfo.city}, UCI: {round(UCI1,3)}")
                InputDocker = GeoInfo.ComputeDf4SimNotChangedMorphology(UCI1,R)
            # Modified Mobility
            else:
                logger.info(f"ID-Mod R: {R}, {GeoInfo.city}, UCI: {round(UCI1,3)}")
                if not ComputedDockerInput:
                    InputDocker = GeoInfo.ComputeDf4SimChangedMorphology(UCI1,R,Modified_Fluxes)
        else:        
            logger.info(f"InputDocker {InputDocker} already exists.")
        # Launch Docker Process
        start = GeoInfo.start
        end = start + 1
        logger.info(f"Launching docker, {GeoInfo.city}, R: {R}, UCI: {round(UCI1,3)}")
        concurrent_manager.queue.put(pid)
        while concurrent_manager.queue.get() != pid:
            time.sleep(0.1)
        # Keep Launching Docker
        while(True):
            logger.info(f"Launch Simulation {check_file}")
            # Check if the GPU is available and Bound The Lock To The GPU Availability
            # NOTE: This Will Be False 
            concurrent_manager.GPULock.acquire(block = True)
            gpu_id = check_first_gpu_available()
            concurrent_manager.GPULock.release()            
            GpuAvailable = gpu_id>=0            
            # Block All The Processes Until The GPU is Available (When Simulation Is Returned)
            concurrent_manager.locks[gpu_id].acquire(block = True)   
            # Tell That You Are Occupying The GPU
            concurrent_manager.shared_dict[pid]['GPU_occupied'] = True    
            save_shared_dict(concurrent_manager.shared_dict, os.path.join(saving_dir, 'shared_dict.json'))
            if GpuAvailable:
                # Change The Configuration File Just One GPU at A Time
                concurrent_manager.lock_Config_ini.acquire(block = True)
                ModifyConfigIni(GeoInfo.city,start,end,R,round(UCI1,3), True)
                logger.info(f"pid {pid}: {R}, {round(UCI1,3)} Free GPU {gpu_id}")
                free_gpu_memory(gpu_id)            
                logger.info(f"Launch Simulation {R}, {round(UCI1,3)} in GPU {gpu_id}")
                process = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
                stdout, stderr = process.communicate()
                concurrent_manager.lock_Config_ini.release()
                # Log the output
#                logger.info(stdout.decode())
#                logger.error(stderr.decode())
                # Check if the command was successful
                ErrorDocker = (process.returncode == 0)
                # Check Errors in GPU
                Bit1,Bit2 = check_gpu_errors()
                ErrorGPU = Bit1 or Bit2
                # Save The Shared Dict With Information About The Process            
                concurrent_manager.shared_dict[pid]["GPU_error"] = ErrorGPU
                concurrent_manager.shared_dict[pid]["docker_error"] = ErrorDocker
                FailedSimulation = ErrorGPU or ErrorDocker
                concurrent_manager.shared_dict[pid]["failed_simulation"] = FailedSimulation
                save_shared_dict(concurrent_manager.shared_dict, os.path.join(saving_dir, 'shared_dict.json'))
                # Bring Information Error Outside the Process
                # NOTE: We are in a lock, so we do not give a shit about other processes
                concurrent_manager.GPU_errors[gpu_id].value = ErrorGPU 
                concurrent_manager.R_errors[gpu_id].value = R
                concurrent_manager.Docker_error.value = ErrorDocker
                # If Failed Simulation 
                if FailedSimulation:
                    concurrent_manager.GPU_error_index.value = gpu_id
                else:
                    pass
                free_gpu_memory(gpu_id)
                concurrent_manager.locks[gpu_id].release()  
                # OutputFile Is There, we can move it
                if not FailedSimulation:
                    output_files = ['0_allPathsinEdgesCUDAFormat{0}to24.csv'.format(start),
                                    '0_indexPathInit{0}to24.csv'.format(start),
                                    '0_people{0}to24.csv'.format(start),
                                    '0_route{0}to24.csv'.format(start)
                                    ]
                    for output_file in output_files:
                        RenameMoveOutput(output_file,GeoInfo.city,R,UCI1,True)
                return not FailedSimulation

#def LaunchProgram(config):
#    subprocess.run(['your_program_executable', config_file])
'''
[General]
GUI=false
USE_CPU=false
NETWORK_PATH=berkeley_2018/boston/
USE_JOHNSON_ROUTING=false
USE_SP_ROUTING=true
USE_PREV_PATHS=false
LIMIT_NUM_PEOPLE=256000
ADD_RANDOM_PEOPLE=false
NUM_PASSES=1
TIME_STEP=1#0.5
START_HR=7
END_HR=24
OD_DEMAND_FILENAME=od_demand_7to8_R_1.csv
SHOW_BENCHMARKS=false
REROUTE_INCREMENT=0
NUM_GPUS=1
'''
