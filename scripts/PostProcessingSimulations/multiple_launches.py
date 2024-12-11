import subprocess
import os
import argparse
import json
import time
import shutil
import psutil
import platform
import numpy as np
from multiprocessing import Pool,log_to_stderr,get_logger
import sys
import logging
from pynvml import *
from GPUHandler import *
from OsFunctions import *
logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.environ["TRAFFIC_DIR"],"scripts","GeometrySphere"))
#from ODfromfma import NUMBER_SIMULATIONS,CityName2RminRmax

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


def check_out_of_memory_error(stderr_output):
    out_of_memory_errors = [
        "out of memory",
        "CUDA_ERROR_OUT_OF_MEMORY",
        "failed to allocate memory",
        "memory allocation error"
        "GPUAssert"
    ]
    
    for error_message in out_of_memory_errors:
        if error_message.lower() in stderr_output.lower():
            return True
    return False


def check_transportation_error(stderr_output):
    """
        @param stderr_output: str -> stderr output from the process
        @return TransportError: bool -> True if a transportation error occurred, False
    """
    transport_errors = [
        "transport: Error while dialing",
        "connection error",
        "timeout"
    ]

    for error_message in transport_errors:
        if error_message.lower() in stderr_output.lower():
            TransportError = True
            return TransportError
    TransportError = False
    return TransportError
    
def CheckBrockenPipeError(stderr_output):
    """
        @param stderr_output: str -> stderr output from the process
        @return TransportError: bool -> True if a transportation error occurred, False
    """
    transport_errors = [
        "BrokenPipeError"
    ]

    for error_message in transport_errors:
        if error_message.lower() in stderr_output.lower():
            TransportError = True
            return TransportError
    TransportError = False
    return TransportError

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







def free_gpu_memory(gpu_id):
    try:
        result = subprocess.run(['nvidia-smi', '--gpu-reset', '-i', str(gpu_id)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            logger.info(f"GPU memory on GPU {gpu_id} has been freed.")
        else:
            logger.warning(f"Failed to free GPU memory on GPU {gpu_id}: {result.stderr.decode()}")
    except FileNotFoundError:
        logger.warning("nvidia-smi command not found. Ensure that NVIDIA drivers are installed.")




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
    try:
        os.chdir(os.getcwd())
        ### ENVIRONMENT VARIABLES ###
        os.environ['PATH'] += os.pathsep + '/usr/bin/'
        container_name = "xuanjiang1998/lpsim:v1"
        PWD = '/home/alberto/LPSim'
        env = os.environ.copy()
        docker_cmd = ['/usr/bin/docker', 'run', '-it', '--rm', '--gpus', 'all', '-v', f'{PWD}:/lpsim', '-w', '/lpsim', f'{container_name}', 'bash', '-c', './LivingCity/LivingCity']

        ##### FILE NAMES #####  /home/alberto/LPSim/LivingCity/berkeley_2018/boston/Output
        dir_new_full_network = os.path.join(GeoInfo.berkeley_2018,'new_full_network')
        saving_dir = os.path.join(GeoInfo.berkeley_2018,GeoInfo.city,'Output')
        output_file = '0_people{0}to24.csv'.format(GeoInfo.start)
        output_file_parquet = output_file.replace('.csv','.parquet')
        check_file = f"R_{R}_UCI_{round(UCI1,3)}_{output_file}"
        destination_file = os.path.join(saving_dir, f"R_{R}_UCI_{round(UCI1,3)}_{output_file}")
        InputFile = f"{GeoInfo.city}_oddemand_{GeoInfo.start}_{GeoInfo.start + 1}_R_{R}_UCI_{round(UCI1,3)}.csv"
        ##### PID ##### 
        pid = os.getpid()
        #### INIT SHARED DICT #####
        concurrent_manager.InitSharedDict(pid,case,R,UCI1,GeoInfo.city,f"{R}, {round(UCI1,3)}, {GeoInfo.city}: Init Dict")
        # BRANCH VARIABLES
        InputDocker = os.path.join(dir_new_full_network,InputFile)
        ComputedDockerInput = os.path.isfile(InputDocker) # Input Exists: True -> SKIP COMPUTATION 
        ComputedOutput = os.path.isfile(destination_file) or os.path.isfile(os.path.join(saving_dir, f"R_{R}_UCI_{round(UCI1,3)}_{output_file_parquet}"))# Output Exists: True -> NOT LAUNCH DOCKER: SuccessSimulation = True
        # Compute The Docker Only If I Do Not Have Already Computed It and The OutputFile does not Exist
    #    Messages = [f"InputDocker: {InputDocker} {os.path.isfile(InputDocker)}",f"OutputFile: {destination_file} {os.path.isfile(destination_file)}"]
    #    concurrent_manager.LogMessage(Messages)
        # Check If The Simulation is Already There (DO NOT REPEAT IT!)
        if ComputedOutput:
            Messages = [f"Simulation {check_file} already exists."]
            concurrent_manager.LogMessage(Messages)
            SuccessSimulation = True
            return SuccessSimulation
        # The Simulation Output is Not There
        else:
            # Check InputDocker (DO NOT REPEAT IT!)
            if not ComputedDockerInput:
                # Not Modified Mobility
                if case == 'NonModified':
                    Messages = [f"ID-Non-Mod R: {R}, {GeoInfo.city}, UCI: {round(UCI1,3)}"]
                    concurrent_manager.LogMessage(Messages)
                    InputDocker = GeoInfo.ComputeDf4SimNotChangedMorphology(UCI1,R)
                # Modified Mobility
                else:
                    Messages = [f"ID-Mod R: {R}, {GeoInfo.city}, UCI: {round(UCI1,3)}"]
                    concurrent_manager.LogMessage(Messages)
                    if not ComputedDockerInput:
                        InputDocker = GeoInfo.ComputeDf4SimChangedMorphology(UCI1,R,Modified_Fluxes)
            else:
                Messages = [f"InputDocker {InputDocker} already exists."]
                concurrent_manager.LogMessage(Messages)
            # Launch Docker Process
            start = GeoInfo.start
            end = start + 1
            # Keep Launching Docker
            while(True):
                # NOTE: This Will Be False 
                concurrent_manager.AcquireLockByName("GPULock", gpu_id = None,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city}: Check GPU Availability")
                try:
                    gpu_id = check_first_gpu_available()
                except Exception as e:
                    concurrent_manager.LogMessage([f"Error check_first_gpu_available: {e}"])
                    gpu_id = -1
                concurrent_manager.ReleaseLockByName("GPULock", gpu_id = None,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city}: Available: {gpu_id}")
                GpuAvailable = gpu_id>=0            
                # Block All The Processes Until The GPU is Available (When Simulation Is Returned)
                # Tell That You Are Occupying The GPU
                try:
                    concurrent_manager.UpdateSharedDict(pid = pid,key = 'GPU_occupied',value = True,Message = f"{R}, {round(UCI1,3)}: Update ShareDic GPU {gpu_id} Occupied: True")
                except Exception as e:
                    concurrent_manager.LogMessage([f"Error UpdateSharedDict: {e}"])

                if GpuAvailable:
                    # Change The Configuration File Just One GPU at A Time
                    try:
                        concurrent_manager.AcquireLockByName("ConfigIniLock", gpu_id = None,Message = f"{R}, {round(UCI1,3)}: Acquire Config_ini Lock")
                        # Copy Cartography
                        CopyCartoGraphy("edges.csv",GeoInfo.city)
                        CopyCartoGraphy("nodes.csv",GeoInfo.city)
                        # Modify Config.ini 
                        ModifyConfigIni(GeoInfo.city,start,end,R,round(UCI1,3), True)
                        concurrent_manager.LogMessage(Messages)
                    except Exception as e:
                        concurrent_manager.LogMessage([f"Error ModifyConfigIni: {e}"])
                    # TransportError, BrockenPipe: 0,0 -> 1 (Continue)  
                    Messages = [f"pid {pid}: {R}, {round(UCI1,3)} Free GPU {gpu_id}",f"Launch Simulation {R}, {round(UCI1,3)} in GPU {gpu_id}"]
                    concurrent_manager.LogMessage(Messages)
                    # Check Docker: If Transport Error -> Stop (We now have that the Process Could Not Continue)
                    try:
                        process = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
                    except Exception as e:
                        concurrent_manager.LogMessage([f"Error check_transportation_error: {e}"])
                        concurrent_manager.ModifyVariablesByNames(name_variable = "RError",pid = None,case = None,R = R,UCI1 = UCI1,city = GeoInfo.city,gpu_id = gpu_id,DockerError = False,GPUError = False,FirstTry = False,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city} RError.value")
#                                concurrent_manager.UpdateRError(gpu_id,R,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city} RError.value")
                        TransportError = True
                        break
                    try:
                        stdout, stderr = process.communicate()
                        BrockenPipe = CheckBrockenPipeError(stderr.decode())
                    except Exception as e:
                        concurrent_manager.LogMessage([f"Error process.communicate: {e}"])
                        try:
                            concurrent_manager.ModifyVariablesByNames(name_variable = "RError",pid = None,case = None,R = R,UCI1 = UCI1,city = GeoInfo.city,gpu_id = gpu_id,DockerError = ErrorDocker,GPUError = False,FirstTry = False,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city} RError.value: {concurrent_manager.RError.value}")
                        except Exception as e:
                            concurrent_manager.LogMessage([f"Error ModifyVariablesByNames: {e}"])
#                                concurrent_manager.UpdateRError(gpu_id,R,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city} RError.value")
                        TransportError = True
                        break
                    try:
                        TransportError = check_transportation_error(stderr.decode())
                    except Exception as e:
                        concurrent_manager.LogMessage([f"Error check_transportation_error: {e}"])
                        TransportError = True
                        break
                    concurrent_manager.ReleaseLockByName("ConfigIniLock", gpu_id = None,Message = f"{R}, {round(UCI1,3)}: Ended Simulation: Release Config_ini Lock")
                    # Check if the command was successful
                    ErrorDocker = (process.returncode != 0)
                    # Check Errors in GPU
                    concurrent_manager.LogMessage([f"Decode/Encode Message: {R}, {round(UCI1,3)}",stdout.decode(),stderr.decode(),f"Error Docker: {ErrorDocker}"])
                    concurrent_manager.AcquireLockByName("GPULock", gpu_id = None,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city}: Check GPU Errors")
                    ErrorGPU = check_out_of_memory_error(stderr.decode())                
                    concurrent_manager.ReleaseLockByName("GPULock", gpu_id = None,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city}: Release Check GPU Errors")                
                    concurrent_manager.LogMessage([f"{R}, {round(UCI1,3)},, {GeoInfo.city} Error GPU: {ErrorGPU}"])
                    # Save The Shared Dict With Information About The Process 
                    concurrent_manager.UpdateSharedDict(pid = pid, key = 'GPU_error',value = ErrorGPU,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city}: Update GPU Error: {ErrorGPU}")           
                    concurrent_manager.UpdateSharedDict(pid = pid, key = 'docker_error',value = ErrorDocker,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city}: Update Docker Error: {ErrorDocker}")
                    FailedSimulation = ErrorGPU or ErrorDocker
                    concurrent_manager.UpdateSharedDict(pid = pid, key = 'failed_simulation',value = FailedSimulation,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city}: Update Failed Simulation: {FailedSimulation}")
                    # NOTE: We are in a lock, so we do not give a shit about other processes                
                    concurrent_manager.ModifyVariablesByNames(name_variable = "GPUError",pid = None,case = None,R = R,UCI1 = UCI1,city = GeoInfo.city,gpu_id = gpu_id,DockerError = False,GPUError = ErrorGPU,FirstTry = False,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city} ErrorGPU.value = {ErrorGPU}")
#                    concurrent_manager.UpdateGPUError(gpu_id, ErrorGPU,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city} ErrorGPU.value = {ErrorGPU}")
                    concurrent_manager.ModifyVariablesByNames(name_variable = "RError",pid = None,case = None,R = R,UCI1 = UCI1,city = GeoInfo.city,gpu_id = gpu_id,DockerError = False,GPUError = False,FirstTry = False,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city} RError.value")
#                    concurrent_manager.UpdateRError(gpu_id,R,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city} RError.value")
                    concurrent_manager.ModifyVariablesByNames(name_variable = "DockerError",pid = None,case = None,R = R,UCI1 = UCI1,city = GeoInfo.city,gpu_id = gpu_id,DockerError = ErrorDocker,GPUError = False,FirstTry = False,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city} Docker_error.value")
#                    concurrent_manager.UpdateDockerError(ErrorDocker,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city} Docker_error.value")
                    concurrent_manager.LogMessage([f"Shared: GPU Error, {concurrent_manager.GPU_errors[gpu_id].value}, R: {concurrent_manager.R_errors[gpu_id].value}, Docker: {concurrent_manager.Docker_error.value}"])
                    # If Failed Simulation GPUErrorIndex
                    if FailedSimulation:
                        concurrent_manager.ModifyVariablesByNames(name_variable = "DockerError",pid = None,case = None,R = R,UCI1 = UCI1,city = GeoInfo.city,gpu_id = gpu_id,DockerError = ErrorDocker,GPUError = False,FirstTry = False,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city} GPUError_index.value: {concurrent_manager.GPU_error_index.value}")
                        concurrent_manager.UpdateGPUErrorIndex(gpu_id,Message = f"{R}, {round(UCI1,3)}, {GeoInfo.city} GPUError_index.value: {concurrent_manager.GPU_error_index.value}")                
                    else:
                        pass
    #               concurrent_manager.AcquireLockByName("GPULock",Message = f"{R}, {round(UCI1,3)} Free Memory")
    #                free_gpu_memory(gpu_id)
    #                concurrent_manager.ReleaseLockByName("GPULock",Message = f"{R}, {round(UCI1,3)} Free Memory")
                    # OutputFile Is There, we can move it
                    if not FailedSimulation:
                        output_files = [
                                        '0_indexPathInit{0}to24.csv'.format(start),
                                        '0_people{0}to24.csv'.format(start),
                                        '0_route{0}to24.csv'.format(start),
                                        '0_allPathsinEdgesCUDAFormat{0}to24.csv'.format(start)
                                        ]
                        for output_file in output_files:
                            RenameMoveOutput(output_file,GeoInfo.city,R,UCI1,True)
                            DeleteFile(InputDocker)
                    return not FailedSimulation
    except Exception as e:
        concurrent_manager.LogMessage([f"Error: {e}"])
        return False

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
