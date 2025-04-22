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

def ModifyConfigIni(start,name_od_file_input_simulation):
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
    logger.info(f"Input File: {name_od_file_input_simulation}")
#    file_txt = '[General]\nGUI=false\nUSE_CPU=false\nNETWORK_PATH=LivingCity/berkeley_2018/new_full_network/\nUSE_JOHNSON_ROUTING=false\nUSE_SP_ROUTING=true\nUSE_PREV_PATHS=false\nLIMIT_NUM_PEOPLE=256000\nADD_RANDOM_PEOPLE=false\nNUM_PASSES=1\nTIME_STEP=1\nSTART_HR={0}\nEND_HR=24\nOD_DEMAND_FILENAME={1}_oddemand_{2}_{3}_R_{4}_UCI_{5}.csv\nSHOW_BENCHMARKS=false\nREROUTE_INCREMENT=0\nPARTITION_FILENAME=ciccio.txt\nNUM_GPUS=1\n'.format(start,CityName,start,end,R,str_UCI)
    file_txt = f"[General]\nGUI=false\nUSE_CPU=false\nNETWORK_PATH=LivingCity/berkeley_2018/new_full_network/\nUSE_JOHNSON_ROUTING=false\nUSE_SP_ROUTING=true\nUSE_PREV_PATHS=false\nLIMIT_NUM_PEOPLE=256000\nADD_RANDOM_PEOPLE=false\nNUM_PASSES=1\nTIME_STEP=1\nSTART_HR={start}\nEND_HR=24\nOD_DEMAND_FILENAME={name_od_file_input_simulation}\nSHOW_BENCHMARKS=false\nREROUTE_INCREMENT=0\nPARTITION_FILENAME=ciccio.txt\nNUM_GPUS=1\n"
    with open(os.path.join(TRAFFIC_DIR,'command_line_options.ini'),'w') as file:
        file.write(file_txt)
    file.close()
#        print(file_txt)    
def _ModifyConfigIni(CityName,start,end,R,str_UCI,verbose = False):
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
    logger.info("Input File: {0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv".format(CityName,start,end,R,str_UCI))
#    file_txt = '[General]\nGUI=false\nUSE_CPU=false\nNETWORK_PATH=LivingCity/berkeley_2018/new_full_network/\nUSE_JOHNSON_ROUTING=false\nUSE_SP_ROUTING=true\nUSE_PREV_PATHS=false\nLIMIT_NUM_PEOPLE=256000\nADD_RANDOM_PEOPLE=false\nNUM_PASSES=1\nTIME_STEP=1\nSTART_HR={0}\nEND_HR=24\nOD_DEMAND_FILENAME={1}_oddemand_{2}_{3}_R_{4}_UCI_{5}.csv\nSHOW_BENCHMARKS=false\nREROUTE_INCREMENT=0\nPARTITION_FILENAME=ciccio.txt\nNUM_GPUS=1\n'.format(start,CityName,start,end,R,str_UCI)
    file_txt = '[General]\nGUI=false\nUSE_CPU=false\nNETWORK_PATH=LivingCity/berkeley_2018/new_full_network/\nUSE_JOHNSON_ROUTING=false\nUSE_SP_ROUTING=true\nUSE_PREV_PATHS=false\nLIMIT_NUM_PEOPLE=256000\nADD_RANDOM_PEOPLE=false\nNUM_PASSES=1\nTIME_STEP=1\nSTART_HR={0}\nEND_HR=24\nOD_DEMAND_FILENAME={2}\nSHOW_BENCHMARKS=false\nREROUTE_INCREMENT=0\nPARTITION_FILENAME=ciccio.txt\nNUM_GPUS=1\n'.format(start,end,name_od_file_input_simulation)
    with open(os.path.join(TRAFFIC_DIR,'command_line_options.ini'),'w') as file:
        file.write(file_txt)
    file.close()



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



# Check Wether The OD Demand File Exists
def CheckODDemandFile(
        complete_path_od_demand_file,
        case,
        GeoInfo,
        UCI1,
        R,
        Modified_Fluxes        
        ):
    """
        @param output_file_analysis: str -> Output file analysis 
            NOTE: This is the file that has been copied in dir_output_simulation_analysis
        @param case: str -> 'NonModified' or 'Modified'
            NOTE: Applies a different strategy to compute the InputDocker
        @param GeoInfo: GeometricalSettingsSpatialPartition object
        @param UCI1: float -> UCI
        @param R: float -> R
        @param Modified_Fluxes: pd.DataFrame -> Modified Fluxes

    """
    is_computed_od_simulation_input = os.path.isfile(complete_path_od_demand_file)
    # Check InputDocker (DO NOT REPEAT IT!)
    if not is_computed_od_simulation_input:
        # Not Modified Mobility
        if case == 'NonModified':
            logger.info(f"ID-Non-Mod R: {R}, {GeoInfo.city}, UCI: {round(UCI1,3)}")
            InputDocker = GeoInfo.ComputeDf4SimNotChangedMorphology(UCI1,
                                                                    R,
                                                                    complete_path_od_demand_file)
        # Modified Mobility
        else:
            logger.info(f"ID-Mod R: {R}, {GeoInfo.city}, UCI: {round(UCI1,3)}")
            if not is_computed_od_simulation_input:
                InputDocker = GeoInfo.ComputeDf4SimChangedMorphology(UCI1,
                                                                     R,
                                                                     Modified_Fluxes,
                                                                     complete_path_od_demand_file)
    else:
        logger.info(f"InputDocker {InputDocker} already exists.")
    return InputDocker





def LaunchSimulation(GeoInfo,
                R,
                str_UCI,
                Name_parameters_output,
                update_dict_df):
    """
        @param case: str -> 'NonModified' or 'Modified'
        @param GeoInfo: GeometricalSettingsSpatialPartition object
        @param Modified_Fluxes: pd.DataFrame
        @param R: float
        @param UCI1: float
        @description:

    """
    assert "complete_path_output_simulation_copied" in update_dict_df.keys(),f"KeyError: complete_path_output_simulation_copied not in update_dict_df"
    assert "complete_path_od_file_input_simulation" in update_dict_df.keys(),f"KeyError: complete_path_od_file_input_simulation not in update_dict_df"
    assert "name_od_file_input_simulation" in update_dict_df.keys(),f"KeyError: name_od_file_input_simulation not in update_dict_df"
    # 1) NOTE: SET CONSTANTS LAUNCH 1 SIMULATION
    os.environ['PATH'] += os.pathsep + '/usr/bin/'
    container_name = "xuanjiang1998/lpsim:v1"
    PWD = os.environ["LPSim"]
    env = os.environ.copy()
    docker_cmd = ['/usr/bin/docker', 'run', '-it', '--rm', '--gpus', 'all', '-v', f'{PWD}:/lpsim', '-w', '/lpsim', f'{container_name}', 'bash', '-c', './LivingCity/LivingCity']  
    start = GeoInfo.start
    end = start + 1
    # 2) NOTE: Prepare Config.ini
    try:
        CopyCartoGraphy("edges.csv",GeoInfo.city)                       # Copy Cartography
        CopyCartoGraphy("nodes.csv",GeoInfo.city)                       # Copy Cartography
        ModifyConfigIni(start,update_dict_df["name_od_file_input_simulation"])   # Modify Config.ini 
    except Exception as e:
        logger.info(f"Error ModifyConfigIni: {e}")
    logger.info(f"{GeoInfo.city}: Launch Simulation {R}, {str_UCI}")
    # Check Docker: If Transport Error -> Stop (We now have that the Process Could Not Continue)
    try:
        process = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    except Exception as e:
        logger.info(f"Error check_transportation_error: {e}")
    try:
        stdout, stderr = process.communicate()
        CheckBrockenPipeError(stderr.decode())
    except Exception as e:
        logger.info(f"Error process.communicate: {e}")
    try:
        check_transportation_error(stderr.decode())
    except Exception as e:
        logger.info(f"Error check_transportation_error: {e}")
    # Check if the command was successful
    ErrorDocker = (process.returncode != 0)
    #
    ErrorGPU = check_out_of_memory_error(stderr.decode())                                                                                                                      # NOTE: Check Errors in GPU             
    logger.info(f"{GeoInfo.city}: {R}, {str_UCI}, Error GPU: {ErrorGPU}")
    logger.info(f"{GeoInfo.city}: {R}, {str_UCI}, Error Docker: {ErrorDocker}")
    FailedSimulation = ErrorGPU or ErrorDocker                                                                                                                                  # NOTE: Condition Failure Run -> ErrorGPU or ErrorDocker
    if not FailedSimulation:                                                                                                                                                    # NOTE: Move the output files just if the simulations is ended well -> as there is just one output dir taking away this would be a problem, risking to move output that are already there for some random reason and copy them into the analysis of another (R,UCI,city)
        # 3) NOTE: Move the output files
        output_files = [
                        '0_indexPathInit{0}to24.csv'.format(start),
                        '0_people{0}to24.csv'.format(start),
                        '0_route{0}to24.csv'.format(start),
                        '0_allPathsinEdgesCUDAFormat{0}to24.csv'.format(start)
                        ]
        for output_file in output_files:
            source_file = os.path.join(update_dict_df["dir_output_simulation_gpu"], output_file)                                                                                # NOTE: Pick the file from the dir_output_gpu 
            #
            destination_file = os.path.join(update_dict_df["dir_output_simulation_analysis"], f"{Name_parameters_output}_{output_file}")                                        # NOTE: Transfer it into the dir_analysis 
            destination_file_parquet = os.path.join(update_dict_df["dir_output_simulation_analysis"], f"{Name_parameters_output}_{output_file.replace('.csv', '.parquet')}")    # NOTE: Rename the .parquet file 
            
            RenameMoveOutput(source_file = source_file,
                             destination_file = destination_file,
                             destination_file_parquet = destination_file_parquet)                                                                                                                          # NOTE: Rename and Move the output file
            DeleteFile(update_dict_df["complete_path_od_file_input_simulation"])                                                                                                # NOTE: Delete the input file (od_demand file)
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
