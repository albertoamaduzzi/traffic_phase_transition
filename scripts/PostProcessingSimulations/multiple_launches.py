import subprocess
import os
import argparse
import json
import time
import shutil
import platform

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
    file_txt = '[General]\nGUI=false\nUSE_CPU=false\nNETWORK_PATH=LivingCity/berkeley_2018/new_full_network/\nUSE_JOHNSON_ROUTING=false\nUSE_SP_ROUTING=true\nUSE_PREV_PATHS=false\nLIMIT_NUM_PEOPLE=256000\nADD_RANDOM_PEOPLE=false\nNUM_PASSES=1\nTIME_STEP=1\nSTART_HR={0}\nEND_HR=24\nOD_DEMAND_FILENAME={1}_oddemand_{2}_{3}_R_{4}_UCI_{5}.csv\nSHOW_BENCHMARKS=false\nREROUTE_INCREMENT=0\nPARTITION_FILENAME=ciccio.txt\nNUM_GPUS=1\n'.format(start,CityName,start,end,R,UCI)
    with open(os.path.join(TRAFFIC_DIR,'command_line_options.ini'),'w') as file:
        file.write(file_txt)
    file.close()
    if verbose:
        print("***** MODIFICATION CONFIG ******")
        print(file_txt)    


def RenameMoveOutput(output_file,CityName,R,UCI,verbose = False):
    saving_dir = os.path.join(HOME_DIR,'berkeley_2018',CityName,'Output')
    source_file = os.path.join(LPSIM_DIR, output_file)
    destination_file = os.path.join(saving_dir, f"R_{R}_UCI_{UCI}_{output_file}")
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    if verbose:
        print("***** RENAME SIMULATION OUTPUT FILES ******")
        print('renamed output file:\t',output_file)
        print('In dir:\t',saving_dir)
    if os.path.exists(source_file):
        os.rename(source_file, destination_file)
    else:
        print(f"File {source_file} does not exist")
#     shutil.move(os.path.join(LPSIM_DIR,f"R_{R}_UCI_{UCI}_{output_file}"),saving_dir)  

  
def CheckUseDocker(verbose = False):
    docker_version = ['docker', '--version']
    try:
        subprocess.run(docker_version, check=True)
        if verbose:
            print("***** CHECK PERMISSION DOCKER COMMAND ******")
            print("User has permission to execute Docker commands.")
    except subprocess.CalledProcessError:
        if verbose:
            print("***** CHECK PERMISSION DOCKER COMMAND ******")
            print("User does not have permission to execute Docker commands.")    


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

def DockerCommand(verbose = False):
    os.environ['PATH'] += os.pathsep + '/usr/bin/'
    PWD ='/home/alberto/LPSim'
    env = os.environ.copy()
    docker_cmd = f'/usr/bin/docker run -it --rm --gpus all -v {PWD}:/lpsim -w /lpsim {container_name} bash'
    docker_cmd = ['/usr/bin/docker', 'run', '-it', '--rm', '--gpus', 'all', '-v', f'{PWD}:/lpsim', '-w', '/lpsim', f'{container_name}', 'bash','-c','./LivingCity/LivingCity']
    if verbose:
        print("***** DOCKER COMMAND ******")
        print('current working dir: ',os.getcwd())
        print(docker_cmd)

    # Execute the Docker command
    if not verbose:
        CheckUseDocker(verbose)
        compare_environment_variables(verbose)
    process = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,env=env)#,shell = True
    stdout, stderr = process.communicate()
    if verbose:
        print(stdout.decode())
        print(stderr.decode())
        print('Executing docker command')

#    exec_commands = [
#        "./LivingCity"
#    ]
#    for cmd in exec_commands:
#        exec_result = subprocess.run(cmd,shell=True)
    return process

def LaunchDockerFromServer(container_name,CityName,start,end,R,UCI,verbose = False):
    if False:
        process = DockerCommand()

        # Check if Docker command was successful
        if process.returncode == 0:
            print("Docker command executed successfully.")
            exec_commands = [
                "cd",
                "cd test",
                "cd LPSim",
                "cd LivingCity",
                "./LivingCity"
            ]
            print('Current directory: ',os.getcwd())
            for cmd in exec_commands:
                exec_result = subprocess.run(['docker', 'exec', '-i', container_name, 'bash', '-c', cmd], capture_output=True, text=True)
                print(exec_result.stdout)
            output_files = ['0_allPathsinEdgesCUDAFormat{0}to{1}.csv'.format(start,end),
                            '0_indexPathInit{0}to{1}.csv'.format(start,end),
                            '0_people{0}to{1}.csv'.format(start,end),
                            '0_route{0}to{1}.csv'.format(start,end)
                            ]
            for output_file in output_files:
                RenameMoveOutput(output_files,CityName,R,UCI)
        else:
            print(f"Error executing Docker command: {stderr.decode()}")
    else:
        print('***** MAIN FUNCTION *****')
#        CheckPlatform()
        ModifyConfigIni(CityName,start,end,R,UCI, verbose)
#        os.chdir(TRAFFIC_DIR)
        CheckUseDocker()
        exec_result = DockerCommand(verbose)   

#        exec_result = subprocess.run(['docker', 'exec', '-i', container_name, 'bash', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#            exec_result = subprocess.run(['docker', 'exec', '-i', container_name, 'bash', '-c', cmd], capture_output=True, text=True)
#        print(exec_result.stdout)
        output_files = ['0_allPathsinEdgesCUDAFormat{0}to24.csv'.format(start),
                        '0_indexPathInit{0}to24.csv'.format(start),
                        '0_people{0}to24.csv'.format(start),
                        '0_route{0}to24.csv'.format(start)
                        ]
#        CheckPlatform()
        for output_file in output_files:
            RenameMoveOutput(output_file,CityName,R,UCI,verbose)


    
    
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
if __name__ == '__main__':
    container_name = "xuanjiang1998/lpsim:v1"
    CityNames = ['BOS']#['BOS','LAX','SFO','RIO','LIS']
    verbose = True
    for CityName in CityNames:
        OD_dir = os.path.join(TRAFFIC_DIR,'berkeley_2018',"new_full_network")
        print(os.getcwd())
        print('OD_dir: ',OD_dir)
        for file in os.listdir(OD_dir):
            if 'od' in file:
                if len(file.split('_')) == 8:
                    start = file.split('_')[2]
                    end = file.split('_')[3]
    #                start = file.split('_')[2].split('to')[0]
    #                end = file.split('_')[2].split('to')[1]
                    R = file.split('_')[5]
                    UCI = file.split('_')[7].split('.csv')[0]
                    print('start: ',start)
                    print('end: ',end)
                    print('R: ',R)
                    print('UCI: ',UCI)
                    print('Launching docker from server')
                    LaunchDockerFromServer(container_name,CityName,start,end,R,UCI,verbose)
                else:
                    pass
