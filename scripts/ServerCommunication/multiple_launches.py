import subprocess
import os
import argparse
import json
import time
import shutil

TRAFFIC_DIR = '/home/alberto/LPSim/LivingCity'
TRAFFIC_DIR = '/lpsim/LivingCity'

def ModifyConfigIni(CityName,start,end,R):
    print('MODIFY CONFIG INI:')
    file_txt = '[General]\nGUI=false\nUSE_CPU=false\nNETWORK_PATH=berkeley_2018/new_full_network/\nUSE_JOHNSON_ROUTING=false\nUSE_SP_ROUTING=true\nUSE_PREV_PATHS=false\nLIMIT_NUM_PEOPLE=256000\nADD_RANDOM_PEOPLE=false\nNUM_PASSES=1\nTIME_STEP=1\nSTART_HR={0}\nEND_HR=24\nOD_DEMAND_FILENAME={1}_oddemand_{2}_{3}_R_{4}.csv\nSHOW_BENCHMARKS=false\nREROUTE_INCREMENT=0\nNUM_GPUS=1\n'.format(start,CityName,start,end,R)
    print(file_txt)    
    with open(os.path.join(TRAFFIC_DIR,'command_line_options.ini'),'w') as file:
        file.write(file_txt)
    file.close()
    

def RenameMoveOutput(output_files,CityName,R):
    for output_file in output_files:
        os.rename(output_file, f"R_{R}_{output_file}")
        saving_dir = os.path.join(TRAFFIC_DIR,CityName,'Output')
        print('renamed output file:\t',output_file)
        print('In dir:\t',saving_dir)
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)
        shutil.move(os.path.join(TRAFFIC_DIR,f"R_{R}_{output_file}"),saving_dir)    

def CheckUseDocker():
    docker_version = ['docker', '--version']
    try:
        subprocess.run(docker_version, check=True)
        print("User has permission to execute Docker commands.")
    except subprocess.CalledProcessError:
        print("User does not have permission to execute Docker commands.")    


def compare_environment_variables():
    shell_env = dict(os.environ)
    python_env = dict()
    # Get environment variables in Python subprocess
    process = subprocess.Popen(['printenv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        python_env = {line.split('=', 1)[0]: line.split('=', 1)[1].strip() for line in stdout.decode().split('\n') if line}
    
    print("Environment variables in shell:")
    for key, value in shell_env.items():
        print(f"{key}: {value}")
    
    print("\nEnvironment variables in Python subprocess:")
    for key, value in python_env.items():
        print(f"{key}: {value}")
    
    # Compare environment variables
    diff_keys = set(shell_env.keys()) - set(python_env.keys())
    if diff_keys:
        print("\nEnvironment variables differ between shell and Python subprocess:")
        for key in diff_keys:
            print(f"{key}: shell = {shell_env[key]}, Python subprocess = not set")
    else:
        print("\nNo differences found in environment variables between shell and Python subprocess.")

def DockerCommand():
    print('current working dir: ',os.getcwd())
    os.environ['PATH'] += os.pathsep + '/usr/bin/'
    print('PATH: ','/usr/bin/' in os.environ['PATH'])
    PWD ='/home/alberto/test/'
    env = os.environ.copy()
    docker_cmd = f'/usr/bin/docker run -it --rm --gpus all -v {PWD}:/lpsim -w /lpsim {container_name} bash'

    # Execute the Docker command
    if 1==0:
        CheckUseDocker()
        compare_environment_variables()
    process = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,env=env)#,shell = True
    stdout, stderr = process.communicate()
    print(stdout.decode())
    print(stderr.decode())
    print('Executing docker command')
    return process

def LaunchDockerFromServer(container_name,CityName,start,end,R):
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
                RenameMoveOutput(output_files,CityName,R)
        else:
            print(f"Error executing Docker command: {stderr.decode()}")
    else:
        print('Launch docker container')
        ModifyConfigIni(CityName,start,end,R)
        exec_commands = [
            "./LivingCity"
        ]
        os.chdir(TRAFFIC_DIR)
        print('Current directory: ',os.getcwd())
        for cmd in exec_commands:
            exec_result = subprocess.run(cmd,shell=True)
            print('Current directory: ',os.getcwd())
#            exec_result = subprocess.run(['docker', 'exec', '-i', container_name, 'bash', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#            exec_result = subprocess.run(['docker', 'exec', '-i', container_name, 'bash', '-c', cmd], capture_output=True, text=True)
            print(exec_result.stdout)
        output_files = ['0_allPathsinEdgesCUDAFormat{0}to{1}.csv'.format(start,end),
                        '0_indexPathInit{0}to{1}.csv'.format(start,end),
                        '0_people{0}to{1}.csv'.format(start,end),
                        '0_route{0}to{1}.csv'.format(start,end)
                        ]
        for output_file in output_files:
            RenameMoveOutput(output_files,CityName,R)


    
    
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
    for CityName in CityNames:
        OD_dir = os.path.join(TRAFFIC_DIR,'berkeley_2018',CityName)
        print(os.getcwd())
        for file in os.listdir(OD_dir):
            if 'od' in file:
                start = file.split('_')[2].split('to')[0]
                end = file.split('_')[2].split('to')[1]
                R = file.split('_')[4].split('.')[0]
                print('start: ',start)
                print('end: ',end)
                print('R: ',R)
                print('Launching docker from server')
                LaunchDockerFromServer(container_name,CityName,start,end,R)