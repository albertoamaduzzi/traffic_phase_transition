'''
    Functions to connect to the server and upload files.
    The informations about the directory structure in the server is hardcoded here.
    No much flexibility is given to the user.
    Everything is saved in TRAFFIC_DIR_SERVER that is a constant that must be changed manually.
'''

import paramiko
import os
import json
import sys
import subprocess
import time
import socket
# SETTING GLOBAL VARIABLES

if socket.gethostname()=='artemis.ist.berkeley.edu':
    TRAFFIC_DIR_LOCAL = '/home/alberto/LPSim/traffic_phase_transition'
else:
    TRAFFIC_DIR_LOCAL = os.getenv('TRAFFIC_DIR') 
TRAFFIC_DIR_SERVER = '/home/alberto/LPSim/LivingCity/berkeley_2018'
sys.path.append(os.path.join(TRAFFIC_DIR_LOCAL,'scripts','GenerationNet'))
from global_functions import *

def OpenConnection(config_dir = os.path.join(TRAFFIC_DIR_LOCAL,'config'),config_file_name = 'artemis.json'):
    '''
        This function opens a connection to the server
        Args:
            config_dir (str): path to the directory where the config file is located
            config_file_name (str): name of the config file
    '''
    with open(os.path.join(config_dir,config_file_name),'r') as f:
        config = json.load(f)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    hostname = config["hostname"]
    username = config["username"]
    password = config["password"]
#    key_filename = config["key_filename"]
    client.connect(hostname, username=username,password=password)
#    client.connect(hostname, username=username, key_filename=key_filename)
    return client

def CloseConnection(client):
    '''
        This function closes the connection to the server
        Args:
            client (paramiko.SSHClient): client to be closed
    '''
    client.close()
    print("Connection closed")

def OpenTransport(config_dir = os.path.join(TRAFFIC_DIR_LOCAL,'config'),config_file_name = 'artemis.json'):
    '''
        This function opens a transport to the server
        Args:
            config_dir (str): path to the directory where the config file is located
            config_file_name (str): name of the config file
    '''
    with open(os.path.join(config_dir,config_file_name),'r') as f:
        config = json.load(f)
    hostname = config["hostname"]
    username = config["username"]
    password = config["password"]
#    key_filename = config["key_filename"]
    transport = paramiko.Transport((hostname, 22))
    transport.connect(username=username,password=password)
#    transport.connect(username=username, key_filename=key_filename)
    return transport


##---------------------------------------- UPLOAD OD_DEMAND ON CLOUD ----------------------------------------##
def Upload2ServerPwd(file2transfer, remote_path,config_dir = os.path.join(TRAFFIC_DIR_LOCAL,'config'),config_file_name = 'artemis.json'):
    '''
        This function upload a file (file2transger) to the server using the password
        Args:
            file2transfer (str): path to the file to be uploaded
            remote_path (str): path to the remote directory where the file will be uploaded
            config_dir (str): path to the directory where the config file is located
            config_file_name (str): name of the config file
    {
    with open(os.path.join(config_dir,config_file_name),'r') as f:
        config = json.load(f) 
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    hostname = config["hostname"]
    username = config["username"]
    password = config["password"]
    client.connect(hostname, username=username, password=password)
    transport = paramiko.Transport((hostname, 22))
    transport.connect(username=username, password=password)
    }
    '''
    client = OpenConnection(config_dir,config_file_name)
    transport = OpenTransport(config_dir,config_file_name)
    sftp = paramiko.SFTPClient.from_transport(transport)
    print(f"File {file2transfer} uploaded to {remote_path}")   
    dir2create = os.path.dirname(remote_path)
    MakeDir(dir2create)
    sftp.put(file2transfer, remote_path)
    CloseConnection(client)
##---------------------------------------- CREATION CONNECTION AND DIRECTORY OPERATIONS ----------------------------------------##
def MakeDir(relative_remote_path,config_dir = os.path.join(TRAFFIC_DIR_LOCAL,'config'),config_file_name = 'artemis.json'):
    '''
        This function upload a file (file2transger) to the server using the key
        Args:
            remote_path (str): path to the remote directory created
            config_dir (str): path to the directory where the config file is located
            config_file_name (str): name of the config file
    '''
    client = OpenConnection(config_dir,config_file_name)
    remote_path = relative_remote_path
#    remote_path = os.path.join(TRAFFIC_DIR_SERVER,relative_remote_path)
    stdin, stdout, stderr = client.exec_command(f'mkdir {remote_path}')
    exit_status = stdout.channel.recv_exit_status()
    if exit_status == 0:
        print(f"Directory '{remote_path}' created successfully")
    else:
        print(f"Failed to create directory '{remote_path}'. Error: {stderr.read().decode()}")    
    CloseConnection(client)

def DeleteDir(relative_remote_path,config_dir = os.path.join(TRAFFIC_DIR_LOCAL,'config'),config_file_name = 'artemis.json'):
    '''
        This function deletes a directory in the server
        Args:
            remote_path (str): path to the remote directory created
            config_dir (str): path to the directory where the config file is located
            config_file_name (str): name of the config file
    '''
    client = OpenConnection(config_dir,config_file_name)
    remote_path = os.path.join(TRAFFIC_DIR_SERVER,relative_remote_path)
    stdin, stdout, stderr = client.exec_command(f'rm {remote_path}')
    exit_status = stdout.channel.recv_exit_status()
    if exit_status == 0:
        print(f"Directory '{remote_path}' deleted successfully")
    else:
        print(f"Failed to delete directory '{remote_path}'. Error: {stderr.read().decode()}")
    CloseConnection(client)
    
def MoveDir(relative_remote_path,relative_destination_remote_path,config_dir = os.path.join(TRAFFIC_DIR_LOCAL,'config'),config_file_name = 'artemis.json'):
    '''
        This function moves a directory in the server
        Args:
            remote_path (str): path to the remote directory created
            config_dir (str): path to the directory where the config file is located
            config_file_name (str): name of the config file
    '''
    client = OpenConnection(config_dir,config_file_name)
    remote_path = os.path.join(TRAFFIC_DIR_SERVER,relative_remote_path)
    destination_path = os.path.join(TRAFFIC_DIR_SERVER,relative_destination_remote_path)
    stdin, stdout, stderr = client.exec_command(f'mv {remote_path} {destination_path}')
    exit_status = stdout.channel.recv_exit_status()
    if exit_status == 0:
        print(f"Directory '{remote_path}' moved successfully")
    else:
        print(f"Failed to move directory '{remote_path}'. Error: {stderr.read().decode()}")
    CloseConnection(client)
##---------------------------------------- LAUNCHING DOCKER ----------------------------------------##
def LaunchDocker(container_name,File2Move,DirDestinationFile,DirFile2Move = '/home/alberto/LPSim/LivingCity'):
    '''
        This function launches the program in the docker container
        Args:
            container_name (str): name of the container
            directory (str): directory where the program is located
    '''
    # Construct the Docker command
    client = OpenConnection()
    docker_cmd = f'docker run -it --rm --gpus all -v "$PWD":/lpsim -w /lpsim {container_name} bash'

    # Execute the Docker command
#    process = subprocess.Popen(docker_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#    stdout, stderr = process.communicate()
    stdin, stdout, stderr = client.exec_command(docker_cmd)
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        print(f"Failed to launch Docker container. Error: {stderr.read().decode()}")
        CloseConnection(client)
        return False        
    else:
    # Check if Docker command was successful
        print("Docker command executed successfully.")

        exec_commands = [
            "cd LivingCity",
            "./LivingCity"
        ]
        for cmd in exec_commands:
            stdin,stdout,stderr = client.exec_command(cmd)
            print(stdout.read().decode())
            if '>>Simulation Ended' in stdout.read().decode():
                print('Simulation ended')
                MoveDir(os.path.join(DirFile2Move,File2Move),os.path.join(DirDestinationFile,File2Move))
                CloseConnection(client)
                return True
            else:
                CloseConnection(client)
                pass
        
def monitor_directory(directory):
    before = set(os.listdir(directory))
    while True:
        time.sleep(1)  # Adjust the interval as needed
        after = set(os.listdir(directory))
        new_files = after - before
        if new_files:
            return new_files.pop()    
