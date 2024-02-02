'''
    Functions to connect to the server and upload files
'''

import paramiko
import os
import json



##---------------------------------------- UPLOAD OD_DEMAND ON CLOUD ----------------------------------------##
def Upload2ServerPwd(file2transfer, remote_path,config_dir = os.path.expanduser('~/Desktop/phd/berkeley/traffic_phase_transition/config/'),config_file_name = 'artemis.json'):
    '''
        This function upload a file (file2transger) to the server using the password
        Args:
            file2transfer (str): path to the file to be uploaded
            remote_path (str): path to the remote directory where the file will be uploaded
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
    client.connect(hostname, username=username, password=password)
    transport = paramiko.Transport((hostname, 22))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.put(file2transfer, remote_path)
    print(f"File {file2transfer} uploaded to {remote_path}")
