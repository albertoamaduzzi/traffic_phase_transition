"""
    Description:
        The script is thought to work as output for the Pipeline written in nextflow.
"""
import os
import sys
import socket
if socket.gethostname()=='artemis.ist.berkeley.edu':
    sys.path.append(os.path.join('/home/alberto/LPSim','traffic_phase_transition','scripts','ServerCommunication'))
else:
    sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','ServerCommunication'))
from HostConnection import *
if socket.gethostname()=='artemis.ist.berkeley.edu':
    sys.path.append(os.path.join('/home/alberto/LPSim','traffic_phase_transition','scripts','PreProcessing'))
else:
    sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','PreProcessing'))

def UploadFiles(FileOrigin2FileDest,ConfigDir):
    '''
        Upload Files to the server
    '''
    for FileOrigin in FileOrigin2FileDest.keys():
        FileDest = FileOrigin2FileDest[FileOrigin]
        Upload2ServerPwd(FileOrigin,FileDest,ConfigDir)
