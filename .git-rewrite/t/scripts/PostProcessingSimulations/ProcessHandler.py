'''
    @description: This file contains the object that is responsible to get the informations about the structure of the cluster you are running
    code. It is thought such that you can use different resources in your cluster. Multiple GPUS for example.
    In the case of this repository, we are using the Artemis cluster from UC Berkeley and we want to run processes in different
    CPUs for the pre-processing and post-processing of the data.
    While different GPUs for the traffic simulations. 
'''
import psutil
import os
import subprocess
import re
class ProcessHandler:
    def __init__(self):
        self.cpu_count = psutil.cpu_count(logical=False)
        self.cpu_count_logical = psutil.cpu_count(logical=True)
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                self.IsGpuAvailable = True
            self.gpu_count =  len(re.findall(r'GPU \d+:', result.stdout))
        except FileNotFoundError:
            self.IsGpuAvailable = False

