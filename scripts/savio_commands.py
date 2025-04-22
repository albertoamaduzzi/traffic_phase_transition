"""
# This script contains functions to run commands on the Savio cluster.
In general contains the commands that I run in my routines to access slurm and run scripts and simulations.
"""
import subprocess

def load_python_module():
    """
    Load the Python module in the Savio cluster.
    """
    command = "module load python"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout
    return output

def which(name_module):
    """
    Check if a module is loaded in the Savio cluster.
    """
    command = f"which {name_module}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout
    return output


def which_partition_do_i_possess():
    """
    Check which partition I possess in the Savio cluster.
    """
    command = "sacctmgr -p show  associations user=$USER"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout
    return output


def which_priority():
    """
    Check which priority I have in the Savio cluster.
    """
    command = "sacctmgr show qos format=Name%24, Priority%8, GrpTRES%22,MinTRES%26"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout
    return output