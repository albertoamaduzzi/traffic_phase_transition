"""
    This file contains all the constant informations used in the project.
    It esentially will contain all the path to directories that are used as default in
    different parts of the pipeline.
"""
import os
# Base dir: NOTE: contains both: - LIVINGCITY
#                                - TRAFFIC_DIR
LPSIM_DIR = os.environ["LPSIM"]
# LivingCity dir: NOTE:
#    - The directories of this section are those used for the simulator
LIVINGCITY_DIR = os.path.join(LPSIM_DIR,"LivingCity")
BERKELEY_2018_DIR = os.path.join(LIVINGCITY_DIR,"berkeley_2018")
# 
TRAFFIC_DIR = os.path.join(LPSIM_DIR,"traffic_phase_transition")
