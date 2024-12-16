import sys
import os
LPSIM_DIR = os.environ["LPSim"]
TRAFFIC_DIR = os.environ["TRAFFIC_DIR"] 
LIVING_CITY_DIR = os.path.join(LPSIM_DIR,"LivingCity")
BERKELEY_DIR = os.path.join(LIVING_CITY_DIR,"berkeley_2018")
NEW_FULL_NETWORK_DIR = os.path.join(BERKELEY_DIR,"new_full_network")
