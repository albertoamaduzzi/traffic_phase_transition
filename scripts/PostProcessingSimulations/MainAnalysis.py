from collections import defaultdict
import os
import json
from TrajectoryAnalysis import *
from GenerateConfiguraiton import *

"""
    Example Config File Read From The Output File System:
    "output_simulation_dir": "/home/alberto/LPSim/LivingCity/berkeley_2018/BOS/Output",
    "delta_t": 15,
    "name": "BOS",
    "graphml_file": "/home/alberto/LPSim/traffic_phase_transition/data/carto/BOS/BOS_new_tertiary_simplified.graphml",
    "0.166": {
        "150": {
            "route_file": "/home/alberto/LPSim/LivingCity/berkeley_2018/BOS/Output/R_150_UCI_0.166_0_route7to24.csv",
            "start_time": "7",
            "end_time": "24",
            "people_file": "/home/alberto/LPSim/LivingCity/berkeley_2018/BOS/Output/R_150_UCI_0.166_0_people7to24.csv"
        },
        ...
    } 
"""

if __name__=='__main__':
    BaseConfig = os.path.join(os.environ["TRAFFIC_DIR"],"config")
    list_cities = os.listdir(os.path.join(os.environ["TRAFFIC_DIR"],'data','carto'))
    ListPeopleFile = []
    ListRoutesFile = []
    for City in list_cities:
        City2Config,Rs,UCIs = InitConfigPolycentrismAnalysis(City)  
        PCTA = Polycentrism2TrafficAnalyzer(City2Config[City],Rs,UCIs)  
        PCTA.CompleteAnalysis()
        with open(os.path.join(BaseConfig,'post_processing_' + City +'.json'),'w') as f:
            json.dump(City2Config,f,indent=4)
    with open(os.path.join(BaseConfig,'post_processing_all.json'),'w') as f:
        json.dump(City2Config,f,indent=4)