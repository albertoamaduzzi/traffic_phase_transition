from collections import defaultdict
import os
import json
from TrajectoryAnalysis import *
from GenerateConfiguraiton import *

if __name__=='__main__':
    BaseConfig = os.path.join(os.environ["TRAFFIC_DIR"],"config")
    list_cities = os.listdir(os.path.join(os.environ["TRAFFIC_DIR"],'data','carto'))
    ListPeopleFile = []
    ListRoutesFile = []
    for City in list_cities:
        City2Config = InitConfigPolycentrismAnalysis(City)  
        PCTA = Polycentrism2TrafficAnalyzer(City2Config[City])  
        PCTA.CompleteAnalysis()
        with open(os.path.join(BaseConfig,'post_processing_' + City +'.json'),'w') as f:
            json.dump(City2Config,f,indent=4)
    with open(os.path.join(BaseConfig,'post_processing_all.json'),'w') as f:
        json.dump(City2Config,f,indent=4)