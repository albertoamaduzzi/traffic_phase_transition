from collections import defaultdict
import os
import json
from TrajectoryAnalysis import *
from GenerateConfiguraiton import *

if __name__=='__main__':
    BaseConfig = os.path.join(os.environ["TRAFFIC_DIR"],"config")
    list_cities = os.listdir(os.path.join(os.environ["TRAFFIC_DIR"],'data','carto'))
    City2Config = defaultdict() 
    ListPeopleFile = []
    ListRoutesFile = []
    for City in list_cities:
        City2Config[City] = defaultdict()
        Config = City2Config[City]
        if "LPSim" not in os.environ.keys():
            BaseData = "/home/alberto/LPSim/LivingCity/berkeley_2018/{}/Output".format(City)
        else:
            BaseData = os.path.join(os.environ["LPSim"],"LivingCity","berkeley_2018",City,"Output")
        Config = GenerateConfig(BaseData,Config,City)
        City2Config[City] = GenerateConfig(BaseData,City2Config[City],City)    
        with open(os.path.join(BaseConfig,'post_processing_' + City +'.json'),'w') as f:
            json.dump(Config,f,indent=4)
    with open(os.path.join(BaseConfig,'post_processing_all.json'),'w') as f:
        json.dump(City2Config,f,indent=4)