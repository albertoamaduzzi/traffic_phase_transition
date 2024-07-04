import os
from collections import defaultdict
import json

def GenerateConfig(BaseData,Config,City):
    Config["output_simulation_dir"] = BaseData
    Config["delta_t"] = 15 # Length in minutes of the interval in which I accumulate the people in the network
    Config['name'] = City
    Config["graphml_file"]= os.path.join(os.environ["TRAFFIC_DIR"],'data','carto',City,City + '_new_tertiary_simplified.graphml')    
    for File in os.listdir(BaseData):
        R  = File.split('_')[1]
        Config[R] = defaultdict()
        UCI = File.split('_')[3]
        Config[R][UCI] = defaultdict()
        if 'people' in File and not ('png' in File):
            StartTime = File.split('_')[5].split('to')[0].split('e')[1]
            EndTime = File.split('_')[5].split('to')[1].split('.')[0]
            Config[R][UCI]["start_time"] = StartTime
            Config[R][UCI]["end_time"] = EndTime
            if "people_file" not in Config[R][UCI].keys():
                Config[R][UCI]["people_file"] = []
            else:
                Config[R][UCI]["people_file"].append(os.path.join(BaseData,File))
        elif 'route' in File and not ('png' in File):
            StartTime = File.split('_')[5].split('to')[0].split('e')[1]
            EndTime = File.split('_')[5].split('to')[1].split('.')[0]
            Config[R][UCI]["start_time"] = StartTime
            Config[R][UCI]["end_time"] = EndTime
            if "route_file" not in Config[R][UCI].keys():
                Config[R][UCI]["route_file"] = []
            else:
                Config[R][UCI]["route_file"].append(os.path.join(BaseData,File))
            pass
        return Config
