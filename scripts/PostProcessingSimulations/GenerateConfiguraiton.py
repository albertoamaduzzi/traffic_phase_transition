import os
from collections import defaultdict,OrderedDict
import json
import numpy as np

def order_subset_of_keys_in_place(input_dict, keys_to_order):
    # Initialize the ordered keys and dictionary
    sorted_keys = np.sort([key for key in input_dict.keys() if key in keys_to_order])
    ordered_dict = defaultdict()
    for key in input_dict.keys():
        if key not in keys_to_order:
            ordered_dict[key] = input_dict[key]
    # Add the sorted keys in order
    for key in sorted_keys:
        if key in keys_to_order:
            ordered_dict[key] = input_dict[key]
    return ordered_dict

def GenerateConfig(BaseData,Config,City):
    Config["output_simulation_dir"] = BaseData
    Config["delta_t"] = 15 # Length in minutes of the interval in which I accumulate the people in the network
    Config['name'] = City
    Config["graphml_file"]= os.path.join(os.environ["TRAFFIC_DIR"],'data','carto',City,City + '_new_tertiary_simplified.graphml')    
    list_UCI = []
    for File in os.listdir(BaseData):
        if os.path.isfile(os.path.join(BaseData,File)):
            UCI = File.split('_')[3]
            R  = File.split('_')[1]
            list_UCI.append(float(UCI))
            if 'people' in File:
                if float(UCI) not in Config.keys():
                    Config[float(UCI)] = defaultdict()
                else:
                    pass
                if int(R) not in Config[float(UCI)].keys():
                    Config[float(UCI)][int(R)] = defaultdict()
                else:
                    pass
                StartTime = File.split('_')[5].split('to')[0].split('le')[1]
                EndTime = File.split('_')[5].split('to')[1].split('.')[0]
                Config[float(UCI)][int(R)]["start_time"] = StartTime
                Config[float(UCI)][int(R)]["end_time"] = EndTime
                Config[float(UCI)][int(R)]["people_file"] = os.path.join(BaseData,File)
            elif 'route' in File:
                if float(UCI) not in Config.keys():
                    Config[float(UCI)] = defaultdict()
                else:
                    pass
                if int(R) not in Config[float(UCI)].keys():
                    Config[float(UCI)][int(R)] = defaultdict()
                else:
                    pass
                Config[float(UCI)][int(R)]["route_file"]= os.path.join(BaseData,File)
        else:
            pass
    for UCI in list_UCI:
        print(Config[UCI])
        Config[UCI] = OrderedDict(sorted(Config[UCI].items()))
    Config = order_subset_of_keys_in_place(Config, list_UCI)

    return Config
