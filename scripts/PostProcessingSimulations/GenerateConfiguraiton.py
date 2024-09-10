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

# Read available UCIs and Rs

def SelectUCIsIfAllR(Rs,UCIs,OD_dir):
    """
        NOTE: Use Just for Boston
        Since not all the UCIs have all the Rs simulated due to the fact that I stopped the simulation before it ended.
        I choose just those that have it.
    """
    UCI2AvailableRs = {uci:[] for uci in UCIs}
    for file in os.listdir(OD_dir):
        if 'od' in file:
            if len(file.split('_')) == 8:
                R = file.split('_')[5]
                UCI = file.split('_')[7].split('.csv')[0]
                if UCI in UCI2AvailableRs.keys():
                    UCI2AvailableRs[UCI].append(R)
    maxR = 0
    U = 0
    for uci in UCI2AvailableRs.keys():
        if len(UCI2AvailableRs[uci]) > maxR:
            maxR = len(UCI2AvailableRs[uci])
            U = uci
    UCIs = [uci for uci in UCI2AvailableRs.keys() if len(UCI2AvailableRs[uci]) == maxR]
    UCIs = ['0.166','0.205','0.214','0.216','0.225','0.235','0.244','0.253','0.256','0.263','0.274','0.309','0.318','0.333','0.349','0.361','0.374','0.392']
    print("Selected UCIs: ",UCIs)
    print("Selected Rs: ",Rs)
    return UCIs

def RsUCIsFromDir(OD_dir):
    Rs = []
    UCIs = []
    for file in os.listdir(OD_dir):
        if 'UCI' in file:
            R = file.split('_')[1]
            UCI = file.split('_')[3].split('.csv')[0]
            if R not in Rs:
                Rs.append(R)
            if float(UCI) not in UCIs:
                UCIs.append(float(UCI))
    Rs = sorted(Rs)
    UCIs = sorted(UCIs)
    UCisJump = []
    for i in range(len(UCIs)):
        if i == 0:
            UCisJump.append(str(UCIs[i]))
        elif (UCIs[i] - float(UCisJump[-1])) > 0.009:
            UCisJump.append(str(UCIs[i]))
    return Rs,UCisJump

# Configuration for the Phase Transition
def GenerateConfig(BaseData,Config,City,Rs,UCIs):
    """
        This Configuration File is Given in Input To the Polycentrism2TrafficAnalyzer Class.
        Contains informations about:
            - Where to Look For All 0_people*.csv, 0_route*.csv Files
            - Time of Aggregation of Data For The Unloading Curve
            - Graphml File of the City (Useful to Plot the )
        NOTE:
            It considers just the Rs, and UCIs that are given in input as plausible.
            This is a work in progress due to the time restriction of the CCS exeter.
    """
    Config["output_simulation_dir"] = BaseData
    Config["delta_t"] = 15 # Length in minutes of the interval in which I accumulate the people in the network
    Config['name'] = City
    Config["graphml_file"]= os.path.join(os.environ["TRAFFIC_DIR"],'data','carto',City,City + '_new_tertiary_simplified.graphml')    
    list_UCI = []
    for File in os.listdir(BaseData):
        if os.path.isfile(os.path.join(BaseData,File)):
            UCI = File.split('_')[3]
            R  = File.split('_')[1]
            if UCI in UCIs:
                list_UCI.append(float(UCI))
            if 'people' in File:
                if R in Rs and UCI in UCIs:
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
                if R in Rs and UCI in UCIs:
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
        Config[UCI] = OrderedDict(sorted(Config[UCI].items()))
    Config = order_subset_of_keys_in_place(Config, list_UCI)

    return Config

def InitConfigPolycentrismAnalysis(CityNames):
    City2Config = {CityName:defaultdict() for CityName in CityNames}
    for CityName in CityNames:
        # 
        City2Config[CityName] = defaultdict()
        if "LPSim" not in os.environ.keys():
            BaseData = "/home/alberto/LPSim/LivingCity/berkeley_2018/{}/Output".format(CityName)
        else:
            BaseData = os.path.join(os.environ["LPSim"],"LivingCity","berkeley_2018",CityName,"Output")
        Rs,UCIs = RsUCIsFromDir(BaseData)
        UCIs = SelectUCIsIfAllR(Rs,UCIs,BaseData)
        City2Config[CityName] = GenerateConfig(BaseData,City2Config[CityName],CityName,Rs,UCIs)  
    return City2Config