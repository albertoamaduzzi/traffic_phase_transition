import os
from collections import defaultdict,OrderedDict
import json
import numpy as np
import sys
import logging
logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.environ['TRAFFIC_DIR'],'scripts','GeometrySphere'))




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


def RsUCIsFromDir(OD_dir):
    """
        @param OD_dir: str -> Directory where the output files are stored
        @description: Force a distance between different UCIs of 0.01.
        In this way we can have
    """
    list_UCIs = [] 
    list_Rs = []
    for file_or_dir in os.listdir(OD_dir):
        if file_or_dir != "Plots" and "parquet" not in file_or_dir and "json" not in file_or_dir and "csv" not in file_or_dir:
            print(file_or_dir)
            UCI = float(file_or_dir)
            DIR_UCI = os.path.join(OD_dir,str(UCI))
            if UCI not in list_UCIs:
                list_UCIs.append(UCI)
            for file in os.listdir(DIR_UCI):
                if "route" in file:
                    R = int(file.split("_")[1])
                    if R not in list_Rs:
                        list_Rs.append(R)
    list_UCIs = sorted(list_UCIs)
    list_Rs = sorted(list_Rs)
    return list_UCIs, list_Rs

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
            logger.info(f"Reading {File}")
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

def InitConfigPolycentrismAnalysis(CityName):
    """
        @param CityName: str -> Name of the City
        This Function Initializes the Configuration for the Polycentrism2TrafficAnalyzer Class.
        It is useful to give in input the Rs and UCIs that are plausible.
        It is useful to give in input the Rs and UCIs that are plausible.
    """
    City2Config = {CityName:defaultdict()}
    if "LPSim" not in os.environ.keys():
        BaseData = "/home/alberto/LPSim/LivingCity/berkeley_2018/{}/Output".format(CityName)
    else:
        BaseData = os.path.join(os.environ["LPSim"],"LivingCity","berkeley_2018",CityName,"Output")
    Rs,UCIs = RsUCIsFromDir(BaseData)
    City2Config[CityName] = GenerateConfig(BaseData,City2Config[CityName],CityName,Rs,UCIs)  
    return City2Config