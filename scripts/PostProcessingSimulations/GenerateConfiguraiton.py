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
        OD_dir: LivingCity/berkeley_2018/CityName/Output
        @param OD_dir: str -> Directory where the output files are stored
        @description: Force a distance between different UCIs of 0.01.
        In this way we can have
    """
    list_UCIs = [] 
    list_Rs = []
    for file_or_dir in os.listdir(OD_dir):
        if file_or_dir != "Plots" and "parquet" not in file_or_dir and "json" not in file_or_dir and "csv" not in file_or_dir:
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
    from collections import defaultdict
#    Config = {float(UCIs): {int(R):defaultdict()} for UCIs in UCIs for R in Rs}
    Config["output_simulation_dir"] = BaseData
    Config["delta_t"] = 15 # Length in minutes of the interval in which I accumulate the people in the network
    Config['name'] = City
    Config["graphml_file"]= os.path.join(os.environ["TRAFFIC_DIR"],'data','carto',City,City + '_new_tertiary_simplified.graphml')    
    for UCI in UCIs:
        UCI_dir = os.path.join(BaseData,str(UCI))
        Config[float(UCI)] = defaultdict()            
        for File in os.listdir(os.path.join(UCI_dir)):
            if os.path.isfile(os.path.join(UCI_dir,File)):
                R  = File.split('_')[1]
                if 'people' in File:
                    StartTime = File.split('_')[5].split('to')[0].split('le')[1]
                    EndTime = File.split('_')[5].split('to')[1].split('.')[0]
                    FileRoute = File.replace('people','route')
                    Config[float(UCI)][int(R)] = {"start_time":StartTime,"end_time":EndTime,"people_file":os.path.join(UCI_dir,File),"route_file":os.path.join(UCI_dir,FileRoute)}
            else:
                pass
    Config = order_subset_of_keys_in_place(Config, UCIs)
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
    UCIs,Rs = RsUCIsFromDir(BaseData)
    City2Config[CityName] = GenerateConfig(BaseData,City2Config[CityName],CityName,Rs,UCIs)  
    return City2Config,Rs,UCIs