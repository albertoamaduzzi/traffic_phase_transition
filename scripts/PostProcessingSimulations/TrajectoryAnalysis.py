"""
    This script contains the analysis of the simulations launched with the LivingCity program.
    The configuration file is for each city and contains the following information:
        - name: name of the city
        - delta_t: time step (in seconds)
        - output_simulation_dir: directory where the simulation output is stored
        - graphml_file: file containing the graph of the city
        - UCI: {
                R: {
                    "route_file": "/home/alberto/LPSim/LivingCity/berkeley_2018/BOS/Output/R_152_UCI_0.666_0_route7to24.csv",
                    "start_time": "7",
                    "end_time": "24",
                    "people_file": "/home/alberto/LPSim/LivingCity/berkeley_2018/BOS/Output/R_152_UCI_0.666_0_people7to24.csv"
                },

"""
import pandas as pd
import os
import matplotlib.pyplot as plt 
import numpy as np
import polars as pl
from collections import defaultdict
from GeoJsonFunctions import *
from OutputStats import *

BASE_PATH = "/home/alberto/LPSim/LivingCity/berkeley_2018/BOS/Output/"
BASE_PATH = "/home/alberto/LPSim/LivingCity/berkeley_2018/BOS/Output/Monocentric"

class Polycentrism2TrafficAnalyzer:
    def __init__(self,config):
        self.config = config
        self.Name = config['name']
        self.delta_t = config['delta_t']
        self.BaseDir = config['output_simulation_dir']
        self.GraphmlFile = config['graphml_file']
        self.R2UCI2OutputStats = defaultdict()
        # Phase Transition
        self.UCI2R2Time2PeopleInNet = defaultdict()
        self.UCI2R2ErrorFit = defaultdict()
        self.UCI2R2Tau = defaultdict()
        self.UCI2NR = defaultdict()
        # FLAGS
        self.GetGeoJsonBool = False
        self.GetGeoJson()
        self.Rs = []
        self.UCIs = []
    def CompleteAnalysis(self):
        """
            Description:
                - For each UCI:
                    - For each R:
                        - Compute the time2road2traveller
                        - Plot the unload curve
                        - Animate the network traffic
        """

    
        for UCI in self.config.keys():
            if UCI != 'name' and UCI != 'delta_t' and UCI != 'output_simulation_dir' and UCI != 'graphml_file':
                self.R2UCI2OutputStats[UCI] = defaultdict()
                for R in self.config[UCI].keys():
                    self.Rs.append(R)
                    OS = OutputStats(R,UCI,self.config,self.GeoJsonEdges)
                    OS.CompleteAnalysis()
                    self.R2UCI2OutputStats[UCI][R] =  OS

        pass


    def GetGeoJson(self):
        """
            Description:
                GeoJsonEdges: gpd.DataFrame -> u,v, uv, geometry, highway, lanes, maxspeed, capacity
        """
        self.GeoJsonNodes,self.GeoJsonEdges,self.GetGeoJsonBool = GetGeopandas(self.GraphmlFile)
        self.GeoJsonEdges = CleanGeojson(self.GeoJsonEdges)







