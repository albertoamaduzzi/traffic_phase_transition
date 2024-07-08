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
        # FLAGS
        self.GetGeoJsonBool = False
        self.GetGeoJson()
        for UCI in config.keys():
            if UCI != 'name' and UCI != 'delta_t' and UCI != 'output_simulation_dir' and UCI != 'graphml_file':
                self.R2UCI2OutputStats[UCI] = defaultdict()
                for R in config[UCI].keys():
                    OS = OutputStats(R,UCI,config,self.GeoJsonEdges)
                    OS.ComputeTime2Road2Traveller()
                    OS.PlotUnloadCurve()
                    OS.AnimateNetworkTraffic()
                    self.R2UCI2OutputStats[UCI][R] =  OS

        pass


    def GetGeoJson(self):
        """
            Description:
                GeoJsonEdges: gpd.DataFrame -> u,v, uv, geometry, highway, lanes, maxspeed, capacity
        """
        self.GeoJsonNodes,self.GeoJsonEdges,self.GetGeoJsonBool = GetGeopandas(self.GraphmlFile)
        self.GeoJsonEdges = CleanGeojson(self.GeoJsonEdges)







