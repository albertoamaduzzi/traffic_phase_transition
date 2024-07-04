import pandas as pd
import os
import matplotlib.pyplot as plt 
import numpy as np
import polars as pl
from collections import defaultdict
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
        for R in config.keys():
            if R != 'name' and R != 'delta_t' and R != 'output_simulation_dir' and R != 'graphml_file':
                self.R2UCI2OutputStats[R] = defaultdict()
                for UCI in config[R].keys():
                    self.R2UCI2OutputStats[R][UCI] = OutputStats(R,UCI,config)        
        pass









