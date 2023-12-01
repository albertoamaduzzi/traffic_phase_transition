import json
import numpy as np
import os
import graph_tool as gt
import pandas as pd
from global_functions import ifnotexistsmkdir

def lognormal(mean, std):
    '''
        Distribution of distances for each trip (the OD pair must be generated in this way)
        For policentric cities the variance is bigger. [ maybe we can invent some way of defining the variance according to the policentricity]
    '''
    return np.random.lognormal(mean, std)

def Weibull(shape, scale):
    '''
        Distribution of distances of trapped cars after 1 hour
    '''
    return np.random.weibull(shape, scale)

    

class OD:
    def __init__(self,config):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
        self.config_dir = os.path.join(root,'config')
        ifnotexistsmkdir(self.config_dir)
        self.config_name = os.listdir(self.config_dir)[conf.index() for conf in os.listdir(self.config_dir) if 'OD' in conf]
        with open(os.path.join(self.config_dir,self.config_name),'r') as f:
            self.config = json.load(f)
        self.graphdf = pd.read_csv(config['file_nodes'],index_col=0)


