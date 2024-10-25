# O
import os
# F
TRAFFIC_DIR = os.environ["TRAFFIC_DIR"]
current_dir = os.path.join(os.getcwd()) 
mother_path = os.path.abspath(os.path.join(current_dir, os.pardir))
print('mother_path:', mother_path)
sys.path.append(os.path.join(mother_path, 'PreProcessing'))
sys.path.append(os.path.join(mother_path))
sys.path.append(os.path.join(TRAFFIC_DIR,'scripts','GeometrySphere'))
sys.path.append(os.path.join(TRAFFIC_DIR,'scripts','ServerCommunication'))
sys.path.append(os.path.join(TRAFFIC_DIR,'scripts','PostProcessingSimulations'))
sys.path.append(os.path.join(TRAFFIC_DIR,'scripts',"InitialConfigProcess"))
# A 
import ast
# C
from collections import defaultdict
# G
import gc
import geopandas as gpd
# J
import json
# M
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
# N
from numba import prange
import numpy as np
# P
import pandas as pd
# S
from shapely.geometry import box,LineString,Point,MultiPoint,MultiLineString,MultiPolygon,Polygon
from shapely.ops import unary_union
import socket
import sys
# T
from termcolor import  cprint
import time

# Project specific
# A
from AlgorithmCheck import *
# C
from ComputeGrid import *
from ComputeHexagon import *
# F
from FittingProcedures import *
# G
from GeometrySphere import *
from GenerateModifiedFluxesSimulation import *
from GravitationalFluxes import *                                               # FIT section
from Grid import *
# H 
from Hexagon import *
if socket.gethostname()=='artemis.ist.berkeley.edu':
    sys.path.append(os.path.join('/home/alberto/LPSim','traffic_phase_transition','scripts','ServerCommunication'))
else:
    sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','ServerCommunication'))
from HostConnection import *
# M
from MainPolycentrism import *
from ModifyPotential import *
# O 
from ODfromfma import *
# P
from plot import *
from Polycentrism import *
from PolycentrismPlot import *
from PolygonSettings import *
from Potential import *
from PreprocessingObj import *



## BASIC PARAMS
gc.set_threshold(10000,50,50)
plt.rcParams.update({
    "text.usetex": False,
})
StateAlgorithm = InitWholeProcessStateFunctions()



if __name__ == '__main__':
    NameCities = os.listdir(os.path.join(TRAFFIC_DIR,'data','carto'))
    container_name = "xuanjiang1998/lpsim:v1"    
    for CityName in NameCities:
        # Everything is handled inside the object
        GeoInfo = GeometricalSettingsSpatialPartition(CityName,TRAFFIC_DIR)
        # Compute the Potential and Vector field for non modified fluxes
        UCI = GeoInfo.RoutineVectorFieldAndPotential()
        # Compute the Fit for the gravity model
        GeoInfo.ComputeFit()
        # Initialize the Concatenated Df for Simulation [It is common for all different R]
        GeoInfo.InitializeDf4Sim()
        # NOTE: Can Parallelize this part and launch the simulations in parallel.
        for R in GeoInfo.ArrayRs:
            # Simulation for the monocentric case.
            NotModifiedInputFile = GeoInfo.ComputeDf4SimNotChangedMorphology(UCI,R)
            # Generate modified Fluxes
            for cov in GeoInfo.config['covariances']:
                for distribution in ['exponential']:
                    for num_peaks in GeoInfo.config['list_peaks']:
                        Modified_Fluxes,UCI1 = GeoInfo.ChangeMorpholgy(cov,distribution,num_peaks)
                        GeoInfo.ComputeDf4SimChangedMorphology(UCI1,R,Modified_Fluxes)

        #NOTE: TODO Change the code in such a way that I compute the UCI,R  -> construct the file.
        # Launch the simulations, delete the input files and save the output in parallel.
        # Then I can run the simulation for the different R in parallel.
        