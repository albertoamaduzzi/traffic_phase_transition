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
from PlotsPercolation import *
import json
BASE_PATH = "/home/alberto/LPSim/LivingCity/berkeley_2018/BOS/Output/"

class Polycentrism2TrafficAnalyzer:
    def __init__(self,config,Rs,UCIs):
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
        self.Rs = Rs
        self.UCIs = UCIs
        # Aggregated Variables
        # One per (UCI,R)
        self.Taus = {round(UCI,3):[] for UCI in UCIs}
        self.Gammmas = {round(UCI,3):[] for UCI in UCIs}
        # One per UCI\
        self.UCI2CriticalParams = {round(UCI,3):defaultdict() for UCI in UCIs}
        self.Rcs = {round(UCI,3):None for UCI in UCIs}
        self.Alphas = {round(UCI,3):[] for UCI in UCIs}
        self.TCritical = {round(UCI,3):[] for UCI in UCIs}


    def CompleteAnalysis(self):
        """
            Description:
                - For each UCI:
                    - For each R:
                        - Compute the time2road2traveller
                        - Plot the unload curve
                        - Animate the network traffic
        """

        # Treat each UCI independently
        for UCI in self.UCIs:    
            self.R2UCI2OutputStats[UCI] = defaultdict()
            FoundRc = False
            R2NtNtFit = {R:{"n":[],"n_fit":[]} for R in sorted(self.Rs)}
            for R in sorted(self.Rs):
                logger.info(f"Generation OS: UCI: {UCI} and R: {R}")
                # Make The Analysis For The sinle Couple Of UCI and R
                OS = OutputStats(R,UCI,self.config,self.GeoJsonEdges)
                logger.info(f"OS Complete Analysis: UCI: {UCI} and R: {R}")
                OS.CompleteAnalysis()
                # Assemble all knowledge.
                # Evidence for Traffic Changing R
                self.Taus[round(UCI,3)].append(OS.Tau)
                self.Gammmas[round(UCI,3)].append(OS.Gamma)
                # NOTE: Each of these is a list, take the smaller to 
                if OS.IsJam:
                    self.Rcs[round(UCI,3)].append(R)
                    self.Alphas[round(UCI,3)].append(OS.CriticalAlpha)
                    self.TCritical[round(UCI,3)].append(OS.TCritical)
                    if not FoundRc:
                        FoundRc = True
                        self.UCI2CriticalParams[round(UCI,3)] = {'Rc':R,'Alpha':OS.CriticalAlpha,'TCritical':OS.TCritical}
                        with open(os.path.join(self.BaseDir,"Plots",f"CriticalParams_UCI_{round(UCI,3)}.json"),'w') as f:
                            json.dump(self.UCI2CriticalParams[round(UCI,3)],f,indent = 4)
                R2NtNtFit[R]["n"] = OS.Time2nt[OS.BestTime]["n"]
                R2NtNtFit[R]["n_fit"] = OS.Time2nt[OS.BestTime]["n_fit"]
            PlotGammaTau(self.Taus[round(UCI,3)],self.Gammas[round(UCI,3)],self.Name,UCI,self.BaseDir)
            Rc = self.UCI2CriticalParams[round(UCI,3)]["Rc"]
            alpha = self.UCI2CriticalParams[round(UCI,3)]["Alpha"]
            R2Epsilon = {R:(R**2 - Rc**2)/Rc**2 for R in sorted(self.Rs)}
            PlotFigure4(R2NtNtFit,R2Epsilon,alpha,OS.DfUnload["Time_hour"].to_numpy(),OS.DfUnload["Time_hour"].to_numpy()[OS.t0],UCI,Rc,self.Name,self.BaseDir)
                # Extract The informations for the analysis to be Global.
                # In this way I do not loose Informations but the process is very power consuming
#                self.R2UCI2OutputStats[UCI][R] =  OS

        pass


    def CollectUnloadAndFitAllRGivenUCI(self,UCI):
        """
            @UCI: float
            This Function is used to collect time and n(t),fit_nt and plot the unload curve for all Rs.
            The DfUnloads: pd.DataFrame -> Columns: nt_{R},n_fit_{R},time_{R} for all Rs
        """
        pass
    def GetGeoJson(self):
        """
            Description:
                GeoJsonEdges: gpd.DataFrame -> u,v, uv, geometry, highway, lanes, maxspeed, capacity
        """
        self.GeoJsonNodes,self.GeoJsonEdges,self.GetGeoJsonBool = GetGeopandas(self.GraphmlFile)
        self.GeoJsonEdges = CleanGeojson(self.GeoJsonEdges)







