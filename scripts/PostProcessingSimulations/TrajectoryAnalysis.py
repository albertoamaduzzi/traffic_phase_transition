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
from Beta import *
BASE_PATH = os.path.join(os.environ["LPSim"],'LivingCity/berkeley_2018/BOS/Output/')


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
        self.Nodes = pl.read_csv(os.path.join(BERKELEY_DIR,self.Name,'nodes.csv'),dtypes={'ref':pl.Utf8})
        self.Edges = pl.read_csv(os.path.join(BERKELEY_DIR,self.Name,'edges.csv'))
        self.Rs = Rs
        self.UCIs = UCIs
        # Aggregated Variables
        # One per (UCI,R)
        self.Taus = {round(UCI,3):[] for UCI in UCIs}
        self.Gammas = {round(UCI,3):[] for UCI in UCIs}
        # One per UCI\
        self.UCI2CriticalParams = {round(UCI,3):defaultdict() for UCI in UCIs}
        self.Rcs = {round(UCI,3):[] for UCI in UCIs}
        self.Alphas = {round(UCI,3):[] for UCI in UCIs}
        # Range Fit
        self.RangeFit = None
        # Time For Plots and Fit Beta
        self.t_vect = None
        # 
        self.UCI2PhaseTransition = {round(UCI,3):{"Beta":None,"best_A_fit_Beta":None,"time_best_fit":None,"int_time_best_fit":None} for UCI in UCIs}
        self.City2Color = {"BOS":"red","LIS":"blue","LAX":"green","SFO":"orange","RIO":"purple"}
        self.City2Marker = {"BOS":"o","LIS":"s","LAX":"^","SFO":"D","RIO":"P"}

    def CompleteAnalysis(self):
        """
            Description:
                - For each UCI:
                    - For each R:
                        - DfUnload: pl.DataFrame -> Columns: 
                            - FractionPeople: np.array (Fraction of people in the network)
                            - Time_hours: np.array (Time in hours from t_start_hour)
                            - Time_hours_shifted: np.array (Time in hours after the unloading period starts) 
                            - IsPeople: np.array (True if people are in network, used as mask for fit)
                            - IsUnload: np.array (True if people are unloading, used to make the y,and y fit coincide at the plotting stage)
                            - NumberPeople: np.array (Number of people, not fraction)
                            - NumberPeopleFit: np.array (Number of people fitted)
                            - Time: np.array (Time in seconds)
                            - y_fit_exp: np.array (Fitted exponential)
                            - y_fit_pl: np.array (Fitted powerlaw)
                            - log_y: np.array (log of the fraction of people)
                            - log_x: np.array (log of the time in hours)
                        - InfoTraffic: dict ->
        """

        # Treat each UCI independently
        for UCI in self.UCIs:    
            self.R2UCI2OutputStats[UCI] = defaultdict()
            FoundRc = False
            R2NtNtFit = {R:defaultdict() for R in sorted(self.Rs)}
            logger.info(f"{self.BaseDir}")
            
            # Compute The Gamma Beta Tau for each UCI for which I did not compute it already
            if not os.path.exists(os.path.join(self.BaseDir,f'{self.Name}_{round(UCI,3)}_GammaTau.parquet')):
                for R in sorted(self.Rs):
                    logger.info(f"Generation OS: UCI: {UCI} and R: {R}")
                    if os.path.isfile(os.path.join(self.BaseDir,str(round(UCI,3)),f"R_{R}_UCI_{round(UCI,3)}_0_people7to24.parquet")):
                        logger.info(f"Generation OS: UCI: {UCI} and R: {R}")
                        # Make The Analysis For The sinle Couple Of UCI and R
                        OS = OutputStats(R,UCI,self.config,self.GeoJsonEdges,self.Nodes,self.Edges)
                        logger.info(f"OS Complete Analysis: UCI: {UCI} and R: {R}")
                        OS.CompleteAnalysis()
                        # NOTE: To look for the best Beta
                        self.DfUnload = OS.DfUnload
                        self.InfoTraffic = OS.InfoTraffic
                        # Evidence for Traffic Changing R
                        self.Taus[round(UCI,3)].append(OS.InfoTraffic["Tau"])
                        self.Gammas[round(UCI,3)].append(OS.InfoTraffic["Gamma"])
                        # NOTE: Each of these is a list, take the smaller to 
                        if OS.InfoTraffic["IsJam"]:
                            self.Rcs[round(UCI,3)].append(R)
                            self.Alphas[round(UCI,3)].append(OS.InfoTraffic["alpha_pl"])
                            if not FoundRc:
                                FoundRc = True
                                self.UCI2CriticalParams[round(UCI,3)] = {'Rc':int(R),
                                                                        'Alpha':float(OS.InfoTraffic["alpha_pl"]),
                                                                        "IndexTau":int(OS.IndexTau)
                                                                        }
                                with open(os.path.join(self.BaseDir,"Plots",f"CriticalParams_UCI_{round(UCI,3)}.json"),'w') as f:
                                    json.dump(self.UCI2CriticalParams,f,indent = 4)
                        # OS.Time2nt contains both n,n_fit, for different times.
                        # We need to find the t that best gives a fit to beta.
                        R2NtNtFit[R]["n"] = OS.DfUnload["FractionPeople"].to_numpy()
#                        R2NtNtFit[R]["t"] = OS.InfoTraffic["Time_hours_shifted"].to_numpy()
                        if self.t_vect is None:
                            self.t_vect = OS.DfUnload["Time_hours_shifted"].to_numpy()
                        if OS.InfoTraffic["IsJam"]:
                            y_fit = OS.DfUnload["y_fit_pl"].to_numpy()
                        else:
                            y_fit = OS.DfUnload["y_fit_exp"].to_numpy()
                        R2NtNtFit[R]["n_fit"] = y_fit
                        del OS
                self.R2NtNtFit = R2NtNtFit
                with open(os.path.join(self.BaseDir,"Plots",f"R2NtNtFit_UCI_{round(UCI,3)}.json"),'w') as f:
                    json.dump(R2NtNtFit,f,indent = 4)
                PlotGammaTau(self.Taus[round(UCI,3)],self.Gammas[round(UCI,3)],self.Name,UCI,self.BaseDir)
                # Compute Beta, such that the fit for an increasing powerlaw is top
                self.RangeFit = range(10,len(self.DfUnload["Time_hours_shifted"]))
                if FoundRc:
                    Time2FitBeta = ComputeTime2FitBeta(self.RangeFit,self.t_vect,self.R2NtNtFit,self.UCI2CriticalParams,self.Rs,UCI)
                    BestBeta,BestA,BestT,t1 = ChooseBestBeta(Time2FitBeta,self.t_vect)
                    PlotBestBeta(Time2FitBeta,BestBeta,BestA,BestT,t1)
                    self.UCI2PhaseTransition[round(UCI,3)] = {"Beta":BestBeta,"best_A_fit_Beta":BestA,"time_best_fit":BestT,"int_time_best_fit":t1}            
                    Rc = self.UCI2CriticalParams[round(UCI,3)]["Rc"]
                    alpha = self.UCI2CriticalParams[round(UCI,3)]["Alpha"]
                    R2Epsilon = {R:(R**2 - Rc**2)/Rc**2 for R in sorted(self.Rs)}
                    PlotFigure4(R2NtNtFit,R2Epsilon,alpha,R2NtNtFit[R]["t"],R2NtNtFit[R]["t"][self.UCI2CriticalParams[round(UCI,3)][round(UCI,3)]["IndexTau"]],UCI,Rc,self.Name,self.BaseDir)
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


    def ChooseBestBeta(self):
        pass



