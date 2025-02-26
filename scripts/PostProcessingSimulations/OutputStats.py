import polars as pl
import pandas as pd
import numpy as np
import osmnx as ox
from FittingProcedures import *
from OsFunctions import *
from DateTimeHandler import *
from Plots import *
from GeoJsonFunctions import *
from StructuresOfInterest import *
from PolarsFunctions import *
from PhaseTransition import *
from Percolation import *
import ast
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from EfficiencyAnalysis import *
from OutputStats_Functions import *
"""
    @ description:
        This script contains the class OutputStats that is used to analyze the output of the simulation.
        - Each Instantiation corresponds to the analysis of a simulation [R,UCI]
        - Consists of:
            - Computing the Unload Curve: DfUnload -> DataFrame
            - Computing the Best Fit for the Power Law and Exponential: Time2ErrorFit -> dict -> {t:{"error_pl":0,"error_exp":0}}
            - Deciding If The Network Is Jammed: IsJam -> bool (Condition: If there exists a time t for which the best fit is a power law)
            - Computing the Gamma and Tau: Gamma -> float, Tau -> float
            - Computing the Fluxes And Speeds in time: EdgesWithFluxesAndSpeed -> DataFrame

"""



def ReadRouteAndPeopleFileFromBaseDir(BaseDir):
    """
        Read the Route and People File from the BaseDir
        Returns:
            RouteFile: str
            PeopleFile: str
    """
    RouteFile = []
    PeopleFile = []
    for file in os.listdir(BaseDir):
        if "route" in file:
            RouteFile.append(file)
        if "people" in file:
            PeopleFile.append(file)
    return RouteFile,PeopleFile

class OutputStats:
    """
        NOTE: 
            @params R: int
            @params UCI: float \in [0,1]
            @params config: dict :
            UCI: {
                R: {
                    "route_file": "/home/alberto/LPSim/LivingCity/berkeley_2018/BOS/Output/R_152_UCI_0.666_0_route7to24.csv",
                    "start_time": "7",
                    "end_time": "24",
                    "people_file": "/home/alberto/LPSim/LivingCity/berkeley_2018/BOS/Output/R_152_UCI_0.666_0_people7to24.csv"
                },...}
            @params GeoJsonEdges: GeoJsonEdges (contains informations about capacity)

    """
    def __init__(self,
                 R,
                 UCI,
                 Df_City2Config,
                 GeoJsonEdges,
                 Nodes,
                 Edges):
        assert type(R) == int, "R must be an integer"
        assert isinstance(Df_City2Config,pl.DataFrame), f"Df_City2Config must be a polars DataFrame instead of {type(Df_City2Config)}"
        assert "complete_path_route_file" in Df_City2Config.columns, "Df_City2Config must contain the column complete_path_route_file"
        assert "complete_path_people_file" in Df_City2Config.columns, "Df_City2Config must contain the column complete_path_people_file"
        assert "complete_path_unloading_curve_base" in Df_City2Config.columns, "Df_City2Config must contain the column complete_path_unloading_curve_base"
        self.verbose = True
        # iter_ indexed variables
        self.RouteFiles = Df_City2Config['complete_path_route_file'].to_numpy()
        self.PeopleFiles = Df_City2Config['complete_path_people_file'].to_numpy()
        self.complete_path_unloading_curve_base = Df_City2Config['complete_path_unloading_curve_base'].to_numpy() 
        self.iters = Df_City2Config['iter'].to_numpy()
        # unique variables
        self.R = R
        self.UCI = UCI
        self.city = Df_City2Config['name'].unique()[0]
        self.delta_t = Df_City2Config['delta_t'].unique()[0]
        self.complete_path_unloading_curve_base =  Df_City2Config["complete_path_unloading_curve_base"].unique().to_numpy()[0]
        self.PlotDir = JoinDir(Df_City2Config['dir_output_simulation_analysis'].unique()[0],'Plots')
        self.PlotDir = JoinDir(self.PlotDir,f'{round(self.UCI,3)}')
        self.LogFile = JoinDir(self.PlotDir,"Log.txt")
        self.CountFunctions = 0
        # Group Control
        self.t_start_control_group = 7*3600
        self.t_end_control_group = 8*3600
        self.t_start_hour = 7
        self.t_end_hour = 8
        # Create the Plot Directory
        os.makedirs(self.PlotDir,exist_ok = True)
        # Flags
        self.ReadPeopleInfoBool = False
        self.ReadRouteInfoBool = False
        self.GetGeopandasBool = False
        # DataFrames
        self.GetPeopleInfo()
        self.GetRouteInfo()    
        self.GetGeopandas(GeoJsonEdges)
        self.Nodes = Nodes
        self.Edges = Edges
        # Edges and Nodes .csv
        self.AddCapacity2Edges()
        self.ConvertSpeedRoad2kmhInEdges()
        # Roads and People
        self.iter_2_DfRoute = defaultdict()
        self.iter_2_DfPeople = defaultdict()
        # TRAFFIC VARIABLES
        self.iter_2_DfUnload = defaultdict()
        self.iter_2_DfControlGroup = defaultdict()
        self.DfUnload = None
        self.DfControlGroup = None
        self.iter_2_VolumeTraffic = defaultdict()
        # Total Volume Traffic
        self.VolumeTraffic = 0
        # Dictionary Traffic
        self.InfoTraffic = {#"n_fit":[],"n":[],"t":[],
                            "ErrorPl":0,"ErrorExp":0,"best_fit":"",
                            "Gamma":None,"Tau":None,
                            "A_exp":0,"A_pl":0,"alpha_exp":0,"alpha_pl":0,
                            "IsJam":None}
        # SKIP
        self.SkipAnalysisR = os.path.exists(os.path.join(self.PlotDir,f"PercolationPlots_{round(self.UCI,3)}_{self.R}")) and os.path.exists(os.path.join(self.PlotDir,f"InfoTraffic_R_{self.R}_UCI_{round(self.UCI,3)}.json"))
        # ENOUGH DATA
        self.IsEnoughData = True
        # Cut Midnight
        self.CutMidnight = 2

#### Preprocess: Road Network -> GeoJsonEdges, Nodes, Edges Static Informations #####
    def GetGeopandas(self,GeoJsonEdges):
        """
            @Description:
                Read the GeoJsonEdges File
            NOTE: Unique for every simulation
        """
        logger.info(f"{self.city} Reading GeoJsonEdges R: {self.R}, UCI: {self.UCI}, City: {self.city}")
        self.GeoJsonEdges = GeoJsonEdges
        self.GetGeopandasBool
        self.CountFunctions += 1
        Message = f"Function {self.CountFunctions}: GetGeopandas: GeoJsonEdges was read"
        AddMessageToLog(Message,self.LogFile)

    def AddCapacity2Edges(self):
        """
            Add the capacity to the Edges DataFrame.
            @ NOTE: Useful For Considerations about traffic.

        """
        logger.info(f"Adding Capacity to Edges R: {self.R}, UCI: {self.UCI}, City: {self.city}")
        df = pl.DataFrame(self.GeoJsonEdges[["capacity","uv"]])
        df = df.with_columns([
            pl.col("capacity").cast(pl.Int64),
            pl.col("uv").cast(pl.Int64)
        ])
        self.Edges = self.Edges.join(df, left_on = "uniqueid",right_on = "uv", how = "inner")


    def ConvertSpeedRoad2kmhInEdges(self):
        """
            Convert the speed in the road to km/h
        """
        self.Edges = self.Edges.with_columns((pl.col("speed_mph")*1.6).alias("speed_kmh"))
        DropColumnsDfIfThere(self.Edges,["speed_mph"])

#### Preprocess: People -> GeoJsonEdges, Nodes, Edges #####


    def GetPeopleInfo(self):
        """
            @description:
                Read the People File:
                NOTE: columns: p, distance, time_departure, last_time_simulated, num_steps, avg_v(mph), a, b, T, gas, co, path_length_cpu, path_length_gpu
        """
        logger.info(f"{self.city} Reading People Files R: {self.R}, UCI: {self.UCI}, City: {self.city}")
        count_people_file = 0
        for PeopleFile in self.PeopleFiles:
            iter_ = self.iters[count_people_file]
            if IsFile(PeopleFile):
                if ".parquet" in PeopleFile:
                    self.iter_2_DfPeople[iter_] = pl.read_parquet(PeopleFile)
                elif ".csv" in PeopleFile:
                    self.iter_2_DfPeople[iter_] = pl.read_csv(PeopleFile)
                self.iter_2_DfPeople[iter_] = self.iter_2_DfPeople[iter_].filter(pl.col("time_departure")>self.t_start_hour*3600)
                logger.info(f"{self.city}: GetPeopleInfo: {PeopleFile} was read, updated iter_2_DfPeople")

#### Preprocess: Trajectories in Road Network  #####


    def GetRouteInfo(self):
        """
            @description:
                Read the Route File:
            NOTE: columns: p, route, distance
        """
        logger.info(f"{self.city} Reading Route Files R: {self.R}, UCI: {self.UCI}, City: {self.city}")
        count_route_file = 0
        for RouteFile in self.RouteFiles:
            iter_ = self.iters[count_route_file]
            if IsFile(RouteFile):
                if ".parquet" in RouteFile:
                    self.iter_2_DfRoute[iter_] = pl.read_parquet(RouteFile)
                elif ".csv" in RouteFile:
                    self.iter_2_DfRoute[iter_] = pl.read_csv(RouteFile,separator = ':')
                    if self.iter_2_DfRoute[iter_].schema["route"] == pl.Utf8:
                        pass
                self.ReadRouteInfoBool = True
                self.CountFunctions += 1
                logger.info(f"{self.city}: GetRouteInfo: {RouteFile} was read -> updated iter_2_DfRoute")
            count_route_file += 1





    @timing_decorator
    def CompleteAnalysis(self):
        """
            This is the Function one wants to Call to get the Analysis of the Simulation
        """
        if not self.SkipAnalysisR:
            # Join DfPeople, Edges And DfRoute 
            self.HavlinPercolation = False
            # 1) NOTE: Compute The Unload Curve DfUnload
            self.ComputeUnloadCurve()   
            # Instantiate All The Variables To Understand If The Network Is Jammed
            if self.IsEnoughData:
                # 2) NOTE: Compute The Best Fit For The Power Law And Exponential
                self.ComputeBestFitPlExpo() 
                # 3) NOTE: Compute The Gamma And Tau
                self.ComputeGammaAndTau()
                # 4) NOTE> Show The Unload Curve
                self.PlotUnloadCurve()
                # Compute The Fluxes And Speeds in time
    #            self.ComputeFluxesAndSpeedDfRoute()
                # Percolation Analysis
    #            AnalysisPercolationSpeed(self.IntTimeArray,self.GeoJsonEdges,self.EdgesWithFluxesAndSpeed,self.UCI,self.R,self.PlotDir)
            else:
                pass
        else:
            # 1) NOTE: Import the DfUnload,InfoTraffic NOTE: Only needed variables 
            import json
            with open(os.path.join(self.PlotDir,f"InfoTraffic_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'r') as f:
                self.InfoTraffic = json.load(f)
            self.DfUnload = pl.read_parquet(self.complete_path_unloading_curve_base)
            logger.info(f"{self.city} Skip Analysis for UCI: {round(self.UCI,3)}, R: {self.R}")



## PREPROCESSING ##
# Trajectories Info
    @timing_decorator
    def ComputeUnloadCurve(self):
        """
            @description:
                - Initialize Time vaiebales
                Interval2NumberPeopleInNet: {0:0,1:0,...,24:0} -> Is The Object That Contains The Number Of People In The Network At Each Time Interval


            Compute the number of people in the network at each time interval
            Returns:
                Interval2NumberPeopleInNet: Dictionary containing the number of people in the network at each time interval
                self.DfUnload: DataFrame
                Columns: 
                    Time: int -> seconds
                    Time_Str: str -> Time in Hours 00:00:00
                    NumberPeople: int -> Number of People in the Network
                    FractionPeople: int -> Fraction of People in the Network
                    DiffNumberPeople: int -> Difference in Number of People (between t+1 and t)
                    Time_hours: int -> Time in hours
                    Time_hours_shifted: int -> Time in hours shifted by start time NOTE: is the one used for fit and plots about fraction people remaining in the road network
            @return:
                self.DfUnload: DataFrame -> This is the object for the traffic analysis for DP.
        """
        logger.info(f"{self.city} Computing Unload Curve R: {self.R}, UCI: {self.UCI}")
        # 1) Initialize time variables
        self.IntTimeArray, self.HourTimeArray, self.Interval2NumberPeopleInNet = InitializeTimeVariablesOutputStats(HOURS_IN_DAY,MINUTES_IN_HOUR,SECONDS_IN_MINUTE,TIMESTAMP_OFFSET,self.delta_t)
        # 2) NOTE: Iter over ensemble realizations
        for iter_ in self.iter_2_DfRoute.keys():
            DfRoute = self.iter_2_DfRoute[iter_]
            DfPeople = self.iter_2_DfPeople[iter_]
            # 3) NOTE: Filter the People in the Control Group
            if self.HavlinPercolation:
                DfRoute = EmbdedTrajectoriesInRoadsAndTime(DfRoute,DfPeople,self.Edges)            
                DfControlGroup = FilterDfPeopleControlGroup(self.t_start_control_group,
                                                            self.t_end_control_group,
                                                            DfRoute,
                                                            "time_departure","last_time_simulated")
            else:
                DfControlGroup = FilterDfPeopleControlGroup(self.t_start_control_group,
                                                            self.t_end_control_group,
                                                            DfPeople,
                                                            "time_departure","last_time_simulated")
                DfControlGroup = EmbdedTrajectoriesInRoadsAndTime(DfRoute,
                                                                  DfControlGroup,
                                                                  self.Edges)
            self.iter_2_DfControlGroup[iter_] = DfControlGroup
            # 4) NOTE: Compute the Unload Curve
            if not os.path.exists(self.complete_path_unloading_curve_iter[iter_]):
                VolumeTraffic, DfUnload = ComputeSingleUnloadCurve(self.Interval2NumberPeopleInNet,
                                                                    self.IntTimeArray,
                                                                    self.HourTimeArray,
                                                                    self.t_start_control_group,
                                                                    self.t_start_hour,
                                                                    self.t_end_control_group,
                                                                    self.DfControlGroup)  
                self.iter_2_VolumeTraffic[iter_] = VolumeTraffic
                self.iter_2_DfUnload[iter_] = DfUnload
            else:
                self.iter_2_DfUnload[iter_] = pl.read_parquet(self.complete_path_unloading_curve_base[iter_])
                self.iter_2_VolumeTraffic[iter_] = self.iter_2_DfUnload[iter_]["NumberPeople"].sum()
                self.HourTimeArray = self.iter_2_DfUnload[iter_]["Time_Str"]
                self.Interval2NumberPeopleInNet = self.iter_2_DfUnload[iter_]["NumberPeople"]
        # 5) NOTE: Compute the average Unloading curve and Volume of traffic
        self.DfUnload = ComputeAverageUnloadCurve(self.iter_2_DfUnload,
                                                  self.t_end_control_group
                                                  )
        self.VolumeTraffic = ComputeAverageVolumeTraffic(self.iter_2_VolumeTraffic)
    
    @timing_decorator
    def ComputeBestFitPlExpo(self):
        """
            Compute the best fit for the Power Law and Exponential:
            @Description:
            Assigns:
                - InfoTraffic = {
                        "ErrorPl":0,"ErrorExp":0,"best_fit":"",
                        "Gamma":None,"Tau":None,
                        "A_exp":0,"A_pl":0,"alpha_exp":0,"alpha_pl":0,
                        "IsJam":None}
                - DfUnload: polars.DataFrame
                    Columns:
                        - FractionPeople: np.array (Fraction of people in the network)
                        - Time_hours: np.array (Time in hours from t_start_hour)
                        - Time_hours_shifted: np.array (Time in hours after the unloading period starts) 
                        - IsPeople: np.array
                        - IsUnload: np.array
                        - NumberPeople: np.array (Number of people, not fraction)
                        - NumberPeopleFit: np.array (Number of people fitted)
                        - Time: np.array (Time in seconds)
                        - y_fit_exp: np.array (Fitted exponential)
                        - y_fit_pl: np.array (Fitted powerlaw)
                        - log_y: np.array (log of the fraction of people)
                        - log_x: np.array (log of the time in hours)
        NOTE: DfUnload is the DataFrame of interest to compute the Unloading curve
        NOTE: InfoTraffic is the dictionary of interest to compute the Traffic Analysis
        """
        import json
        if not os.path.exists(self.complete_path_unloading_curve_base):
            logger.info(f"{self.city} Computing Best Fit Pl Exponential R: {self.R}, UCI: {self.UCI}")
            # 1) NOTE: Compute the Best Fit for the Power Law and Exponential
            self.DfUnload,ErrorExp,A0,alpha_exp,ErrorPL,A_pl,alpha_pl = ComparePlExpoDf(self.DfUnload,
                                                                                        maxfev = 10000)
            
            
            # 2) NOTE: Error & Fit
            self.InfoTraffic = FillInfoTraffic(ErrorExp, ErrorPL, A0, A_pl, alpha_exp, alpha_pl,self.city)
            # 3) NOTE: Compute Gamma and Tau
            self.ComputeGammaAndTau()
            self.InfoTraffic["Gamma"] = self.Gamma
            # NOTE: Writing InfoTraffic, DfUnload 
            with open(os.path.join(self.PlotDir,f"InfoTraffic_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'w') as f:
                json.dump(self.InfoTraffic,f,indent = 4)
            self.DfUnload.write_parquet(self.complete_path_unloading_curve_base)

        else:
            logger.info(f"{self.city} Reading InfoTraffic File R: {self.R}, UCI: {self.UCI}")
            with open(os.path.join(self.PlotDir,f"InfoTraffic_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'r') as f:
                self.InfoTraffic = json.load(f)
            self.InfoTraffic["Tau"] = - 1/self.InfoTraffic["alpha_exp"]
            self.DfUnload = pl.read_parquet(self.complete_path_unloading_curve_base)
### FLUXES AND GAMMA
    @timing_decorator
    def ComputeFluxesAndSpeedDfRoute(self):
        """
            @description:
                - Computes Fluxes And Average Speeds in The Road Network in Intervals Of Time of delta_t
                @ return self.EdgesWithFluxesAndSpeed: DataFrame with the fluxes and speeds in the road network
        """
        if os.path.exists(os.path.join(self.PlotDir,f"R_{self.R}_UCI_{self.UCI}_traffic.parquet")):
            logger.info(f"{self.city} Reading Traffic File R: {self.R}, UCI: {self.UCI}")
            self.EdgesWithFluxesAndSpeed = pl.read_parquet(os.path.join(self.PlotDir,f"R_{self.R}_UCI_{self.UCI}_traffic.parquet"))
        else:
            logger.info(f"{self.city} Computing Traffic File R: {self.R}, UCI: {self.UCI}")
            for t in range(len(self.IntTimeArray)):
                DfRoadsFluxes = self.DfRoute.with_columns(
                    pl.lit(0).alias(f"flux_{t}")
                )
                DfRoadsFluxes = DfRoadsFluxes.with_columns(
                    pl.lit(0).alias(f"speed_kmh_{t}")
                )

            for t in range(len(self.IntTimeArray)):
                ConditionUsersInRoad = (pl.col("time_departure") <= self.IntTimeArray[t]) & (pl.col("time_leaving_road") >= self.IntTimeArray[t])
                Df2Join = self.DfRoute.filter(ConditionUsersInRoad).group_by("route").agg([
                    (pl.col("avg_v(km/h)")).mean().alias(f"speed_kmh_{t}"),
                    pl.col("p").count().alias(f"flux_{t}")
                ])
                if Df2Join.is_empty(): 
                    pass
                else:
                    DfRoadsFluxes = DfRoadsFluxes.join(Df2Join, on="route", how="left")
            EdgesWithFluxesAndSpeed = self.Edges.join(DfRoadsFluxes, left_on='uniqueid', right_on='route', how='left')    
            EdgesWithFluxesAndSpeed = EdgesWithFluxesAndSpeed.unique(subset=["u","v"], keep='first')
            self.EdgesWithFluxesAndSpeed = DropColumnsDfIfThere(EdgesWithFluxesAndSpeed,["osmid_u","osmid_v","p","distance","init_intersection","end_intersection","time_departure","last_time_simulated","distance_km","avg_v(m/s)","time_leaving_road","length_km","avg_v(km/h)","length_right","u_right","v_right","capacity_right"])            
            self.EdgesWithFluxesAndSpeed.write_parquet(os.path.join(self.PlotDir,f"R_{self.R}_UCI_{self.UCI}_traffic.parquet"))




### TRAFFIC MEASURES ###
    @timing_decorator
    def ComputeGammaAndTau(self):
        """
            Compute the Gamma and Tau for the Route and People Files
            Returns:
                Gamma: float is time that the trafficked city would require to obtain
                    the integrated fluxes of maximum capacity
                Tau: float
        """
        logger.info(f"{self.city} Computing Gamma And Tau R: {self.R}, UCI: {self.UCI}")
        self.Gamma = ComputeGamma(self.DfControlGroup)
        if self.InfoTraffic["IsJam"]:
            nt0 = max(self.DfUnload["FractionPeople"].to_numpy())
            # Choose the index for which the fraction of people in the network is less than 1/e of the maximum
            self.IndexTau = np.where(self.DfUnload["FractionPeople"].to_numpy() <= nt0*np.exp(-1))[0][0]
            self.Tau = self.DfUnload["Time_hours_shifted"].to_numpy()[self.IndexTau]
            #self.InfoTraffic["Tau"] = self.Tau
        else:
            # NOTE: in this case Tau was already computed in DecideIfJam.
            pass




### NETWORK FEATURES ###
    



### PLOTS ####

    @timing_decorator
    def PlotUnloadCurve(self):
        """
            @Decsription:
                For each time window checks wether exponential is better than powerlaw. If powerlaw is better than exponential, then we have a jam.
            NOTE: The time is shifted to be close to 0 when it starts to decrease since the fit otherwhise would work.
        """
        if self.InfoTraffic["IsJam"]:
            y_fit = self.DfUnload["y_fit_pl"].to_numpy()
        else:
            y_fit = self.DfUnload["y_fit_exp"].to_numpy()
        logger.info(f"{self.city} Plot Unload Curve R: {self.R}, UCI: {self.UCI}")
        PlotPeopleInNetwork(self.DfUnload["FractionPeople"],
                            y_fit,self.DfUnload["Time_hours_shifted"],
                            self.InfoTraffic["best_fit"],
                            JoinDir(self.PlotDir,"UnloadingCurve_R_{0}_UCI_{1}.png".format(self.R,self.UCI)))
        logger.info(f"{self.city} Plotting dN/dt R: {self.R}, UCI: {self.UCI}")
#        PlotGt(self.DfUnload,
#               JoinDir(self.PlotDir,"dN_dt_R_{0}_UCI_{1}.png".format(self.R,self.UCI)))

    @timing_decorator
    def PlotNtAndFitSingleR(self):
        """
            Plots One of the curves in Fig 4 Paper Marta.
            @Description:
                LastTime is set to Plot all the data.

        """
        LastTime = list(self.Time2ErrorFit.keys())[-1]
        PlotNtAndFitSingleR(self.DfUnload["Time_hours"].to_numpy(),self.Time2nt[LastTime]["n"],self.Tau,self.Time2nt[LastTime]["n_fit"],self.R,self.UCI,self.PlotDir)


