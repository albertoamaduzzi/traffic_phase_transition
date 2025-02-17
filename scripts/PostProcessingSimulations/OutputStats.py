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
    def __init__(self,R,UCI,config,GeoJsonEdges,Nodes,Edges):
        
        self.verbose = True
        # Add Logic to Take All The Files From The Base Dir
        self.RouteFile = config[UCI][R]['route_file']
        self.PeopleFile = config[UCI][R]['people_file']
        self.R = R
        self.UCI = UCI
        self.city = config['name']
        self.delta_t = config['delta_t']
        self.PlotDir = JoinDir(config['output_simulation_dir'],'Plots')
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
        # TRAFFIC VARIABLES
        self.DfUnload = None
        self.DfControlGroup = None
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
        logger.info(f"{self.city} Reading People File R: {self.R}, UCI: {self.UCI}, City: {self.city}")
        if IsFile(self.PeopleFile):
            if ".parquet" in self.PeopleFile:
                self.DfPeople = pl.read_parquet(self.PeopleFile)
            elif ".csv" in self.PeopleFile:
                self.DfPeople = pl.read_csv(self.PeopleFile)
            self.DfPeople = self.DfPeople.filter(pl.col("time_departure")>self.t_start_hour*3600)
            self.ReadPeopleInfoBool = True
            self.CountFunctions += 1
            Message = f"Function {self.CountFunctions}: GetPeopleInfo: {self.PeopleFile} was read"
            AddMessageToLog(Message,self.LogFile)

#### Preprocess: Trajectories in Road Network  #####


    def GetRouteInfo(self):
        """
            @description:
                Read the Route File:
            NOTE: columns: p, route, distance
        """
        logger.info(f"{self.city} Reading Route File R: {self.R}, UCI: {self.UCI}, City: {self.city}")
        if IsFile(self.RouteFile):
            if ".parquet" in self.RouteFile:
                self.DfRoute = pl.read_parquet(self.RouteFile)
            elif ".csv" in self.RouteFile:
                self.DfRoute = pl.read_csv(self.RouteFile,separator = ':')
                if self.DfRoute.schema["route"] == pl.Utf8:
                    pass
            self.ReadRouteInfoBool = True
            self.CountFunctions += 1
            Message = f"Function {self.CountFunctions}: GetRouteInfo: {self.RouteFile} was read"
            AddMessageToLog(Message,self.LogFile)





    @timing_decorator
    def CompleteAnalysis(self):
        """
            This is the Function one wants to Call to get the Analysis of the Simulation
        """
        if not self.SkipAnalysisR:
            # Join DfPeople, Edges And DfRoute 
            self.HavlinPercolation = False
            if self.HavlinPercolation:
                self.PreprocessDfRouteUnionDfPeople() # It consumes a lot of resources and it is not actually needed for our DP pourposes
            # Compute The Unload Curve
            self.ComputeUnloadCurve()   # Output: DfUnload
            # Instantiate All The Variables To Understand If The Network Is Jammed
            if self.IsEnoughData:
                # Compute The Best Fit For The Power Law And Exponential
                self.ComputeBestFitPlExpo() 
                # Decide If The Network Is Jammed NOTE: (sel.TCritical, self.CriticalNt, self.CriticalAlpha)
    #            self.DecideIfJam()
                # Compute The Gamma And Tau
                self.ComputeGammaAndTau()
                # Show The Unload Curve
                self.PlotUnloadCurve()
                # Compute The Fluxes And Speeds in time
    #            self.ComputeFluxesAndSpeedDfRoute()
                # Percolation Analysis
    #            AnalysisPercolationSpeed(self.IntTimeArray,self.GeoJsonEdges,self.EdgesWithFluxesAndSpeed,self.UCI,self.R,self.PlotDir)
            else:
                pass
        else:
            # Import the Unloading Curve, The InfoTraffic and the DfUnload
            import json
            with open(os.path.join(self.PlotDir,f"InfoTraffic_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'r') as f:
                self.InfoTraffic = json.load(f)
            self.DfUnload = pl.read_parquet(JoinDir(self.PlotDir,"UnloadCurve_R_{0}_UCI_{1}.parquet".format(self.R,self.UCI)))
            logger.info(f"{self.city} Skip Analysis for UCI: {round(self.UCI,3)}, R: {self.R}")



## PREPROCESSING ##
    def PreprocessDfRouteUnionDfPeople(self):
        """
            @description:
                Preprocess the DfRoute with the DfPeople  and the Edges
                NOTE: This Object is The Main Object We are Going to Use To Compute Everything about the Traffic
                NOTE: columns
                - p: User Id
                - distance: Distance Travelled By User
                - init_intersection: Origin (in uniqueid Nodes index)
                - end_intersection: Destination (in uniqueid Nodes index)
                - time_departure: Time of Departure (in seconds)	
                - last_time_simulated: Time of Arrival (in seconds)
                - route: List of Edges in the route
                - avg_v(km/h): Average Velocity of the User (in km/h)
                - osmid_u: Index Node Front in Osmid Code	
                - osmid_v: Index Node Back in Osmid Code
                - length: Length of the Road (in m)	
                - u: Index Node Front in Uniqueid Code
                - v: Index Node Back in Uniqueid Code
                - length_km: Length of the Road (in km)
                - distance_km: Distance Travelled By User (in km)
                - avg_v(m/s): Average Velocity of the User (in m/s)
                - time_leaving_road: Time of Leaving the Road (in seconds)
                NOTE: See EmbdedTrajectoriesInRoadsAndTime
        """
        logger.info(f"{self.city} Preprocessing DfRoute Union DfPeople R: {self.R}, UCI: {self.UCI}, City: {self.city}")
        # Prepare the DataFrame: NOTE: Compute the Fluxes and Speeds is Deprecated
        if self.HavlinPercolation:
            self.DfRoute = EmbdedTrajectoriesInRoadsAndTime(self.DfRoute,self.DfPeople,self.Edges)
        else:
            # This one contains informations just about the control group, NOTE: the logic is sloppy but should work
            self.DfRoute = EmbdedTrajectoriesInRoadsAndTime(self.DfRoute,self.DfControlGroup,self.Edges)
            

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
        """
        logger.info(f"{self.city} Computing Unload Curve R: {self.R}, UCI: {self.UCI}")
        self.IntTimeArray, self.HourTimeArray, self.Interval2NumberPeopleInNet = InitializeTimeVariablesOutputStats(HOURS_IN_DAY,MINUTES_IN_HOUR,SECONDS_IN_MINUTE,TIMESTAMP_OFFSET,self.delta_t)
        if self.HavlinPercolation:
            self.DfRoute = EmbdedTrajectoriesInRoadsAndTime(self.DfRoute,self.DfPeople,self.Edges)            
            self.DfControlGroup = FilterDfPeopleControlGroup(self.t_start_control_group,self.t_end_control_group,self.DfRoute,"time_departure","last_time_simulated")
        else:
            self.DfControlGroup = FilterDfPeopleControlGroup(self.t_start_control_group,self.t_end_control_group,self.DfPeople,"time_departure","last_time_simulated")
            self.DfControlGroup = EmbdedTrajectoriesInRoadsAndTime(self.DfRoute,self.DfControlGroup,self.Edges)

        if not os.path.exists(JoinDir(self.PlotDir,"UnloadCurve_R_{0}_UCI_{1}.parquet".format(self.R,self.UCI))):
            # Initialize Time Variables IntTimeArray = [0,delta_t,2*delta_t,...,SecondsInDay], HourTimeArray = [0,1,2,...,24], Interval2NumberPeopleInNet = {0:0,1:0,...,24:0}
            # NOTE: Filter the People in the Control Group  
            for t in range(len(self.Interval2NumberPeopleInNet.keys())-1):
                # DataFrame Filtered With People in the Network at time t in the time interval t,t+1
                DfPeopleInNetAtTimet = FilterDfPeopleStilInNet(self.IntTimeArray[t],self.IntTimeArray[t+1],"last_time_simulated",self.DfControlGroup)
                # Number Of People in the Network at time t
                self.Interval2NumberPeopleInNet[self.HourTimeArray[t]] += len(DfPeopleInNetAtTimet) 
            data = {
                "Time": list(self.IntTimeArray[:-1]),
                "Time_Str": list(self.Interval2NumberPeopleInNet.keys()),
                "NumberPeople": list(self.Interval2NumberPeopleInNet.values())
            }
            self.DfUnload = pl.DataFrame(data)
            self.VolumeTraffic = np.sum(self.DfUnload["NumberPeople"].to_numpy())
            self.DfUnload = self.DfUnload.with_columns((pl.col("NumberPeople")/self.VolumeTraffic).alias("FractionPeople"),
                                                       (pl.col("NumberPeople").diff().shift(1).fill_null(0)).alias("DiffNumberPeople")
                                                       )            
            # Add Time_hours column
            self.DfUnload = self.DfUnload.with_columns(
                (pl.col("Time") / 3600).alias("Time_hours")
            )
            self.DfUnload = self.DfUnload.filter(pl.col("Time") > self.t_start_control_group)
            self.DfUnload = self.DfUnload.with_columns((pl.col("Time_hours") - self.t_start_hour).alias("Time_hours_shifted"))
            self.DfUnload = self.DfUnload.with_columns((pl.col("FractionPeople") > 0).alias("IsPeople"))
            self.DfUnload = self.DfUnload.with_columns((pl.col("Time") >= self.t_end_control_group).alias("IsUnload"))
        else:
            self.DfUnload = pl.read_parquet(JoinDir(self.PlotDir,"UnloadCurve_R_{0}_UCI_{1}.parquet".format(self.R,self.UCI)))
            self.HourTimeArray = self.DfUnload["Time_Str"]
            self.Interval2NumberPeopleInNet = self.DfUnload["NumberPeople"]
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
        if not os.path.exists(JoinDir(self.PlotDir,"UnloadCurve_R_{0}_UCI_{1}.parquet".format(self.R,self.UCI))):
            logger.info(f"{self.city} Computing Best Fit Pl Exponential R: {self.R}, UCI: {self.UCI}")
            # Consider Just the period after the load finishes.
            # Consider just when traffic is there             
            self.DfUnload,ErrorExp,A0,alpha_exp,ErrorPL,A_pl,alpha_pl = ComparePlExpoDf(self.DfUnload,
                                                                                        maxfev = 10000)
            
            
            # Error & Fit
            self.InfoTraffic["ErrorExp"] = ErrorExp
            self.InfoTraffic["ErrorPl"] = ErrorPL
            self.InfoTraffic["A_exp"] = A0
            self.InfoTraffic["A_pl"] = A_pl
            self.InfoTraffic["alpha_exp"] = alpha_exp
            self.InfoTraffic["alpha_pl"] = alpha_pl
            if ErrorExp<ErrorPL:
                self.InfoTraffic["best_fit"] = "Exponential"
                self.InfoTraffic["IsJam"] = False
                self.InfoTraffic["Tau"] = - 1/self.InfoTraffic["alpha_exp"]       # In hours
                logger.info("{0} Network is not jammed at time with alpha = None and Tau = {1}".format(self.city,self.InfoTraffic["Tau"]))
            else:
                self.InfoTraffic["best_fit"] = "PowerLaw"
                self.InfoTraffic["IsJam"] = True
                self.InfoTraffic["Tau"] = - 1/self.InfoTraffic["alpha_exp"]
                logger.info(f"{self.city} Network is jammed with alpha = {alpha_pl} and Tau = None")
            self.ComputeGammaAndTau()
            self.InfoTraffic["Gamma"] = self.Gamma
            # Save 
            with open(os.path.join(self.PlotDir,f"InfoTraffic_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'w') as f:
                json.dump(self.InfoTraffic,f,indent = 4)
            self.DfUnload.write_parquet(JoinDir(self.PlotDir,"UnloadCurve_R_{0}_UCI_{1}.parquet".format(self.R,self.UCI)))

        else:
            logger.info(f"{self.city} Reading InfoTraffic File R: {self.R}, UCI: {self.UCI}")
            with open(os.path.join(self.PlotDir,f"InfoTraffic_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'r') as f:
                self.InfoTraffic = json.load(f)
            self.InfoTraffic["Tau"] = - 1/self.InfoTraffic["alpha_exp"]
            self.DfUnload = pl.read_parquet(JoinDir(self.PlotDir,"UnloadCurve_R_{0}_UCI_{1}.parquet".format(self.R,self.UCI)))
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
        PlotGt(self.DfUnload,
               JoinDir(self.PlotDir,"dN_dt_R_{0}_UCI_{1}.png".format(self.R,self.UCI)))

    @timing_decorator
    def PlotNtAndFitSingleR(self):
        """
            Plots One of the curves in Fig 4 Paper Marta.
            @Description:
                LastTime is set to Plot all the data.

        """
        LastTime = list(self.Time2ErrorFit.keys())[-1]
        PlotNtAndFitSingleR(self.DfUnload["Time_hours"].to_numpy(),self.Time2nt[LastTime]["n"],self.Tau,self.Time2nt[LastTime]["n_fit"],self.R,self.UCI,self.PlotDir)


