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
        self.complete_path_unloading_curve_iter = Df_City2Config["complete_path_unloading_curve_iter"].to_numpy()
        self.iters = Df_City2Config['iter'].to_numpy()
        # unique variables
        self.R = R
        self.UCI = UCI
        self.city = Df_City2Config['name'].unique().to_numpy()[0]
        self.delta_t = Df_City2Config['delta_t'].unique().to_numpy()[0]
        self.complete_path_unloading_curve_base =  Df_City2Config["complete_path_unloading_curve_base"].unique().to_numpy()[0]
        self.dir_output_simulation_analysis = JoinDir(Df_City2Config['dir_output_simulation_analysis'].unique().to_numpy()[0],'Plots')
        self.PlotDir = JoinDir(Df_City2Config['dir_output_simulation_analysis'].unique().to_numpy()[0],'Plots')
        self.PlotDir = JoinDir(self.PlotDir,f'{round(self.UCI,3)}')
        self.LogFile = JoinDir(self.PlotDir,"Log.txt")
        self.CountFunctions = 0
        logger.info(f"OutputStats: {self.city} R: {self.R}, UCI: {self.UCI}, Delta_t: {self.delta_t}, PlotDir: {self.PlotDir}")
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
        # Roads and People
        self.iter_2_DfRoute = {iter_: None for iter_ in self.iters}
        self.iter_2_DfPeople = {iter_: None for iter_ in self.iters}
        # TRAFFIC VARIABLES
        self.iter_2_DfUnload = {iter_: None for iter_ in self.iters}
        self.iter_2_DfControlGroup = {iter_: None for iter_ in self.iters}

        # NOTE: fixing the problem arising by lack of memory -> check availability
        self.iter_2_is_available_locally_DfPeople = {iter_: False for iter_ in self.iters}
        self.iter_2_is_available_locally_DfRoute = {iter_: False for iter_ in self.iters}
        self.iter_2_is_available_locally_DfUnload = {iter_: False for iter_ in self.iters}
        self.iter_2_is_compute_DfUnload = {iter_: False for iter_ in self.iters}                                            # Do we have DfRoute,DfPeople? Therefore I can compute DfUnload per iter?
        self.iter_2_is_already_computed_DfUnload = {iter_: False for iter_ in self.iters}                                   # Have I already computed the DfUnload for this iter?
        # 

        self.DfUnload = None
        self.DfControlGroup = None
        self.iter_2_VolumeTraffic = {iter_: None for iter_ in self.iters}
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
        ## INITIALIZATION
        # DataFrames
        self.GetPeopleInfo()
        self.GetRouteInfo()    
        self.GetGeopandas(GeoJsonEdges)
        self.Nodes = Nodes
        self.Edges = Edges
        # Edges and Nodes .csv
        self.AddCapacity2Edges()
        self.ConvertSpeedRoad2kmhInEdges()
        # Error Fit Expo, Power Law for the curve n_R(t)
        self.err_exp_nt_R = 0
        self.err_pl_nt_R = 0

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
                NOTE: I also add a condtion in: iter_2_is_available_locally_DfPeople[iter_] = False -> check this to decide when to compute the unload curve
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
                self.iter_2_DfPeople[iter_] = self.iter_2_DfPeople[iter_].filter(pl.col("time_departure")>=self.t_start_hour*3600)
                logger.info(f"{self.city}: GetPeopleInfo: {PeopleFile} was read, updated iter_2_DfPeople")
                self.iter_2_is_available_locally_DfPeople[iter_] = True
            else:
                logger.error(f"{self.city}: GetPeopleInfo: {PeopleFile} was not read, updated iter_2_DfPeople")
                self.iter_2_DfPeople[iter_] = None
                self.iter_2_is_available_locally_DfPeople[iter_] = False
            count_people_file += 1
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
                self.iter_2_is_available_locally_DfRoute[iter_] = True
                self.ReadRouteInfoBool = True
                self.CountFunctions += 1
                logger.info(f"{self.city}: GetRouteInfo: {RouteFile} was read -> updated iter_2_DfRoute")
            else:
                logger.error(f"{self.city}: GetRouteInfo: {RouteFile} was not read -> updated iter_2_DfRoute")
                self.iter_2_DfRoute[iter_] = None
                self.iter_2_is_available_locally_DfRoute[iter_] = False
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
#                self.ComputeGammaAndTau()
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
            self.DfUnload = pl.read_parquet(os.path.join(self.PlotDir,f"UnloadCurve_R_{self.R}_UCI_{round(self.UCI,3)}.parquet"))
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


            NOTE:
            Once I decide to compute and save the new data I must also decide to move data back to savio
        """
        from json import dump,load
        logger.info(f"{self.city} Computing Unload Curve R: {self.R}, UCI: {self.UCI}")
        # 1) Initialize time variables
        self.IntTimeArray, self.HourTimeArray, self.Interval2NumberPeopleInNet = InitializeTimeVariablesOutputStats(HOURS_IN_DAY,MINUTES_IN_HOUR,SECONDS_IN_MINUTE,TIMESTAMP_OFFSET,self.delta_t)
        # 2) NOTE: Iter over ensemble realizations
        for iter_ in self.iters:
            logger.info(f"{self.city} iter_: {iter_}")
            # NOTE: RouteFile and People file are available for iter -> iter_2_is_compute will play the role of a mask
            self.iter_2_is_compute_DfUnload[iter_] = self.iter_2_is_available_locally_DfRoute[iter_] and self.iter_2_is_available_locally_DfPeople[iter_] 
            self.complete_path_unloading_curve_iter[iter_] = os.path.join(self.dir_output_simulation_analysis,str(self.R),f"UnloadCurve_R_{self.R}_UCI_{round(self.UCI,3)}_iter_{iter_}.parquet")#os.path.join(os.path.dirname(self.complete_path_unloading_curve_iter[iter_]),f"UnloadCurve_R_{self.R}_UCI_{round(self.UCI,3)}_iter_{iter_}.parquet")
            self.iter_2_is_already_computed_DfUnload[iter_] = os.path.exists(self.complete_path_unloading_curve_iter[iter_])

###################### UPLOADING ITER UNLOAD CURVE ##########################            
            # NOTE: Upload the Unload Curve if it is already computed
            if self.iter_2_is_already_computed_DfUnload[iter_]:
                logger.info(f"{self.city} Uploading Unload Curve R: {self.R}, UCI: {self.UCI}, UnloadCurve_iter: {self.complete_path_unloading_curve_iter[iter_]}")
                self.iter_2_DfUnload[iter_] = pl.read_parquet(self.complete_path_unloading_curve_iter[iter_])
                self.iter_2_VolumeTraffic[iter_] = self.iter_2_DfUnload[iter_]["NumberPeople"].sum()
                self.HourTimeArray = self.iter_2_DfUnload[iter_]["Time_Str"]
                self.Interval2NumberPeopleInNet = dict(zip(
                    self.iter_2_DfUnload[iter_]["Time_hours"], 
                    self.iter_2_DfUnload[iter_]["NumberPeople"]
                ))
                self.iter_2_is_available_locally_DfUnload[iter_] = True
########################## COMPUTING ITER UNLOAD CURVE ##########################
            # NOTE: Compute the Unload Curve if it is not already computed
            elif self.iter_2_is_compute_DfUnload[iter_]:                                          # Compute iter_2_DfUnload[iter_] just if you have the right files
                logger.info(f"{self.city} Computing Unload Curve R: {self.R}, UCI: {self.UCI}, UnloadCurve_iter: {self.complete_path_unloading_curve_iter[iter_]}")
                DfRoute = self.iter_2_DfRoute[iter_]
                DfPeople = self.iter_2_DfPeople[iter_]
                DfPeople = DfPeople.sort("time_departure")
                logger.info(f"{self.city}, {self.R}, {self.UCI} DfRoute just uploaded: {DfRoute.head()}")
                logger.info(f"{self.city}, {self.R}, {self.UCI} DfPeople just uploaded: {DfPeople.head()}")
                # 3) NOTE: Filter the People in the Control Group
                if self.HavlinPercolation:
                    DfRoute = EmbdedTrajectoriesInRoadsAndTime(DfRoute,DfPeople,self.Edges)            
                    DfControlGroup = FilterDfPeopleControlGroup(self.t_start_control_group,
                                                                self.t_end_control_group,
                                                                DfRoute,
                                                                "time_departure",
                                                                "last_time_simulated")
                else:
#                    print(f"{self.city}, {self.R}, {self.UCI}, Before filtering 1, DfControlGroup:\n",DfPeople.head()," time start: ",self.t_start_control_group," time end: ",self.t_end_control_group)
                    DfControlGroup = FilterDfPeopleControlGroup(self.t_start_control_group,
                                                                self.t_end_control_group,
                                                                DfPeople,
                                                                "time_departure",
                                                                "last_time_simulated")
#                    print(f"{self.city}, {self.R}, {self.UCI}, After filtering 1, DfControlGroup:\n",DfControlGroup.head())
                    DfControlGroup = EmbdedTrajectoriesInRoadsAndTime(DfRoute,
                                                                    DfControlGroup,
                                                                    self.Edges)
                
                logger.info(f"{self.city} Control group prepared")
                self.iter_2_DfControlGroup[iter_] = DfControlGroup
#                print(f"{self.city}, {self.R}, {self.UCI}, DfControlGroup:\n{DfControlGroup.head()}")
                assert os.path.exists(self.complete_path_unloading_curve_iter[iter_]) or DfControlGroup is not None, f"DfControlGroup is None for iter_: {iter_} and UnloadCurve_iter: {self.complete_path_unloading_curve_iter[iter_]} does not exist, I cannot compute the UnloadCurve_iter, and I cannot upload it either"
                # 4) NOTE: Compute the Unload Curve
                # NOTE: Add here the clause that the simulation file is on the system: NOTE -> they well not be in the system since I moved them for problems in memory
                print(f"{self.city}, {self.R}, {self.UCI}, IntTimeArray:\n{self.IntTimeArray}")
                print(f"{self.city}, {self.R}, {self.UCI}, HourTimeArray:\n{self.HourTimeArray}")
                VolumeTraffic, DfUnload = ComputeSingleUnloadCurve(self.Interval2NumberPeopleInNet,
                                                                    self.IntTimeArray,
                                                                    self.HourTimeArray,
                                                                    self.t_start_control_group,
                                                                    self.t_start_hour,
                                                                    self.t_end_control_group,
                                                                    self.iter_2_DfControlGroup[iter_])  
#                print(f"DfUnload {self.R}, {self.UCI}:\n{DfUnload.head()}")
                self.iter_2_VolumeTraffic[iter_] = VolumeTraffic
                self.iter_2_DfUnload[iter_] = DfUnload
                os.makedirs(os.path.dirname(self.complete_path_unloading_curve_iter[iter_]),exist_ok = True)
                logger.info(f"{self.city} Writing Unload Curve R: {self.R}, UCI: {self.UCI}, UnloadCurve_iter: {self.complete_path_unloading_curve_iter[iter_]}")
                DfUnload.write_parquet(self.complete_path_unloading_curve_iter[iter_])
#                with open(self.complete_path_unloading_curve_iter[iter_],'w') as f:
#                    dump(self.Interval2NumberPeopleInNet,f,indent = 4)
                self.iter_2_is_available_locally_DfUnload[iter_] = True
            else:
                logger.info(f"{self.city} Unload Curve R: {self.R}, UCI: {self.UCI}, UnloadCurve_iter: {self.complete_path_unloading_curve_iter[iter_]} does not exist, I cannot compute the UnloadCurve_iter, and I cannot upload it either")
                self.iter_2_DfUnload[iter_] = None
                self.iter_2_is_available_locally_DfUnload[iter_] = False
##################### COMPUTING AVERAGE UNLOAD CURVE ##########################
            if os.path.isfile(os.path.join(self.PlotDir,f"VolumeTraffic_R_{self.R}_UCI_{round(self.UCI,3)}.json")):
                with open(os.path.join(self.PlotDir,f"VolumeTraffic_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'r') as f:
                    VolumeTrafficPreComputed = load(f)
            else:
                VolumeTrafficPreComputed = None
                pass
        self.VolumeTraffic = ComputeAverageVolumeTraffic(self.iter_2_VolumeTraffic,
                                                            self.iter_2_is_available_locally_DfUnload,
                                                            VolumeTrafficPreComputed)
        # 5) NOTE: Compute the average Unloading curve and Volume of traffic
        if os.path.exists(os.path.join(self.PlotDir,f"UnloadCurve_R_{self.R}_UCI_{round(self.UCI,3)}.parquet")):
            logger.info(f"{self.city} Computing Unload Curve R: {self.R}, UCI: {self.UCI}, Unload.parquet exists = {os.path.exists(os.path.join(self.PlotDir,f"UnloadCurve_R_{self.R}_UCI_{round(self.UCI,3)}.parquet"))}")
            DfUnloadPrecomputed = pl.read_parquet(os.path.join(self.PlotDir,f"UnloadCurve_R_{self.R}_UCI_{round(self.UCI,3)}.parquet"))
        else:
            DfUnloadPrecomputed = None
            pass
        logger.info(f"{self.city} Computing Average Unloading Curve R: {self.R}, UCI: {self.UCI}, Unload.parquet ")
        self.DfUnload = ComputeAverageUnloadCurve(self.iter_2_DfUnload,
                                                self.t_end_control_group,
                                                self.iter_2_is_available_locally_DfUnload,
                                                DfUnloadPrecomputed,
                                                os.path.join(self.PlotDir,f"UnloadCurve_R_{self.R}_UCI_{round(self.UCI,3)}.parquet")
                                                )
        # 6) NOTE: Writing DfUnload
        if not os.path.exists(os.path.join(self.PlotDir,f"UnloadCurve_R_{self.R}_UCI_{round(self.UCI,3)}.parquet")):
            self.DfUnload.write_parquet(os.path.join(self.PlotDir,f"UnloadCurve_R_{self.R}_UCI_{round(self.UCI,3)}.parquet"))
            with open(os.path.join(self.PlotDir,f"VolumeTraffic_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'w') as f:
                dump(self.VolumeTraffic,f,indent = 4)
        else:
            pass    

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
#        if not os.path.exists(os.path.join(self.PlotDir,f"InfoTraffic_R_{self.R}_UCI_{round(self.UCI,3)}.json")):
        logger.info(f"{self.city} Computing Best Fit Pl Exponential R: {self.R}, UCI: {self.UCI}")
        # 1) NOTE: Compute the Best Fit for the Power Law and Exponential
        # Fit the exponential and power law        
        self.DfUnload,ErrorExp,A0,alpha_exp,ErrorPL,A_pl,alpha_pl = ComparePlExpoDf(self.DfUnload,
                                                                                    init_time_hour_cut_for_fit = 1,
                                                                                    minimum_time_with_people= 10,  
                                                                                    max_time_for_fit = 7,                                                                                  
                                                                                    maxfev = 10000)
        
        logger.info(f"{self.city} Filling info traffic R: {self.R}, UCI: {self.UCI}")
        # 2) NOTE: Error & Fit
        self.InfoTraffic = FillInfoTraffic(ErrorExp, ErrorPL, A0, A_pl, alpha_exp, alpha_pl,self.city)
        logger.info(f"{self.city} Computing Gamma And Tau R: {self.R}, UCI: {self.UCI}")   
        # 3) NOTE: Compute Gamma and Tau
        self.ComputeGammaAndTau()
        self.InfoTraffic["Gamma"] = self.Gamma
        logger.info(f"{self.city} saving info traffic")
        # NOTE: Writing InfoTraffic, DfUnload 
        with open(os.path.join(self.PlotDir,f"InfoTraffic_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'w') as f:
            json.dump(self.InfoTraffic,f,indent = 4)
        
        # 4) NOTE: take care about the errors in fit to later see how they evolve with R 
        self.err_exp_nt_R = ErrorExp
        self.err_pl_nt_R = ErrorPL
            

#        else:
#            logger.info(f"{self.city} Reading InfoTraffic File R: {self.R}, UCI: {self.UCI}")
#            with open(os.path.join(self.PlotDir,f"InfoTraffic_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'r') as f:
#                self.InfoTraffic = json.load(f)
#            self.InfoTraffic["Tau"] = - 1/self.InfoTraffic["alpha_exp"]
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
        from json import load
        logger.info(f"{self.city} Computing Gamma And Tau R: {self.R}, UCI: {self.UCI}")
        if os.path.exists(os.path.join(self.PlotDir,f"InfoTraffic_R_{self.R}_UCI_{round(self.UCI,3)}.json")):
            logger.info(f"{self.city} Reading InfoTraffic File R: {self.R}, UCI: {self.UCI}")
            with open(os.path.join(self.PlotDir,f"InfoTraffic_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'r') as f:
                self.InfoTraffic = load(f)
            self.Gamma = self.InfoTraffic["Gamma"]
            self.Tau = self.InfoTraffic["Tau"]
        else:
            count_iters = 0
            Gamma = 0
            for iter_ in self.iters:
                if self.iter_2_is_available_locally_DfRoute[iter_] and self.iter_2_is_available_locally_DfPeople[iter_]: 
                    DfRoute = self.iter_2_DfRoute[iter_]
                    DfPeople = self.iter_2_DfPeople[iter_]
                    DfPeople = DfPeople.sort("time_departure")

                    DfControlGroup = FilterDfPeopleControlGroup(self.t_start_control_group,
                                                                self.t_end_control_group,
                                                                DfPeople,
                                                                "time_departure",
                                                                "last_time_simulated")
    #                    print(f"{self.city}, {self.R}, {self.UCI}, After filtering 1, DfControlGroup:\n",DfControlGroup.head())
                    DfControlGroup = EmbdedTrajectoriesInRoadsAndTime(DfRoute,
                                                                    DfControlGroup,
                                                                    self.Edges)
                
                logger.info(f"{self.city} Control group prepared")
                self.iter_2_DfControlGroup[iter_] = DfControlGroup

                if self.iter_2_DfControlGroup[iter_] is not None:
                    logger.info(f"{self.city} Computing Gamma And Tau R: {self.R}, UCI: {self.UCI}, iter_: {iter_}")
                    # NOTE: Handling the strange error that occurs whenthe simulation has not started R = 210, Realization = 1 UCI = 0.2
                    try:
                        Gamma += ComputeGamma(self.iter_2_DfControlGroup[iter_])
                        count_iters += 1
                    except ZeroDivisionError:
                        logger.error(f"{self.city} ZeroDivisionError Gamma R: {self.R}, UCI: {self.UCI}, iter_: {iter_}")
                        Gamma += 0
                else:
                    logger.info(f"{self.city} No Gamma And Tau R: {self.R}, UCI: {self.UCI}, iter_: {iter_}")
                    pass
            self.Gamma = Gamma/count_iters
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


