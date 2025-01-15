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
    def __init__(self,R,UCI,config,GeoJsonEdges):
        
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
        os.makedirs(self.PlotDir,exist_ok = True)
        # Flags
        self.ReadPeopleInfoBool = False
        self.ReadRouteInfoBool = False
        self.GetGeopandasBool = False
        # DataFrames
        self.GetPeopleInfo()
        self.GetRouteInfo()    
        self.GetGeopandas(GeoJsonEdges)
        # Edges and Nodes .csv
        self.Nodes = pl.read_csv(os.path.join(BERKELEY_DIR,self.city,'nodes.csv'))
        self.Edges = pl.read_csv(os.path.join(BERKELEY_DIR,self.city,'edges.csv'))
        self.AddCapacity2Edges()
        self.ConvertSpeedRoad2kmhInEdges()
        # Group Control
        self.t_start_control_group = 7*3600
        self.t_end_control_group = 8*3600
        # TRAFFIC VARIABLES
        self.DfUnload = None
        self.DfControlGroup = None
        # Total Volume Traffic
        self.VolumeTraffic = 0
        # Is Jam (Variable To Encode The Fact That PowerLaw Is Better Then Exponential, after Some Point)
        self.IsJam = False
    @timing_decorator
    def CompleteAnalysis(self):
        """
            This is the Function one wants to Call to get the Analysis of the Simulation
        """
        # Join DfPeople, Edges And DfRoute
        self.PreprocessDfRouteUnionDfPeople()
        # Compute The Unload Curve
        self.ComputeUnloadCurve()   # Output: DfUnload
        # Instantiate All The Variables To Understand If The Network Is Jammed
        self.ComputeBestFitPlExpo() 
        # Decide If The Network Is Jammed NOTE: (sel.TCritical, self.CriticalNt, self.CriticalAlpha)
        self.DecideIfJam()
        # Compute The Gamma And Tau
        self.ComputeGammaAndTau()
        # Show The Unload Curve
        self.PlotUnloadCurve()
        # Compute The Fluxes And Speeds in time
        self.ComputeFluxesAndSpeedDfRoute()
        # Percolation Analysis
        AnalysisPercolationSpeed(self.IntTimeArray,self.GeoJsonEdges,self.EdgesWithFluxesAndSpeed,self.UCI,self.R,self.PlotDir)

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

    def GetPeopleInfo(self):
        """
            @description:
                Read the People File:
                NOTE: columns: p, distance, time_departure, last_time_simulated, num_steps, avg_v(mph), a, b, T, gas, co, path_length_cpu, path_length_gpu
        """
        logger.info(f"Reading People File R: {self.R}, UCI: {self.UCI}, City: {self.city}")
        if IsFile(self.PeopleFile):
            if ".parquet" in self.PeopleFile:
                self.DfPeople = pl.read_parquet(self.PeopleFile)
            elif ".csv" in self.PeopleFile:
                self.DfPeople = pl.read_csv(self.PeopleFile)
            self.ReadPeopleInfoBool = True
            self.CountFunctions += 1
            Message = f"Function {self.CountFunctions}: GetPeopleInfo: {self.PeopleFile} was read"
            AddMessageToLog(Message,self.LogFile)



    def GetRouteInfo(self):
        """
            @description:
                Read the Route File:
            NOTE: columns: p, route, distance
        """
        logger.info(f"Reading Route File R: {self.R}, UCI: {self.UCI}, City: {self.city}")
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


    def GetGeopandas(self,GeoJsonEdges):
        logger.info(f"Reading GeoJsonEdges R: {self.R}, UCI: {self.UCI}, City: {self.city}")
        self.GeoJsonEdges = GeoJsonEdges
        self.GetGeopandasBool
        self.CountFunctions += 1
        Message = f"Function {self.CountFunctions}: GetGeopandas: GeoJsonEdges was read"
        AddMessageToLog(Message,self.LogFile)


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
        logger.info(f"Preprocessing DfRoute Union DfPeople R: {self.R}, UCI: {self.UCI}, City: {self.city}")
        # Prepare the DataFrame: NOTE: Compute the Fluxes and Speeds is Deprecated
        self.DfRoute = EmbdedTrajectoriesInRoadsAndTime(self.DfRoute,self.DfPeople,self.Edges)


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
            NOTE: self.DfUnload: DataFrame
        """
        logger.info(f"Computing Unload Curve R: {self.R}, UCI: {self.UCI}")
        self.IntTimeArray, self.HourTimeArray, self.Interval2NumberPeopleInNet = InitializeTimeVariablesOutputStats(HOURS_IN_DAY,MINUTES_IN_HOUR,SECONDS_IN_MINUTE,TIMESTAMP_OFFSET,self.delta_t)
        self.DfControlGroup = FilterDfPeopleControlGroup(self.t_start_control_group,self.t_end_control_group,self.DfRoute,"time_departure","last_time_simulated")
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
            self.DfUnload = self.DfUnload.with_columns((pl.col("NumberPeople")/self.VolumeTraffic).alias("FractionPeople"))            
            # Add Time_hours column
            self.DfUnload = self.DfUnload.with_columns(
                (pl.col("Time") / 3600).alias("Time_hours")
            )
            self.DfUnload.write_parquet(JoinDir(self.PlotDir,"UnloadCurve_R_{0}_UCI_{1}.parquet".format(self.R,self.UCI)))
        else:
            self.DfUnload = pl.read_parquet(JoinDir(self.PlotDir,"UnloadCurve_R_{0}_UCI_{1}.parquet".format(self.R,self.UCI)))
            self.HourTimeArray = self.DfUnload["Time_Str"]
            self.Interval2NumberPeopleInNet = self.DfUnload["NumberPeople"]
    @timing_decorator
    def ComputeBestFitPlExpo(self):
        """
            Compute the best fit for the Power Law and Exponential
            Returns:
                ErrorExp: float
                A0: float
                tau: float
                PowerLawFitted: bool
            self.Time2ErrorFit: dict -> {t:{"error_pl":0,"error_exp":0}}
            I want t extract informations about the best fit for the power law and the exponential
            NOTE: 
                The nt is cut in the time window. The cut is done in hours
        """
        import json
        if not os.path.isfile(os.path.join(self.PlotDir,f"Time2ErrorFit_R_{self.R}_UCI_{round(self.UCI,3)}.json")):
            t_vect = self.DfUnload["Time_hours"].to_numpy()
            nt = self.DfUnload["NumberPeople"].to_numpy()
            Z_n = np.sum(nt)
            mask = nt>0
            nt = nt[mask]
            t_vect = t_vect[mask]
            # # Look at just the last two, if one of the two is a powerlaw, then I have a powerlaw
            if len(t_vect[4:])>= 4:
                self.RangePlFit = range(9,len(t_vect))   
                # Check The t For Which The Power Law Fit the Best and Is Better Than The Exponential
                self.Time2ErrorFit,self.Time2BestFit,self.Time2BestFit,self.Time2nt,self.Time2Fit = InitializeDictionariesFit(t_vect,nt,Z_n,self.RangePlFit)
                for t0 in self.RangePlFit:
                    # Constraint to have a minimum of 1 hour of observation
                    if t0 - 4 >= 6:
                        t = t_vect[t0]
                        # Consider 1 hour before fitting, NOTE: Scale the time to close to 0
                        ErrorExp,A0,alpha_exp,y_fit_exp,ErrorPL,A_pl,alpha_pl,y_fit_pl = ComparePlExpo(np.array(t_vect[4:t0])-np.array(t_vect[3]),nt[3:t0],initial_guess_powerlaw = (max(nt),-1), initial_guess_expo = (max(nt),-1),maxfev = 10000)
                        # Standardize for a DataFrame the y_fit
                        if len(y_fit_exp) != 0:
                            y_fit_exp = np.append(y_fit_exp,np.zeros(len(nt)-len(y_fit_exp)))
                            MultiplicativeFactor = nt[4]/y_fit_exp[0]
                            y_fit_exp = y_fit_exp*MultiplicativeFactor
                            y_fit_exp = np.concatenate((nt[:4],y_fit_exp))
                            self.t0 = 28

                        if len(y_fit_pl) != 0:
                            y_fit_pl = np.append(y_fit_pl,np.zeros(len(nt)-len(y_fit_pl)))
                            MultiplicativeFactor = nt[4]/y_fit_pl[0]
                            y_fit_pl = y_fit_pl*MultiplicativeFactor
                            y_fit_pl = np.concatenate((nt[:4],y_fit_pl))
                            self.t0 = 28
                        if ErrorExp<ErrorPL:
                            self.Time2BestFit[t] = "Exponential"
                            self.Time2nt[t]["n_fit"] = list(y_fit_exp/Z_n)
                        else:
                            self.Time2BestFit[t] = "PowerLaw"
                            self.Time2nt[t]["n_fit"] = list(y_fit_pl/Z_n)
                        # Save the Error
                        self.Time2ErrorFit[t]["Exponential"] = ErrorExp
                        self.Time2ErrorFit[t]["PowerLaw"] = ErrorPL
                        # Save the Fitted Parameters
                        self.Time2Fit[t]["A_exp"] = A0
                        self.Time2Fit[t]["A_pl"] = A_pl
                        self.Time2Fit[t]["alpha_exp"] = alpha_exp
                        self.Time2Fit[t]["alpha_pl"] = alpha_pl
                    
            else:
                logger.info(f"Time Vector is too short to compute the best fit for the Power Law and Exponential")
                self.Time2ErrorFit = None
                self.Time2BestFit = None
                self.Time2nt = None
                self.Time2Fit = None
            if self.Time2ErrorFit is not None:
                with open(os.path.join(self.PlotDir,f"Time2ErrorFit_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'w') as f:
                    json.dump(self.Time2ErrorFit,f,indent = 4)
            if self.Time2BestFit is not None:
                with open(os.path.join(self.PlotDir,f"Time2BestFit_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'w') as f:
                    json.dump(self.Time2BestFit,f,indent = 4)
            if self.Time2nt is not None:
                with open(os.path.join(self.PlotDir,f"Time2nt_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'w') as f:
                    json.dump(self.Time2nt,f,indent = 4)
                for t in self.Time2nt.keys():
                    self.Time2nt[t]["n"] = np.array(self.Time2nt[t]["n"])
                    self.Time2nt[t]["n_fit"] = np.array(self.Time2nt[t]["n_fit"])
            if self.Time2Fit is not None:
                with open(os.path.join(self.PlotDir,f"Time2Fit_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'w') as f:
                    json.dump(self.Time2Fit,f,indent = 4)
                with open(os.path.join(self.PlotDir,f"RangePlFit_{self.R}_UCI_{round(self.UCI,3)}.json"),'w') as f:
                    json.dump({"RangePl":list(self.RangePlFit)},f,indent = 4)
            
        else:
            if os.path.isfile(os.path.join(self.PlotDir,f"Time2ErrorFit_R_{self.R}_UCI_{round(self.UCI,3)}.json")):
                with open(os.path.join(self.PlotDir,f"Time2ErrorFit_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'r') as f:
                    self.Time2ErrorFit = json.load(f)
                with open(os.path.join(self.PlotDir,f"Time2BestFit_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'r') as f:
                    self.Time2BestFit = json.load(f)
                with open(os.path.join(self.PlotDir,f"Time2nt_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'r') as f:
                    self.Time2nt = json.load(f)
                for t in self.Time2nt.keys():
                    self.Time2nt[t]["n"] = np.array(self.Time2nt[t]["n"])
                    self.Time2nt[t]["n_fit"] = np.array(self.Time2nt[t]["n_fit"])
                with open(os.path.join(self.PlotDir,f"Time2Fit_R_{self.R}_UCI_{round(self.UCI,3)}.json"),'r') as f:
                    self.Time2Fit = json.load(f)
                with open(os.path.join(self.PlotDir,f"RangePlFit_{self.R}_UCI_{round(self.UCI,3)}.json"),'r') as f:
                    self.RangePlFit = json.load(f)["RangePl"]
    @timing_decorator
    def GetBestFitTimes(self):
        """
        Extract the t values such that self.Time2ErrorFit[t]["PowerLaw"] or self.Time2ErrorFit[t]["Exponential"] are minimum.
        
        :return: Tuple containing the t values for minimum PowerLaw and Exponential errors.
        NOTE: I accept traffic if there is at least one time in which the powerlaw wins
                """
        min_powerlaw_error = float('inf')
        min_exponential_error = float('inf')
        best_t_powerlaw = None
        best_t_exponential = None
        old = False
        if old:
            for t, errors in self.Time2ErrorFit.items():
                if errors["PowerLaw"] < min_powerlaw_error:
                    min_powerlaw_error = errors["PowerLaw"]
                    best_t_powerlaw = t
                if errors["Exponential"] < min_exponential_error:
                    min_exponential_error = errors["Exponential"]
                    best_t_exponential = t
            if min_exponential_error < min_powerlaw_error:
                logger.info(f"Exponential fit is better than PowerLaw fit at time {best_t_exponential}")
                self.BestFunctionFit = "Exponential"
                return "Exponential", best_t_exponential
            else:
                logger.info(f"PowerLaw fit is better than Exponential fit at time {best_t_powerlaw}")
                self.BestFunctionFit = "PowerLaw"
                return "PowerLaw", best_t_powerlaw
        else:
            for t, fit in self.Time2ErrorFit.items():
                if fit == "PowerLaw":
                    if self.Time2ErrorFit[t]["PowerLaw"] < min_powerlaw_error:
                        min_powerlaw_error = self.Time2ErrorFit[t]["PowerLaw"]
                        best_t_powerlaw = t
                if fit == "Exponential":
                    if self.Time2ErrorFit[t]["Exponential"] < min_exponential_error:
                        min_exponential_error = self.Time2ErrorFit[t]["Exponential"]
                        best_t_exponential = t
            if min_powerlaw_error == float('inf'):
                self.BestFunctionFit = "Exponential"
                logger.info(f"Exponential fit is better than PowerLaw fit at time {best_t_exponential}")
                return "Exponential", best_t_exponential
            else:
                self.BestFunctionFit = "PowerLaw"
                logger.info(f"PowerLaw fit is better than Exponential fit at time {best_t_powerlaw}")
                return "PowerLaw", best_t_powerlaw
    @timing_decorator
    def DecideIfJam(self):
        """
            @description:
            Looking at each self.Time2BestFit, if there is powerlaw at some point, then we have a jam
            NOTE: Decides wether the network is jammed or not, if it is Jammed saves CrticalNt (UnoloadCurve) and TCritical (Critical Time To Consider For Alpha)
            self.CriticalAlpha
            NOTE: The Criticality Of Behavior Is Defined For R,UCI couple here in isJam that corresponds to the condition that the best fit is an exponential
        """
        self.TCritical = None
        self.CriticalNt = None
        self.CrirticalAlpha = None
        BestFunctionFit, BestTime = self.GetBestFitTimes()

        if BestFunctionFit == "PowerLaw":
            self.IsJam = True
            self.TCritical = BestTime
            self.CriticalNt = self.Time2nt[BestTime]["n"]
            self.CriticalAlpha = self.Time2Fit[BestTime]["alpha_pl"]
            self.Tau = None
            self.BestTime = BestTime
            logger.info(f"Network is jammed at time {BestTime} with alpha = {self.CriticalAlpha} and Tau = None")
        else:
            self.IsJam = False
            self.TCritical = None
            self.CriticalNt = None
            self.CriticalAlpha = None
            self.BestTime = BestTime
            self.Tau = - 1/self.Time2Fit[BestTime]["alpha_exp"]       # In hours
            logger.info(f"Network is not jammed at time {BestTime} with alpha = None and Tau = {self.Tau}")
### FLUXES AND GAMMA
    @timing_decorator
    def ComputeFluxesAndSpeedDfRoute(self):
        """
            @description:
                - Computes Fluxes And Average Speeds in The Road Network in Intervals Of Time of delta_t
                @ return self.EdgesWithFluxesAndSpeed: DataFrame with the fluxes and speeds in the road network
        """
        if os.path.exists(os.path.join(self.PlotDir,f"R_{self.R}_UCI_{self.UCI}_traffic.parquet")):
            logger.info(f"Reading Traffic File R: {self.R}, UCI: {self.UCI}")
            self.EdgesWithFluxesAndSpeed = pl.read_parquet(os.path.join(self.PlotDir,f"R_{self.R}_UCI_{self.UCI}_traffic.parquet"))
        else:
            logger.info(f"Computing Traffic File R: {self.R}, UCI: {self.UCI}")
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
        logger.info(f"Computing Gamma And Tau R: {self.R}, UCI: {self.UCI}")
        self.Gamma = ComputeGamma(self.DfControlGroup)
        if self.IsJam:
            nt0 = max(self.DfUnload["FractionPeople"].to_numpy())
            # Choose the index for which the fraction of people in the network is less than 1/e of the maximum
            self.IndexTau = np.where(self.DfUnload["FractionPeople"].to_numpy() <= nt0*np.exp(-1))[0][0]
            self.Tau = self.DfUnload["Time_hours"].to_numpy()[self.IndexTau]
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
        count = 0
        n = self.DfUnload["NumberPeople"].to_numpy()
        mask = n > 0
        t_vect = self.DfUnload["Time_hours"].to_numpy()[mask]
        SmallerVect = min([len(t_vect),len(self.Time2nt[self.BestTime]["n"]),len(self.Time2nt[self.BestTime]["n_fit"])])
        PlotPeopleInNetwork(self.Time2nt[self.BestTime]["n"][:SmallerVect],self.Time2nt[self.BestTime]["n_fit"][:SmallerVect],t_vect[:SmallerVect],self.BestFunctionFit,self.BestTime,JoinDir(self.PlotDir,"UnloadingCurve_R_{0}_UCI_{1}.png".format(self.R,self.UCI)))
        count += 1
        Message = f"PlotUnloadCurve: Plot the number of people in the network at each time interval"
        AddMessageToLog(Message,self.LogFile)

    @timing_decorator
    def PlotNtAndFitSingleR(self):
        """
            Plots One of the curves in Fig 4 Paper Marta.
            @Description:
                LastTime is set to Plot all the data.

        """
        LastTime = list(self.Time2ErrorFit.keys())[-1]
        PlotNtAndFitSingleR(self.DfUnload["Time_hours"].to_numpy(),self.Time2nt[LastTime]["n"],self.Tau,self.Time2nt[LastTime]["n_fit"],self.R,self.UCI,self.PlotDir)


    ## HERE WOULD START PROPERTIES OF THE CONTROL GROUP




    


