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
import ast

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
        self.Nodes = pl.read_csv(os.path.join(BERKELEY_DIR,self.city,'Nodes.csv'))
        self.Edges = pl.read_csv(os.path.join(BERKELEY_DIR,self.city,'Edges.csv'))
        self.AddCapacity2Edges()
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


    def AddCapacity2Edges(self):
        """
            Add the capacity to the Edges DataFrame.
            @ NOTE: Useful For Considerations about traffic.
        """
        df = pl.DataFrame(self.GeoJsonEdges[["capacity","uv"]])
        df = df.with_columns([
            pl.col("capacity").cast(pl.Int64),
            pl.col("uv").cast(pl.Int64)
        ])
        self.Edges = self.Edges.join(df, left_on = "uniqueid",right_on = "uv", how = "inner")


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
        # Show The Unload Curve
        self.PlotUnloadCurve()
        # Compute The Gamma And Tau
        self.ComputeGammaAndTau()
        # DEPRECATED
        self.ComputeTime2Road2Traveller()
        self.AnimateNetworkTraffic()

    def GetPeopleInfo(self):
        """
            @description:
                Read the People File:
                NOTE: columns: p, distance, time_departure, last_time_simulated, num_steps, avg_v(mph), a, b, T, gas, co, path_length_cpu, path_length_gpu
        """
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
        # Prepare the DataFrame: NOTE: Compute the Fluxes and Speeds is Deprecated
        self.DfRoute = EmbdedTrajectoriesInRoadsAndTime(self.DfRoute,self.DfPeople,self.Edges)


# Trajectories Info
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
        if not os.path.exists(JoinDir(self.PlotDir,"UnloadCurve_R_{0}_UCI_{1}.csv".format(self.R,self.UCI))):
            # Initialize Time Variables IntTimeArray = [0,delta_t,2*delta_t,...,SecondsInDay], HourTimeArray = [0,1,2,...,24], Interval2NumberPeopleInNet = {0:0,1:0,...,24:0}
            self.IntTimeArray, self.HourTimeArray, self.Interval2NumberPeopleInNet = InitializeTimeVariablesOutputStats(HOURS_IN_DAY,MINUTES_IN_HOUR,SECONDS_IN_MINUTE,TIMESTAMP_OFFSET,self.delta_t)
            # NOTE: Filter the People in the Control Group  
            self.DfControlGroup = FilterDfPeopleControlGroup(self.t_start_control_group,self.t_end_control_group,self.DfRoute,"time_departure","last_time_simulated")
            for t in range(len(self.Interval2NumberPeopleInNet.keys())-1):
                # DataFrame Filtered With People in the Network at time t in the time interval t,t+1
                DfPeopleInNetAtTimet = FilterDfPeopleStilInNet(self.IntTimeArray[t],self.IntTimeArray[t+1],"last_time_simulated",self.DfControlGroup)
                # Number Of People in the Network at time t
                self.Interval2NumberPeopleInNet[self.HourTimeArray[t]] += len(DfPeopleInNetAtTimet) 
            self.VolumeTraffic = np.sum(DfUnload["NumberPeople"].to_numpy())
            DfUnload["FractionPeople"] = DfUnload["NumberPeople"].to_numpy()/self.VolumeTraffic
            DfUnload = pl.DataFrame(self.Interval2NumberPeopleInNet.items(),columns = ["Time","NumberPeople","FractionPeople"])
            DfUnload["Time_seconds"] = self.IntTimeArray
            DfUnload["Time_hours"] = self.IntTimeArray/3600
            DfUnload.to_csv(JoinDir(self.PlotDir,"UnloadCurve_R_{0}_UCI_{1}.csv".format(self.R,self.UCI)),index = False)
            self.DfUnload = DfUnload
        else:
            self.DfUnload = pl.read_csv(JoinDir(self.PlotDir,"UnloadCurve_R_{0}_UCI_{1}.csv".format(self.R,self.UCI)))
            self.HourTimeArray = self.Interval2NumberPeopleInNet["Time"]
            self.Interval2NumberPeopleInNet = self.Interval2NumberPeopleInNet["NumberPeople"]

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
        t_vect = self.DfUnload["Time_hours"].to_numpy()
        nt = self.DfUnload["FractionPeople"].to_numpy()
        # Check The t For Which The Power Law Fit the Best and Is Better Than The Exponential
        self.Time2ErrorFit = {t:{"PowerLaw":0,"Exponential":0} for t in t_vect[10:-1]}
        self.Time2BestFit = {t:"" for t in t_vect[10:-1]}
        self.Time2nt = {t:{"n":nt[:t],"n_fit":[]} for t in t_vect[10:-1]}
        self.Time2Fit = {t:{"A_exp":0,"A_pl":0,"alpha_exp":0,"alpha_pl":0} for t in t_vect[10:-1]}
        for t in t_vect[10:-10]:
            ErrorExp,A0,alpha_exp,y_fit_exp,ErrorPL,A_pl,alpha_pl,y_fit_pl = ComparePlExpo(t_vect[:t],nt[:t],initial_guess_powerlaw = (0,-1), initial_guess_expo = (1,-1),maxfev = 10000)
            # Standardize for a DataFrame the y_fit
            if len(y_fit_exp) != 0:
                y_fit_exp = np.append(y_fit_exp,np.zeros(len(nt)-len(y_fit_exp)))
            if len(y_fit_pl) != 0:
                y_fit_pl = np.append(y_fit_pl,np.zeros(len(nt)-len(y_fit_pl)))
            if ErrorExp<ErrorPL:
                self.Time2BestFit[t] = "Exponential"
                self.Time2nt[t]["n_fit"] = y_fit_exp
            else:
                self.Time2BestFit[t] = "PowerLaw"
                self.Time2nt[t]["n_fit"] = y_fit_pl
            # Save the Error
            self.Time2ErrorFit[t]["Exponential"] = ErrorExp
            self.Time2ErrorFit[t]["PowerLaw"] = ErrorPL
            # Save the Fitted Parameters
            self.Time2Fit[t]["A_exp"] = A0
            self.Time2Fit[t]["A_pl"] = A_pl
            self.Time2Fit[t]["alpha_exp"] = alpha_exp
            self.Time2Fit[t]["alpha_pl"] = alpha_pl

    def GetBestFitTimes(self):
        """
        Extract the t values such that self.Time2ErrorFit[t]["PowerLaw"] or self.Time2ErrorFit[t]["Exponential"] are minimum.
        
        :return: Tuple containing the t values for minimum PowerLaw and Exponential errors.
        """
        min_powerlaw_error = float('inf')
        min_exponential_error = float('inf')
        best_t_powerlaw = None
        best_t_exponential = None

        for t, errors in self.Time2ErrorFit.items():
            if errors["PowerLaw"] < min_powerlaw_error:
                min_powerlaw_error = errors["PowerLaw"]
                best_t_powerlaw = t
            if errors["Exponential"] < min_exponential_error:
                min_exponential_error = errors["Exponential"]
                best_t_exponential = t
        if min_exponential_error < min_powerlaw_error:
            return "Exponential", best_t_exponential
        else:
            return "PowerLaw", best_t_powerlaw

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
        else:
            self.IsJam = False
            self.TCritical = None
            self.CriticalNt = None
            self.CriticalAlpha = None

### FLUXES AND GAMMA
    def ComputeFluxesAndSpeedDfRoute(self):
        """
            @description:
                Transform DfRoute: [p,distance,route] -> [p,distance,route,avg_v(km/h)]
        """
        if os.path.exists(os.path.join(self.PlotDir,f"R_{self.R}_UCI_{self.UCI}_traffic.parquet")):
            self.DfRoute = pl.read_parquet(os.path.join(self.PlotDir,f"R_{self.R}_UCI_{self.UCI}_traffic.parquet"))
        else:
            self.DfRoute = self.DfRoute.join(self.DfPeople, on="p", how="left")
            self.DfRoute = self.DfRoute.with_columns((pl.col("avg_v(mph)")*1.6).alias("avg_v(km/h)"))
            self.DfRoute = self.DfRoute.drop(["distance_right","a","b","T","gas","co","path_length_cpu","path_length_gpu","avg_v(mph)","init_intersection","end_intersection"])
            self.DfRoute = self.DfRoute.filter(pl.col("num_steps") != 0)    
            self.DfRoute.with_columns(pl.when(not isinstance(pl.col("avg_v(km/h)"),pl.Float64)).then(pl.col("distance")/(pl.col("last_time_simulated")- pl.col("time_departure"))*3.6).alias("avg_v(km/h)"))
            for t in range(len(self.IntTimeArray)):
                self.DfRoute = self.DfRoute.with_columns(
                    pl.lit(0).alias(f"flux_{t}")
                )
                self.DfRoute = self.DfRoute.with_columns(
                    pl.lit(0).alias(f"speed_kmh_{t}")
                )
            for t in range(len(self.IntTimeArray)):
                print(t)
                UsersPerAvgSpeedRoads = (pl.col("time_departure") <= self.IntTimeArray[t]) & (pl.col("last_time_simulated") >= self.IntTimeArray[t])
                self.DfRoute.filter(UsersPerAvgSpeedRoads).group_by("route").agg([
                    (pl.col("avg_v(km/h)")).mean().alias(f"speed_kmh_{t}"),
                    pl.col("p").count().alias(f"flux_{t}")
                ])
            self.DfRoute.write_parquet(os.path.join(self.PlotDir,f"R_{self.R}_UCI_{self.UCI}_traffic.parquet"))




### TRAFFIC MEASURES ###

    def ComputeGammaAndTau(self):
        """
            Compute the Gamma and Tau for the Route and People Files
            Returns:
                Gamma: float is time that the trafficked city would require to obtain
                    the integrated fluxes of maximum capacity
                Tau: float
        """
        self.Gamma = ComputeGamma(self.DfControlGroup)
        if self.IsJam:
            nt0 = max(self.DfUnload["FractionPeople"].to_numpy())
            # Choose the index for which the fraction of people in the network is less than 1/e of the maximum
            self.IndexTau = np.where(self.DfUnload["FractionPeople"].to_numpy() <= nt0*np.exp(-1))[0][0]
            self.Tau = self.DfUnload["Time_hours"].to_numpy()[self.IndexTau]
            # TODO: Compute the Gamma -> Look post_processing_simulations
        pass




### NETWORK FEATURES ###
    
# Road Network Info # DEPRECATED!
    def ComputeTime2Road2Traveller(self):
        """
            Description:
            Time2Road2MFDNotProcessed: 
                 {TimeHour0:{Road0:{"p":[],"avg_v(mph)":[],"NumberPeople":0},
                                    Road1:{"p":[],"avg_v(mph)":[],"NumberPeople":0},
                                    ...}
                        TimeHour1:{Road0:{"p":[],"avg_v(mph)":[],"NumberPeople":0},
                                    ....
                                    }
                        ...
                        }
        """
        if self.ReadRouteInfoBool and self.ReadPeopleInfoBool:
            # Generate {TimeHour0:{Road0:}}
            StrSpeed = "avg_v(mph)"
            StrUsersId = "p"
            StrNumberPeople = "NumberPeople"
            # vec(Time): vec(Road): vec(StrSpeed),vec(StrUsersId),StrNumberPeople
            IntRoads = [int(value) for value in self.GeoJsonEdges["uv"].to_list()]
            self.Time2Road2MFDNotProcessed = Init_Time2Road2MFDNotProcessed(self.HourTimeArray,IntRoads,StrSpeed,StrUsersId,StrNumberPeople)
            self.Time2Road2MFD = Init_Time2Road2MFD(self.Time2Road2MFDNotProcessed,StrSpeed,StrNumberPeople)
            self.DfControlGroup = FilterDfPeopleControlGroup(self.t_start_control_group,self.t_end_control_group,self.DfPeople,"time_departure","last_time_simulated")
            for t  in range(len(self.IntTimeArray)-1):    
                # People in Array at time t
                DfPeopleInNetAtTimet = FilterDfPeopleStilInNet(self.IntTimeArray[t],self.IntTimeArray[t+1],"last_time_simulated",self.DfControlGroup)
                # People in the network at time t NOTE: route df
            self.Time2Road2MFD = ComputeAvgTime2Road2MFD(self.Time2Road2MFDNotProcessed,self.Time2Road2MFD,StrSpeed,StrNumberPeople)
            self.Road2MFD2Plot = Init_Road2MFD2Plot(self.Time2Road2MFD,StrSpeed,StrNumberPeople,self.GeoJsonEdges)
            self.MFD2Plot = Init_MFD2Plot(self.Road2MFD2Plot,StrSpeed,StrNumberPeople)
            self.AddGeoJsonTimeColumns(StrNumberPeople,StrSpeed)
            self.CountFunctions += 1
            ReturnMessageTime2Road(self.CountFunctions,self.LogFile,self.Time2Road2MFDNotProcessed,self.Time2Road2MFD,self.Road2MFD2Plot)
    





    def AddGeoJsonTimeColumns(self,StrNumberPeople,StrSpeed):
        """
            Add the columns to the GeoJsonEdges that contains the number of people and the speed in the
        """
        if self.verbose:
            print("AddGeoJsonTimeColumns")
        for t in self.Time2Road2MFD.keys():
            self.GeoJsonEdges[StrNumberPeople + "_" + t] = self.GeoJsonEdges["uv"].apply(lambda x: self.Time2Road2MFD[t][x][StrNumberPeople])
            self.GeoJsonEdges[StrSpeed + "_" + t] = self.GeoJsonEdges["uv"].apply(lambda x: self.Time2Road2MFD[t][x][StrSpeed])
            self.GeoJsonEdges["timepercorrence_" + t] = self.GeoJsonEdges.apply(lambda x: self.Time2Road2MFD[t][x["uv"]][StrSpeed]/x["maxspeed_int"],axis = 1)
            self.GeoJsonEdges["q_"+t] = self.GeoJsonEdges.apply(lambda x: self.Time2Road2MFD[t][x["uv"]][StrSpeed]/x["length"],axis = 1)
        self.InitColumn2InfoSavePlot(StrNumberPeople,StrSpeed)
        self.CountFunctions += 1
        Message = f"Function {self.CountFunctions}: AddGeoJsonTimeColumns: Add the columns to the GeoJsonEdges that contains the number of people and the speed in the"
        AddMessageToLog(Message,self.LogFile)

    def InitColumn2InfoSavePlot(self,StrNumberPeople,StrSpeed):
        """
            Description:
                Create dictionary that contains the right information to plot the data in the animation.
            StrNumberPeople: str -> Number of People (Column of Output)
        """
        self.Column2InfoSavePlot = defaultdict()
        for t in self.HourTimeArray:
            self.Column2InfoSavePlot[StrNumberPeople + "_" + t] = {"title":f"Number of People Road at {t}",
                                                                   "savefile":StrNumberPeople + "_" + t +".png",
                                                                   "colorbar":"Number People",
                                                                   "animationfile":"NumberPeople.gif"}
            self.Column2InfoSavePlot[StrSpeed + "_" + t] = {"title": f"Speed at {t} (mph)",
                                                            "savefile":StrSpeed + "_" + t +".png",
                                                            "colorbar":"Speed (mph)",
                                                            "animationfile":"Speed.gif"}
            self.Column2InfoSavePlot["timepercorrence_" + t] = {"title": f"time percorrence roads at {t}",
                                                                "savefile":"timepercorrence_" + t +".png",
                                                                "colorbar":"Time Percorrence (s)",
                                                                "animationfile":"timepercorrence.gif"}
            self.Column2InfoSavePlot["q_" + t] = {"title": f"fraction max speed at {t}",
                                                  "savefile":"q_" + t +".png",
                                                  "colorbar":"Fraction Maximum Speed",
                                                  "animationfile":"q.gif"}
        self.CountFunctions += 1
        Message = f"Function {self.CountFunctions}: InitColumn2InfoSavePlot: Create dictionary that contains the right information to plot the data in the animation."
        AddMessageToLog(Message,self.LogFile)





### PLOTS ####


    def PlotUnloadCurve(self):
        """
            @Decsription:
                For each time window checks wether exponential is better than powerlaw. If powerlaw is better than exponential, then we have a jam.
        """
        for t in self.Time2ErrorFit.keys():
            # Plots The Fraction Of People That are Stuck In The Network at Time T
            PlotPeopleInNetwork(self.Time2nt[t]["n"],self.Time2nt[t]["n_fit"],self.HourTimeArray[:t],JoinDir(self.PlotDir,"UnloadingCurve_R_{0}_UCI_{1}.png".format(self.R,self.UCI)))
        self.CountFunctions += 1
        Message = f"Function {self.CountFunctions}: PlotUnloadCurve: Plot the number of people in the network at each time interval"
        AddMessageToLog(Message,self.LogFile)

    
    def PlotNtAndFitSingleR(self):
        """
            Plots One of the curves in Fig 4 Paper Marta.
            @Description:
                LastTime is set to Plot all the data.

        """
        LastTime = list(self.Time2ErrorFit.keys())[-1]
        PlotNtAndFitSingleR(self.HourTimeArray,self.Time2nt[LastTime]["n"],tau,self.Time2nt[LastTime]["n_fit"],self.R,self.UCI,self.PlotDir)






    def AnimateNetworkTraffic(self):
        # TODO: Create a function that quantifies for each hour the velocity in the road and compute the traffic.
        # NOTE: The dependence of TrafficLevel to the FilePlots, does not allow the nesting I thought for this function.
        AnimateNetworkTraffic(self.PlotDir,self.GeoJsonEdges,self.Column2InfoSavePlot,dpi = 300,IsLognorm = False)
        self.CountFunctions += 1
        Message = f"Function {self.CountFunctions}: AnimateNetworkTraffic: Create the animation of the traffic in the network"
        AddMessageToLog(Message,self.LogFile)



