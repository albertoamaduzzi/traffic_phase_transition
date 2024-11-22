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
    def __init__(self,R,UCI,config,GeoJsonEdges):
        
        self.verbose = True
        # Add Logic to Take All The Files From The Base Dir
        self.RouteFile = config[UCI][R]['route_file']
        self.PeopleFile = config[UCI][R]['people_file']
        self.R = R
        self.UCI = UCI
        self.Name = config['name']
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
        # Group Control
        self.t_start_control_group = 7*3600
        self.t_end_control_group = 8*3600
        # Total Volume Traffic
        self.VolumeTraffic = 0
    def CompleteAnalysis(self):
        """
            This is the Function one wants to Call to get the Analysis of the Simulation
        """
        self.ComputeUnloadCurve()
        self.PlotUnloadCurve()
        self.ComputeTime2Road2Traveller()
        self.AnimateNetworkTraffic()

    def GetPeopleInfo(self):
        if IsFile(self.PeopleFile):
            dtypes = {"p":pl.Int64,
                      "init_intersection":pl.Int64,
                      "end_intersection":pl.Int64,
                      "time_departure":pl.Float64,
                      "num_steps":pl.Int64,
                      "co":pl.Float32,
                      "gas":pl.Float32,
                      "distance":pl.Float32,
                      "a":pl.Float32,
                      "b":pl.Float32,
                      "T":pl.Float32,"avg_v(mph)":pl.Float32,
                      "active":pl.Int32,
                      "last_time_simulated":pl.Int64,
                      "path_length_cpu":pl.Int64,
                      "path_length_gpu":pl.Int64}
            self.DfPeople = pl.read_parquet(self.PeopleFile,dtypes = dtypes,ignore_errors = True)
#            self.DfPeople = pl.read_csv(self.PeopleFile,dtypes = dtypes,ignore_errors = True)
            self.ReadPeopleInfoBool = True
            self.CountFunctions += 1
            Message = f"Function {self.CountFunctions}: GetPeopleInfo: {self.PeopleFile} was read"
            AddMessageToLog(Message,self.LogFile)


    def GetRouteInfo(self):
        """
            NOTE: Translate The column of the RouteDf into array from string 
        """
        if IsFile(self.RouteFile):
            self.DfRoute = pd.read_csv(self.RouteFile,sep = ':')
            self.DfRoute = pl.from_pandas(self.DfRoute)
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


# Trajectories Info
    def ComputeUnloadCurve(self):
        """
            Compute the number of people in the network at each time interval
            Returns:
                Interval2NumberPeopleInNet: Dictionary containing the number of people in the network at each time interval
            NOTE: self.DfUnload: DataFrame
        """
        if not os.path.exists(JoinDir(self.PlotDir,"UnloadCurve_R_{0}_UCI_{1}.csv".format(self.R,self.UCI))):
            SecondsInDay = int(HOURS_IN_DAY*MINUTES_IN_HOUR*SECONDS_IN_MINUTE)
            MinutesInDay = int(HOURS_IN_DAY*MINUTES_IN_HOUR)
            # I count people in Network at each time interval (15 mins)
            NumIntervals = int(MinutesInDay/self.delta_t)
            # Create the Int Time Array (0 -> Seconds In Day)
            self.IntTimeArray = np.linspace(0,TIMESTAMP_OFFSET + SecondsInDay,NumIntervals,dtype = int)       
            # Convert it to minutes seconds and hour for the plot Labels (0 -> 24)        
            self.HourTimeArray = ConvertArray2HMS(np.linspace(TIMESTAMP_OFFSET,TIMESTAMP_OFFSET + SecondsInDay,NumIntervals))
            self.Interval2NumberPeopleInNet = {Interval:0 for Interval in self.HourTimeArray}
            self.DfControlGroup = FilterDfPeopleControlGroup(self.t_start_control_group,self.t_end_control_group,self.DfPeople,"time_departure","last_time_simulated")
            for t in range(len(self.Interval2NumberPeopleInNet.keys())-1):
                # DataFrame Filtered With People in the Network at time t in the time interval t,t+1
                DfPeopleInNetAtTimet = FilterDfPeopleStilInNet(self.IntTimeArray[t],self.IntTimeArray[t+1],"last_time_simulated",self.DfControlGroup)
                # Number Of People in the Network at time t
                self.Interval2NumberPeopleInNet[self.HourTimeArray[t]] += len(DfPeopleInNetAtTimet) 
            self.VolumeTraffic = np.sum(DfUnload["NumberPeople"].to_numpy())
            DfUnload["FractionPeople"] = DfUnload["NumberPeople"].to_numpy()/self.VolumeTraffic
            DfUnload = pd.DataFrame(self.Interval2NumberPeopleInNet.items(),columns = ["Time","NumberPeople","FractionPeople"])
            DfUnload["Time_seconds"] = self.IntTimeArray
            DfUnload.to_csv(JoinDir(self.PlotDir,"UnloadCurve_R_{0}_UCI_{1}.csv".format(self.R,self.UCI)),index = False)
            self.DfUnload = DfUnload
        else:
            self.DfUnload = pd.read_csv(JoinDir(self.PlotDir,"UnloadCurve_R_{0}_UCI_{1}.csv".format(self.R,self.UCI)))
            self.HourTimeArray = self.Interval2NumberPeopleInNet["Time"]
            self.Interval2NumberPeopleInNet = self.Interval2NumberPeopleInNet["NumberPeople"]


    def FindRcritical(self):
        """
            Find the critical value of R
        """
        t_vect = self.DfUnload["Time_seconds"].to_numpy()
        dt = np.diff(t_vect)
        nt = self.DfUnload["NumberPeople"].to_numpy()
        # Check The t For Which The Power Law Fit the Best and Is Better Than The Exponential
        ErrorFitExp = []
        ErrorFitPl = []
        for t in t_vect[10:-1]:
            # Cut Until t
            WeightFit = len(t_vect[:t]) 
            Z = np.sum(nt[:t]*dt[:t])
            nt_norm = nt[:t]/Z
            initial_guess_powerlaw = (1,0.3) 
            initial_guess_exp = (1,-1)
            fitpl = Fitting(t[:t], nt_norm[:t],'powerlaw', initial_guess_powerlaw)
            fitexp = Fitting(t[:t], nt_norm[:t],'exponential', initial_guess_exp)

    def PlotUnloadCurve(self):
        PlotPeopleInNetwork(self.Interval2NumberPeopleInNet,self.HourTimeArray,JoinDir(self.PlotDir,"UnloadingCurve_R_{0}_UCI_{1}.png".format(self.R,self.UCI)))
        self.CountFunctions += 1
        Message = f"Function {self.CountFunctions}: PlotUnloadCurve: Plot the number of people in the network at each time interval"
        AddMessageToLog(Message,self.LogFile)

    




# Road Network Info
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
                # Add time_departure: int last_time_simulated: int avg_v(mph): float time: np.array
                self.DfRoute,self.Column2IndexDfRoute = AddTimeList2DfRoute(self.DfRoute,self.DfPeople)
                # People in the network at time t NOTE: route df
                user_ids_in_net = DfPeopleInNetAtTimet[StrUsersId].to_list()
                if len(user_ids_in_net) != 0:
                    DfRouteInInterval = self.DfRoute.filter(pl.col(StrUsersId).is_in(user_ids_in_net))
                    for Personrow in DfRouteInInterval.rows():
                        # The Roads that are Travelled by Person Id_
                        Roads = ast.literal_eval(Personrow[self.Column2IndexDfRoute["route"]])
                        Id_ = Personrow[self.Column2IndexDfRoute[StrUsersId]]
                        for Road in Roads:
                            # Time: Road
                            self.Time2Road2MFDNotProcessed[self.HourTimeArray[t]][Road][StrUsersId].append(Id_) 
                            self.Time2Road2MFDNotProcessed[self.HourTimeArray[t]][Road][StrSpeed].append(Road)
                            self.Time2Road2MFDNotProcessed[self.HourTimeArray[t]][Road][StrNumberPeople] += len(DfRouteInInterval)
                else:
                    pass
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



    def AnimateNetworkTraffic(self):
        # TODO: Create a function that quantifies for each hour the velocity in the road and compute the traffic.
        # NOTE: The dependence of TrafficLevel to the FilePlots, does not allow the nesting I thought for this function.
        AnimateNetworkTraffic(self.PlotDir,self.GeoJsonEdges,self.Column2InfoSavePlot,dpi = 300,IsLognorm = False)
        self.CountFunctions += 1
        Message = f"Function {self.CountFunctions}: AnimateNetworkTraffic: Create the animation of the traffic in the network"
        AddMessageToLog(Message,self.LogFile)



