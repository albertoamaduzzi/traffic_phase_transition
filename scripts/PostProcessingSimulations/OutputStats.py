import polars as pl
import pandas as pd
import numpy as np
import osmnx as ox
from OsFunctions import *
from DateTimeHandler import *
from Plots import *
from GeoJsonFunctions import *
from StructuresOfInterest import *
from PolarsFunctions import *
class OutputStats:
    def __init__(self,R,UCI,config,GeoJsonEdges):
        self.RouteFile = config[R][UCI]['route_file']
        self.PeopleFile = config[R][UCI]['people_file']
        self.R = R
        self.UCI = UCI
        self.Name = config['name']
        self.delta_t = config['delta_t']
        self.PlotDir = JoinDir(config['output_simulation_dir'],'Plots')
        # Flags
        self.ReadPeopleInfoBool = False
        self.ReadRouteInfoBool = False
        self.GetGeopandasBool = False
        # DataFrames
        self.GetPeopleInfo()
        self.GetRouteInfo()    
        self.GetGeopandas(GeoJsonEdges)

    def GetPeopleInfo(self):
        if IsFile(self.PeopleFile):
            self.DfPeople = pl.read_csv(self.PeopleFile)
            self.ReadPeopleInfoBool = True

    def GetRouteInfo(self):
        if IsFile(self.RouteFile):
            self.DfRoute = pd.read_csv(self.RouteFile,sep = ':')
            self.ReadRouteInfoBool = True

    def GetGeopandas(self,GeoJsonEdges):
        self.GeoJsonEdges = GeoJsonEdges
        self.GetGeopandasBool


# Trajectories Info
    def ComputeUnloadCurve(self):
        """
            Compute the number of people in the network at each time interval
            Returns:
                Interval2NumberPeopleInNet: Dictionary containing the number of people in the network at each time interval
        """
        SecondsInDay = HOURS_IN_DAY*MINUTES_IN_HOUR*SECONDS_IN_MINUTE
        NumIntervals = HOURS_IN_DAY*MINUTES_IN_HOUR/self.delta_t 
        # Create the Int Time Array (0 -> Seconds In Day)
        self.IntTimeArray = np.linspace(0,TIMESTAMP_OFFSET + SecondsInDay,NumIntervals)       
        # Convert it to minutes seconds and hour for the plot Labels (0 -> 24)        
        self.HourTimeArray = ConvertArray2HMS(np.linspace(TIMESTAMP_OFFSET,TIMESTAMP_OFFSET + SecondsInDay,NumIntervals))
        self.Interval2NumberPeopleInNet = {Interval:0 for Interval in self.HourTimeArray}
        for t in range(len(self.Interval2NumberPeopleInNet.keys())-1):
            DfPeopleInNetAtTimet = FilterDfPeopleInNetInTSlice(self.IntTimeArray[t],self.IntTimeArray[t+1],self.DfPeople,"time_departure","last_time_simulated")
            print("Interval: ",self.IntTimeArray[t],self.IntTimeArray[t+1])
            print("DfPeopleInNetAtTimet:\n",DfPeopleInNetAtTimet)
            self.Interval2NumberPeopleInNet[self.IntTimeArray[t]] += len(DfPeopleInNetAtTimet) 
        
    def PlotUnloadCurve(self):
        self.ComputeUnloadCurve()
        PlotPeopleInNetwork(self.Interval2NumberPeopleInNet,self.HourTimeArray,JoinDir(self.PlotDir,"UnloadingCurve_R_{0}_UCI_{1}.png".format(self.R,self.UCI)))


# Road Network Info
    def ComputeTime2Road2Traveller(self):
        """
            Description:
                Compute {TimeHour0:{Road0:{"p":[],"avg_v(mph)":[],"NumberPeople":0},
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
            self.Time2Road2MFD = Init_Time2Road2MFD(self.HourTimeArray,self.GeoJsonEdges["uv"].to_list(),StrSpeed,StrUsersId,StrNumberPeople)
            for t  in range(len(self.IntTimeArray)):    
                DfPeopleInNetAtTimet = FilterDfPeopleInNetInTSlice(self.IntTimeArray[t],self.IntTimeArray[t+1],self.DfPeople,"time_departure","last_time_simulated")
                self.DfRoute = AddTimeList2DfRoute(self.DfRoute,self.DfPeople)
                DfRouteInInterval = self.DfRoute[self.DfRoute[StrUsersId].isin(DfPeopleInNetAtTimet[StrUsersId].to_list())] 
                for Personrow in DfRouteInInterval.iterrows():
                    for Road in Personrow["route"]:
                        self.Time2Road2MFD[self.HourTimeArray[t]][Road][StrUsersId].append(Personrow[StrUsersId]) 
                        self.Time2Road2MFD[self.HourTimeArray[t]][Road].append(Personrow[StrSpeed])
                self.Time2Road2MFD[self.HourTimeArray[t]][Road][StrNumberPeople] += len(DfRouteInInterval)


    def ComputeMFD(self):
        if self.GetGeopandasBool:


    def AnimateNetworkTraffic(self,GeoJsonEdges):
        # TODO: Create a function that quantifies for each hour the velocity in the road and compute the traffic.
        # NOTE: The dependence of TrafficLevel to the FilePlots, does not allow the nesting I thought for this function.
        FilePlots =[JoinDir(self.PlotDir,Column2Plot) for Column2Plot in self.HourTimeArray]
        AnimateNetworkTraffic(GeoJsonEdges,
                              AnimationFile,
                              TrafficGdf,
                              TrafficLevel,
                              ColorBarExplanation,
                              Title,
                              dpi = 300,
                              IsLognorm = False)


