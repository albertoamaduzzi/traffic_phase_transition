import polars as pl
import numpy as np
import osmnx as ox
from OsFunctions import *
from DateTimeHandler import *
from Plots import *
class OutputStats:
    def __init__(self,R,UCI,config):
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
        self.GetGeopandas()    

    def GetPeopleInfo(self):
        if IsFile(self.PeopleFile):
            self.DfPeople = pl.read_csv(self.PeopleFile)
            self.ReadPeopleInfoBool = True

    def GetRouteInfo(self):
        if IsFile(self.RouteFile):
            self.DfRoute = pl.read_csv(self.RouteFile)
            self.ReadRouteInfoBool = True

    def GetGeopandas(self):
        if os.path.isfile(self.config["graphml_file"]):
            G = ox.load_graphml(self.config["graphml_file"])
            self.GeoJson = ox.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
        self.GetGeopandasBool = True

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
            condition = (
                (pl.col("time_departure") >= int(self.IntTimeArray[t])) &
                (pl.col("time_departure") < int(self.IntTimeArray[t+1])) &  # Corrected condition
                (int(self.IntTimeArray[t+1]) < pl.col("last_time_simulated"))
            )
            DfPeopleInNetAtTimet = self.DfPeople.filter(condition)
            print("Interval: ",self.IntTimeArray[t],self.IntTimeArray[t+1])
            print("DfPeopleInNetAtTimet:\n",DfPeopleInNetAtTimet)
            self.Interval2NumberPeopleInNet[self.IntTimeArray[t]] += len(DfPeopleInNetAtTimet) 
        
    def PlotUnloadCurve(self):
        self.ComputeUnloadCurve()
        PlotPeopleInNetwork(self.Interval2NumberPeopleInNet,self.HourTimeArray,JoinDir(self.PlotDir,"UnloadingCurve_R_{0}_UCI_{1}.png".format(self.R,self.UCI)))


    def AnimateNetworkTraffic(self):
        # TODO: Create a function that quantifies for each hour the velocity in the road and compute the traffic.
        # NOTE: The dependence of TrafficLevel to the FilePlots, does not allow the nesting I thought for this function.
        FilePlots =[JoinDir(self.PlotDir,Column2Plot) for Column2Plot in self.HourTimeArray]
        AnimateNetworkTraffic(FilePlots,
                              AnimationFile,
                              TrafficGdf,
                              TrafficLevel,
                              ColorBarExplanation,
                              Title,
                              dpi = 300,
                              IsLognorm = False)


