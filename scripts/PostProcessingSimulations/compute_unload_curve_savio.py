import polars as pl
import numpy as np
import os
import datetime
import pandas as pd
import logging 
logger = logging.getLogger(__name__)
HOURS_IN_DAY = 24
MINUTES_IN_HOUR = 60
SECONDS_IN_MINUTE = 60
SECONDS_IN_DAY = SECONDS_IN_MINUTE*MINUTES_IN_HOUR*HOURS_IN_DAY
DAYS_IN_WEEK = 7
DAYS_IN_YEAR = 365
DAYS_IN_BISESTIL_YEAR = 366
TIMESTAMP_OFFSET = (datetime.datetime(1970,1,1,0,0,0).timestamp())

def ConvertArray2HMS(array):
    """
        Convert an array of timestamps to an array of hours
        Input:
            array: Array of timestamps
        Output:
            Array of hours
        """
    if isinstance(array[0],float):
        array = [int(time) for time in array]
    if isinstance(array[0],datetime.datetime):
        return [time.hour for time in array]
    elif isinstance(array[0],int):
        datetime_ = [datetime.datetime.fromtimestamp(time) for time in array]
        return [dt.strftime("%Y-%m-%d %H:%M:%S").split(" ")[1] for dt in datetime_]
    
    elif isinstance(array[0],str):
        return [datetime.datetime.strptime(time,"%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S").split(" ")[1] for time in array]
    else:
        raise ValueError("Array type not supported")


def InitializeTimeVariablesOutputStats(HOURS_IN_DAY,MINUTES_IN_HOUR,SECONDS_IN_MINUTE,TIMESTAMP_OFFSET,delta_t):
    """
        @params:
            - HOURS_IN_DAY: int
            - MINUTES_IN_HOUR: int
            - SECONDS_IN_MINUTE: int
            - TIMESTAMP_OFFSET: int
            - delta_t: int
        @return:
            - IntTimeArray: np.array
            - HourTimeArray: np.array
            - Interval2NumberPeopleInNet: dict (Hourly Time -> Number of People in Network)
    """
    SecondsInDay = int(HOURS_IN_DAY*MINUTES_IN_HOUR*SECONDS_IN_MINUTE)
    MinutesInDay = int(HOURS_IN_DAY*MINUTES_IN_HOUR)
    # I count people in Network at each time interval (15 mins)
    NumIntervals = int(MinutesInDay/delta_t)
    # Create the Int Time Array (0 -> Seconds In Day)
    IntTimeArray = np.linspace(0,TIMESTAMP_OFFSET + SecondsInDay,NumIntervals,dtype = int)       
    # Convert it to minutes seconds and hour for the plot Labels (0 -> 24)        
    HourTimeArray = ConvertArray2HMS(np.linspace(TIMESTAMP_OFFSET,TIMESTAMP_OFFSET + SecondsInDay,NumIntervals))
    Interval2NumberPeopleInNet = {Interval:0 for Interval in HourTimeArray}
    return IntTimeArray, HourTimeArray, Interval2NumberPeopleInNet



def FilterDfPeopleControlGroup(t0,t1,DfPeople,StrTimeDeparture,StrLastTimeSimulated):
    """
        Description:
            Filter the people in the network at a given time interval
        Args:
            t0: int -> Initial time of the interval
            t1: int -> Final time of the interval
            DfPeople: pl.DataFrame -> People DataFrame
            StrTimeDeparture: str -> Column name of the departure time
            StrLastTimeSimulated: str -> Column name of the last time simulated
        Returns:
            DfPeopleInNetAtTimet: pl.DataFrame -> People in the network at the given time interval
    """
    condition = (
        (pl.col(StrTimeDeparture) >= int(t0)) &
        (pl.col(StrTimeDeparture) < int(t1))
#          &  # Corrected condition
#        (int(t1) < pl.col(StrLastTimeSimulated))
    )
    DfPeopleInNetAtTimet = DfPeople.filter(condition)
    return DfPeopleInNetAtTimet

def EmbdedTrajectoriesInRoadsAndTime(DfRoute,DfPeople,Edges):
    """
        @param DfRoute: DataFrame with the routes of the people in the network
        @param DfPeople: DataFrame with the properties of the people
        @param Edges: DataFrame with the properties of the edges
        @return DfRoute: DataFrame with columns:
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
        - time_entering_road: Time of Entering the Road (in seconds)
    """
    Edges.with_columns((pl.col("speed_mph")*1.6).alias("speed_limit_kmh"))
    # Inport Properties Of People into Routes
    DfRoute = DfRoute.join(DfPeople, on="p", how="left")
    # Extract ],[ from string and replace with empty string"" 
    DfRoute = DfRoute.with_columns(pl.col("route").str.replace("]", "",literal = True).str.replace("[", "",literal = True).alias("no_brackets"))
    # Transform to list of strings
    DfRoute = DfRoute.with_columns(no_coma = pl.col("no_brackets").str.split(","))
    # Cast 2 int
    DfRoute = DfRoute.explode("no_coma")
    DfRoute = DfRoute.filter(pl.col("no_coma") != "")
    DfRoute = DfRoute.with_columns(route_list = pl.col("no_coma").cast(pl.Int64))
    #DfRoute = DfRoute.groupby("p",mantain_order = True).agg(pl.col("route_list").agg_groups().alias("route_list"))
    DfRoute = DfRoute.drop(["route","no_brackets","no_coma"])
    DfRoute = DfRoute.with_columns(pl.col("route_list").alias("route"))
    DfRoute = DfRoute.drop("route_list")
    DfRoute = DfRoute.with_columns((pl.col("avg_v(mph)")*1.6).alias("avg_v(km/h)"))
    DfRoute = DfRoute.drop(["distance_right","a","b","T","gas","co","path_length_cpu","path_length_gpu","avg_v(mph)"]) # "init_intersection","end_intersection"
    DfRoute = DfRoute.filter(pl.col("num_steps") != 0)    
    DfRoute = DfRoute.with_columns(pl.when(not isinstance(pl.col("avg_v(km/h)"),pl.Float64)).then(pl.col("distance")/(pl.col("last_time_simulated")- pl.col("time_departure"))*3.6).alias("avg_v(km/h)"))
    DfRoute = DfRoute.filter(pl.col("avg_v(km/h)") <150)
    DfRoute = DfRoute.join(Edges, left_on='route', right_on='uniqueid', how='inner')
    DfRoute = DfRoute.drop(["osmid_u","osmid_v","speed_mph","active","num_steps","lanes"])
    # Convert m to km
    DfRoute = DfRoute.with_columns((pl.col("length")/1000).alias("length_km"),
                                (pl.col("distance")/1000).alias("distance_km"),
                                (pl.col("avg_v(km/h)")/3.6).alias("avg_v(m/s)")) 
    DfRoute = DfRoute.with_columns(
        (pl.col("length")/pl.col("avg_v(m/s)")).alias("time_interval_in_road"))
    DfRoute = DfRoute.with_columns(
        (pl.cum_sum("time_interval_in_road").over("p")).alias("cumulative_time_spent_in_road")
    )
    DfRoute = DfRoute.with_columns((pl.col("time_departure") + pl.col("cumulative_time_spent_in_road")).alias("time_leaving_road"))
    DfRoute = DfRoute.with_columns(
        (pl.col("time_leaving_road") - pl.col("time_interval_in_road")).alias("time_entering_road")
    )    
    DfRoute = DfRoute.drop(["time_interval_in_road","cumulative_time_spent_in_road"])
    if "speed_mph" in DfRoute.columns:
        DfRoute = DfRoute.drop("speed_mph")
    return DfRoute



def FilterDfPeopleStilInNet(t_start_interval,t_end_interval,StrLastTimeSimulated,DfPeople):
    """
        Description:
            Filter the people still in the network
        Args:
            t1: int -> Time
            StrLastTimeSimulated: str -> Column name of the last time simulated
            DfPeople: pl.DataFrame -> People DataFrame
        Returns:
            DfPeopleStillInNet: pl.DataFrame -> People still in the network
    """
    condition = (
        (pl.col(StrLastTimeSimulated) >= int(t_start_interval)) &
        (pl.col(StrLastTimeSimulated) < int(t_end_interval))
    )
    DfPeopleStillInNet = DfPeople.filter(condition)
    return DfPeopleStillInNet

def GetPeopleInfo(iter_2_DfPeople,PeopleFiles,iters,t_start_hour):
    count_people_file = 0
    for PeopleFile in PeopleFiles:
        iter_ = iters[count_people_file]
        if os.path.isfile(PeopleFile):
            if ".parquet" in PeopleFile:
                iter_2_DfPeople[iter_] = pl.read_parquet(PeopleFile)
            elif ".csv" in PeopleFile:
                iter_2_DfPeople[iter_] = pl.read_csv(PeopleFile)
            iter_2_DfPeople[iter_] = iter_2_DfPeople[iter_].filter(pl.col("time_departure")>t_start_hour*3600)
    return iter_2_DfPeople


def GetRouteInfo(iter_2_DfRoute,RouteFiles,iters):
    count_route_file = 0
    for RouteFile in RouteFiles:
        iter_ = iters[count_route_file]
        if os.path.isfile(RouteFile):
            if ".parquet" in RouteFile:
                iter_2_DfRoute[iter_] = pl.read_parquet(RouteFile)
            elif ".csv" in RouteFile:
                iter_2_DfRoute[iter_] = pl.read_csv(RouteFile,separator = ':')
                if iter_2_DfRoute[iter_].schema["route"] == pl.Utf8:
                    pass
        count_route_file += 1
    return iter_2_DfRoute



def ComputeSingleUnloadCurve(Interval2NumberPeopleInNet,
                             IntTimeArray,
                             HourTimeArray,
                             t_start_control_group,
                             t_start_hour,
                             t_end_control_group,
                             DfControlGroup):
    """

    """
    for t in range(len(Interval2NumberPeopleInNet.keys())-1):
        # DataFrame Filtered With People in the Network at time t in the time interval t,t+1
        DfPeopleInNetAtTimet = FilterDfPeopleStilInNet(IntTimeArray[t],IntTimeArray[t+1],"last_time_simulated",DfControlGroup)
        # Number Of People in the Network at time t
        Interval2NumberPeopleInNet[HourTimeArray[t]] += len(DfPeopleInNetAtTimet) 
    data = {
        "Time": list(IntTimeArray[:-1]),
        "Time_Str": list(Interval2NumberPeopleInNet.keys()),
        "NumberPeople": list(Interval2NumberPeopleInNet.values())
    }
    DfUnload = pl.DataFrame(data)
    VolumeTraffic = np.sum(DfUnload["NumberPeople"].to_numpy())
    DfUnload = DfUnload.with_columns((pl.col("NumberPeople")/VolumeTraffic).alias("FractionPeople"),
                                    (pl.col("NumberPeople").diff().shift(1).fill_null(0)).alias("DiffNumberPeople")
                                                )            
    # Add Time_hours column
    DfUnload = DfUnload.with_columns(
        (pl.col("Time") / 3600).alias("Time_hours")
    )
    DfUnload = DfUnload.filter(pl.col("Time") > t_start_control_group)
    DfUnload = DfUnload.with_columns((pl.col("Time_hours") - t_start_hour).alias("Time_hours_shifted"))
    DfUnload = DfUnload.with_columns((pl.col("FractionPeople") > 0).alias("IsPeople"))
    DfUnload = DfUnload.with_columns((pl.col("Time") >= t_end_control_group).alias("IsUnload"))
    return VolumeTraffic, DfUnload


if __name__ == "__main__":
    City = "SFO"
    savio_base_dir_data = "/global/scratch/p2p3/pl1_lbs/data/artemis_data/alberto/"
    # Initialize Rs and UCIs
    Df_Info_Simulations = pl.DataFrame(pd.read_csv(os.path.join(savio_base_dir_data,"info_simulations.csv")))
    Rs = Df_Info_Simulations.filter(pl.col("city") == City)["R"].unique().to_numpy()
    UCIs = Df_Info_Simulations.filter(pl.col("city") == City)["UCI"].unique().to_numpy()
    Df_City2Config = Df_Info_Simulations.filter(pl.col("city") == City)
    delta_t = Df_City2Config['delta_t'].unique().to_numpy()[0]
    # NOTE: Initialize the time variables
    logger.info(f"Initializing Time Variables")
    IntTimeArray, HourTimeArray, Interval2NumberPeopleInNet = InitializeTimeVariablesOutputStats(HOURS_IN_DAY,
                                                                                                MINUTES_IN_HOUR,
                                                                                                SECONDS_IN_MINUTE,
                                                                                                TIMESTAMP_OFFSET,
                                                                                                delta_t)

    # NOTE: Read Edges
    logger.info(f"Reading Edges")
    Edges = pl.DataFrame(pd.read_csv(os.path.join(savio_base_dir_data,City,"edges.csv")))
    # Time Variables
    t_start_control_group = 7*3600
    t_end_control_group = 8*3600
    t_start_hour = 7
    t_end_hour = 8
    
    for UCI in UCIs: 
        logger.info(f"Starting computation for UCI: {UCI}")   
        UCI_plot_dir = os.path.join(savio_base_dir_data,City,"Output",str(UCI),"Plots")
        UCI_route_people_dir = os.path.join(savio_base_dir_data,City,"Output",str(UCI))
        Df_config_UCI = Df_City2Config.filter(pl.col("UCI") == UCI)        
        # NOTE: Put a cut on R according to some considerations or not
        for R in sorted(Rs):
            # NOTE: Comupte the iters for the UCI, R at hand -> may be spourious since we had memory problem that stopped randomly the simulation
            Df_config_UCI_R = Df_config_UCI.filter(pl.col("R") == int(R),
                                                  )
            iters = Df_config_UCI_R['iter'].to_numpy()
            # NOTE: Different UCIs have different Rs, so you must condition the fact that the UCI under -> 
            # consideration contains the R we are sampling, given Rs is the union of R over all UCIs
            if len(Df_config_UCI_R) == 0:
                pass
            else:
                # Suffixes
                suffix_parquet_people_file = f'0_people{t_start_hour}to24.parquet'                                                                                                     # NOTE:   
                suffix_parquet_route_file = f'0_route{t_start_hour}to24.parquet'                                                                                           # Files Simulation                 
                # R,UCIs -> lists
                for iter in iters:
                    logger.info(f"starting computation UCI: {UCI}, R: {R}, iter: {iter}")
                    # NOTE: Compute the Unload Curve
                    name_unloading_curve_iter = f"UnloadCurve_R_{R}_UCI_{UCI}_iter_{iter}.parquet"                    
                    complete_path_unloading_curve_iter = os.path.join(UCI_plot_dir,str(R),name_unloading_curve_iter)
                    os.makedirs(os.path.join(UCI_plot_dir,str(R)),exist_ok=True)
                    # Buliding Complete Path: NOTE: fixed R,UCI -> varyiing the ensemble number
                    Name_parameters_output = f"R_{R}_UCI_{UCI}_realization_{iter}"
                    name_people_simulation_copied = f"{Name_parameters_output}_{suffix_parquet_people_file}"                                                                        # NOTE: name of the file that will be checked to be sure the output of the simulation is already there and we do not need to recompute it
                    name_route_simulation_copied = f"{Name_parameters_output}_{suffix_parquet_route_file}"                                                                          # NOTE: name of the file that will be checked to be sure the output of the simulation is already there and we do not need to recompute it
                    complete_path_people_file = os.path.join(UCI_route_people_dir,name_people_simulation_copied)                                                                 # NOTE: complete path of the file that will be checked to be sure the output of the simulation is already there and we do not need to recompute it
                    complete_path_route_file = os.path.join(UCI_route_people_dir,name_route_simulation_copied)                                                                 # NOTE: complete path of the file that will be checked to be sure the output of the simulation is already there and we do not need to recompute it
                    # NOTE: Check if the file exists  
                    logger.info(f"people file: {complete_path_people_file} exists?")              
                    if os.path.isfile(complete_path_people_file):
                        if ".parquet" in complete_path_people_file:
                            DfPeople = pl.DataFrame(pd.read_parquet(complete_path_people_file))
                        elif ".csv" in complete_path_people_file:
                            DfPeople = pl.DataFrame(pd.read_csv(complete_path_people_file))
                        is_DfPeople = True
                    else:
                        logger.error(f"File {complete_path_people_file} is not a valid file")
                        is_DfPeople = False
                    logger.info(f"route file: {complete_path_route_file} exists?")
                    if os.path.isfile(complete_path_route_file):
                        if ".parquet" in complete_path_route_file:
                            DfRoute = pl.DataFrame(pd.read_parquet(complete_path_route_file))
                        elif ".csv" in complete_path_route_file:
                            DfRoute = pl.DataFrame(pd.read_csv(complete_path_route_file,separator = ':'))
                            if DfRoute.schema["route"] == pl.Utf8:
                                pass
                        is_DfRoute = True
                    else:
                        logger.error(f"File {complete_path_route_file} is not a valid file")
                        is_DfRoute = False
                    
                    if is_DfPeople and is_DfRoute:
                        logger.info(f"People and Route DataFrames exist R: {R}, UCI: {UCI}, iter: {iter}")
                        # NOTE: initialize Route and People DataFrames
                        if not os.path.isfile(complete_path_unloading_curve_iter):
                            logger.info(f"Filter Control group {iter}")
                            #
                            DfControlGroup = FilterDfPeopleControlGroup(t_start_control_group,
                                                                        t_end_control_group,
                                                                        DfPeople,
                                                                        "time_departure",
                                                                        "last_time_simulated")
                            logger.info(f"Embed Control group in Roads and Time")
                            #
                            DfControlGroup = EmbdedTrajectoriesInRoadsAndTime(DfRoute,
                                                                            DfControlGroup,
                                                                            Edges)
                            
                            logger.info(f"Computing Unload Curve for iter {iter}")
                            #
                            VolumeTraffic, DfUnload = ComputeSingleUnloadCurve(Interval2NumberPeopleInNet,
                                                        IntTimeArray,
                                                        HourTimeArray,
                                                        t_start_control_group,
                                                        t_start_hour,
                                                        t_end_control_group,
                                                        DfControlGroup)   
                            logger.info(f"Saving Unload Curve for iter {iter} -> {complete_path_unloading_curve_iter}")
                            # NOTE: save the file              
                            DfUnload.write_csv(complete_path_unloading_curve_iter)
                        else:
                            logger.info(f"File {complete_path_unloading_curve_iter} already exists")
                else:
                    logger.error(f"People and Route DataFrames do not exist R: {R}, UCI: {UCI}, iter: {iter}")