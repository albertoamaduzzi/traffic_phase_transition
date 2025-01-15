import polars as pl
import numpy as np
from EfficiencyAnalysis import *
### GENERIC PREPROCESSING FUNCTIONS

def DropColumnsDfIfThere(Df,columns):
    """
        @param Df: DataFrame
        @param columns: List of columns to drop
        @return Df: DataFrame without the columns in columns
    """
    Column2Drop = []
    for column in columns:
        if column in Df.columns:
            Column2Drop.append(column)
    Df = Df.drop(Column2Drop)
    return Df

def DropDuplicateFromSubsetColumns(Df,columns):
    """
        @param Df: DataFrame
        @param columns: List of columns to consider for the duplicates
        @return Df: DataFrame without the duplicates in the subset columns
    """
    return Df.unique(subset=columns, keep='first')


####




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

def ApplyPeople2Time(personId,People2Time,i):
    return People2Time[personId][i]

@timing_decorator
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

