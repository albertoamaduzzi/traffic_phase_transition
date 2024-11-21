import polars as pl
import numpy as np
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

def AddTimeList2DfRoute(DfRoute,DfPeople):
    """
        Description:
            Add the time of departure and the last time simulated to the DfRoute DataFrame
    """
    Column2IndexDfPeople = GetColumn2Index(DfPeople)
    Column2IndexDfRoute = GetColumn2Index(DfRoute)
    People2Time = {row[Column2IndexDfPeople["p"]]: [row[Column2IndexDfPeople["time_departure"]], row[Column2IndexDfPeople["last_time_simulated"]], row[Column2IndexDfPeople["avg_v(mph)"]]] for row in DfPeople.rows()}
#    People2Time = {row["p"]:[row["time_departure"],row["last_time_simulated"],row["avg_v(mph)"]] for row in DfPeople.iterrows()} 
    DfRoute = DfRoute.to_pandas()
    DfRoute["time_departure"] = DfRoute["p"].apply(lambda x: ApplyPeople2Time(x,People2Time,0))
    DfRoute["last_time_simulated"] = DfRoute["p"].apply(lambda x: ApplyPeople2Time(x,People2Time,1))
    DfRoute["avg_v(mph)"] = DfRoute["p"].apply(lambda x: ApplyPeople2Time(x,People2Time,2))
#    DfRoute = DfRoute.with_columns(DfRoute["p"].apply(lambda x: ApplyPeople2Time(x,People2Time,0),return_dtype = pl.Int32).alias("time_departure"))
#    DfRoute = DfRoute.with_columns(DfRoute["p"].apply(lambda x: ApplyPeople2Time(x,People2Time,1),return_dtype = pl.Int32).alias("last_time_simulated"))
#    DfRoute = DfRoute.with_columns(DfRoute["p"].apply(lambda x: ApplyPeople2Time(x,People2Time,2),return_dtype = pl.Int32).alias("avg_v(mph)"))
#    DfRoute = DfRoute.to_pandas()
    DfRoute["time"] = DfRoute.apply(lambda x: np.linspace(x["time_departure"],x["last_time_simulated"],len(x["route"])),axis = 1)
    DfRoute = pl.DataFrame(DfRoute)
    Column2IndexDfRoute = GetColumn2Index(DfRoute)
    return DfRoute,Column2IndexDfRoute

def GetColumn2Index(Df):
    """
        Description:
            Get the column names and the index of the DataFrame
        Args:
            Df: pl.DataFrame -> DataFrame
        Returns:
            Column2Index: dict -> Dictionary with the column names as keys and the index as values
    """
    Column2Index = {col: i for i, col in enumerate(Df.columns)}
    return Column2Index