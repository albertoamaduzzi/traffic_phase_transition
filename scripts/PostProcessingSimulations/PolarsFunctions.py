import polars as pl
import numpy as np
def FilterDfPeopleInNetInTSlice(t0,t1,DfPeople,StrTimeDeparture,StrLastTimeSimulated):
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
        (pl.col(StrTimeDeparture) < int(t1)) &  # Corrected condition
        (int(t1) < pl.col(StrLastTimeSimulated))
    )
    DfPeopleInNetAtTimet = DfPeople.filter(condition)
    return DfPeopleInNetAtTimet

def AddTimeList2DfRoute(DfRoute,DfPeople):
    """
        Description:
            Add the time of departure and the last time simulated to the DfRoute DataFrame
    """
    People2Time = {row["p"]:[row["time_departure"],row["last_time_simulated"],row["avg_v(mph)"]] for row in DfPeople.iterrows()} 
    DfRoute["time_departure"] = DfRoute["p"].apply(lambda x: People2Time[x][0])
    DfRoute["last_time_simulated"] = DfRoute["p"].apply(lambda x: People2Time[x][1])
    DfRoute["avg_v(mph)"] = DfRoute["p"].apply(lambda x: People2Time[x][2])
    DfRoute["time"] = DfRoute.apply(lambda x: np.linspace(x["time_departure"],x["last_time_simulated"],(x["last_time_simulated"] - x["time_departure"])/len(x["route"])),axis = 1)
    return DfRoute
