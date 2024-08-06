from collections import defaultdict
import numpy as np
import polars as pl
def AddMessageToLog(Message,LogFile):
    with open(LogFile,'a') as f:
        f.write(Message+'\n')


def Init_Time2Road2MFDNotProcessed(HourTimeArray,ListIntRoads,StrSpeed,StrUsersId,StrNumberPeople):
    """
        Description:
            Initialize the dictionary that will contain the number of people in each road at each time interval
        Args:
            HourTimeArray: List of strings containing the time intervals
            ListIntRoads: List of strings containing the road names
        Returns:
            Time2Road2Traveller: Dictionary containing the number of people in each road at each time interval
    """
    return {t: {IntRoad:{StrUsersId:[],StrSpeed:[],StrNumberPeople:0} for IntRoad in ListIntRoads} for t in HourTimeArray}


def Init_Time2Road2MFD(Time2Road2MFDNotProcessed,StrSpeed,StrNumberPeople):
    """
        Description:
            Initialize the dictionary that will contain the number of people in each road at each time interval
        Args:
            HourTimeArray: List of strings containing the time intervals
            ListIntRoads: List of strings containing the road names
        Returns:
            Time2Road2Traveller: Dictionary containing the number of people in each road at each time interval
    """
    return {t: {IntRoad:{StrSpeed:0,StrNumberPeople:0} for IntRoad in Time2Road2MFDNotProcessed[t].keys()} for t in Time2Road2MFDNotProcessed.keys()}  

def ComputeAvgTime2Road2MFD(Time2Road2MFDNotProcessed,Time2Road2MFD,StrSpeed,StrNumberPeople):
    """
        Description:
            Compute the average speed and the number of people in each road at each time interval
        Args:
            Time2Road2MFDNotProcessed: Dictionary containing the number of people in each road at each time interval
        Returns:
            Time2Road2MFD: Dictionary containing the average speed and the number of people in each road at each time interval
    """
    for t in Time2Road2MFDNotProcessed.keys():
        for Road in Time2Road2MFDNotProcessed[t].keys():
            Time2Road2MFD[t][Road][StrSpeed] = np.mean(Time2Road2MFDNotProcessed[t][Road][StrSpeed])
            Time2Road2MFD[t][Road][StrNumberPeople] = Time2Road2MFDNotProcessed[t][Road][StrNumberPeople]
    return Time2Road2MFD

def Init_Road2MFD2Plot(Time2Road2MFD,StrSpeed,StrNumberPeople,GeoJsonEdges):
    """
        For each Road Add the Speed and the Number of People per time

    """
#    SpeedList = np.array((len(list(Time2Road2MFD.keys())),1))
#    NumberPeopleList = np.array((len(list(Time2Road2MFD.keys())),1))
    Road2Time2MFDPlot = {Road:{t:{StrSpeed:np.empty(len(list(Time2Road2MFD.keys()))),StrNumberPeople:np.empty(len(list(Time2Road2MFD.keys())))} for t in Time2Road2MFD.keys()}for Road in Time2Road2MFD[list(Time2Road2MFD.keys())[0]]}
    Road2MFDPlot = {Road:{StrSpeed:np.empty(len(list(Road2Time2MFDPlot[Road].keys()))),StrNumberPeople:np.empty(len(list(Road2Time2MFDPlot[Road].keys())))} for Road in Road2Time2MFDPlot.keys()}
    for Road in Time2Road2MFD[list(Time2Road2MFD.keys())[0]]:
        for t in Time2Road2MFD.keys():
            Road2Time2MFDPlot[Road][t][StrSpeed] = np.append(Road2Time2MFDPlot[Road][t][StrSpeed],Time2Road2MFD[t][Road][StrSpeed])
            Road2Time2MFDPlot[Road][t][StrNumberPeople] = np.append(Road2Time2MFDPlot[Road][t][StrNumberPeople],Time2Road2MFD[t][Road][StrNumberPeople])
    #            SpeedList = np.append(SpeedList,Time2Road2MFD[t][Road][StrSpeed],axis =1)
    #            NumberPeopleList = np.append(NumberPeopleList,Time2Road2MFD[t][Road][StrNumberPeople],axis =1)
    for Road in Road2Time2MFDPlot.keys():
        for t in range(len(Road2Time2MFDPlot[Road].keys())):
            TmpList = list(Road2Time2MFDPlot[Road].keys())
            if len(Road2Time2MFDPlot[Road][TmpList[t]][StrSpeed])!=0:
                Road2MFDPlot[Road][StrSpeed][t] = np.mean(Road2Time2MFDPlot[Road][TmpList[t]][StrSpeed])
            else:
                Road2MFDPlot[Road][StrSpeed][t] = GeoJsonEdges.filter(pl.col("uv") == Road)["maxspeed_int"].to_list()[0]
            if len(Road2Time2MFDPlot[Road][TmpList[t]][StrNumberPeople])!=0:
                Road2MFDPlot[Road][StrNumberPeople][t] = np.mean(Road2Time2MFDPlot[Road][TmpList[t]][StrNumberPeople])
            else:
                Road2MFDPlot[Road][StrNumberPeople][t] = 0
#    Road2MFDPlot = {Road:{StrSpeed:np.mean(Road2Time2MFDPlot[Road][t][StrSpeed]),StrNumberPeople:np.sum(NumberPeopleList,axis = 1)} for Road in Road2MFDPlot}
    return Road2MFDPlot


def Init_MFD2Plot(Road2MFD2Plot,StrSpeed,StrNumberPeople):
    """
        Average the speed on the road averaged for the number of people.
        I will have information aggregated among all possible roads.
        Super Quick, Super Slow, al together
    """
    MFDPlot = {StrSpeed:[],StrNumberPeople:[]}
    for Road in Road2MFD2Plot.keys():
        MFDPlot[StrSpeed].append(Road2MFD2Plot[Road][StrSpeed]*Road2MFD2Plot[Road][StrNumberPeople])
        MFDPlot[StrNumberPeople].append(Road2MFD2Plot[Road][StrNumberPeople]) 
    return MFDPlot


# MESSAGE
def ReturnMessageTime2Road(CountFunctions,LogFile,Time2Road2MFDNotProcessed,Time2Road2MFD,Road2MFD2Plot):
    Message = f"Function {CountFunctions}: ComputeTime2Road2Traveller: Compute the number of people in the road network at each time interval"
    AddMessageToLog(Message,LogFile)
    Message = f"Time2Road2MFDNotProcessed:\n"
    Count0 = 0
    for key in Time2Road2MFDNotProcessed.keys():
        Count1 = 0
        if Count0 < 3:
            Count0 += 1
            for Road in Time2Road2MFDNotProcessed[key].keys():
                if Count1 < 3:
                    Message += f"{key}: {Road} -> p:{Time2Road2MFDNotProcessed[key][Road]["p"][:3]},avg_v(mph):{Time2Road2MFDNotProcessed[key][Road]["avg_v(mph)"][:3]},NumberPeople: {Time2Road2MFDNotProcessed[key][Road]["NumberPeople"]}\n"
                    Count1 += 1
                else:
                    break
        else:
            break
    AddMessageToLog(Message,LogFile)
    Message = f"Time2Road2MFD:\n"
    Count0 = 0
    for key in Time2Road2MFD.keys():
        Count1 = 0
        if Count0 < 3:
            Count0 += 1
            for Road in Time2Road2MFD[key].keys():
                if Count1 < 3:
                    Message += f"{key}: {Road} -> p:{Time2Road2MFD[key][Road]["p"][:3]},avg_v(mph):{Time2Road2MFD[key][Road]["avg_v(mph)"][:3]},NumberPeople: {Time2Road2MFD[key][Road]["NumberPeople"]}\n"
                    Count1 += 1
                else:
                    break
        else:
            break
    AddMessageToLog(Message,LogFile)
    Message = f"Road2MFD2Plot:\n"
    Count0 = 0
    for Road in Road2MFD2Plot.keys():
        if Count0 < 3:
            Count0 += 1
            Message += f"{Road}: {Road2MFD2Plot[Road]}\n"
        else:
            break
    AddMessageToLog(Message,LogFile)