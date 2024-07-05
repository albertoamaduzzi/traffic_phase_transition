from collections import defaultdict
import numpy as np
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
      
def Init_Road2MFD2Plot(Time2Road2MFD,StrSpeed,StrNumberPeople):
    """
        For each Road Add the Speed and the Number of People per time

    """
    SpeedList = np.array((len(list(Time2Road2MFD.keys())),1))
    NumberPeopleList = np.array((len(list(Time2Road2MFD.keys())),1))
    for Road in Time2Road2MFD[list(Time2Road2MFD.keys())[0]]:
        for t in Time2Road2MFD.keys():
            SpeedList = np.append(SpeedList,Time2Road2MFD[t][Road][StrSpeed],axis =1)
            NumberPeopleList = np.append(NumberPeopleList,Time2Road2MFD[t][Road][StrNumberPeople],axis =1)
    Road2MFDPlot = {Road:{StrSpeed:np.mean(SpeedList,axis = 1),StrNumberPeople:np.sum(NumberPeopleList,axis = 1)} for Road in Road2MFDPlot}
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
