from collections import defaultdict

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
      
def Init_Road2MFD(Time2Road2MFD):


