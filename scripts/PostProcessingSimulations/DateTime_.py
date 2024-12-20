import datetime
import numpy as np
def date_to_timestamp(date_str):
    """
    Convert a date string in "%Y-%m-%d" format to a timestamp.    
    Parameters:
    date_str (str): The date string in "%Y-%m-%d" format.    
    Returns:
    float: The corresponding timestamp.
    """
    # Parse the date string into a datetime object
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    # Convert the datetime object to a timestamp
    timestamp = date_obj.timestamp()
    return timestamp
def ConvertMinuteVectorToSeconds(VectorMinutes):
    """
        @param VectorMinutes: Vector with minutes
        @return VectorSeconds: Vector with seconds
    """
    VectorSeconds = [minutes*60 for minutes in VectorMinutes]
    return np.array(VectorSeconds)
def ConvertVectorMinutesToDatetime(VectorMinutes,Start_Day_In_Hour,Day = "2023-08-15"):
    """
        @param VectorMinutes: Vector with minutes
        @return VectorDatetime: Vector with datetime
    """
    StartTimeStamp = date_to_timestamp(Day)
    ShiftFromMidnight = Start_Day_In_Hour*3600
    VectorSeconds = ConvertMinuteVectorToSeconds(VectorMinutes)
    # Vector [timestamp(Day,Shift),..,timestamp(Day,Shift) + N*dt)]
    VectorSeconds = VectorSeconds + StartTimeStamp + ShiftFromMidnight
    # [datetime(Day,Shift),..,datetime(Day,Shift) + N*dt]
    VectorDatetime = [datetime.datetime.fromtimestamp(seconds).strftime('%Y-%m-%d %H:%M:%S') for seconds in VectorSeconds]
    return VectorDatetime
def CastVectorDateTime2Hours(VectorDateTime):
    """
        @param VectorDateTime: Vector with datetime
        @return VectorHours: Vector with hours
    """
    VectorHMS = [datetime.datetime.strptime(Datetime, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S").split(" ")[1] for Datetime in VectorDateTime]
#    VectorHMS = [(datetime.datetime.strptime(Datetime, "%Y-%m-%d %H:%M:%S").hour,
#                  datetime.datetime.strptime(Datetime, "%Y-%m-%d %H:%M:%S").minute,
#                  datetime.datetime.strptime(Datetime, "%Y-%m-%d %H:%M:%S").second)
#                 for Datetime in VectorDateTime]
    return VectorHMS