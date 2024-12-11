import datetime
from numpy import linspace
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
    SecondsInDay = int(HOURS_IN_DAY*MINUTES_IN_HOUR*SECONDS_IN_MINUTE)
    MinutesInDay = int(HOURS_IN_DAY*MINUTES_IN_HOUR)
    # I count people in Network at each time interval (15 mins)
    NumIntervals = int(MinutesInDay/delta_t)
    # Create the Int Time Array (0 -> Seconds In Day)
    IntTimeArray = linspace(0,TIMESTAMP_OFFSET + SecondsInDay,NumIntervals,dtype = int)       
    # Convert it to minutes seconds and hour for the plot Labels (0 -> 24)        
    HourTimeArray = ConvertArray2HMS(linspace(TIMESTAMP_OFFSET,TIMESTAMP_OFFSET + SecondsInDay,NumIntervals))
    Interval2NumberPeopleInNet = {Interval:0 for Interval in HourTimeArray}
    return IntTimeArray, HourTimeArray, Interval2NumberPeopleInNet

