import datetime
HOURS_IN_DAY = 24
MINUTES_IN_HOUR = 60
SECONDS_IN_MINUTE = 60
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
    
