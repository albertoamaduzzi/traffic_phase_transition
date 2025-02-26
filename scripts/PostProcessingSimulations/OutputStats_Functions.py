"""
    This files contains the functions, organized in blocks of usage to compute
    quantities of interest in OutputStats.py
"""
from collections import defaultdict
import logging
import polars as pl
import numpy as np
from PolarsFunctions import FilterDfPeopleStilInNet
logger = logging.getLogger(__name__)


############# Definition control group ############




############ Unload Curves ############
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


def ComputeAverageUnloadCurve(iter_2_DfUnload,
                              t_end_control_group):
    """
        @params iter_2_DfUnload: Iterator of DataFrame with the unload curve
        @description: Compute the average unload curve
        @return: Average unload curve
    """
    DfUnload = iter_2_DfUnload[0]
    for i in range(1,len(iter_2_DfUnload)):
        DfUnload = DfUnload.vstack(iter_2_DfUnload[i])
    DfUnload = DfUnload.groupby("Time_hours_shifted").agg(
        pl.col("FractionPeople").mean().alias("FractionPeople"),
        pl.col("DiffNumberPeople").mean().alias("DiffNumberPeople"),
    )
    DfUnload = DfUnload.with_columns((pl.col("FractionPeople") > 0).alias("IsPeople"))
    DfUnload = DfUnload.with_columns((pl.col("Time") >= t_end_control_group).alias("IsUnload"))
    return DfUnload


def ComputeAverageVolumeTraffic(iter_2_VolumeTraffic):
    """
        @params iter_2_VolumeTraffic: Iterator of VolumeTraffic
        @description: Compute the average volume traffic
        @return: Average volume traffic
    """
    VolumeTraffic = iter_2_VolumeTraffic[0]
    for i in range(1,len(iter_2_VolumeTraffic)):
        VolumeTraffic += iter_2_VolumeTraffic[i]
    VolumeTraffic = VolumeTraffic/len(iter_2_VolumeTraffic)
    return VolumeTraffic

############ Fit ################ 
def FillInfoTraffic(ErrorExp, ErrorPL, A0, A_pl, alpha_exp, alpha_pl,city):
    """
        @params ErrorExp: Error of the exponential fit
        @params ErrorPL: Error of the power law fit
        @params A0: Amplitude of the exponential fit
        @params A_pl: Amplitude of the power law fit
        @params alpha_exp: Exponent of the exponential fit
        @params alpha_pl: Exponent of the power law fit
        @params city: Name of the city
        @description: Fill the InfoTraffic dictionary with the results of the fits
        @return: InfoTraffic dictionary
    """
    InfoTraffic = defaultdict()
    InfoTraffic["ErrorExp"] = ErrorExp
    InfoTraffic["ErrorPl"] = ErrorPL
    InfoTraffic["A_exp"] = A0
    InfoTraffic["A_pl"] = A_pl
    InfoTraffic["alpha_exp"] = alpha_exp
    InfoTraffic["alpha_pl"] = alpha_pl
    if ErrorExp<ErrorPL:
        InfoTraffic["best_fit"] = "Exponential"
        InfoTraffic["IsJam"] = False
        InfoTraffic["Tau"] = - 1/InfoTraffic["alpha_exp"]       # In hours
        logger.info("{0} Network is not jammed at time with alpha = None and Tau = {1}".format(city,InfoTraffic["Tau"]))
    else:
        InfoTraffic["best_fit"] = "PowerLaw"
        InfoTraffic["IsJam"] = True
        InfoTraffic["Tau"] = - 1/InfoTraffic["alpha_exp"]
        logger.info(f"{city} Network is jammed with alpha = {alpha_pl} and Tau = None")
    return InfoTraffic