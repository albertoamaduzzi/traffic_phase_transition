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
                              t_end_control_group,
                              iter_2_is_available_locally_DfUnload,
                              DfUnloadPrecomputed,
                              complete_path_unloading_curve_base):
    """
        @params iter_2_DfUnload: Iterator of DataFrame with the unload curve
        @description: Compute the average unload curve
        @return: Average unload curve
    """
    t_start_control_group  = t_end_control_group - 3600
    t_start_hour = t_start_control_group/3600
    columns_2_consider = ['Time_hours_shifted', 'FractionPeople', 'DiffNumberPeople', 'Time', 'Time_hours', 'IsPeople', 'IsUnload']        
    from os.path import exists
    # NOTE: If I have that the Unload curve exists, I need to average it with the files that I have already
    if DfUnloadPrecomputed is not None:
        logger.info(f"Having Unload Curve from {complete_path_unloading_curve_base}")        
        list_DfUnload = [DfUnloadPrecomputed.with_columns(pl.col("Time").cast(pl.Int64),
                                                          pl.col("DiffNumberPeople").cast(pl.Int64))]
        for i in iter_2_DfUnload.keys():
            if iter_2_is_available_locally_DfUnload[i]:
                list_DfUnload.append(iter_2_DfUnload[i].with_columns(pl.col("Time").cast(pl.Int64),
                                                          pl.col("DiffNumberPeople").cast(pl.Int64)).select(columns_2_consider))
                logger.info(f"Loaded {i} Unload Curve, shape: {iter_2_DfUnload[i].shape}")
            else:
                logger.info(f"Not loaded {i} Unload Curve")
        if len(list_DfUnload) == 1:
            logger.info("Not available new Unloading Curves")
            return DfUnloadPrecomputed
        else:
            logger.info(f"Computing Average Unload Curve from {iter_2_DfUnload.keys()}, Unload curves")
            DfUnload = pl.concat(list_DfUnload, how="vertical")
            del list_DfUnload
            DfUnload = DfUnload.group_by("Time_hours_shifted").agg(
                pl.col("FractionPeople").mean().alias("FractionPeople"),
                pl.col("DiffNumberPeople").mean().alias("DiffNumberPeople"),
            )
            # rebuild time
            DfUnload = DfUnload.with_columns(
                ((pl.col("Time_hours_shifted") + t_start_hour) * 3600).alias("Time"),
                (pl.col("Time_hours_shifted") + t_start_hour).alias("Time_hours"),
                (pl.col("FractionPeople") > 0).alias("IsPeople"),
            )
            DfUnload = DfUnload.with_columns((pl.col("Time") >= t_end_control_group).alias("IsUnload"))
            DfUnload = DfUnload.sort("Time_hours_shifted")

    # Check if the file exists
    if not exists(complete_path_unloading_curve_base):
        logger.info(f"Computing Average Unload Curve from {iter_2_DfUnload.keys()}, Unload curves")
        list_DfUnload = []
        for i in iter_2_DfUnload.keys():
            if iter_2_is_available_locally_DfUnload[i]:
                list_DfUnload.append(iter_2_DfUnload[i])
                logger.info(f"Loaded {i} Unload Curve, shape: {iter_2_DfUnload[i].shape}")
            else:
                logger.info(f"Not loaded {i} Unload Curve")
        DfUnload = pl.concat(list_DfUnload, how="vertical")
        # Free some memory as soon as you can
        del list_DfUnload
        DfUnload = DfUnload.group_by("Time_hours_shifted").agg(
            pl.col("FractionPeople").mean().alias("FractionPeople"),
            pl.col("DiffNumberPeople").mean().alias("DiffNumberPeople"),
        )
        # rebuild time
        DfUnload = DfUnload.with_columns(
            ((pl.col("Time_hours_shifted") + t_start_hour) * 3600).alias("Time"),
            (pl.col("Time_hours_shifted") + t_start_hour).alias("Time_hours"),
            (pl.col("FractionPeople") > 0).alias("IsPeople"),
        )
        DfUnload = DfUnload.with_columns((pl.col("Time") >= t_end_control_group).alias("IsUnload"))
        DfUnload = DfUnload.sort("Time_hours_shifted")
    return DfUnload


def ComputeAverageVolumeTraffic(iter_2_VolumeTraffic,
                                iter_2_is_available_locally_DfUnload,
                                VolumeTrafficPreComputed):
    """
        @params iter_2_VolumeTraffic: Iterator of VolumeTraffic
        @params iter_2_is_available_locally_DfUnload: Iterator of boolean values indicating if the DfUnload is available
        @params VolumeTrafficPreComputed: Precomputed VolumeTraffic
        @description: Compute the average volume traffic, if there exist some precomputed Volume then it comes from the set
            of 
        
        @return: Average volume traffic
    """
    # Initialize VolumeTraffic: 0 if not precomputed, otherwise use the precomputed value
    if VolumeTrafficPreComputed is not None:
        VolumeTraffic = VolumeTrafficPreComputed                                        # Already averaged: The one for which it is not available the DfUnload
    else:
        VolumeTraffic = 0

    # Count the number of iter for which we can compute the component of volume traffic 
    number_available_DfUnload = 0
    for i in range(0,len(iter_2_VolumeTraffic)):
        if iter_2_is_available_locally_DfUnload[i]:
            number_available_DfUnload += 1
            VolumeTraffic += iter_2_VolumeTraffic[i]
        else:
            pass
    # NOTE: case where VolumeTraffic is not computed nor precomputed with the new files
    if number_available_DfUnload == 0:
        pass    
    else:
        VolumeTraffic = VolumeTraffic/number_available_DfUnload
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