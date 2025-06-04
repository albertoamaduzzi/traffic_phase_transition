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
    if len(IntTimeArray) != len(HourTimeArray):
        mask = np.array(IntTimeArray) >= t_start_control_group
        IntTimeArray = IntTimeArray[mask]
    keys_Interval2NumberPeopleInNet = list(Interval2NumberPeopleInNet.keys())
#    print("len(hourTimeArray)",len(HourTimeArray)," len(IntTimeArray)",len(IntTimeArray)," len(Interval2NumberPeopleInNet)",len(Interval2NumberPeopleInNet))
#    print("HourTimeArray[t]",HourTimeArray)
#    print("Interval2NumberPeopleInNet.keys()",Interval2NumberPeopleInNet.keys())
#    print("IntTimeArray",IntTimeArray)

    for t in range(len(Interval2NumberPeopleInNet.keys())-1):
        # DataFrame Filtered With People in the Network at time t in the time interval t,t+1
        DfPeopleInNetAtTimet = FilterDfPeopleStilInNet(IntTimeArray[t],
                                                       IntTimeArray[t+1],
                                                       "last_time_simulated",
                                                       DfControlGroup)
        # Number Of People in the Network at time t

        Interval2NumberPeopleInNet[keys_Interval2NumberPeopleInNet[t]] += len(DfPeopleInNetAtTimet) 
    data = {
        "Time": list(IntTimeArray[:-1]),
        "Time_Str": list(HourTimeArray[:-1]),
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
        if 1:
            - The unload curve is not precomputed, then it is computed from the iter_2_DfUnload, 
            NOTE: this happens for the cases for which I had
                previously computed the unload curve with the ensemble that was smaller. 
                So it allows to unify the case where I have already computed DfUnload and the one which I compute it with all iter_2_DfUnolad.
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
        # Compute the average with the unloading curve of iter i if available
        for i in iter_2_DfUnload.keys():
            if iter_2_is_available_locally_DfUnload[i]:
                list_DfUnload.append(iter_2_DfUnload[i].with_columns(pl.col("Time").cast(pl.Int64),
                                                          pl.col("DiffNumberPeople").cast(pl.Int64)).select(columns_2_consider))
                logger.info(f"Loaded {i} Unload Curve, shape: {iter_2_DfUnload[i].shape}")
            # If not available locally, then I do not load it 
            # NOTE:the case for which I have not computed the Unload curve in that 
            # iter is not taken into account, it avoids to add something that does not exist. It may occur when I have computed the iter 
            # in some PC and I did not move locally the files. So I just ignore them even though I know I computed them since they
            # are written in info_simulations.csv  
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
#            print("aggregated DfUnload\n",DfUnload)
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
        print("aggregated DfUnload\n",DfUnload)
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
        Keys are defined for the triplet (City,UCI,R): 
            - ErrorExp: Error of the exponential fit 
            - ErrorPL: Error of the power law fit
            - A_exp: Amplitude of the exponential fit
            - A_pl: Amplitude of the power law fit
            - alpha_exp: Exponent of the exponential fit
            - alpha_pl: Exponent of the power law fit
            - best_fit: Best fit (Exponential or PowerLaw)
            - IsJam: True if the network is jammed, False otherwise
            - Tau: Time constant of the exponential fit
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

def Modify_InfoTraffic_Tau_IfAlreadyJammed(InfoTraffic,found_jam, DfUnload,t0 = 0.5):
    """
        @params InfoTraffic: InfoTraffic dictionary
        @params found_jam: Boolean indicating if the jam is already found
        @params DfUnload: DataFrame with the unload curve
    """
    if found_jam:
        found_jam = True
        # Pick the nt0 after 30 mins after 8
        index_nt0 = np.where(DfUnload["Time_hours_shifted"].to_numpy() >= t0)[0][0]
#        nt0 = max(DfUnload["FractionPeople"].to_numpy())
        nt0 = DfUnload["FractionPeople"].to_numpy()[index_nt0]
        IndexTau = np.where(DfUnload["FractionPeople"].to_numpy()[index_nt0:] <= nt0*np.exp(-1))[0][0]
        Tau = DfUnload["Time_hours_shifted"].to_numpy()[IndexTau]
        InfoTraffic["Tau"] = Tau
        print(f"Jam found: index_nt0 {index_nt0}, t0: {DfUnload["Time_hours_shifted"].to_numpy()[index_nt0]}, nt0 {nt0}, IndexTau {IndexTau}, Tau = {Tau}")
        return InfoTraffic, found_jam
    else:
        return InfoTraffic, found_jam
