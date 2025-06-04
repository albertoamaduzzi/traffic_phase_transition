from collections import defaultdict
import polars as pl
import os
import json
import numpy as np

def choose_columns_DfUnload_from_default(DfUnload,
                                        list_cols_to_choose = ['Time_hours_shifted',
                                                                'FractionPeople',
                                                                'DiffNumberPeople',
                                                                'Time',
                                                                'Time_hours',
                                                                'IsPeople',
                                                                'IsUnload']):
    
    return DfUnload.select(list_cols_to_choose)

# -------- Choice of Rs that are available for the UCIs in a UCI interval --------- #

def compute_UCI_interval_2_Rs_available_to_all_UCIs(UCI_interval_2_UCISimulation,
                                                    Df_Info_Simulations,
                                                    City):
    """
        Compute the Rs available to all UCIs in the UCI interval.
        NOTE: Take just those Rs for which we have the DfUnload 
    """
    UCI_interval_2_Rs_available_to_all_UCIs = defaultdict(list)
    for UCI_interval, UCIs in UCI_interval_2_UCISimulation.items():
        Rs_UCI = Df_Info_Simulations.filter(pl.col("city") == City).filter(pl.col("UCI").is_in(UCIs))["R"].unique().to_numpy()
        Rs_2_is_in_all_UCIs = {R:True for R in Rs_UCI}
        # TODO: Control that the Rs are present for each UCI
        for UCI in UCIs:
            # For each UCI I check all the Rs avaiable, if I find that for that particular UCI no R then the R will be discarded
            Rs_UCI = Df_Info_Simulations.filter(pl.col("city") == City).filter(pl.col("UCI") == UCI)["R"].unique().to_numpy()
            for R in Rs_2_is_in_all_UCIs.keys():
                # NOTE: if any of the simulations in the UCI_interval has not that R, then do not consider that R
                if R not in Rs_UCI:
                    Rs_2_is_in_all_UCIs[R] = False    
        UCI_interval_2_Rs_available_to_all_UCIs[UCI_interval] = [R for R in Rs_2_is_in_all_UCIs.keys() if Rs_2_is_in_all_UCIs[R] == True]
    return UCI_interval_2_Rs_available_to_all_UCIs

# -------------- Select the UCIs that are in the UCI interval -------------- #

def compute_UCI_interval_2_UCIs(UCI_intervals,
                                UCIs):
    """
        Compute the UCI interval to UCIs available to all UCI intervals.
    """
    UCI_interval_2_UCISimulation = defaultdict(list)
    for UCI in UCIs:
        for i in range(len(UCI_intervals)-1):
            if UCI_intervals[i] <= UCI < UCI_intervals[i+1]:
                UCI_interval_2_UCISimulation[UCI_intervals[i]].append(UCI)
    return UCI_interval_2_UCISimulation


def compute_average_Df_Unload_UCI_interval(UCIs,
                                           UCI_interval,
                                           dir_base_analyzed_simulations,
                                           R,
                                           t_start_hour,
                                           t_end_control_group,
                                           City
                                           ):
    """
        @params: UCIs list
        @params: UCI_interval float
        @params: dir_base_analyzed_simulations str -> path to the average DfUnload
        NOTE: Computes the average DfUnload for the R given the UCI interval.     
    """
    # NOTE: Average over All UCIs 
    list_DfUnload_R = []
    print(f"Append DfUnload case R: {R}, UCI_interval: {UCI_interval}")
    for UCI in UCIs:
        DfUnload_R = pl.read_parquet(os.path.join(dir_base_analyzed_simulations,str(UCI),"Plots",str(UCI),f"UnloadCurve_R_{R}_UCI_{UCI}.parquet"))
        DfUnload_R = choose_columns_DfUnload_from_default(DfUnload_R)
        list_DfUnload_R.append(DfUnload_R)
    print(f"Computing Average Unload Curve R: {R}, Unload curves")
    DfUnload = pl.concat(list_DfUnload_R, how="vertical")
    del list_DfUnload_R
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
    return DfUnload

def compute_average_Df_Unload_from_iterations(R,
                                              UCI,
                                              dir_base_analyzed_simulations,
                                              t_start_hour,
                                              t_end_control_group,
                                              ):
    """
        NOTE: This function reads from the file system the DfUnload for the R and UCI given.
        if there are the iteratations available then computes the average from them
    """
    # NOTE: Average over All UCIs 
    list_DfUnload_R = []
    print(f"Append DfUnload case R: {R}, UCI: {UCI}")
    dir_df_unload_iters = os.path.join(dir_base_analyzed_simulations,str(UCI),"Plots",str(R))
    list_files_unload = os.listdir(dir_df_unload_iters)
    # NOTE: If no iters available in the R dir, then upload the DfUnload.
    if len(list_files_unload) == 0:
        DfUnload = pl.read_parquet(os.path.join(dir_base_analyzed_simulations,str(UCI),"Plots",str(UCI),f"UnloadCurve_R_{R}_UCI_{UCI}.parquet"))
        DfUnload = choose_columns_DfUnload_from_default(DfUnload)
        return DfUnload
    # NOTE: Compute the average over the iterations of the ensemble
    else:
        try:
            for file in list_files_unload:
                if "UnloadCurve_R" in file and file.endswith(".parquet"):
                    print("uploading: ",os.path.join(dir_df_unload_iters,file))
                    try:
                        DfUnload_R = pl.read_parquet(os.path.join(dir_df_unload_iters,file))
                    except:
                        DfUnload_R = pl.read_csv(os.path.join(dir_df_unload_iters,file))      
                    DfUnload_R = choose_columns_DfUnload_from_default(DfUnload_R)
                    list_DfUnload_R.append(DfUnload_R)
            DfUnload = pl.concat(list_DfUnload_R, how="vertical")
            del list_DfUnload_R
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
        except:
            print(f"Error in computing the average DfUnload for R: {R}, UCI: {UCI}. Returning the DfUnload for the R and UCI.")
            DfUnload = pl.read_parquet(os.path.join(dir_base_analyzed_simulations,str(UCI),"Plots",str(UCI),f"UnloadCurve_R_{R}_UCI_{UCI}.parquet"))
            DfUnload = choose_columns_DfUnload_from_default(DfUnload)
        return DfUnload

def compute_average_gamma(UCIs,R,dir_base_analyzed_simulations):
    """
        @return the average of the gamma for the R given averaging over the UCIs that belong to the UCI_nterval at hand.
    """
    list_VolumeTraffic = []
    for UCI in UCIs:
        with open(os.path.join(dir_base_analyzed_simulations,str(UCI),"Plots",str(UCI),f"VolumeTraffic_R_{R}_UCI_{UCI}.json"), 'r') as f:
            VolumeTraffic = json.load(f)
        list_VolumeTraffic.append(float(VolumeTraffic))
    return np.mean(list_VolumeTraffic)
