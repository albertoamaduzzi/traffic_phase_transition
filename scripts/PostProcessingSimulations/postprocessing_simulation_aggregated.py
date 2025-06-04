"""
    NOTE: For each City,UCI we define Rs_interval: 
        - set of Rs that are shared by the UCIs of certain UCI_interval -> In this way we can average DfUnload on UCI intervals
        - The UCIs available are those that have been computed in UCI_simulations
        NOTE: This just works if we UCIs,Rs are sub sample of all the available UCIs,Rs
        #   1) UCI_2_R2InfoTraffic {UCI: R: {ErrorExp,ErrorPL,A0,A_pl,alpha_exp,alpha_pl,Gamma,DfUnload,IsJam}}
        #   2) UCI_2_CriticalParams {UCI: {Rc,Alpha,Tauc,Gammac}}
        #   3) UCI_2_R2NtNtFit {UCI: {R: {[n,n_fit]}}}
        #   4) UCI_2_R2DfUnload {UCI: {R: {Df[Time,FractionPeople,DiffNumberPeople]}}}
        #   5) UCI_2_R2gamma
        #   6) UCI_2_Time2FitBeta, {UCI: {[n(R,epsilon)_fit], [n(R,epsilon)],[epsilon],ErrorPL,Beta}}
        #   7) UCI_2_gamma_tau
        
"""
from PlotsPercolationAggregated import *
from dictionaries_percolation_analysis import *
from pipeline_InfoTraffic import *
from pipeline_aggregation import *
from pipeline_relevant_choose_compute_df_unload import *
from Beta import *

def compute_City_2_UCI_2_var_interest(case_,
                                    Df_Info_Simulations,
                                    list_cities,
                                    t_start_hour,
                                    t_end_control_group,
                                    start_bin_time_considered_in_beta_fit,
                                    init_time_hour_cut_for_fit = 1,
                                    minimum_time_with_people = 10,
                                    max_time_for_fit = 6,
                                    UCI_intervals = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                                    base_dir_before_city = "/data_2/alberto"):
    """
        @cases: list of cases to analyze
        @Df_Info_Simulations: DataFrame with info about simulations
        @list_cities: list of cities to analyze
        @t_start_hour: hour to start the analysis
        @t_end_control_group: end hour of the control group
        @start_bin_time_considered_in_beta_fit: start bin time considered in beta fit
        @init_time_hour_cut_for_fit: initial time hour cut for fit
        @minimum_time_with_people: minimum time with people
        @max_time_for_fit: maximum time for fit
    """
    City_2_UCI_2_R2InfoTraffic = defaultdict() 
    City_2_UCI_2_CriticalParams = defaultdict()
    City_2_UCI_2_R2NtNtFit = defaultdict()
    City_2_UCI_2_Time2FitBeta = defaultdict()
    City_2_UCI_2_ErrorBeta_in_Time = defaultdict()
    City_2_UCI_interval_2_Rs_available_to_all_UCIs = defaultdict()
    City_2_UCI_interval_2_UCISimulation = defaultdict()
    # Filter by city
    for City in list_cities:
        # -- Analized simulations -> Put data There
        dir_base_analyzed_simulations = os.path.join(base_dir_before_city,City,"Output")                                               # NOTE: base dir for input
        dir_output_aggregated = os.path.join(dir_base_analyzed_simulations,case_)                                            # NOTE: base dir for output
        dir_output_aggregated_all_cities = os.path.join(base_dir_before_city,case_)                                             # NOTE: base dir for output when we aggregate over all cities
        os.makedirs(dir_output_aggregated,exist_ok=True)
        # NOTE: all available Rs, UCIs for city: -> for different UCIs, you have different Rs -> so must filter somehow
        UCIs = Df_Info_Simulations.filter(pl.col("city") == City)["UCI"].unique().to_numpy()
        # NOTE: {UCI_interval:[UCIs]}
        print(f"{City}: Computing map UCI_interval -> UCIs")
        UCI_interval_2_UCISimulation = compute_UCI_interval_2_UCIs(UCI_intervals,UCIs)
        print(f"{City}: UCI intervals -> UCI: ",UCI_interval_2_UCISimulation)
        # NOTE: {UCI_interval: [UsedRs]}
        print(f"{City}: Computing map UCI_interval -> Rs")
        UCI_interval_2_Rs_available_to_all_UCIs = compute_UCI_interval_2_Rs_available_to_all_UCIs(UCI_interval_2_UCISimulation,Df_Info_Simulations,City)
        print(f"{City}: UCI intervals -> Rs: ",UCI_interval_2_Rs_available_to_all_UCIs)
        # NOTE: Fill the dictionaries of UCIs and Rs used
        City_2_UCI_interval_2_Rs_available_to_all_UCIs[City] = UCI_interval_2_Rs_available_to_all_UCIs
        City_2_UCI_interval_2_UCISimulation[City] = UCI_interval_2_UCISimulation
        # NOTE: Initialize the variables that will contain informations about the DP analysis
        UCI_2_R2InfoTraffic = defaultdict() 
        UCI_2_CriticalParams = defaultdict()
        UCI_2_R2NtNtFit = defaultdict()
        UCI_2_Time2FitBeta = defaultdict()
        UCI_2_gamma_tau = defaultdict()
        UCI_2_R2DfUnload = defaultdict()
        UCI_2_R2gamma = defaultdict()
        UCI_2_ErrorBeta_in_Time = defaultdict()
        # NOTE: Compute things UCI by UCI
        for UCI_interval, UCIs in UCI_interval_2_UCISimulation.items():
            # NOTE: Compute the Rs about which we are interested.
            Rs_interval = UCI_interval_2_Rs_available_to_all_UCIs[UCI_interval]
            for UCI in UCIs:
                print(f"{City}: Available Rs: ",Rs_interval)        
                UCI_2_R2InfoTraffic, UCI_2_CriticalParams, UCI_2_R2NtNtFit, UCI_2_R2DfUnload, UCI_2_R2gamma, UCI_2_Time2FitBeta, UCI_2_gamma_tau,found_jam, is_found_beta,Errors,Betas,time_,t_vect = pipeline_single_UCI(UCI_2_R2InfoTraffic,
                                                                                                                                                                                                                                                UCI_2_CriticalParams,
                                                                                                                                                                                                                                                UCI_2_R2NtNtFit,
                                                                                                                                                                                                                                                UCI_2_R2DfUnload,
                                                                                                                                                                                                                                                UCI_2_R2gamma,
                                                                                                                                                                                                                                                UCI_2_Time2FitBeta,
                                                                                                                                                                                                                                                UCI_2_gamma_tau,
                                                                                                                                                                                                                                                None,
                                                                                                                                                                                                                                                Rs_interval,                            # NOTE: The Rs are associated beforehand to the UCI at hand
                                                                                                                                                                                                                                                UCI,
                                                                                                                                                                                                                                                None,
                                                                                                                                                                                                                                                City,
                                                                                                                                                                                                                                                case_,
                                                                                                                                                                                                                                                t_start_hour,
                                                                                                                                                                                                                                                t_end_control_group,
                                                                                                                                                                                                                                                start_bin_time_considered_in_beta_fit,
                                                                                                                                                                                                                                                dir_base_analyzed_simulations,
                                                                                                                                                                                                                                                init_time_hour_cut_for_fit = init_time_hour_cut_for_fit,
                                                                                                                                                                                                                                                minimum_time_with_people = minimum_time_with_people,
                                                                                                                                                                                                                                                max_time_for_fit = max_time_for_fit

                                                                                                                                                                                                                                                )        # NOTE: Output from the previous line: 
                UCI_2_ErrorBeta_in_Time[UCI] = fill_UCI_2_ErrorBeta_in_Time(Errors,Betas,time_)
        # NOTE: Writing InfoTraffic, DfUnload 
        save_jsons_DP(dir_output_aggregated,
                UCI_2_R2InfoTraffic,
                UCI_2_CriticalParams,
                UCI_2_R2NtNtFit,
                UCI_2_Time2FitBeta,
                UCI_2_ErrorBeta_in_Time,
                case_)        
        R2NtNtFit_converted = convert_ndarray(UCI_2_R2NtNtFit)        
        City_2_UCI_2_R2InfoTraffic[City] = UCI_2_R2InfoTraffic
        City_2_UCI_2_CriticalParams[City] = UCI_2_CriticalParams
        City_2_UCI_2_R2NtNtFit[City] = R2NtNtFit_converted
        City_2_UCI_2_Time2FitBeta[City] = UCI_2_Time2FitBeta
        City_2_UCI_2_ErrorBeta_in_Time[City] = UCI_2_ErrorBeta_in_Time
    os.makedirs(dir_output_aggregated_all_cities, exist_ok=True)
    save_jsons_DP(dir_output_aggregated_all_cities,
                   City_2_UCI_2_R2InfoTraffic,
                   City_2_UCI_2_CriticalParams,
                   City_2_UCI_2_R2NtNtFit,
                   City_2_UCI_2_Time2FitBeta,
                   City_2_UCI_2_ErrorBeta_in_Time,
                   case_)
    
    return City_2_UCI_2_R2InfoTraffic, City_2_UCI_2_CriticalParams, City_2_UCI_2_R2NtNtFit, City_2_UCI_2_Time2FitBeta, City_2_UCI_2_ErrorBeta_in_Time,City_2_UCI_interval_2_UCISimulation, City_2_UCI_interval_2_Rs_available_to_all_UCIs,t_vect
