# Parameters
from PlotsPercolationAggregated import *
from dictionaries_percolation_analysis import *
from pipeline_InfoTraffic import *
from pipeline_aggregation import *
from pipeline_relevant_choose_compute_df_unload import *
from Beta import *
from postprocessing_simulation_aggregated import *
# Input
t_start_hour = 7
t_end_control_group = 8*3600
start_bin_time_considered_in_beta_fit = 39
# --- Completed Simulations
BaseConfig = os.path.join(os.environ["TRAFFIC_DIR"],"config")                           # Info about simulations
list_cities = os.listdir(os.path.join(os.environ["TRAFFIC_DIR"],'data','carto'))        # Info about cities
list_cities = ["RIO","SFO"]#["SFO","RIO"]#["BOS","SFO","RIO"]
ListPeopleFile = []
ListRoutesFile = []
TRAFFIC_DIR = os.environ["TRAFFIC_DIR"]                                                 # NOTE: base dir project
complete_path_info_simulations_df = os.path.join(TRAFFIC_DIR,"info_simulations.csv")
# Available simulations
Df_Info_Simulations = pl.read_csv(complete_path_info_simulations_df)
# The cases are thought as:
# 1) Aggregated_UCIs: -> UCI interval -> each DfUnload corresponds to the average of the DfUnload of all the UCIs in the UCI interval
# 2) Single_UCI: -> UCI interval -> each DfUnload corresponds to the DfUnload of the UCI in the UCI interval
cases = ["Single_UCI"]                                                                  #["Aggregated_UCIs","Single_UCI"]
case_ = "Single_UCI"                                                                    # Choose the case to analyze
delta_RC_plot_before = 120
delta_RC_plot_after = 80
delta_RC_plot_1 = 200

# NOTE: FIT Params
init_time_hour_cut_for_fit = 1
minimum_time_with_people = 10
max_time_for_fit = 6

# UCI
UCI_intervals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # UCI intervals to consider

colors = [
    # Original 20 colors
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
    '#aec7e8',  # Light blue
    '#ffbb78',  # Light orange
    '#98df8a',  # Light green
    '#ff9896',  # Light red
    '#c5b0d5',  # Light purple
    '#c49c94',  # Light brown
    '#f7b6d2',  # Light pink
    '#c7c7c7',  # Light gray
    '#dbdb8d',  # Light olive
    '#9edae5',  # Light cyan
    
    # Darker versions of the first 10
    '#0d5a8c',  # Dark blue
    '#d35400',  # Dark orange
    '#1e6b1e',  # Dark green
    '#b21e1e',  # Dark red
    '#6d4394',  # Dark purple
    '#6d4331',  # Dark brown
    '#c2549c',  # Dark pink
    '#4d4d4d',  # Dark gray
    '#9a9c18',  # Dark olive
    '#0e8c98',  # Dark cyan
    
    # Additional colors
    '#8a2be2',  # Blueviolet
    '#a52a2a',  # Brown
    '#deb887',  # Burlywood
    '#5f9ea0',  # Cadetblue
    '#7fff00',  # Chartreuse
    '#d2691e',  # Chocolate
    '#dc143c',  # Crimson
    '#00ffff',  # Cyan/Aqua
    '#00008b',  # Dark blue
    '#008b8b',  # Dark cyan
    
    # Lighter versions of additional colors
    '#c299ff',  # Light blueviolet
    '#d48a8a',  # Light brown
    '#ffe4c4',  # Light burlywood
    '#a3d1d3',  # Light cadetblue
    '#c0ff84',  # Light chartreuse
    '#e8ad7d',  # Light chocolate
    '#f08080',  # Light crimson
    '#e0ffff',  # Light cyan
    '#7171c6',  # Light dark blue
    '#66cccc'   # Light dark cyan
]

UCI_interval_2_color = get_UCI_interval_2_colormap(UCI_intervals)


################ COMPUTE THE CRITICAL DICTIONARIES ################


City_2_UCI_2_R2InfoTraffic, City_2_UCI_2_CriticalParams, City_2_UCI_2_R2NtNtFit, City_2_UCI_2_Time2FitBeta, City_2_UCI_2_ErrorBeta_in_Time,City_2_UCI_interval_2_UCISimulation, City_2_UCI_interval_2_Rs_available_to_all_UCIs,t_vect = compute_City_2_UCI_2_var_interest(case_,
                                                                                                                                                                                                                                                                Df_Info_Simulations,
                                                                                                                                                                                                                                                                list_cities,
                                                                                                                                                                                                                                                                t_start_hour,
                                                                                                                                                                                                                                                                t_end_control_group,
                                                                                                                                                                                                                                                                start_bin_time_considered_in_beta_fit,
                                                                                                                                                                                                                                                                init_time_hour_cut_for_fit = init_time_hour_cut_for_fit,
                                                                                                                                                                                                                                                                minimum_time_with_people = minimum_time_with_people,
                                                                                                                                                                                                                                                                max_time_for_fit = max_time_for_fit)

################# PLOT CITIES SEPARATELY #################

for City in list_cities:
    dir_base_analyzed_simulations = os.path.join("/data_2/alberto",City,"Output")                                               # NOTE: base dir for input
    dir_output_aggregated = os.path.join(dir_base_analyzed_simulations,case_)                                            # NOTE: base dir for output
    dir_output_aggregated_all_cities = os.path.join("/data_2/alberto",case_)                                             # NOTE: base dir for output when we aggregate over all cities    
    print(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxx {City} Plotting xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    color_idx = 0
    fig_nr,ax_nr = plt.subplots(1,1,figsize = (10,10))
    ax_inset = create_inset_plot(fig_nr,ax_nr)
    fig_gamma_tau,ax_gamma_tau = plt.subplots(1,1,figsize = (10,10))
    fig_tau_alpha,ax_tau_alpha = plt.subplots(1,1,figsize = (10,10))
    fig_alpha_beta,ax_alpha_beta = plt.subplots(1,1,figsize = (10,10))
    UCI_2_Time2FitBeta = City_2_UCI_2_Time2FitBeta[City]
    UCI_2_R2NtNtFit = City_2_UCI_2_R2NtNtFit[City]
    UCI_2_CriticalParams = City_2_UCI_2_CriticalParams[City]
    UCI_2_R2InfoTraffic = City_2_UCI_2_R2InfoTraffic[City]
    UCI_interval_2_UCISimulation = City_2_UCI_interval_2_UCISimulation[City]
    UCI_interval_2_Rs_available_to_all_UCIs = City_2_UCI_interval_2_Rs_available_to_all_UCIs[City]
    for UCI_interval, UCIs in UCI_interval_2_UCISimulation.items():
        # NOTE: Clarify whether the phase transition is found or not
        for UCI in UCIs:
            if City == "SFO":
                condition = UCI > 0.4 and UCI != 0.564 and UCI < 0.9
            elif City == "RIO":
                condition = True
            if condition:
                Gammas, Taus, Rs_plot_tau = get_vecs_gammas_taus_from_R2Infotraffic(UCI_2_R2InfoTraffic[UCI])
                print(f"{City}: get gammas,taus from UCI: {UCI}")
                Rc,Tauc,alpha,Gammac,is_phase_transition_found = get_Rc_Tc_Gc_is_pht_from_CritPar(UCI_2_CriticalParams,UCI)            
                # NOTE: plot_gamma_tau needed variables                
                UCI_plots = UCI
                idx_color = get_UCI_interval_from_UCI(UCI,UCI_intervals)
                color =  colors[idx_color]
                
                Rs_interval = UCI_interval_2_Rs_available_to_all_UCIs[UCI_interval] if UCI_interval in UCI_interval_2_Rs_available_to_all_UCIs else []
                if is_phase_transition_found:
                    Time2FitBeta = UCI_2_Time2FitBeta[UCI]
                    R2NtNtFit = UCI_2_R2NtNtFit[UCI]
                    CriticalParams = UCI_2_CriticalParams[UCI]
                    t_star = CriticalParams["t_infty"]      
                    beta = CriticalParams["beta"]
                    IndexRc = [i for i, R in enumerate(Rs_interval) if int(R) == int(Rc)][0]
                    if beta > 0:
                        # NOTE: plot n(R)
                        int_t_star = CriticalParams["int_t_infty"]
                        nRs,RsOverRc, epsilons = get_nR_R2Rc_epsilons_from_R2NtNtFit(R2NtNtFit, int_t_star,Rc)
                        epsilon_fit = Time2FitBeta[int_t_star]["epsilon"]
                        nR_fit = Time2FitBeta[int_t_star]["n(R,epsilon)_fit"]
                        nR_region_fit = Time2FitBeta[int_t_star]["n(R,epsilon)"]
                        print(f"{City}: Phase Trans - Plotting n(R) UCI: {UCI}") 
                        print("nR: ",nRs)
                        print("nR_fit: ",nR_fit)
                        ax_nr =     single_plot_nr(RsOverRc,
                                                    nRs,
                                                    epsilons,
                                                    City,
                                                    ax_nr,
                                                    color,
                                                    beta_ = beta,
                                                    UCI = UCI,                                                # NOTE: UCI -> UCI of the city
                                                    )
                        ax_inset = single_plot_inset_nR(ax_inset,
                                        nR_fit,
                                        epsilon_fit,
                                        nR_region_fit,
                                        City,  
                                        color)    
                        ax_gamma_tau = single_plot_gamma_tau(Taus,
                                    Gammas,
                                    Gammac,
                                    Tauc,
                                    IndexRc,
                                    City,
                                    UCI,
                                    color,                                # NOTE: color -> color of the plot
                                    ax_gamma_tau)   
                        ax_tau_alpha = single_plot_R_tau_alpha(Rs_plot_tau,
                                                                Taus,
                                                                alpha,
                                                                Tauc,
                                                                Rc,
                                                                City,
                                                                ax_tau_alpha,
                                                                color,
                                                                UCI=UCI)
                    else:
                        print(f"{City}: UCI: {UCI}, Error in plotting n(R), gamma, tau, alpha")
                    color_idx += 1                                                              # NOTE: Update the color idx
    finalize_inset(ax_nr, ax_inset)
    # NOTE: Plot alpha betas in terms of UCI after having plot everything for R
    ax_alpha_beta = single_plot_alpha_beta_UCI(UCI_2_CriticalParams,
                               City,
                               ax_alpha_beta,
                               colors,
                               UCI_intervals)                           
    fig_nr.savefig(f"{dir_output_aggregated}/{City}_nr.png")
    fig_gamma_tau.savefig(f"{dir_output_aggregated}/{City}_gamma_tau.png")
    fig_tau_alpha.savefig(f"{dir_output_aggregated}/{City}_tau_alpha.png")
    fig_alpha_beta.savefig(f"{dir_output_aggregated}/{City}_alpha_beta.png")
    plt.close(fig_nr)
    plt.close(fig_gamma_tau)
    plt.close(fig_tau_alpha)
    plt.close(fig_alpha_beta)
    

################# PLOT SINGLE CITY SINGLE UCI #################

for City in list_cities:
    dir_base_analyzed_simulations = os.path.join("/data_2/alberto",City,"Output")                                               # NOTE: base dir for input
    dir_output_aggregated = os.path.join(dir_base_analyzed_simulations,case_)                                            # NOTE: base dir for output
    dir_output_aggregated_all_cities = os.path.join("/data_2/alberto",case_)                                             # NOTE: base dir for output when we aggregate over all cities    
    print(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxx {City} Plotting xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    color_idx = 0
    UCI_2_Time2FitBeta = City_2_UCI_2_Time2FitBeta[City]
    UCI_2_R2NtNtFit = City_2_UCI_2_R2NtNtFit[City]
    UCI_2_CriticalParams = City_2_UCI_2_CriticalParams[City]
    UCI_2_R2InfoTraffic = City_2_UCI_2_R2InfoTraffic[City]
    UCI_interval_2_UCISimulation = City_2_UCI_interval_2_UCISimulation[City]
    UCI_interval_2_Rs_available_to_all_UCIs = City_2_UCI_interval_2_Rs_available_to_all_UCIs[City]
    UCI_2_ErrorBeta_in_Time = City_2_UCI_2_ErrorBeta_in_Time[City]
    for UCI_interval, UCIs in UCI_interval_2_UCISimulation.items():
        # NOTE: Clarify whether the phase transition is found or not
        for UCI in UCIs:
            if City == "SFO":
                condition = UCI > 0.4 and UCI != 0.564
            elif City == "RIO":
                condition = True
            if condition:
                Gammas, Taus, Rs_plot_tau = get_vecs_gammas_taus_from_R2Infotraffic(UCI_2_R2InfoTraffic[UCI])
                print(f"{City}: get gammas,taus from UCI: {UCI}")
                Rc,Tauc,alpha,Gammac,is_phase_transition_found = get_Rc_Tc_Gc_is_pht_from_CritPar(UCI_2_CriticalParams,UCI)            
                # NOTE: plot_gamma_tau needed variables                
                UCI_plots = UCI
#                color = UCI_interval_2_color[UCI_interval] if UCI_interval in UCI_interval_2_color else colors[color_idx]
                idx_color = get_UCI_interval_from_UCI(UCI,UCI_intervals)
                color =  colors[idx_color]

                Rs_interval = UCI_interval_2_Rs_available_to_all_UCIs[UCI_interval] if UCI_interval in UCI_interval_2_Rs_available_to_all_UCIs else []
                fig_nt_scaled,ax_nt_scaled = plt.subplots(1,1,figsize = (10,10))
                fig_nt,ax_nt = plt.subplots(1,1,figsize = (10,10))
                fig_gamma_tau,ax_gamma_tau = plt.subplots(1,1,figsize = (10,10))
                fig_tau_alpha,ax_tau_alpha = plt.subplots(1,1,figsize = (10,10))
                fig_nr,ax_nr = plt.subplots(1,1,figsize = (10,10))
                ax_inset = create_inset_plot(fig_nr,ax_nr,"lower right")
                for R in Rs_interval:
                    print(f"---------------------------- {City} - {UCI} - R = {R} ----------------------------")
                    # NOTE: Plot being aware of whether the phase transtion occured
                    if is_phase_transition_found:
                        # NOTE: Initialize features that are of interest for time evolution
                        vec_nt = np.array(UCI_2_R2NtNtFit[UCI][R]["n"])               # NOTE: vector of fraction of people in the road in time
                        vec_nt_fit = np.array(UCI_2_R2NtNtFit[UCI][R]["n_fit"])       # NOTE: vector of fit for fraction of people in the road in time -> TimeHourArray
                        vec_t = np.array(t_vect)
                        is_alpha = int(R) == int(Rc)                                           # NOTE: is_alpha -> boolean if the alpha needs to be plotted
                        if City == "RIO":
                            delta_RC_plot_before = 400
                            delta_RC_plot_after = 20
                            delta_RC_plot_1 = 320
                        else:
                            pass
                        is_label_below = (int(R) > int(Rc) - delta_RC_plot_before) and int(R) < int(Rc)
                        is_label_above0 = int(R) > int(Rc) + delta_RC_plot_after
                        is_label_above1 = int(R) < int(Rc) + delta_RC_plot_1
                        is_label_above = is_label_above0 and is_label_above1
                        is_label = is_label_below or is_label_above
                        print(f"{City}: R {R} is_label_below: {is_label_below}, is_label_above0: {is_label_above0}, is_label_above1: {is_label_above1}, is_label_above: {is_label_above}, -> is_label: {is_label}")
                        # NOTE: needed -> single_plot_nt_alpha
                        epsilon = np.abs(int(R)**2 - int(Rc**2))/int(Rc)**2                                              # NOTE: epsilon -> value of the epsilon parameter
                        # NOTE: plot_gamma_tau needed variables                
                        IndexRc = [i for i, R in enumerate(Rs_interval) if int(R) == int(Rc)][0]
                        UCI_plots = UCI
                        Time2FitBeta = UCI_2_Time2FitBeta[UCI]
                        R2NtNtFit = UCI_2_R2NtNtFit[UCI]
                        CriticalParams = UCI_2_CriticalParams[UCI]
                        t_star = CriticalParams["t_infty"]
                        print(f"{City}: Phase Trans - Plotting nt R: {R}, UCI: {UCI}")
                        ax_nt = single_plot_nt_aggregated_single_city(vec_nt,                  # NOTE: vector of  fraction of people in the road in time
                                                                    vec_t,                    # NOTE: vector of time -> TimeHourArray
                                                                    is_alpha,                 # NOTE: is_alpha -> boolean if the alpha needs to be plotted
                                                                    alpha,                    # NOTE: alpha -> power-law for the phase transition
                                                                    R,                        # NOTE: R -> value of the R parameter
                                                                    t_star,                   # NOTE: t_star -> time of best fit of beta 
                                                                    vec_nt_fit,               # NOTE: vector of fit for fraction of people in the road in time -> TimeHourArray
                                                                    is_label,                 # NOTE: is_label -> boolean if the label needs to be plotted
                                                                    City,                     # NOTE: city -> name of the city
                                                                    ax_nt,                    # NOTE: ax1 -> axis of the plot
                                                                    is_savgol = False,        # NOTE: is_savgol -> boolean if the savgol filter needs to be applied
                        )
                        # NOTE: Condition on the existence of epsilon
                        print(f"{City}: Phase Trans - Plotting nt^alpha R: {R}, UCI: {UCI}")
                        ax_nt_scaled = single_plot_nt_alpha_aggregated_single_city(vec_nt,                   # NOTE: vector of  fraction of people in the road in time
                                                                vec_t,                          # NOTE: vector of time -> TimeHourArray
                                                                epsilon,                        # NOTE: epsilon -> value of the epsilon parameter
                                                                is_alpha,                       # NOTE: is_alpha -> boolean if the alpha needs to be plotted
                                                                alpha,                          # NOTE: alpha -> power-law for the phase transition
                                                                R,                              # NOTE: R -> value of the R parameter
                                                                vec_nt_fit,                     # NOTE: vector of fit for fraction of people in the road in time -> TimeHourArray
                                                                is_label,                       # NOTE: is_label -> boolean if the label needs to be plotted
                                                                City,                           # NOTE: city -> name of the city
                                                                ax_nt_scaled,                   # NOTE: ax1 -> axis of the plot
                                                                init_time_hour_cut_for_fit = init_time_hour_cut_for_fit, # NOTE: init_time_hour_cut_for_fit -> initial time for the fit  
                                                                max_time_for_fit = max_time_for_fit,           # NOTE: max_time_for_fit -> maximum time for the fit                                                    
                                                                is_savgol = False,              # NOTE: is_savgol -> boolean if the savgol filter needs to be applied
                                                                savgol_window = 3,              # NOTE: savgol_window -> window for the savgol filter
                                                                City_2_Marker = CITY2MARKER,    # NOTE: City_2_Marker -> dictionary of the city and the marker
                                                                City_2_Color = CITY2COLOR,      # NOTE: City_2_Color -> dictionary of the city and the color
                        )
                print(f"{City}: Phase Trans - Plotting Error Beta in Time R: {R}, UCI: {UCI}")  
                print(f"{City}: UCI: {UCI}, t_star: {t_star}")
                print(f"{City}: UCI: {UCI}, beta_fit_time: {UCI_2_ErrorBeta_in_Time[UCI]['beta_fit_time']}")
                print(f"{City}: UCI: {UCI}, error_beta_fit_time: {UCI_2_ErrorBeta_in_Time[UCI]['error_beta_fit_time']}")
                print(f"{City}: UCI: {UCI}, time_fit: {UCI_2_ErrorBeta_in_Time[UCI]['time_fit']}") 
                try:
                    fig_error_beta_in_time,ax_error_beta_in_time = plt.subplots(1,1,figsize = (10,10))
                    beta_fit_time = [float(beta_exp) for beta_exp in UCI_2_ErrorBeta_in_Time[UCI]["beta_fit_time"]]
                    error_fit_beta = [float(error_beta) for error_beta in UCI_2_ErrorBeta_in_Time[UCI]["error_beta_fit_time"]]
                    time_fit = [t for t in UCI_2_ErrorBeta_in_Time[UCI]["time_fit"]]
                    ax_error_beta_in_time = single_plot_error_associated_to_powerlaw_beta(beta_fit_time,
                                                                                    error_fit_beta,
                                                                                    time_fit,
                                                                                    t_star,
                                                                                    City,
                                                                                    UCI,
                                                                                    ax_error_beta_in_time)
                except Exception as e:
                    print(f"{City}: UCI: {UCI}, Error in plotting Error Beta in Time: {e}")
                    ax_error_beta_in_time = None
                try:
                    int_t_star = CriticalParams["int_t_infty"]
                    nRs,RsOverRc, epsilons = get_nR_R2Rc_epsilons_from_R2NtNtFit(R2NtNtFit, int_t_star,Rc)
                    epsilon_fit = Time2FitBeta[int_t_star]["epsilon"]
                    nR_fit = Time2FitBeta[int_t_star]["n(R,epsilon)_fit"]
                    nR_region_fit = Time2FitBeta[int_t_star]["n(R,epsilon)"]

                    print(f"{City}: Phase Trans - Plotting n(R) R: {R}, UCI: {UCI}") 
                    print("nR: ",nRs)
                    print("nR_fit: ",nR_fit)
                    ax_nr =     single_plot_nr(RsOverRc,
                                            nRs,
                                            epsilons,
                                            City,
                                            ax_nr,
                                            color,
                                            beta_ = beta,
                                            UCI = UCI,                                                # NOTE: UCI -> UCI of the city
                                            )
                    ax_inset = single_plot_beta_fit(Time2FitBeta,
                         t_star,
                         int_t_star,
                         t_vect,
                         ax_inset)
                    
                    ax_gamma_tau = single_plot_gamma_tau(Taus,
                                Gammas,
                                Gammac,
                                Tauc,
                                IndexRc,
                                City,
                                UCI,
                                color,                                # NOTE: color -> color of the plot
                                ax_gamma_tau)   
                    ax_tau_alpha = single_plot_R_tau_alpha(Rs_plot_tau,
                                            Taus,
                                            alpha,
                                            None,
                                            Rc,
                                            City,
                                            ax_tau_alpha,
                                            color,
                                            UCI=UCI)
                    finalize_inset(ax_nr, ax_inset)
                except Exception as e:
                    print(f"{City}: UCI: {UCI}, Error in plotting n(R), gamma, tau, alpha: {e}")
                    ax_nr = None
                    ax_gamma_tau = None
                    ax_tau_alpha = None
                fig_nt.savefig(f"{dir_output_aggregated}/nt_{City}_{UCI}.png")
                fig_nt_scaled.savefig(f"{dir_output_aggregated}/scaled_nt_{City}_{UCI}.png")
                fig_error_beta_in_time.savefig(f"{dir_output_aggregated}/error_beta_in_time_{City}_{UCI}.png")
                fig_gamma_tau.savefig(f"{dir_output_aggregated}/gamma_tau_{City}_{UCI}.png")
                fig_tau_alpha.savefig(f"{dir_output_aggregated}/tau_alpha_{City}_{UCI}.png")
                fig_nr.savefig(f"{dir_output_aggregated}/nr_{City}_{UCI}.png")
                if ax_nt is not None:
#                        plt.show(fig_nt)
                    plt.close(fig_nt)
                if ax_nt_scaled is not None:
#                        plt.show(fig_nt_scaled)
                    plt.close(fig_nt_scaled)
                if ax_error_beta_in_time is not None:
#                        plt.show(fig_error_beta_in_time)
                    plt.close(fig_error_beta_in_time)
                if ax_gamma_tau is not None:
#                        plt.show(fig_gamma_tau)
                    plt.close(fig_gamma_tau)
                if ax_tau_alpha is not None:
#                        plt.show(fig_tau_alpha)
                    plt.close(fig_tau_alpha)
                if ax_nr is not None:
#                        plt.show(fig_nr)
                    plt.close(fig_nr)
                color_idx += 1                                                              # NOTE: Update the color idx
