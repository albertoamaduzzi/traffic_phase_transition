from PlotsPercolationAggregated import *
from dictionaries_percolation_analysis import *
from pipeline_InfoTraffic import *
from FittingProcedures import *
from Beta import *
from pipeline_relevant_choose_compute_df_unload import *


def pipeline_sigle_R(R2InfoTraffic,                     # Output -> The InfoTraffic associated to the R NOTE: (is a leaf so can be: case,City,UCI,R)
                     R2NtNtFit,                         # Output -> The NtNtFit associated to the R NOTE: (is a leaf so can be: case,City,UCI,R)
                     CriticalParams,                    # Output -> The CriticalParams associated to the R NOTE: (is a leaf so can be: case,City,UCI,R)
                     R2DfUnload,                        # Output -> The DfUnload associated to the R NOTE: (is a leaf so can be: case,City,UCI,R)
                     R2gamma,                           # Output -> The gamma associated to the R NOTE: (is a leaf so can be: case,City,UCI,R)
                     UCIs,
                     City,
                     R,
                     UCI,
                     UCI_interval,
                     case_,
                     t_start_hour,
                     t_end_control_group,
                     dir_base_analyzed_simulations,
                     found_jam,
                     init_time_hour_cut_for_fit = 1.5,
                     minimum_time_with_people = 10,
                     max_time_for_fit = 7,
                     t_vect = [],
                     is_compute_DfUnload_from_iters = True
                     ):
    """
        @Description:
            - Choose the case: Aggregated_UCIs or Single_UCI
            - if aggregated_UCIs:
                - Compute the average DfUnload for the R in the UCI interval
                - Compute the average gamma for the R in the UCI interval
                - Compute the fit of the average FractionPeople
                - if the powerlaw fits better than:
                    - found_jam = True

        NOTE: It is contained in the pipeline for the single UCI and called as many times as the number of Rs
        """
    # NOTE: (City,UCI,R) -> depending on the case these are just leaves of the tree City_2_UCI_2_R2InfoTraffic ...
    R2InfoTraffic[R] = defaultdict(dict)
    R2NtNtFit[R] = defaultdict(dict)
    # NOTE: Compute average gamma and DfUnload according to the case
    if case_ == "Aggregated_UCIs":
        assert len(UCIs) >= 1, f"Case Aggregated_UCIs but UCIs: {UCIs} is not a list of length > 1"
        print(f"{City}: case: {case_}: Compute <DfUnload>_ucis, R: {R} in UCI interval: {UCI}")
        R2DfUnload[R] = compute_average_Df_Unload_UCI_interval(UCIs,
                                                            UCI_interval,
                                                            dir_base_analyzed_simulations,
                                                            R,
                                                            t_start_hour,
                                                            t_end_control_group,
                                                            City
                                                            )

        # 3) NOTE: Compute Gamma as the average of the Unload curves 
        print(f"{City}: case: {case_}: Compute <Gamma>_ucis, R: {R} in UCI interval: {UCI_interval}")
        R2gamma[R] = compute_average_gamma(UCIs,R,dir_base_analyzed_simulations)
    else:
        if is_compute_DfUnload_from_iters:
            print(f"{City}: case: {case_}: Compute <DfUnload>_iters, R: {R} in UCI: {UCI}")            
            # NOTE: If the iter unload curves are not available then upload the DfUnload
            R2DfUnload[R] = compute_average_Df_Unload_from_iterations(R,
                                              UCI,
                                              dir_base_analyzed_simulations,
                                              t_start_hour,
                                              t_end_control_group,

                                              )
        else:
            # NOTE: Upload single DfUnload
            print(f"{City}: case: {case_}: Get DfUnload for R: {R} in UCI: {UCI}")
            R2DfUnload[R] = pl.read_parquet(os.path.join(dir_base_analyzed_simulations,City,"Output",str(UCI),"Plots",str(UCI),f"UnloadCurve_R_{R}_UCI_{UCI}.parquet")) 
        print(f"{City}: case: {case_}: Get <Gamma>_iters, R: {R} in UCI: {UCI}")
        with open(os.path.join(dir_base_analyzed_simulations,str(UCI),"Plots",str(UCI),f"VolumeTraffic_R_{R}_UCI_{UCI}.json"), 'r') as f:
            VolumeTraffic = json.load(f)
        R2gamma[R] = float(VolumeTraffic)
    # TODO: Compute the fit of the average FractionPeople
    if case_ == "Aggregated_UCIs":
        print(f"{City}: case: {case_}: Compute fit for R: {R} in UCI interval: {UCI_interval}")
    else:
        print(f"{City}: case: {case_}: Compute fit for R: {R} in UCI: {UCI}")
    DfUnload,ErrorExp,A0,alpha_exp,ErrorPL,A_pl,alpha_pl = ComparePlExpoDf(R2DfUnload[R],
                                                                            init_time_hour_cut_for_fit = init_time_hour_cut_for_fit,
                                                                            minimum_time_with_people = minimum_time_with_people,  
                                                                            max_time_for_fit = max_time_for_fit,                                                                                  
                                                                            maxfev = 10000)
    if case_ == "Aggregated_UCIs":        
        print(f"{City} Filling info traffic R: {R}, UCI: {UCI}")
        # 2) NOTE: Extract info about traffi (City,UCI,R) 
        InfoTraffic, found_jam = pipeline_fill_InfoTraffic(ErrorExp,               
                                                            ErrorPL,                      
                                                            A0,                                 
                                                            A_pl,                         
                                                            alpha_exp,                    
                                                            alpha_pl,                     
                                                            City,                         
                                                            Gamma = R2gamma[R],                        
                                                            DfUnload = DfUnload,                     
                                                            found_jam = found_jam,            
                                                            R = R,
                                                            UCI = UCI_interval)
    else:
        print(f"{City} Filling info traffic R: {R}, UCI: {UCI}")
        # 2) NOTE: Extract info about traffi (City,UCI,R) 
        InfoTraffic, found_jam = pipeline_fill_InfoTraffic(ErrorExp,               
                                                            ErrorPL,                      
                                                            A0,                                 
                                                            A_pl,                         
                                                            alpha_exp,                    
                                                            alpha_pl,                     
                                                            City,                         
                                                            Gamma = R2gamma[R],                        
                                                            DfUnload = DfUnload,                     
                                                            found_jam = found_jam,            
                                                            R = R,
                                                            UCI = UCI)
    # NOTE: Update informations about traffic for higher level of analysis
    R2InfoTraffic[R] = InfoTraffic
    # NOTE: Store Critical Parameters -> NOTE: InfoTraffic["IsJam"] != found_jam since found_jam is True forever when at least one InfoTraffic["IsJam"]
    R2NtNtFit[R] = fill_R2NtNtFit(R2NtNtFit[R],
                                DfUnload,
                                InfoTraffic["IsJam"])
    
    if len(t_vect) == 0:
        t_vect = DfUnload["Time_hours_shifted"].to_numpy()[:-1]
    return R2InfoTraffic, R2NtNtFit, CriticalParams, R2DfUnload, R2gamma, t_vect,found_jam


def pipeline_single_UCI(UCI_2_R2InfoTraffic,
                        UCI_2_CriticalParams,
                        UCI_2_R2NtNtFit,
                        UCI_2_R2DfUnload,
                        UCI_2_R2gamma,
                        UCI_2_Time2FitBeta,
                        UCI_2_gamma_tau,
                        UCI_interval,
                        Rs_interval,                            # NOTE: The Rs are associated beforehand to the UCI at hand
                        UCI,
                        UCIs,
                        City,
                        case_,
                        t_start_hour,
                        t_end_control_group,
                        start_bin_time_considered_in_beta_fit,
                        dir_base_analyzed_simulations,
                        init_time_hour_cut_for_fit = 1.5,
                        minimum_time_with_people = 10,
                        max_time_for_fit = 7,
                        ):
    if case_ == "Aggregated_UCIs":
        # NOTE: Dictionaries Output DfUnload, Criticality, InfoTraffic
        UCI_2_gamma_tau[UCI_interval] = {"R":[],"Gamma":[],"Tau":[]}
    else:
        # NOTE: Dictionaries Output DfUnload, Criticality, InfoTraffic
        UCI_2_gamma_tau[UCI] = {"R":[],"Gamma":[],"Tau":[]}
    # -------------------------- COMPUTE SECTION -------------------------- #
    found_jam = False
    first_found_jam = False
    is_found_beta = False
    t_vect = []
    # NOTE: Pick all the Rs that are shared by the UCIs in UCI_interval
    R2InfoTraffic = defaultdict(dict)
    R2NtNtFit = defaultdict(dict)
    R2DfUnload = defaultdict(dict)
    R2gamma = defaultdict(dict)
    CriticalParams = defaultdict(dict)
    # NOTE: Loop over the Rs in the UCI interval
    for R in Rs_interval:
        R2InfoTraffic, R2NtNtFit, CriticalParams, R2DfUnload, R2gamma, t_vect,found_jam = pipeline_sigle_R(R2InfoTraffic,                     # Output -> The InfoTraffic associated to the R NOTE: (is a leaf so can be: case,City,UCI,R)
                                                                                                            R2NtNtFit,                         # Output -> The NtNtFit associated to the R NOTE: (is a leaf so can be: case,City,UCI,R)
                                                                                                            CriticalParams,                    # Output -> The CriticalParams associated to the R NOTE: (is a leaf so can be: case,City,UCI,R)
                                                                                                            R2DfUnload,                        # Output -> The DfUnload associated to the R NOTE: (is a leaf so can be: case,City,UCI,R)
                                                                                                            R2gamma,                           # Output -> The gamma associated to the R NOTE: (is a leaf so can be: case,City,UCI,R)
                                                                                                            UCIs,
                                                                                                            City,
                                                                                                            R,
                                                                                                            UCI,
                                                                                                            UCI_interval,
                                                                                                            case_,
                                                                                                            t_start_hour,
                                                                                                            t_end_control_group,
                                                                                                            dir_base_analyzed_simulations,
                                                                                                            found_jam,
                                                                                                            init_time_hour_cut_for_fit = init_time_hour_cut_for_fit,
                                                                                                            minimum_time_with_people = minimum_time_with_people,
                                                                                                            max_time_for_fit = max_time_for_fit,
                                                                                                            t_vect = t_vect) 

        if case_ == "Aggregated_UCIs":
            UCI_2_gamma_tau[UCI_interval]["Tau"].append(R2InfoTraffic[R]["Tau"])
            UCI_2_gamma_tau[UCI_interval]["R"].append(R)
            UCI_2_gamma_tau[UCI_interval]["Gamma"].append(R2gamma[R])
            if not first_found_jam and found_jam:
                # NOTE: This is unique for the UCIs and computed just the first time the jam is found: i.e. n(t) is better fitted by pl then exp
                CriticalParams = compute_UCI2CriticalParams(R,
                                                            R2InfoTraffic[R])

                first_found_jam = True
                UCI_2_CriticalParams[UCI_interval] = CriticalParams

        else:
            UCI_2_gamma_tau[UCI]["Tau"].append(R2InfoTraffic[R]["Tau"])
            UCI_2_gamma_tau[UCI]["R"].append(R)
            UCI_2_gamma_tau[UCI]["Gamma"].append(R2gamma[R])
            if not first_found_jam and found_jam:
                # NOTE: This is unique for the UCIs and computed just the first time the jam is found: i.e. n(t) is better fitted by pl then exp
                CriticalParams = compute_UCI2CriticalParams(R,
                                                            R2InfoTraffic[R])
                first_found_jam = True
                UCI_2_CriticalParams[UCI] = CriticalParams

        print(f"{City}: case: {case_}: R: {R}: R2InfoTraffic: ",R2InfoTraffic[R])
        print(f"{City}: case: {case_}: R: {R}: R2NtNtFit: ",R2NtNtFit[R]["n"][:4])
        print(f"{City}: case: {case_}: R: {R}: R2NtNtFit: ",R2NtNtFit[R]["n_fit"][:4])
        print(f"{City}: case: {case_}: R: {R}: CriticalParams: ",CriticalParams)
        print(f"{City}: case: {case_}: R: {R}: R2gamma: ",R2gamma[R])
        print(f"{City}: case: {case_}: R: {R}: t_vect: ",t_vect[:4])
        print(f"{City}: case: {case_}: R: {R}: found_jam: ",found_jam)

    # NOTE: If the analysis is aggregated -> UCI_interval is going to be the key for the dictionaries
    if case_ == "Aggregated_UCIs":
        # NOTE: (City,UCI,R) -> NOTE: Inititalize before computation fit beta
        UCI_2_R2InfoTraffic[UCI_interval] = R2InfoTraffic
        UCI_2_R2NtNtFit[UCI_interval] = R2NtNtFit
        UCI_2_R2DfUnload[UCI_interval] = R2DfUnload
        UCI_2_R2gamma[UCI_interval] = R2gamma
        # Beta Pipeline -> (City,UCI)
        if found_jam:
            print(f"{City}: case: {case_}: Compute Beta for UCI interval: {UCI_interval}, len(t_vect): {len(t_vect)}")
            Time2FitBeta = defaultdict(dict)
            BestBeta,BestA,t_star,int_t_star,is_found_beta,Time2FitBeta,Errors,Betas,time_ = pipeline_BestBeta(Time2FitBeta,
                                                                                                                UCI_2_R2NtNtFit[UCI_interval],
                                                                                                                UCI_2_CriticalParams[UCI_interval],
                                                                                                                Rs_interval,
                                                                                                                UCI_interval,
                                                                                                                City,
                                                                                                                t_vect,
                                                                                                                start_bin_time_considered_in_beta_fit,
                                                                                                                minimum_number_simulations_for_fit = 3)
            UCI_2_Time2FitBeta[UCI_interval] = Time2FitBeta
            UCI_2_CriticalParams[UCI_interval]["beta"] = BestBeta
            UCI_2_CriticalParams[UCI_interval]["A_pl_beta"] = BestA
            UCI_2_CriticalParams[UCI_interval]["t_infty"] = t_star
            UCI_2_CriticalParams[UCI_interval]["int_t_infty"] = int_t_star

            print(f"{City}: case: {case_}: BestBeta: ",BestBeta)
            print(f"{City}: case: {case_}: BestA: ",BestA)
            print(f"{City}: case: {case_}: t_star: ",t_star)
            print(f"{City}: case: {case_}: int_t_star: ",int_t_star)
            print(f"{City}: case: {case_}: is_found_beta: ",is_found_beta)
            print(f"{City}: case: {case_}: Errors: ",Errors[:4])
            print(f"{City}: case: {case_}: Betas: ",Betas[:4])
            print(f"{City}: case: {case_}: time_: ",time_[:4])
        else:
            BestBeta,BestA,t_star,int_t_star,is_found_beta,Time2FitBeta,Errors,Betas,time_,int_t_star = None,None,None,None,False,None,None,None,None,None
    else:
        # NOTE: (City,UCI)
        UCI_2_R2InfoTraffic[UCI] = R2InfoTraffic
        UCI_2_R2NtNtFit[UCI] = R2NtNtFit
        UCI_2_R2gamma[UCI] = R2gamma
        UCI_2_R2DfUnload[UCI] = R2DfUnload        
        # Beta Pipeline -> (City,UCI)
        if found_jam:
            Time2FitBeta = defaultdict(dict)
            print(f"{City}: case: {case_}: Compute Beta for UCI: {UCI}, len(t_vect): {len(t_vect)}")
            BestBeta,BestA,t_star,int_t_star,is_found_beta,Time2FitBeta,Errors,Betas,time_ = pipeline_BestBeta(Time2FitBeta,
                                                                                                                UCI_2_R2NtNtFit[UCI],
                                                                                                                UCI_2_CriticalParams[UCI],
                                                                                                                Rs_interval,
                                                                                                                UCI,
                                                                                                                City,
                                                                                                                t_vect,
                                                                                                                start_bin_time_considered_in_beta_fit,
                                                                                                                minimum_number_simulations_for_fit = 3)
            
            UCI_2_Time2FitBeta[UCI] = Time2FitBeta
            UCI_2_CriticalParams[UCI]["beta"] = BestBeta
            UCI_2_CriticalParams[UCI]["A_pl_beta"] = BestA
            UCI_2_CriticalParams[UCI]["t_infty"] = t_star
            UCI_2_CriticalParams[UCI]["int_t_infty"] = int_t_star
            print(f"{City}: case: {case_}: BestBeta: ",BestBeta)
            print(f"{City}: case: {case_}: BestA: ",BestA)
            print(f"{City}: case: {case_}: t_star: ",t_star)
            print(f"{City}: case: {case_}: int_t_star: ",int_t_star)
            print(f"{City}: case: {case_}: is_found_beta: ",is_found_beta)
            print(f"{City}: case: {case_}: Errors: ",Errors[:4])
            print(f"{City}: case: {case_}: Betas: ",Betas[:4])
            print(f"{City}: case: {case_}: time_: ",time_[:4])

        else:
            BestBeta,BestA,t_star,int_t_star,is_found_beta,Time2FitBeta,Errors,Betas,time_,int_t_star = None,None,None,None,False,None,None,None,None,None

    return UCI_2_R2InfoTraffic, UCI_2_CriticalParams, UCI_2_R2NtNtFit, UCI_2_R2DfUnload, UCI_2_R2gamma, UCI_2_Time2FitBeta, UCI_2_gamma_tau,found_jam, is_found_beta,Errors,Betas,time_,t_vect
