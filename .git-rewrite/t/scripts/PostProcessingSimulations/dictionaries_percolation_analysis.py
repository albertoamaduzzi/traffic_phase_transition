"""
    This script contains functions whose goal is to initialize and fill the dictionaries
    that are used to store informations about the directed percolation transition.
    The dictionaries we talk about are:
    ----- SHOT CRITICAL PARAMETERS -----
        - CriticalParams = {'Rc':int(R),
                            'Alpha':float(InfoTraffic["alpha_pl"]),
                            "Tauc":float(InfoTraffic["Tau"]),
                            "Gammac":float(InfoTraffic["Gamma"]),
                            }
    ------ CITY,UCI ------
        - R2NtNtFit = {R: {n:[],
                        n_fit: [],}
                       } 

        - Time2FitBeta = {index_time_in_which_beta_fit_is_computed:{
                                                                    n(R,epsilon)_fit: [],       # NOTE: The size of these vector is the number of Rs
                                                                    n(R,epsilon): [],
                                                                    ErrorPL: float,
                                                                    A_pl: float,
                                                                    Beta: float,
                                                                    epsilon: []}
                        }    NOTE: This one needs the set of Rs that are used to compute the 

                       
        ------ CITY, UCI, R ------
        - InfoTraffic = {
                        "ErrorExp": 0.034653040180289474,
                        "ErrorPl": 1000000,
                        "A_exp": 36.23633873148154,
                        "A_pl": 0.5011225497563125,
                        "alpha_exp": -4.745873455713671,
                        "alpha_pl": -7.632132614660605,
                        "best_fit": "Exponential",
                        "IsJam": false,
                        "Tau": 0.2107093687456154,
                        "Gamma": 0.04439294534356198
                        }
    UCI_interval_2_gamma_tau = defaultdict()

    
"""
from collections import defaultdict
import os
import json
import numpy as np

def convert_ndarray(d):
    """
        Recursively convert numpy arrays to lists and numpy int64 to int in a dictionary.
    """
    if isinstance(d, dict):
        return {convert_ndarray(k): convert_ndarray(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_ndarray(i) for i in d]
    elif isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, np.int64):
        return int(d)
    else:
        return d



#----- ONE SHOT CRITICAL PARAMETERS -----#

# Critical Parameters - > One shot at criticality
def compute_UCI2CriticalParams(R,InfoTraffic):
    """
    Compute the critical parameters for the UCI2 model
    """
    return {'Rc':int(R),
        'Alpha':float(InfoTraffic["alpha_pl"]),
        "Tauc":float(InfoTraffic["Tau"]),
        "Gammac":float(InfoTraffic["Gamma"]),
        }

#------ CITY ------#
def init_City2UCI2ImportantFeaturesForComparisonCities(Cities,UCIs):
    City2UCI2ImportantFeaturesForComparisonCities = defaultdict(dict)
    for City in Cities:
        City2UCI2ImportantFeaturesForComparisonCities[City] = {str(round(UCI,3)):{"Rc":None,                                           # Plot fig 5c/5a
                                                                        "Tauc":None,                                         # Plot fig 5c/5a
                                                                        "index_Rc":None,                                    # Plot fig 5c/5a
                                                                        "Taus":[],                                          # Plot fig 5c
                                                                        "Rs":[],                                            # Plot fig 5c
                                                                        "alpha":None,                                       # Plot fig 5b
                                                                        "beta":None,                                        # Plot fig 5b
                                                                        "n(R)":[]                                           # Plot fig 5a
                                                                        } for UCI in UCIs
                                                                    }
    return City2UCI2ImportantFeaturesForComparisonCities


# ------ CITY, UCI -----#
def fill_R2NtNtFit(R2NtNtFit,DfUnload,IsJam):
    """
        Fill the dictionary R2NtNtFit with the values of n and n_fit from DfUnload
    """
    R2NtNtFit["n"] = DfUnload["FractionPeople"].to_numpy()[:-1]
    if IsJam:
        y_fit = DfUnload["y_fit_pl"].to_numpy()
    else:
        y_fit = DfUnload["y_fit_exp"].to_numpy()
    R2NtNtFit["n_fit"] = y_fit[:-1]
    return R2NtNtFit


def fill_UCI_2_ErrorBeta_in_Time(Errors,Betas,time_):
    return {"error_beta_fit_time":Errors,
            "beta_fit_time":Betas,
            "time_fit":time_}


# ------- GETTER FUNCTIONS ------- #
def get_Rc_Tc_Gc_is_pht_from_CritPar(UCI_2_CriticalParams,UCI_interval):

        try:
            Rc = UCI_2_CriticalParams[UCI_interval]["Rc"]              # NOTE: Critical loading rate for which at first the power law drop is observed
            Tauc = UCI_2_CriticalParams[UCI_interval]["Tauc"]          # NOTE: Critical characteristic time for which the power law drop is observed
            alpha = UCI_2_CriticalParams[UCI_interval]["Alpha"]        # NOTE: Power law exponent that fits better the drop of fraction of people
            Gammac = UCI_2_CriticalParams[UCI_interval]["Gammac"]      # NOTE: Critical Gamma for which the power law drop is observed
            is_phase_transition_found = True
        except:
            Rc = None
            Tauc = None
            alpha = None
            Gammac = None
            is_phase_transition_found = False
        return Rc,Tauc,alpha,Gammac,is_phase_transition_found


def get_vecs_gammas_taus_from_R2Infotraffic(R2InfoTraffic):
    """
        Get the vectors of gammas and taus from the R2InfoTraffic dictionary
        @R2Infotraffic: dict -> {R: {"Gamma": float,
                                     "Tau": float
                                     ErrorExp": float,
                                     "ErrorPl": float,
                                     "A_exp": float,
                                     "A_pl": float,
                                     "alpha_exp": float,
                                     "alpha_pl": float,
                                     "best_fit": str,
                                     "IsJam": bool}}
        @return: vec_gammas: list -> [Gamma1,...,GammaN]
                vec_taus: list -> [Tau1,...,TauN]
        Description:
            - The function takes the R2InfoTraffic dictionary and extracts the gamma and tau values
            - It returns two lists: one for gammas and one for taus
            - The function filters out any None values and only keeps the positive values
            NOTE: the positive values are chosen since in the pipeline when there are no people left from 
                  the unloading group in the computation of tau I put -1.0
    """
    vec_gammas = []
    vec_taus = []
    Rs = []
    for R in R2InfoTraffic.keys():
        gamma = R2InfoTraffic[R]["Gamma"]
        tau = R2InfoTraffic[R]["Tau"]
        if gamma is not None and tau is not None:
            if tau > 0 and gamma > 0:
                vec_gammas.append(gamma)
                vec_taus.append(tau)
                Rs.append(R)
    return vec_gammas,vec_taus,Rs


# --- SAVE FUNCTIONS --- #
def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_numpy_types(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif hasattr(obj, 'to_dict'):
        # Handle objects like pandas DataFrames or Series
        return convert_numpy_types(obj.to_dict())
    else:
        return obj

def save_jsons_DP(dir_output_aggregated_all_cities,
                   City_2_UCI_2_R2InfoTraffic,
                   City_2_UCI_2_CriticalParams,
                   City_2_UCI_2_R2NtNtFit,
                   City_2_UCI_2_Time2FitBeta,
                   City_UCI_2_ErrorBeta_in_Time,
                   case_):
    """
        Save the jsons for the DP analysis
    """
    converted_InfoTraffic = convert_numpy_types(City_2_UCI_2_R2InfoTraffic)
    with open(os.path.join(dir_output_aggregated_all_cities,f"InfoTraffic_{case_}.json"),'w') as f:
        json.dump(converted_InfoTraffic,f,indent = 4)
    # NOTE: Critical params
    converted_CriticalParams = convert_numpy_types(City_2_UCI_2_CriticalParams)
    with open(os.path.join(dir_output_aggregated_all_cities,f"CriticalParams_{case_}.json"),'w') as f:
        json.dump(City_2_UCI_2_CriticalParams,f,indent = 4)
    # NOTE: DfUnload with fit.
    R2NtNtFit_converted = convert_ndarray(City_2_UCI_2_R2NtNtFit)
    R2NtNtFit_converted = convert_numpy_types(R2NtNtFit_converted)
    with open(os.path.join(dir_output_aggregated_all_cities,f"R2NtNtFit_{case_}.json"),'w') as f:
        json.dump(R2NtNtFit_converted,f,indent = 4)
    # NOTE: Informations about beta fit
    converted_Time2FitBeta = convert_numpy_types(City_2_UCI_2_Time2FitBeta)
    with open(os.path.join(dir_output_aggregated_all_cities,f"Time2FitBeta_{case_}.json"),'w') as f:
        json.dump(convert_ndarray(converted_Time2FitBeta),f,indent = 4)
    # 
    City_UCI_2_ErrorBeta_in_Time_converted = convert_ndarray(City_UCI_2_ErrorBeta_in_Time)
    City_UCI_2_ErrorBeta_in_Time_converted = convert_numpy_types(City_UCI_2_ErrorBeta_in_Time_converted)
    with open(os.path.join(dir_output_aggregated_all_cities,f"ErrorBetaInTime_{case_}.json"),'w') as f:
        json.dump(City_UCI_2_ErrorBeta_in_Time_converted,f,indent = 4)


def read_json_DP(dir_output_aggregated_all_cities,case_):
    """
        Read the jsons for the DP analysis
    """
    with open(os.path.join(dir_output_aggregated_all_cities,f"InfoTraffic_{case_}.json"),'r') as f:
        City_2_UCI_2_R2InfoTraffic = json.load(f)
    with open(os.path.join(dir_output_aggregated_all_cities,f"CriticalParams_{case_}.json"),'r') as f:
        City_2_UCI_2_CriticalParams = json.load(f)
    with open(os.path.join(dir_output_aggregated_all_cities,f"R2NtNtFit_{case_}.json"),'r') as f:
        City_2_UCI_2_R2NtNtFit = json.load(f)
    with open(os.path.join(dir_output_aggregated_all_cities,f"Time2FitBeta_{case_}.json"),'r') as f:
        City_2_UCI_2_Time2FitBeta = json.load(f)
    with open(os.path.join(dir_output_aggregated_all_cities,f"ErrorBetaInTime_{case_}.json"),'r') as f:
        City_UCI_2_ErrorBeta_in_Time = json.load(f)
    return City_2_UCI_2_R2InfoTraffic,City_2_UCI_2_CriticalParams,City_2_UCI_2_R2NtNtFit,City_2_UCI_2_Time2FitBeta,City_UCI_2_ErrorBeta_in_Time
