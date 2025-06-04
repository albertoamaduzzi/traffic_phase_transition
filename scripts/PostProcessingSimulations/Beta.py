from FittingProcedures import FittingPowerlaw
import matplotlib.pyplot as plt
import logging
import numpy as np
from collections import defaultdict
import os
from matplotlib.ticker import ScalarFormatter
logger = logging.getLogger(__name__)



def ComputeTime2FitBeta(RangeFit,
                        t_vect,
                        R2NtNtFit,
                        UCI2CriticalParams,
                        Rs,
                        UCI,
                        City,
                        minimum_number_simulations_for_fit = 3):
    """
        Description:
            - Compute the time2fitbeta for a given UCI
        NOTE: We are avoiding to consider the R that contains 0 at t0 > t_critical. 
            We are assuming that 0s are appearing just in low time where the transition is not yet complete
            
            """
    logger.info(f"{City}: Computing Time2FitBeta for UCI = {UCI}")
    Time2FitBeta = defaultdict()#{t0:{"y_fit_pl":[],"ErrorPL":[],"A_pl":[],"Beta":[]} for t0 in RangeFit}
    # 
    for t0 in RangeFit:                             # NOTE: Checking just big values of t0
        NRsGivent0 = []
        FRcR = []
        RoverRc = []
        t = t_vect[t0]
        Rs1 = [R for R in sorted(Rs) if R > UCI2CriticalParams[round(UCI,3)]["Rc"]]
        for R in sorted(Rs1):
            # Avoid those casesfor which I have R but no simulation is computed
            if "n" in R2NtNtFit[R].keys() and len(R2NtNtFit[R]["n"]) > t0:
                # Check PowerLaw fixed t0, between NRsGivent0 and (R^2 - Rc^2)/Rc^2
#                if R2NtNtFit[R]["n"][t0] > 0:
                NRsGivent0.append(R2NtNtFit[R]["n"][t0])
                FRcR.append((R**2 - UCI2CriticalParams[round(UCI,3)]["Rc"]**2)/UCI2CriticalParams[round(UCI,3)]["Rc"]**2)
                RoverRc.append(R/UCI2CriticalParams[round(UCI,3)]["Rc"])
            if np.shape(NRsGivent0) != np.shape(FRcR):  
                logger.error(f"shape NRsGivent0: {len(NRsGivent0)}, shape FRcR: {len(FRcR)}")
                logger.error(f"NRsGivent0: {NRsGivent0}")
                logger.error(f"FRcR: {FRcR}")
                raise ValueError("NRsGivent0 and FRcR must have the same shape")
        if len(NRsGivent0) == len(FRcR) and len(NRsGivent0) > minimum_number_simulations_for_fit:
            # Fit Power Law
            y_fit_pl,ErrorPL,A_pl,Beta = FittingPowerlaw(FRcR,NRsGivent0)
            if y_fit_pl is not None:
#                y_fit_pl = np.array(y_fit_pl)/np.sum(y_fit_pl)
#                NRsGivent0 = np.array(NRsGivent0)/np.sum(NRsGivent0)
                Time2FitBeta[t0] = {"n(R,epsilon)_fit":y_fit_pl,
                                    "n(R,epsilon)":NRsGivent0,
                                    "ErrorPL":ErrorPL,
                                    "A_pl":A_pl,
                                    "Beta":Beta,
                                    "epsilon":FRcR,
                                    "R/Rc":RoverRc}

    return Time2FitBeta


def pipeline_ComputeTime2FitBeta(RangeFit,
                        t_vect,
                        R2NtNtFit,
                        CriticalParams,
                        Rs,
                        UCI,
                        City,
                        minimum_number_simulations_for_fit = 2):
    """
        Description:
            - Compute the time2fitbeta for a given UCI
        NOTE: We are avoiding to consider the R that contains 0 at t0 > t_critical. 
            We are assuming that 0s are appearing just in low time where the transition is not yet complete
            
            """
    print(f"{City}: Computing Time2FitBeta for UCI = {UCI}")
    Time2FitBeta = defaultdict()#{t0:{"y_fit_pl":[],"ErrorPL":[],"A_pl":[],"Beta":[]} for t0 in RangeFit}
    # 
    for t0 in RangeFit:                             # NOTE: Checking just big values of t0
        NRsGivent0 = []
        FRcR = []
        RoverRc = []
        t = t_vect[t0]
        # NOTE: The fit chooses just the simulations that are above Rc
        Rs1 = [R for R in sorted(Rs) if R >= CriticalParams["Rc"]]
        for R in sorted(Rs1):
            # Avoid those casesfor which I have R but no simulation is computed
            if "n" in R2NtNtFit[R].keys() and len(R2NtNtFit[R]["n"]) > t0:
                # Check PowerLaw fixed t0, between NRsGivent0 and (R^2 - Rc^2)/Rc^2
#                if R2NtNtFit[R]["n"][t0] > 0:
                NRsGivent0.append(R2NtNtFit[R]["n"][t0])
                FRcR.append(np.abs(R**2 - CriticalParams["Rc"]**2)/CriticalParams["Rc"]**2)
                RoverRc.append(R/CriticalParams["Rc"])
            if np.shape(NRsGivent0) != np.shape(FRcR):  
                logger.error(f"shape NRsGivent0: {len(NRsGivent0)}, shape FRcR: {len(FRcR)}")
                logger.error(f"NRsGivent0: {NRsGivent0}")
                logger.error(f"FRcR: {FRcR}")
                raise ValueError("NRsGivent0 and FRcR must have the same shape")
        if len(NRsGivent0) == len(FRcR) and len(NRsGivent0) > minimum_number_simulations_for_fit:
            # Fit Power Law
            y_fit_pl,ErrorPL,A_pl,Beta = FittingPowerlaw(FRcR,NRsGivent0)
            if y_fit_pl is not None:
#                y_fit_pl = np.array(y_fit_pl)/np.sum(y_fit_pl)
#                NRsGivent0 = np.array(NRsGivent0)/np.sum(NRsGivent0)
                Time2FitBeta[t0] = {"n(R,epsilon)_fit":y_fit_pl,
                                    "n(R,epsilon)":NRsGivent0,
                                    "ErrorPL":ErrorPL,
                                    "A_pl":A_pl,
                                    "Beta":Beta,
                                    "epsilon":FRcR,
                                    "R/Rc":RoverRc}

    return Time2FitBeta




def Compute_vecs_error_pl_and_betas(Time2FitBeta,t_vect):
    """
        Description:
            - Compute the vectors of the error of the power law fit and the betas
        NOTE: This is useful if we want to plot:
            - the error of beta fit
            - beta fit over time.
        NOTE: It is extendible if one wants to compute the average fo the exponent and of the error.

            """
    Betas = [Time2FitBeta[t0]["Beta"] for t0 in Time2FitBeta.keys()]
    Errors = [Time2FitBeta[t0]["ErrorPL"] for t0 in Time2FitBeta.keys()]
    time_ = [t_vect[t0] for t0 in list(Time2FitBeta.keys())]
    return time_,Betas,Errors



def ChooseBestBeta(Time2FitBeta,t_vect):
    """
        Description:
            Choose the best Beta, A and T for the powerlaw fit.
        NOTE:
            - beta is not negative. That is, we must not consider the cases where the successive Rs have less people.
    """

    logger.info("Choosing Best Beta")
    MinError = 1e10
    BestBeta = -1
    BestA = -1
    BestT = -1
    t1 = -1
    for t0 in Time2FitBeta.keys():
        print(f"t0: {t0}, t_vect[t0]: {t_vect[t0]}, Beta: {Time2FitBeta[t0]['Beta']}, ErrorPL: {Time2FitBeta[t0]['ErrorPL']}, MinError: {MinError}, t1: {t1}")
        if Time2FitBeta[t0]["ErrorPL"] < MinError:
            if Time2FitBeta[t0]["Beta"]>0:
                MinError = Time2FitBeta[t0]["ErrorPL"]
                BestBeta = Time2FitBeta[t0]["Beta"]
                BestA = Time2FitBeta[t0]["A_pl"]
                BestT = t_vect[t0]
                t1 = t0
    if BestBeta < 0:
        is_found_beta = False
    else:
        is_found_beta = True 
    return BestBeta,BestA,BestT,t1,is_found_beta         

def pipeline_BestBeta(Time2FitBeta,
                      R2NtNtFit,
                      CriticalParams,
                      Rs_interval,
                      UCI,
                      City,
                      t_vect,
                      start_bin_time_considered_in_beta_fit,
                      minimum_number_simulations_for_fit = 2):
    """
        Description:
            - Compute the best beta for a given UCI i.e. choose the best fit of n(R),epsilon for all the times for which n(R,t)>0 for at least 3 times
    """
    RangeFit = range(start_bin_time_considered_in_beta_fit,min(len(t_vect),100))
    print(f"{City}: Computing Best Beta for UCI = {UCI}, size RangeFit: {len(RangeFit)}, len(t_vect): {len(t_vect)}")
    Time2FitBeta = pipeline_ComputeTime2FitBeta(RangeFit,                                    # NOTE: The time intervals I compute the error for beta
                                    t_vect,                                                  # NOTE: All the time intervals
                                    R2NtNtFit,                                               # NOTE: The n(t) for the different Rs -> compute iterating on Range fit n(R,t)
                                    CriticalParams,                                          # NOTE: The critical params Rc (choose just the curves in the transition),Tauc
                                    Rs_interval,                                             # NOTE: The Rs available for the UCI interval
                                    UCI,                                                     # NOTE: The UCI
                                    City,
                                    minimum_number_simulations_for_fit)                                                    # NOTE: The city

    BestBeta,BestA,t_star,int_t_star,is_found_beta = ChooseBestBeta(Time2FitBeta,t_vect)
    if is_found_beta:
        time_,Betas,Errors = Compute_vecs_error_pl_and_betas(Time2FitBeta,t_vect)
    else:
        time_ = []
        Betas = []
        Errors = []
    return BestBeta,BestA,t_star,int_t_star,is_found_beta,Time2FitBeta,Errors,Betas,time_


def PlotTime2FitBetaError(Time2FitBeta,t_vect,t_star,City,UCI,PlotDir):
    """
        Description:
            - Plot the Error of the Power Law Fit
    """
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    time_,Betas,Errors = Compute_vecs_error_pl_and_betas(Time2FitBeta,t_vect)
    ax.scatter(time_,Errors,marker = "o",label = f"error")
    ax.scatter(time_,Betas,marker = "*",label = r"$\beta$")
    int_t_star = int(t_star)
    ax.vlines(x = t_vect[int_t_star],ymin = 0,ymax = 1,color = "red",linestyles = "dashed")
    ax2 = ax.twinx()
    ax2.spines['left'].set_position(('outward', 60))
    ax2.set_xlabel("t (h)")
    ax2.set_ylabel("error")
    ax2.set_title(f"{City}, UCI: {UCI}")
    plt.savefig(os.path.join(PlotDir,f'{City}_{UCI}_ErrorBeta.png'))
    plt.close(fig)




def PlotBestBeta(Time2FitBeta,BestT,t1,City,t_vect,UCI,PlotDir):
    """
        Description:

    """   
    logger.info(f"{City}:Plotting Best Beta")
    fig,ax = plt.subplots(figsize = (10,10))
    for t0 in Time2FitBeta.keys():
        if t1 == t0:
            ax.scatter(Time2FitBeta[t0]["epsilon"],Time2FitBeta[t0]["n(R,epsilon)"],marker ='+',label = f"t = {round(BestT,3)}")
            ax.plot(Time2FitBeta[t0]["epsilon"],Time2FitBeta[t0]["n(R,epsilon)_fit"], label = "")
            ax.text(min(Time2FitBeta[t0]["epsilon"]) + 0.1,min(Time2FitBeta[t0]["n(R,epsilon)_fit"]) + 0.0001,r"$\beta$ = " + f"{round(Time2FitBeta[t0]['Beta'],3)}")
        else:
            if t0 > t1 - 3 and t0 < t1 + 3:
                ax.plot(Time2FitBeta[t0]["epsilon"],Time2FitBeta[t0]["n(R,epsilon)"],label = f"t = {round(t_vect[t0],3)}",linestyle = '--')
#    ax.plot(BestA,BestBeta,'*',label = f"Beta: {round(BestBeta,3)} A: {round(BestA,2)}")
    ax.set_xlabel(r"$\epsilon = \frac{R^{2}-R^{2}_c}{R^{2}_c}$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(min(Time2FitBeta[t1]["epsilon"]),max(Time2FitBeta[t1]["epsilon"]))
    ax.set_ylim(10^(-4),3*10^(-1))
    ax.set_ylabel(r"$\beta$")
    ax.legend()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)

#    plt.title(f"Best Beta: {BestBeta} and Best A: {BestA} at t = {BestT}")
    plt.savefig(os.path.join(PlotDir,f'{City}_BestBeta_{UCI}.png'))
    plt.close(fig)
    

def safe_float(x):
    if isinstance(x, str):
        return float(x)
    elif isinstance(x,complex):
        return float(x.real)
    elif isinstance(x, list):
        return [safe_float(i) for i in x]
    elif isinstance(x, tuple):
        return tuple(safe_float(i) for i in x)
    elif isinstance(x,int):
        return float(x)
    elif isinstance(x, float):
        return x
    elif x is None:
        return np.nan
    elif isinstance(x, np.ndarray):
        return np.array([safe_float(i) for i in x])
    else:
        return x
def PlotRRcTauTauc(Taus,Tauc,Rs,Rc,alpha,City,UCI,PlotDir):
    """
        @params:
            - Tauc: float (is the critical tau: self.Taus[UCI])
            - Taus: list -> [Tau1,...,TauN]
            - Rc: float (is the critical R: self.Rs[UCI])
            - Rs: list -> [R1,...,RN]
            - alpha: float: self.Alphas[UCI]
        Description:
            - Plots the Beta vs R/Rc for different t

    """
    import os
    import json
    fig,ax = plt.subplots(figsize = (10,10))
    # Tau
    TausOverTauc = [safe_float((Taus[i]/Tauc)**(-alpha)) for i in range(len(Taus))]
    TausOverTaucMax = max(TausOverTauc) 
    # R
    ROverRc = [Rs[i]/Rc for i in range(len(Rs))]
    # Plot
    ax.scatter(ROverRc,TausOverTauc,label = r"$\frac{\tau}{\tau_c}^{\alpha}$")
    ax.vlines(x=1, ymin = 0, ymax = TausOverTaucMax, linestyles = '--')
    ax.set_xlabel(r"$\frac{R}{R_c}$")
    ax.set_ylabel(r"$\frac{\tau}{\tau_c}^{\alpha}$")
    ax.legend()
    plt.savefig(os.path.join(PlotDir,f'{City}_{UCI}_RRcTau.png'))
    plt.close(fig)
    with open(os.path.join(PlotDir,f'{City}_{UCI}_RRcTau.json'),'w') as f:
        json.dump({"tau_tauc_alpha":TausOverTauc,"R_over_Rc":ROverRc},f,indent=4)