from FittingProcedures import FittingPowerlaw
import matplotlib.pyplot as plt
import logging
import numpy as np
from collections import defaultdict
import os
from matplotlib.ticker import ScalarFormatter
logger = logging.getLogger(__name__)

def ComputeTime2FitBeta(RangeFit,t_vect,R2NtNtFit,UCI2CriticalParams,Rs,UCI,City):
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
        t = t_vect[t0]
        Rs1 = [R for R in sorted(Rs) if R > UCI2CriticalParams[round(UCI,3)]["Rc"]]
        for R in sorted(Rs1):
            # Avoid those casesfor which I have R but no simulation is computed
            if "n" in R2NtNtFit[R].keys():
                # Check PowerLaw fixed t0, between NRsGivent0 and (R^2 - Rc^2)/Rc^2
#                if R2NtNtFit[R]["n"][t0] > 0:
                NRsGivent0.append(R2NtNtFit[R]["n"][t0])
                FRcR.append((R**2 - UCI2CriticalParams[round(UCI,3)]["Rc"]**2)/UCI2CriticalParams[round(UCI,3)]["Rc"]**2)
            if np.shape(NRsGivent0) != np.shape(FRcR):  
                logger.error(f"shape NRsGivent0: {len(NRsGivent0)}, shape FRcR: {len(FRcR)}")
                logger.error(f"NRsGivent0: {NRsGivent0}")
                logger.error(f"FRcR: {FRcR}")
                raise ValueError("NRsGivent0 and FRcR must have the same shape")
        if len(NRsGivent0) == len(FRcR) and len(NRsGivent0) > 4:
            # Fit Power Law
            y_fit_pl,ErrorPL,A_pl,Beta = FittingPowerlaw(FRcR,NRsGivent0)
            if y_fit_pl is not None:
                y_fit_pl = np.array(y_fit_pl)/np.sum(y_fit_pl)
                NRsGivent0 = np.array(NRsGivent0)/np.sum(NRsGivent0)
                Time2FitBeta[t0] = {"n(R,epsilon)_fit":y_fit_pl,"n(R,epsilon)":NRsGivent0,"ErrorPL":ErrorPL,"A_pl":A_pl,"Beta":Beta,"epsilon":FRcR}

    return Time2FitBeta

def PlotTime2FitBetaError(Time2FitBeta,t_vect,t_star,City,UCI,PlotDir):
    """
        Description:
            - Plot the Error of the Power Law Fit
    """
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    Betas = [Time2FitBeta[t0]["Beta"] for t0 in Time2FitBeta.keys()]
    Errors = [Time2FitBeta[t0]["ErrorPL"] for t0 in Time2FitBeta.keys()]
    time_ = [t_vect[t0] for t0 in list(Time2FitBeta.keys())]
    ax.scatter(time_,Errors,marker = "o",label = f"error")
    ax.scatter(time_,Betas,marker = "*",label = r"$\beta$")
    int_t_star = int(t_star)
    ax.vlines(x = t_vect[int_t_star],ymin = 0,ymax = 0.05,color = "red",linestyles = "dashed")
    ax2 = ax.twinx()
    ax2.spines['left'].set_position(('outward', 60))
    ax2.set_xlabel("t (h)")
    ax2.set_ylabel("error")
    ax2.set_title(f"{City}, UCI: {UCI}")
    plt.savefig(os.path.join(PlotDir,f'{City}_{UCI}_ErrorBeta.png'))
    plt.close(fig)
def ChooseBestBeta(Time2FitBeta,t_vect):
    """
        Description:
            Choose the best Beta, A and T for the powerlaw fit.
    """

    logger.info("Choosing Best Beta")
    MinError = 1e10
    for t0 in Time2FitBeta.keys():
        if Time2FitBeta[t0]["ErrorPL"] < MinError:
            MinError = Time2FitBeta[t0]["ErrorPL"]
            BestBeta = Time2FitBeta[t0]["Beta"]
            BestA = Time2FitBeta[t0]["A_pl"]
            BestT = t_vect[t0]
            t1 = t0
    return BestBeta,BestA,BestT,t1         


def PlotBestBeta(Time2FitBeta,BestBeta,BestA,BestT,t1,City,t_vect,UCI,PlotDir):
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
                ax.plot(Time2FitBeta[t0]["epsilon"],Time2FitBeta[t0]["n(R,epsilon)"],label = f"t = {round(t_vect[t0],3)}")
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
    TausOverTauc = [(Taus[i]/Tauc)**(-alpha) for i in range(len(Taus))]
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