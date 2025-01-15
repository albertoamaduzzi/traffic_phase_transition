from FittingProcedures import FittingPowerlaw
import matplotlib.pyplot as plt
def ComputeTime2FitBeta(RangeFit,t_vect,R2NtNtFit,UCI2CriticalParams,Rs,UCI):
    """
        Description:
            - Compute the time2fitbeta for a given UCI
    """
    Time2FitBeta = {t0:{"y_fit_pl":[],"ErrorPL":[],"A_pl":[],"Beta":[]} for t0 in RangeFit}
    for t0 in RangeFit:
        NRsGivent0 = []
        FRcR = []
        t = t_vect[t0]
        Rs1 = [R for R in sorted(Rs) if R > UCI2CriticalParams[round(UCI,3)]["Rc"]]
        for R in sorted(Rs1):
            # Check PowerLaw fixed t0, between NRsGivent0 and (R^2 - Rc^2)/Rc^2
            NRsGivent0.append(R2NtNtFit[R][t]["n"][t])
            FRcR.append((R**2 - UCI2CriticalParams[round(UCI,3)]["Rc"]**2)/UCI2CriticalParams[round(UCI,3)]["Rc"]**2)
        y_fit_pl,ErrorPL,A_pl,Beta = FittingPowerlaw(FRcR,NRsGivent0)
        Time2FitBeta[t0] = {"n(R,epsilon)_fit":y_fit_pl,"n(R,epsilon)":NRsGivent0,"ErrorPL":ErrorPL,"A_pl":A_pl,"Beta":Beta,"epsilon":FRcR}
    return Time2FitBeta

def ChooseBestBeta(Time2FitBeta,t_vect):
    """
        Description:
            Choose the best Beta, A and T for the powerlaw fit.
    """
    MinError = 1e10
    for t0 in Time2FitBeta.keys():
        if Time2FitBeta[t0]["ErrorPL"] < MinError:
            MinError = Time2FitBeta[t0]["ErrorPL"]
            BestBeta = Time2FitBeta[t0]["Beta"]
            BestA = Time2FitBeta[t0]["A_pl"]
            BestT = t_vect[t0]
            t1 = t0
    return BestBeta,BestA,BestT,t1         


def PlotBestBeta(Time2FitBeta,BestBeta,BestA,BestT,t1):
    """
        Description:

    """   
    fig,ax = plt.subplots(figsize = (10,10))
    
    for t0 in Time2FitBeta.keys():
        if t1 == t0:
            ax.scatter(Time2FitBeta[t0]["n(R,epsilon)"],Time2FitBeta[t0]["epsilon"],'+')
            ax.plot(Time2FitBeta[t0]["n(R,epsilon)_fit"],Time2FitBeta[t0]["epsilon"])
            ax.text(0.1,0.1,r"$\beta$ = " + f"{Time2FitBeta[t0]['Beta']}")
        else:
            plt.plot(Time2FitBeta[t0]["n(R,epsilon)"],Time2FitBeta[t0]["epsilon"],'--')

    plt.plot(BestA,BestBeta,'*',label = f"Best Beta: {BestBeta} and Best A: {BestA}")
    plt.xlabel("A")
    plt.ylabel("Beta")
    plt.legend()
    plt.title(f"Best Beta: {BestBeta} and Best A: {BestA} at t = {BestT}")
    plt.show()
    return