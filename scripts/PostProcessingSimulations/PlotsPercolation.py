import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import os
import numpy as np
def PlotGammaTau(Taus,Gammas,IndexRc,City,UCI,PlotDir):
    """
        Description:
            - Plots the Gamma vs Tau for each UCI
        Args:
            - Taus: dict -> [Tau_R0,...,Tau_RN]
            - Gammas: dict -> [Gamma]
            - City: str -> City Name
            - PlotDir: str -> Directory where the plot will be saved
        NOTE: The Plot is Saved with the corresponding curve so that it can be used for collective study.
    """
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    ax.scatter(Gammas,Taus)
    ax.vlines(x = Gammas[IndexRc],ymin = min(Taus),ymax = max(Taus),linestyles = '--',color = 'black')
    ax.set_xlabel(r'$\Gamma$',fontsize=15)
    ax.set_ylabel(r'$\tau$',fontsize=15)
    plt.savefig(os.path.join(PlotDir,f'{City}_{round(UCI,3)}_GammaTau.png'))
    plt.close()
    pd.DataFrame({'Tau':Taus,'Gamma':Gammas}).to_parquet(os.path.join(PlotDir,f'{City}_{round(UCI,3)}_GammaTau.parquet'))


def PlotFigure4(R2NtNtFit,R2epsilon,alpha,Time,t0,UCI,Rc,City,BestT,delta_RC_plot_before,delta_RC_plot_after,delta_RC_plot_1,PlotDir):
    """
        @params:
            - UCI2NtNtFit: dict -> {UCI:{R:{'nt':Nt,'nt_fit':NtFit,'t':t}}}
            - epsilon: float
            - alpha: float
    """
    from scipy.signal import savgol_filter
    print(f"{City}:Plotting Figure 4 for UCI = {UCI}")
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    fig1,ax1 = plt.subplots(1,1,figsize = (10,10))

    for R in R2NtNtFit.keys():
#        SmallestIndexForPlot = min(len(R2NtNtFit[R]['n']),Time[t0:],len(R2NtNtFit[R]['n_fit']))
#        nttalpha = R2NtNtFit[R]['n'][:SmallestIndexForPlot]*Time[t0:SmallestIndexForPlot]**(-alpha)
        # NOTE: since we are considering at the same time different sets of R (different UCI have different ranges to transition to gridlock)
        # NOTE: for a fixed UCI is necessary to add this condition since we avoid the case when the size of N is bigger than the size of t
        if "n" in R2NtNtFit[R].keys():
            assert len(R2NtNtFit[R]['n']) == len(R2NtNtFit[R]['n_fit']),f"The length of n {len(R2NtNtFit[R]['n'])} and n_fit {len(R2NtNtFit[R]['n_fit'])} should be the same"
            size_N_minus_size_t = len(R2NtNtFit[R]['n']) - len(Time[t0:])
            if size_N_minus_size_t == 0:
                #savitzky_golay
                n = savgol_filter(R2NtNtFit[R]['n'], 7, 2)
                n_fit = R2NtNtFit[R]['n_fit']
                t = np.array(Time[t0:]) - Time[t0]
                t_alpha_bigger_then_0 = np.array(t[1:])**(-alpha) > 0
                nttalpha = (n[1:]*np.array(t[1:])**(-alpha))[t_alpha_bigger_then_0]

            elif size_N_minus_size_t > 0:
                #savitzky_golay
                n = savgol_filter(R2NtNtFit[R]['n'][:-size_N_minus_size_t], 7, 2)
                n_fit = R2NtNtFit[R]['n_fit'][:-size_N_minus_size_t]
                t = np.array(Time[t0:]) - Time[t0]
                t_alpha_bigger_then_0 = np.array(t[1:])**(-alpha) > 0
                nttalpha = (n[1:]*np.array(t[1:])**(-alpha))[t_alpha_bigger_then_0]
            else:
                #savitzky_golay
                n = savgol_filter(R2NtNtFit[R]['n'], 7, 2)
                n_fit = R2NtNtFit[R]['n_fit']
                t = np.array(Time[t0:-size_N_minus_size_t]) - Time[t0]
                t_alpha_bigger_then_0 = np.array(t[1:])**(-alpha) > 0
                nttalpha = (n[1:]*np.array(t[1:])**(-alpha))[t_alpha_bigger_then_0]
            tminust0epsilon = (np.array(t[1:]))*R2epsilon[R]
            # Plot Bottom
            if int(R) == int(Rc):
                ax.scatter(tminust0epsilon,nttalpha,label = rf'Rc = {R} veh/$\Delta$t')
            elif (int(R) > int(Rc) - delta_RC_plot_before and  int(R) < int(Rc) + delta_RC_plot_after) or (int(R) > int(Rc) + delta_RC_plot_after and int(R) < int(Rc) + delta_RC_plot_1): #and int(R) < int(Rc) + 40
                ax.plot(tminust0epsilon,nttalpha,linestyle = '--',label = rf'R = {R} veh/$\Delta$t')
#                ax.plot(tminust0epsilon[t0:],nttalpha,linestyle = '--',label = rf'R = {R} veh/$\Delta$t')
#            else:
#                ax.plot(tminust0epsilon[t0:],nttalpha,linestyle = '--',label = '')
            if int(R) == int(Rc):
                ax1.scatter(t,n,label = rf'$R_c$ = {R} veh/$\Delta$t')
                ax1.plot(t,n_fit,linestyle = '--',label = '')
            elif (int(R) > int(Rc) - delta_RC_plot_before and  int(R) < int(Rc) + delta_RC_plot_after) or (int(R) > int(Rc) + delta_RC_plot_after and int(R) < int(Rc) + delta_RC_plot_1): # and int(R) < int(Rc) + 200
                ax1.plot(t,n,linestyle = '--',label = rf'R = {R} veh/$\Delta$t')
#            else:
#                ax1.plot(Time[t0:],R2NtNtFit[R]['n'][1:],linestyle = '--',label = '')
#    ax.hlines(y=1)#(y=1)
#    ax.text(0.1,2,r'$R_c = $'+ str(Rc),fontsize=15)
    ax.set_xlabel(r'$(t-t_0)\epsilon$',fontsize=15)
    ax.set_ylabel(r'$n(t)t^{\alpha} [h]$',fontsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(10**(-2),15)
#    ax.set_ylim(10**(-5),10)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.legend()
    fig.savefig(os.path.join(PlotDir,f'{City}_{round(UCI,3)}_Nt_Scaled.png'))
    ax1.text(0.1,2,r'$R_c = $'+ str(Rc),fontsize=15)
    ax1.set_xlabel(r'$(t-t_0)[h]$',fontsize=15)
    ax1.set_ylabel(r'$n(t)[h]$',fontsize=15)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_xlim(10**(0),15)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.xaxis.get_major_formatter().set_scientific(False)
    ax1.xaxis.get_major_formatter().set_useOffset(False)
    ax1.vlines(x = BestT, ymin = 0, ymax = 1, linestyles = '--',color = 'black')
    fig1.savefig(os.path.join(PlotDir,f'{City}_{round(UCI,3)}_Nt.png'))
    plt.close(fig)
    plt.close(fig1)


## TRANSITION BETA ALPHA
def PlotFig5A5B(Rs,                                          # NOTE: Rs
                Rc,                                          # NOTE: Rc
                R2NtNtFit,                                   # NOTE: N(R), by choosing the int_t_star position of the vector associated to R
                int_t_star,                                                         # NOTE: t*, time at which the best transition occurs.
                BestBeta,                                                           # NOTE: Best Beta, is the coefficient for the increase of the N(R) at time t*    
                City,
                UCI,
                PlotDir):
    """
        Description:
            - Plots the N(R) vs R for the best Beta
        Args:
            - Rs: list -> List of Rs
            - Rc: float -> Rc
            - R2NtNtFit: dict -> {R:{'n':n,'n_fit':n_fit}}
            - int_t_star: int -> t*
            - BestBeta: float -> Best Beta
    """
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    n_R = []
    new_Rs = []
    for R in Rs:
        if R in R2NtNtFit.keys():
            n_R.append(R2NtNtFit[R]['n'][int_t_star])
            new_Rs.append(R)
    new_Rs = np.array(new_Rs)/Rc
    ax.scatter(new_Rs,n_R)
    ax.plot(new_Rs[new_Rs>=Rc],new_Rs[new_Rs>=Rc]**BestBeta)
    ax.set_xlabel(r'$R/R_c$',fontsize=15)
    ax.set_ylabel(r'$n(R,t_{\infty})$',fontsize=15)
    plt.savefig(os.path.join(PlotDir,f'{City}_{round(UCI,3)}_NtStar.png'))
    plt.close(fig)

