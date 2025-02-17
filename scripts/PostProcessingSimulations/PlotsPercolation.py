import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os
import numpy as np
def PlotGammaTau(Taus,Gammas,City,UCI,PlotDir):
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
    ax.set_xlabel(r'$\Gamma$',fontsize=15)
    ax.set_ylabel(r'$\tau$',fontsize=15)
    plt.savefig(os.path.join(PlotDir,f'{City}_{round(UCI,3)}_GammaTau.png'))
    plt.close()
    pd.DataFrame({'Tau':Taus,'Gamma':Gammas}).to_parquet(os.path.join(PlotDir,f'{City}_{round(UCI,3)}_GammaTau.parquet'))


def PlotFigure4(R2NtNtFit,R2epsilon,alpha,Time,t0,UCI,Rc,City,BestT,PlotDir):
    """
        @params:
            - UCI2NtNtFit: dict -> {UCI:{R:{'nt':Nt,'nt_fit':NtFit,'t':t}}}
            - epsilon: float
            - alpha: float
    """
    print(f"{City}:Plotting Figure 4 for UCI = {UCI}")
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    fig1,ax1 = plt.subplots(1,1,figsize = (10,10))
    for R in R2NtNtFit.keys():
#        SmallestIndexForPlot = min(len(R2NtNtFit[R]['n']),Time[t0:],len(R2NtNtFit[R]['n_fit']))
        tminust0epsilon = (np.array(Time) -t0)*R2epsilon[R]
#        nttalpha = R2NtNtFit[R]['n'][:SmallestIndexForPlot]*Time[t0:SmallestIndexForPlot]**(-alpha)
        if "n" in R2NtNtFit[R].keys():
            nttalpha = R2NtNtFit[R]['n'][1:]*Time[t0:]**(-alpha)

            # Plot Bottom
            if int(R) == int(Rc):
                ax.scatter(tminust0epsilon[t0:],nttalpha,label = rf'Rc = {R} veh/$\Delta$t')
            elif int(R) > int(Rc) - 100 and int(R) < int(Rc) + 100:
                ax.plot(tminust0epsilon[t0:],nttalpha,linestyle = '--',label = rf'R = {R} veh/$\Delta$t')
#            else:
#                ax.plot(tminust0epsilon[t0:],nttalpha,linestyle = '--',label = '')
            if int(R) == int(Rc):
                ax1.scatter(Time[t0:],R2NtNtFit[R]['n'][1:],label = rf'$R_c$ = {R} veh/$\Delta$t')
                ax1.plot(Time[t0:],R2NtNtFit[R]['n_fit'][1:],linestyle = '--',label = '')
            elif int(R) > int(Rc) - 100 and int(R) < int(Rc) + 100:
                ax1.plot(Time[t0:],R2NtNtFit[R]['n'][1:],linestyle = '--',label = rf'R = {R} veh/$\Delta$t')
#            else:
#                ax1.plot(Time[t0:],R2NtNtFit[R]['n'][1:],linestyle = '--',label = '')
#    ax.hlines(y=1)#(y=1)
    ax.text(0.1,2,r'$R_c = $'+ str(Rc),fontsize=15)
    ax.set_xlabel(r'$(t-t_0)\epsilon$',fontsize=15)
    ax.set_ylabel(r'$n(t)t^{\alpha} [h]$',fontsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(10**(-2),15)
    ax.set_ylim(10**(-5),10)
    ax.legend()
    fig.savefig(os.path.join(PlotDir,f'{City}_{round(UCI,3)}_Nt_Scaled.png'))
    ax1.text(0.1,2,r'$R_c = $'+ str(Rc),fontsize=15)
    ax1.set_xlabel(r'$(t-t_0)$',fontsize=15)
    ax1.set_ylabel(r'$n(t)[h]$',fontsize=15)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_xlim(10**(-1),15)
    ax1.vlines(x = BestT -t0, ymin = 0, ymax = 1, linestyles = '--',color = 'black')
    fig1.savefig(os.path.join(PlotDir,f'{City}_{round(UCI,3)}_Nt.png'))
    plt.close(fig)
    plt.close(fig1)
