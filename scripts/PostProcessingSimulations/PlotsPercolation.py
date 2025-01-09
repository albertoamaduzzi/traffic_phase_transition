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
    for UCI in Taus.keys():
        ax.scatter(Taus,Gammas)
    ax.set_xlabel(r'$\\tau$',fontsize=15)
    ax.set_ylabel(r'$\\gamma$',fontsize=15)
    plt.savefig(os.path.join(PlotDir,f'{City}_{round(UCI,3)}_GammaTau.png'))
    plt.close()
    pd.to_parquet(os.path.join(PlotDir,f'{City}_{round(UCI,3)}_GammaTau.parquet'),pd.DataFrame({'Tau':Taus,'Gamma':Gammas}))


def PlotFigure4(R2NtNtFit,R2epsilon,alpha,Time,t0,UCI,Rc,City,PlotDir):
    """
        @params:
            - UCI2NtNtFit: dict -> {UCI:{R:{'nt':Nt,'nt_fit':NtFit,'t':t}}}
            - epsilon: float
            - alpha: float
    """
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    fig1,ax1 = plt.subplots(1,1,figsize = (10,10))
    for R in R2NtNtFit.keys():
        SmallestIndexForPlot = min(len(R2NtNtFit[R]['n']),Time[t0:],len(R2NtNtFit[R]['n_fit']))
        tminust0epsilon = (np.array(R2NtNtFit[R]['t']) -t0)*R2epsilon[R]
        nttalpha = R2NtNtFit[R]['n'][:SmallestIndexForPlot]*Time[t0:SmallestIndexForPlot]**(-alpha)
        # Plot Bottom
        ax.scatter(tminust0epsilon,nttalpha,linestyle = '--')
        # Plot Top
        ax1.scatter(Time[t0:SmallestIndexForPlot],R2NtNtFit[R]['n'][:SmallestIndexForPlot])
        ax1.scatter(Time[t0:SmallestIndexForPlot],R2NtNtFit[R]['n_fit'][:SmallestIndexForPlot],linestyle = '--')
    ax.ahline(y=1)
    ax.text(0.01,2,r'$R_c = $'+ str(Rc),fontsize=15)
    ax.set_xlabel(r'$(t-t_0)\epsilon$',fontsize=15)
    ax.set_ylabel(r'$n(t)t^{\alpha} [h]$',fontsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(os.path.join(PlotDir,f'{City}_{round(UCI,3)}_Nt_Scaled.png'))
    ax1.text(0.01,2,r'$R_c = $'+ str(Rc),fontsize=15)
    ax1.set_xlabel(r'$(t-t_0)$',fontsize=15)
    ax1.set_ylabel(r'$n(t)[h]$',fontsize=15)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    fig1.savefig(os.path.join(PlotDir,f'{City}_{round(UCI,3)}_Nt.png'))
    fig.close()
    fig1.close()
