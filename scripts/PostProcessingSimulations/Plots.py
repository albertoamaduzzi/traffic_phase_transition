import matplotlib.pyplot as plt
import geopandas as gpd
#import geoplot as gplt
#import geoplot.crs as gcrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from DateTime_ import *
#import imageio
import os

# Unloading Curve
def PlotPeopleInNetwork(Interval2NumberPeopleInNet,interval15,FileName):
    '''
        Input:
            df_people: DataFrame containing the people information (stay time in the network, time of departure, travel time, etc.)
            save_dir: Directory where the plot will be saved
            name: Name of the plot
            hour_in_day: Number of hours in a day
            minutes_in_hour: Number of minutes in an hour
            seconds_in_minute: Number of seconds in a minute
            interval_in_minutes: Number of minutes in each interval
    '''
    fig,ax = plt.subplots(1,1,figsize = (15,15))
    Interval2Plot = [t for t in interval15 if Interval2NumberPeopleInNet[t] > 0]
    NPeople2Count = [Interval2NumberPeopleInNet[t] for t in interval15 if Interval2NumberPeopleInNet[t] > 0]
    ax.plot(Interval2Plot[1:-1],NPeople2Count[1:-1])
#    ax.set_xticks(second2hour(interval15))
#    ax.set_xticklabels([str(t) for t in interval15])
    ax.set_xlabel('time')
    ax.set_ylabel('Number people in graph')
    ax.set_xticks(ticks = interval15[1:-1],labels = interval15[1:-1],rotation=90)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig(FileName,dpi = 200)
    plt.show()

# Hopefully traffic in the road network
'''def PlotTrafficInGeopandasNet(TrafficGdf,TrafficLevel,ColorBarExplanation,PlotFile,Title,dpi = 300,IsLognorm = False):
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    gs = GridSpec(1, 2, width_ratios=[20, 1])
    ax = fig.add_subplot(gs[0,0],projection=gcrs.AlbersEqualArea())
    if IsLognorm:
        gplt.sankey(
            TrafficGdf,
            scale= TrafficLevel,
            limits=(0.1, 3),
            hue= TrafficLevel,
            cmap = 'inferno',
            norm = LogNorm(),
            ax=ax  # Use the created axes
        )
        sm = plt.cm.ScalarMappable(cmap='inferno', norm=LogNorm(vmin=min(TrafficGdf[TrafficLevel].to_numpy()), vmax=max(TrafficGdf[TrafficLevel].to_numpy())))

    else:
        gplt.sankey(
            TrafficGdf,
            scale= TrafficLevel,
            limits=(0.1, 3),
            hue= TrafficLevel,
            cmap = 'inferno',
            ax=ax  # Use the created axes
        )
        sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=min(TrafficGdf[TrafficLevel].to_numpy()), vmax=max(TrafficGdf[TrafficLevel].to_numpy())))

    cax = fig.add_subplot(gs[0,1])
    # Create a ScalarMappable object for the colorbar
    # Empty array for the data range
    sm.set_array([])
    # Add the colorbar to the figure
    cbar = fig.colorbar(sm, cax=cax)
    # Set the colorbar label
    cbar.set_label(ColorBarExplanation)
    ax.set_title(Title)
    plt.savefig(PlotFile,dpi = dpi)

    plt.show()
    return PlotFile

# Animation traffic in the road network
def AnimateNetworkTraffic(PlotDir,TrafficGdf,Column2InfoSavePlot,dpi = 300,IsLognorm = False):
    images = []
    
    for Column in Column2InfoSavePlot:
        PlotFile = os.path.join(PlotDir,Column2InfoSavePlot[Column]["savefile"])
        PlotTrafficInGeopandasNet(TrafficGdf,Column,Column2InfoSavePlot[Column]["colorbar"],PlotFile,Column2InfoSavePlot[Column]["title"],dpi,IsLognorm)
        images.append(imageio.imread(PlotFile))
    imageio.mimsave(Column2InfoSavePlot[Column]["animationfile"], images, duration = 0.5)
    return 'movie.gif'
'''

# Fondamental Diagram


# Phase Transition
def PlotAvailableUCIs(UCIs,PlotDir):
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    ax.scatter(UCIs,[0]*len(UCIs))
    ax.set_xlabel('UCI',fontsize=15)
    ax.set_title('Available UCIs',fontsize=15)
    plt.savefig(os.path.join(PlotDir,'AvailableUCIs.png'))


def PlotAlphaBeta(Alphas,Betas,PlotDir):
    """
        Description:
            - Plots the Alpha and Beta
        Args:
            - Alphas: list -> [alpha1,alpha2,...,alphaN]
            - Betas: list -> [beta1,beta2,...,betaN]
    """
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    ax.scatter(Alphas,Betas,label = 'Alpha')
    ax.set_xlabel(r'$\alpha$',fontsize=15)
    ax.set_ylabel(r'$\beta$',fontsize=15)
    ax.set_title('Critical Exponent',fontsize=15)
    ax.axhline(y=0.27, xmin=0 , xmax=1, color='black', linestyle='--')
    ax.axhline(y=0.25, xmin=0 , xmax=1, color='blue', linestyle='--')
    ax.axhline(y=0.29, xmin=0 , xmax=1, color='blue', linestyle='--')
    ax.text(0.28,0.2,'monocentric', ha='right', va='bottom', color='black',fontsize=15)
    ax.text(0.9,0.2,'1DP', ha='right', va='bottom', color='black',fontsize=15)
    ax.axhline(y=0.58, xmin=0 , xmax=1, color='black', linestyle='--')
    ax.axhline(y=0.6, xmin=0 , xmax=1, color='red', linestyle='--')
    ax.axhline(y=0.56, xmin=0 , xmax=1, color='red', linestyle='--')
    ax.text(0.22,0.62,'polycentric', ha='right', va='bottom', color='black',fontsize=15)
    ax.text(0.9,0.62,'2DP', ha='right', va='bottom', color='black',fontsize=15)
    plt.savefig(os.path.join(PlotDir,'AlphaBeta.png'))

def PlotNR(Rs,Rc,NR,PlotDir):
    """
        Description:
            - Plots the NR given in input the array of Rs and NR
        Args:
            - Rs: list -> [R1,R2,...,RN]
            - Rc: int -> Rc
            - NR: list -> [NR1,NR2,...,NRN]
    """
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    Rs = Rs/Rc
    ax.scatter(Rs,NR,label = 'NR')
    ax.set_xlabel(r'\\frac{R}{R_{c}}',fontsize=15)
    ax.set_ylabel('N(R)',fontsize=15)
    ax.set_title('NR',fontsize=15)
    plt.savefig(os.path.join(PlotDir,'NR.png'))

def PlotNtAndFitSingleR(t,Nt,tau,NtFitted,R,UCI,PlotDir):
    """
        Description:
            - Plots the Nt given in input the array of Nt
        Args:
            - Nt: list -> [Nt1,Nt2,...,NtN]
    """
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    ax.scatter(t,Nt)
    ax.plot(t,NtFitted,label = 'Fitted')
    ax.axvline(x=tau, color='red', linestyle='--')

    ax.set_xlabel('t',fontsize=15)
    ax.set_ylabel('N(t)',fontsize=15)
    ax.set_title('Number of People in network',fontsize=15)
    plt.savefig(os.path.join(PlotDir,f'{R}_{round(UCI,3)}_Nt.png'))

def PlotNtAndFit(Rs,Nt,tau,NtFitted,PlotDir):
    """
        Description:
            - Plots the Nt given in input the array of Nt
        Args:
            - Nt: list -> [Nt1,Nt2,...,NtN]
    """
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    for i in range(len(Rs)):
        ax.scatter(Rs[i],Nt[i])
        ax.plot(Rs[i],NtFitted[i],label = 'Fitted')
    ax.axvline(x=tau, color='red', linestyle='--')

    ax.set_xlabel('R',fontsize=15)
    ax.set_ylabel('N(t)',fontsize=15)
    ax.set_title('Number of People in network',fontsize=15)
    plt.savefig(os.path.join(PlotDir,'Nt.png'))

def PlotErrorFitAlphaWindow(Time2ErrorFit,R,UCI,PlotDir):
    """
        Description:
            - Plots the Error Fit given in input the dictionary Time2ErrorFit
        Args:
            - Time2ErrorFit: dict -> {time:{'error':0,"A":0,"b":0,"is_powerlaw":False}}
    """
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    ax.scatter(list(Time2ErrorFit.keys()),list(Time2ErrorFit.values()))
    ax.set_xlabel('Window Time',fontsize=15)
    ax.set_ylabel('Error',fontsize=15)
    ax.set_title('Error Fit',fontsize=15)
    plt.savefig(os.path.join(PlotDir,f'{R}_{round(UCI,3)}_ErrorFitAlphaWindow.png'))




################### AGGREGATE AT R LEVEL ############################
# Fig. 10 Supplementay

def PlotR2Tau(R2Tau,PlotDir):
    """
        Description:
            - Plots the R2Tau
        Args:
            - R2Tau: dict -> {R:tau}
    """
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    ax.scatter(list(R2Tau.keys()),list(R2Tau.values()))
    ax.set_xlabel('R',fontsize=15)
    ax.set_ylabel(r'$\\tau(R)$',fontsize=15)
    ax.set_title('Tau',fontsize=15)
    plt.savefig(os.path.join(PlotDir,'R2Tau.png'))