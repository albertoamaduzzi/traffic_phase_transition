import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import os
import numpy as np

# --------------- Section 1: Plot for single (UCI,City) --------------------- #

# NOTE: figure 3.c

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


def equalize_vectors(vec1, vec2, vec3):
    """
    Truncate larger vectors to match the length of the smallest vector.
    
    Args:
        vec1, vec2, vec3: Input vectors (lists, numpy arrays, etc.)
        
    Returns:
        tuple: (vec1_truncated, vec2_truncated, vec3_truncated)
    """
    # Get the length of each vector
    len1, len2, len3 = len(vec1), len(vec2), len(vec3)
    
    # Find the minimum length
    min_len = min(len1, len2, len3)
    
    
    # Truncate vectors to the minimum length
    vec1_truncated = vec1[:min_len]
    vec2_truncated = vec2[:min_len]
    vec3_truncated = vec3[:min_len]
    
    return vec1_truncated, vec2_truncated, vec3_truncated


    
def Extract_n_n_fit_t(R2NtNtFit,Time,R,savgol_window,alpha):
    from scipy.signal import savgol_filter
    consider_2_plot = True
    assert len(R2NtNtFit[R]['n']) == len(R2NtNtFit[R]['n_fit']),f"The length of n {len(R2NtNtFit[R]['n'])} and n_fit {len(R2NtNtFit[R]['n_fit'])} should be the same"
    assert len(R2NtNtFit[R]['n']) == len(Time),f"The length of n {len(R2NtNtFit[R]['n'])} and Time {len(Time)} should be the same"
    data_length = len(R2NtNtFit[R]['n'])
    try:            
        # Ensure the window is odd (required by savgol) and at least 3 (> polyorder of 2)
        current_savgol_window = min(savgol_window, (data_length))
        if current_savgol_window < 3:
            current_savgol_window = 3
        if current_savgol_window % 2 == 0:  # Make odd if even
            current_savgol_window = current_savgol_window - 1
        
        if current_savgol_window < 3:
            # If we can't use savgol_filter, just use the original data
            print(f"Warning: Can't use savgol_filter for R={R}, data too short. Using original data.")
            n = R2NtNtFit[R]['n']
            n_fit = R2NtNtFit[R]['n_fit']
            t = np.array(Time[:])# - Time[t0]
            nttalpha = (n*np.array(t)**(-alpha))

        else:
            #savitzky_golay
            n = savgol_filter(R2NtNtFit[R]['n'], current_savgol_window, 2)
            n_fit = R2NtNtFit[R]['n_fit']
            t = np.array(Time)# - Time[t0]
            nttalpha = (n*np.array(t)**(-alpha))
    except Exception as e:
        print(f"Error: {e}")
        n = R2NtNtFit[R]['n']
        n_fit = R2NtNtFit[R]['n_fit']
        t = np.array(Time)# - Time[t0]
        nttalpha = (n*np.array(t)**(-alpha))
        consider_2_plot = False
    if len(nttalpha) < 4:
        print(f"Warning: nttalpha is too short for R={R}, consider_2_plot = False")
        consider_2_plot = False
    # NOTE: avoid cases where the number of people drops too quickly     
    return n, n_fit, t, nttalpha, consider_2_plot

# NOTE: figure 4.a,b,c

def PlotFigure4(R2NtNtFit,
                R2epsilon,
                alpha,
                Time,
                UCI,
                Rc,
                City,
                BestT,
                delta_RC_plot_before,
                delta_RC_plot_after,
                delta_RC_plot_1,
                PlotDir,
                savgol_window=5):
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
            # NOTE: avoid cases where the number of people drops too quickly
            R2NtNtFit[R]['n'], R2NtNtFit[R]['n_fit'], Time = equalize_vectors(R2NtNtFit[R]['n'], R2NtNtFit[R]['n_fit'], Time)
            n, n_fit, t, nttalpha, consider_2_plot = Extract_n_n_fit_t(R2NtNtFit,Time,R,savgol_window,alpha)
            if consider_2_plot:
                type_key_R2epsilon = type(list(R2epsilon.keys())[0])
                tminust0epsilon = (np.array(t))*R2epsilon[type_key_R2epsilon(R)]
                # Plot Bottom
                if int(R) == int(Rc):
                    ax.plot(tminust0epsilon,nttalpha,label = rf'Rc = {R} veh/$\Delta$t',color = 'black')
                elif (int(R) > int(Rc) - delta_RC_plot_before and  int(R) < int(Rc) + delta_RC_plot_after) or (int(R) > int(Rc) + delta_RC_plot_after and int(R) < int(Rc) + delta_RC_plot_1): #and int(R) < int(Rc) + 40
                    ax.plot(tminust0epsilon,nttalpha,linestyle = '--',label = rf'R = {R} veh/$\Delta$t')
                if int(R) == int(Rc):
                    ax1.scatter(t,n,label = rf'$R_c$ = {R} veh/$\Delta$t')
                    ax1.plot(t,n_fit,linestyle = '--',label = '')
                elif (int(R) > int(Rc) - delta_RC_plot_before and  int(R) < int(Rc) + delta_RC_plot_after) or (int(R) > int(Rc) + delta_RC_plot_after and int(R) < int(Rc) + delta_RC_plot_1): # and int(R) < int(Rc) + 200
                    ax1.plot(t,n,linestyle = '--',label = rf'R = {R} veh/$\Delta$t')
            else:
                pass
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

# NOTE: figure 5.a,b

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


def plot_error_fit_expo_powerlaw(Rs,err_exp_nt_R,err_pl_nt_R,PlotDir):
    """
        Description:
            - Plots the error of the fit for the powerlaw and exponential
        Args:
            - Rs: list -> List of R
            - err_exp_nt_R: list -> Error of the fit for the exponential
            - err_pl_nt_R: list -> Error of the fit for the powerlaw
    """
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    ax.scatter(Rs,err_exp_nt_R,label = 'Exponential')
    ax.scatter(Rs,err_pl_nt_R,label = 'Powerlaw')
    ax.set_xlabel(r'$R$',fontsize=15)
    ax.set_ylabel(r'$\epsilon$',fontsize=15)
    ax.set_yscale('log')
    ax.set_ylim(0,10**(1))
    ax.legend()
    plt.savefig(os.path.join(PlotDir,f'ErrorFit.png'))
    plt.close(fig)

# -------------- Compute average quantities of interest in UCI intervals ---------------------- # 

# NOTE: These plot are taken for all the cities together. We give as a starting object the Info_Simulation_DP

def PlotAverageRcUCITauc_from_InfoSimulationDP(Info_Simulation_DP,
                    PlotDir,
                    City2Color = {"BOS":"red","LIS":"blue","LAX":"green","SFO":"orange","RIO":"purple"},
                    City2Marker = {"BOS":"o","LIS":"s","LAX":"^","SFO":"D","RIO":"P"}):
    """
        @params:
            - Info_Simulation_DP: dict -> {City:{UCI:{'alpha':alpha,            # NOTE: exponent powerlaw better then expo -> n(t)
                                                    'beta':beta,                # NOTE: exponent best fit powerlaw in t -> to verify transition
                                                    'Rc':Rc,                    # NOTE: Critical loading rate for the transition, the R for which alpha is defined
                                                    'index_Rc':index_Rc,        # NOTE: index of Rc in the list of Rs
                                                    'Tauc':Tauc,                # NOTE: Tauc, the time at which the transition occurs
                                                    'n(R)':[n(R)],              # NOTE: n(R), the number of vehicles at the transition time
                                                    'Rs':[R],                   # NOTE: Rs, the list of Rs
                                                    'Taus':[Tau],               # NOTE: Taus, the list of Taus
                                                    }
                                                }
                                            }       
        @description:
            This plot will show a 3d representation of UCI = x-axis, Rc = y-axis 
    """
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    fig1,ax1 = plt.subplots(1,1,figsize=(10,10))
    keys = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # Pick the city
    for City in Info_Simulation_DP.keys():
        # Pick the UCIs
        uci_2_Rc = {i:{"Rc":[],"Tauc":[]} for i in keys}
        for UCI in Info_Simulation_DP[City].keys():
            float_UCI = float(UCI)
            # NOTE: Pick the UCI in the range of 0.1
            # Pick the alpha and beta
            if 'Rc' in Info_Simulation_DP[City][UCI].keys() and 'Tauc' in Info_Simulation_DP[City][UCI].keys():
                for i in range(len(keys)-1):
                    if float_UCI >= keys[i] and float_UCI < keys[i+1]:
                        uci_2_Rc[keys[i]]["Rc"].append(Info_Simulation_DP[City][UCI]['Rc'])
                        uci_2_Rc[keys[i]]["Tauc"].append(Info_Simulation_DP[City][UCI]['Tauc'])
                        break
        # Compute the average of the alpha beta
        UCIs = []
        Rcs = []
        Tauc = []
        for interva_uci_val in uci_2_Rc.keys():
            if len(uci_2_Rc[interva_uci_val]["Rc"]) > 0:
                UCIs.append(interva_uci_val)
                Rcs.append(np.nanmean(uci_2_Rc[interva_uci_val]["Rc"]))
                Tauc.append(np.nanmean(uci_2_Rc[interva_uci_val]["Tauc"]))
        # Plot Rc
        ax.scatter(UCIs,Rcs,marker = City2Marker[City],color = City2Color[City],label = City)
        ax.set_xlabel('UCI',fontsize=15)
        ax.set_ylabel(r'$R_c$',fontsize=15)
        plt.savefig(os.path.join(PlotDir,f'Rc_UCI.png'))
        plt.close(fig)
        # Plot Tauc
        ax1.scatter(UCIs,Tauc,marker = City2Marker[City],color = City2Color[City],label = City)
        ax1.set_xlabel('UCI',fontsize=15)
        ax1.set_ylabel(r'$\tau_{c}$',fontsize=15)
        plt.savefig(os.path.join(PlotDir,f'Tauc_UCI.png'))
        plt.close(fig1)


def PlotAlphaBetaUCI_from_InfoSimulationDP(Info_Simulation_DP,
                    PlotDir,
                    City2Color = {"BOS":"red","LIS":"blue","LAX":"green","SFO":"orange","RIO":"purple"},
                    City2Marker = {"BOS":"o","LIS":"s","LAX":"^","SFO":"D","RIO":"P"}):
    """
        @params:
            - Info_Simulation_DP: dict -> {City:{UCI:{'alpha':alpha,            # NOTE: exponent powerlaw better then expo -> n(t)
                                                    'beta':beta,                # NOTE: exponent best fit powerlaw in t -> to verify transition 
                                                    'Rc':Rc,                    # NOTE: Critical loading rate for the transition, the R for which alpha is defined
                                                    'index_Rc':index_Rc,        # NOTE: index of Rc in the list of Rs
                                                    'Tauc':Tauc,                # NOTE: Tauc, the time at which the transition occurs
                                                    'n(R)':[n(R)],              # NOTE: n(R), the number of vehicles at the transition time
                                                    'Rs':[R],                   # NOTE: Rs, the list of Rs
                                                    'Taus':[Tau],               # NOTE: Taus, the list of Taus
                                                    }
                                                }
                                            }
        @description:
            This plot will show a 3d representation of UCI = x-axis, alpha = y-axis and beta = z-axis
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    keys = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]    
    # Pick the city
    for City in Info_Simulation_DP.keys():
        # Pick the UCIs
        uci_2_alpha_beta = {i:{"alpha":[],"beta":[]} for i in keys}
        for UCI in Info_Simulation_DP[City].keys():
            float_UCI = float(UCI)
            # NOTE: Pick the UCI in the range of 0.1
            # Pick the alpha and beta
            if 'alpha' in Info_Simulation_DP[City][UCI].keys() and 'beta' in Info_Simulation_DP[City][UCI].keys():
                for i in range(len(keys)-1):
                    if float_UCI >= keys[i] and float_UCI < keys[i+1]:
                        uci_2_alpha_beta[keys[i]]["alpha"].append(Info_Simulation_DP[City][UCI]['alpha'])
                        uci_2_alpha_beta[keys[i]]["beta"].append(Info_Simulation_DP[City][UCI]['beta'])
                        break
        print(uci_2_alpha_beta)
        # Compute the average of the alpha beta
        UCIs = []
        alphas = []
        betas = []
        for interva_uci_val in uci_2_alpha_beta.keys():
            if len(uci_2_alpha_beta[interva_uci_val]["alpha"]) > 0:
                UCIs.append(interva_uci_val)
                alphas.append(np.nanmean(uci_2_alpha_beta[interva_uci_val]["alpha"]))
                betas.append(np.nanmean(uci_2_alpha_beta[interva_uci_val]["beta"])) 
    # Plot
    ax.scatter(UCIs,alphas,betas,marker = City2Marker[City],color = City2Color[City],label = City)
    ax.set_xlabel('UCI',fontsize=15)
    ax.set_ylabel(r'$\alpha$',fontsize=15)
    ax.set_zlabel(r'$\beta$',fontsize=15)
    # Add plane of DP
    x_range = np.linspace(min(UCIs), max(UCIs), 10)
    y_range = np.linspace(min(alphas), max(alphas), 10)
    X, Y = np.meshgrid(x_range, y_range)

    for z in [0.276,0.578]:
        Z = np.full_like(X, z)
        ax.plot_surface(X, Y, Z, alpha=0.1, color="grey", rstride=100, cstride=100)
    plt.savefig(os.path.join(PlotDir,f'AlphaBeta_UCI.png'))
    plt.close(fig)