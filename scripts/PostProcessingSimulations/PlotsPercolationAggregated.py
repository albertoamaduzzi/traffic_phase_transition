import matplotlib.pyplot as plt
from PlotsPercolation import equalize_vectors
from PlotsPercolationAggregated_related_functions import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter

# GLOBAL CONSTANTS 
CITY2COLOR = {"BOS":"red",
              "LIS":"blue",
              "LAX":"green",
              "SFO":"orange",
              "RIO":"purple"
              }
CITY2MARKER = {"BOS":"o",
               "LIS":"s",
               "LAX":"^",
               "SFO":"D",
               "RIO":"P"}


def get_UCI_interval_2_colormap(UCIs):
    # Creo una colormap
    cmap = plt.get_cmap('viridis')  # Puoi cambiare con 'plasma', 'inferno', 'magma', ecc.
    # Creo il dizionario UCI_interval -> colore
    UCI_interval_to_color = {}
    n_intervals = len(UCIs) - 1  # Numero di intervalli (non valori)

    for i in range(n_intervals):
        uci_value = UCIs[i]
        # Normalizzo l'indice tra 0 e 1 per la colormap
        normalized_index = i / (n_intervals - 1) if n_intervals > 1 else 0
        color = cmap(normalized_index)
        UCI_interval_to_color[uci_value] = color    
    return UCI_interval_to_color

def apply_safe_savgol_filter(n, savgol_window):
    from scipy.signal import savgol_filter
    current_savgol_window = min(savgol_window, (len(n)))
    if current_savgol_window < 3:
        current_savgol_window = 3
    if current_savgol_window % 2 == 0:  # Make odd if even
        current_savgol_window = current_savgol_window - 1
    
    if current_savgol_window < 3:
        return n
    else:
        return savgol_filter(n, current_savgol_window, 2)

#-------------------- SINGLE CITY PLOTS --------------------#

def single_plot_nt_aggregated_single_city(vec_nt,                   # NOTE: vector of  fraction of people in the road in time
                                          vec_t,                    # NOTE: vector of time -> TimeHourArray
                                          is_alpha,                 # NOTE: is_alpha -> boolean if the alpha needs to be plotted
                                          alpha,                    # NOTE: alpha -> power-law for the phase transition
                                          R,                        # NOTE: R -> value of the R parameter
                                          t_star,                   # NOTE: t_star -> time of best fit of beta 
                                          vec_fit_nt,               # NOTE: vector of fit for fraction of people in the road in time -> TimeHourArray
                                          is_label,                 # NOTE: is_label -> boolean if the label needs to be plotted
                                          city,                     # NOTE: city -> name of the city
                                          ax1,                      # NOTE: ax1 -> axis of the plot
                                          is_savgol = False,                # NOTE: is_savgol -> boolean if the savgol filter needs to be applied
                                          savgol_window = 3,            # NOTE: savgol_window -> window for the savgol filter
                                          City_2_Marker = CITY2MARKER, # NOTE: City_2_Marker -> dictionary of the city and the marker
                                          City_2_Color = CITY2COLOR,  # NOTE: City_2_Color -> dictionary of the city and the color
):
    """
        Plots the figure of the average Unload with corrispective fit with the alpha and the 
    """
    # NOTE: avoid cases where the number of people drops too quickly
    vec_nt, vec_fit_nt, vec_t = equalize_vectors(vec_nt, vec_fit_nt, vec_t)
    if is_savgol:
        vec_nt = apply_safe_savgol_filter(vec_nt, savgol_window)
    # NOTE: is_fit is true just for the case of R = Rc
    if is_alpha:
        # NOTE: is_alpha = condition R = R_c 
        ax1.scatter(vec_t,vec_nt,label = rf'$R_c$ = {R} veh/$\Delta$t',marker = City_2_Marker[city],color = City_2_Color[city],s = 10)
        ax1.plot(vec_t,vec_fit_nt,linestyle = '--',label = '')
        ax1.text(1,0.003,r'$R_c = $'+ str(R),fontsize=15)
        ax1.text(1,0.005,r'$\alpha = $'+ str(round(alpha,2)),fontsize=15)
        ax1.vlines(x = t_star, ymin = 0, ymax = 1, linestyles = '--',color = 'black')
    else:
        if is_label:
            ax1.plot(vec_t,vec_nt,linestyle = '--',label = rf'R = {R} veh/$\Delta$t')
#        else:
#            ax1.plot(vec_t,vec_nt,linestyle = '--',color = "grey",label = '')

    ax1.set_xlabel(r'$(t-t_0)[h]$',fontsize=15)
    ax1.set_ylabel(r'$n(t)[h]$',fontsize=15)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_xlim(10**(0),15)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.xaxis.get_major_formatter().set_scientific(False)
    ax1.xaxis.get_major_formatter().set_useOffset(False)
    return ax1


from numpy import logical_and,abs

def prepend_initial_values_if_needed(x, y, init_time_hour_cut_for_fit, normalized_first_value_wanted):
    """
    Check if the smallest value of x is bigger than init_time_hour_cut_for_fit.
    If so, prepend the initial values to x and y vectors.
    
    Args:
        x (array-like): Input x vector
        y (array-like): Input y vector  
        init_time_hour_cut_for_fit (float): Value to prepend to x if min(x) > this value
        normalized_first_value_wanted (float): Value to prepend to y
        
    Returns:
        tuple: (x_new, y_new) with potentially prepended values
    """
    import numpy as np
    
    # Convert to numpy arrays if they aren't already
    x = np.array(x)
    y = np.array(y)
    
    # Check if the smallest value of x is bigger than init_time_hour_cut_for_fit
    if len(x) > 0 and np.min(x) > init_time_hour_cut_for_fit:
        # Prepend the initial values
        x_new = np.concatenate(([init_time_hour_cut_for_fit], x))
        y_new = np.concatenate(([normalized_first_value_wanted], y))
        print(f"Prepended values: min(x) = {np.min(x)} > {init_time_hour_cut_for_fit}")
    else:
        # No need to prepend, return original arrays
        x_new = x
        y_new = y
        if len(x) > 0:
            print(f"No prepending needed: min(x) = {np.min(x)} <= {init_time_hour_cut_for_fit}")
        else:
            print("Empty array, no prepending performed")
    
    return x_new, y_new

def single_plot_nt_alpha_aggregated_single_city(vec_nt,
                                                vec_t, 
                                                epsilon, 
                                                is_alpha, 
                                                alpha, 
                                                R, 
                                                vec_fit_nt, 
                                                is_label, 
                                                city, 
                                                ax,
                                                init_time_hour_cut_for_fit = 1,
                                                max_time_for_fit = 7,          
                                          is_savgol=False, savgol_window=3, 
                                          City_2_Marker=CITY2MARKER, City_2_Color=CITY2COLOR):
    """
    Plots the figure of the average Unload with corrispective fit with the alpha and the 
    """
    # NOTE: avoid cases where the number of people drops too quickly
    vec_nt, vec_fit_nt, vec_t = equalize_vectors(vec_nt, vec_fit_nt, vec_t)
    if is_savgol:
        vec_nt = apply_safe_savgol_filter(vec_nt, savgol_window)
    upper_limit_time = (max_time_for_fit)
#    upper_limit_time = max_time_for_fit + 1
    # avoid 0s
    mask_bigger_t0 = vec_t > init_time_hour_cut_for_fit 
    mask_smaller_t_fit = vec_t < upper_limit_time
    mask_fit = logical_and(mask_bigger_t0, mask_smaller_t_fit)
    vec_t = vec_t[mask_fit]
    vec_nt = vec_nt[mask_fit]
    vec_fit_nt = vec_fit_nt[mask_fit]    
    # Calculate scaled values
    tminust0epsilon = vec_t*epsilon
#    tminust0epsilon = vec_t#*epsilon
    nttalpha = vec_nt*vec_t**(abs(alpha))
    print("vec_t in ntalpha",vec_t)
#    print("vec_nt in ntalpha",vec_nt)
#    print(f"nntalpha: {nttalpha}")
    max_nttalpha = nttalpha[0]
#    normalized_first_value_wanted = max_nttalpha
    normalized_first_value_wanted = 0.01
    nttalpha = array(nttalpha) / max_nttalpha * normalized_first_value_wanted
    tminust0epsilon, nttalpha = prepend_initial_values_if_needed(tminust0epsilon, nttalpha, init_time_hour_cut_for_fit, normalized_first_value_wanted)
    upper_limit_time = 30#max(10,tminust0epsilon[-2])
    if len(nttalpha) == 0:
        print(f"Warning: No data to plot for R = {R} in city {city}.")
        return ax
    else:        
        # Plot logic - unchanged from your original code
        if is_alpha:
#            ax.plot(tminust0epsilon, nttalpha, label=rf'Rc = {R} veh/$\Delta$t', color='black')
            ax.hlines(y=normalized_first_value_wanted, xmin=init_time_hour_cut_for_fit, xmax=upper_limit_time, color='black',label=rf'$R_c$ = {R} veh/$\Delta$t')
            if is_label:            
                ax.plot(tminust0epsilon, nttalpha, linestyle='--', 
                       label=rf'R = {R} veh/$\Delta$t', color=City_2_Color[city],)
        else:
            if is_label:
                ax.plot(tminust0epsilon, nttalpha, linestyle='--', 
                       label=rf'R = {R} veh/$\Delta$t')
        
        # Only set these the first time or they'll reset with each call
        ax.set_xlabel(r'$(t-t_0)\epsilon$', fontsize=15) #\epsilon
        ax.set_ylabel(r'$n(t)t^{\alpha} [h]$', fontsize=15)
        ax.set_xscale('log')
        ax.set_yscale('log')
#        ax.set_xlim(init_time_hour_cut_for_fit, upper_limit_time)
        ax.set_xlim(init_time_hour_cut_for_fit, upper_limit_time)
        # Format axes display
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.xaxis.get_major_formatter().set_useOffset(False)
        
        ax.legend()
        return ax
# ----- GAMMA TAU PLOTS ----- #
def single_plot_gamma_tau(Taus,
                 Gammas,
                 Gammac,
                 Tauc,
                 IndexRc,
                 City,
                 UCI,
                 color,
                 ax):
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
    from numpy import array
    if Gammac == None or Tauc == None or IndexRc == None:
        Gammac = 0
        Tauc = 0
        gamma_label = r'$\Gamma$'
        tau_label = r'$\tau$'
        is_transition = False
    else:
        gamma_label = r'$\Gamma - \Gamma_0$'
        tau_label = r'$\tau - \tau_0$'
        is_transition = True
    Taus = array(Taus) - Tauc
    Gammas = array(Gammas) - Gammac
    if IndexRc is not None:
        idx = find_last_stable_point(Taus, 0.1)
        coeffs = np.polyfit(Gammas[idx:],Taus[idx:], 1)
        gammas_fit = np.linspace(Gammas[idx], max(Gammas),20)
        taus_fit = coeffs[0] * gammas_fit + coeffs[1]
        ax.plot(gammas_fit, taus_fit, color=color, linestyle='--', label="")
    print("Gammas\n",Gammas)
    print("Taus\n",Taus)
    ax.scatter(Gammas,Taus,color = color,marker = CITY2MARKER[City],label = f"UCI: {UCI}",s = 30)
#    if is_transition:
#        ax.vlines(x = 0,ymin = min(Taus),ymax = max(Taus),linestyles = '--',color = 'black')
    ax.set_xlabel(gamma_label,fontsize=15)
    ax.set_ylabel(tau_label,fontsize=15)
    ax.set_ylim(-2)
    ax.legend()
    return ax

def single_plot_R_tau_alpha(Rs_plot_tau,
                            Taus,
                            alpha,
                            Tauc,
                            Rc,
                            City,
                            ax,
                            color,
                            UCI=None):
    """
    Plots the R vs Tau for each UCI with alpha and Rc
    Args:
        Rs_plot_tau (list): List of R values to plot.
        Taus (list): List of Tau values corresponding to Rs_plot_tau.
        alpha (float): The alpha value to display on the plot.
        Tauc (float): The critical Tau value to display on the plot.
        Rc (float): The critical R value to display on the plot.
        City (str): The name of the city for labeling.
        ax (matplotlib.axes.Axes): The axes on which to plot.
        UCI (str, optional): The UCI value for labeling. Defaults to None.
    """
    from numpy import array, abs
    import numpy as np
    
    # Convert Rs_plot_tau and Taus to numpy arrays for easier manipulation
    Rs_plot_tau2Rc = array(Rs_plot_tau)/Rc
    Taus2Tauc = array(Taus)/Tauc
    
    # Create masks for data before and after R/Rc = 1
    mask_before = Rs_plot_tau2Rc <= 1.0
    mask_after = Rs_plot_tau2Rc > 1.0
    
    # Get data points for each region
    Rs_before = Rs_plot_tau2Rc[mask_before]
    Taus_before = Taus2Tauc[mask_before]
    Rs_after = Rs_plot_tau2Rc[mask_after]
    Taus_after = Taus2Tauc[mask_after]
    
    # Fit and plot line for R/Rc <= 1 (if we have at least 2 points)
    if len(Rs_before) >= 2:
        try:
            coeffs_before = np.polyfit(Rs_before, Taus_before, 1)
            Rs_fit_before = np.linspace(min(Rs_before), max(Rs_before), 20)
            Taus_fit_before = coeffs_before[0] * Rs_fit_before + coeffs_before[1]
            ax.plot(Rs_fit_before, Taus_fit_before, color=color, linestyle='--', label="")
        except np.RankWarning:
            print(f"Warning: Could not fit line for R/Rc <= 1 region for {City}, UCI {UCI}")
        except Exception as e:
            print(f"Error fitting line for R/Rc <= 1 region for {City}, UCI {UCI}: {e}")
    
    # Fit and plot line for R/Rc > 1 (if we have at least 2 points)
    if len(Rs_after) >= 2:
        try:
            coeffs_after = np.polyfit(Rs_after, Taus_after, 1)
            Rs_fit_after = np.linspace(min(Rs_after), max(Rs_after), 20)
            Taus_fit_after = coeffs_after[0] * Rs_fit_after + coeffs_after[1]
            ax.plot(Rs_fit_after, Taus_fit_after, color=color, linestyle='--', label="")
        except np.RankWarning:
            print(f"Warning: Could not fit line for R/Rc > 1 region for {City}, UCI {UCI}")
        except Exception as e:
            print(f"Error fitting line for R/Rc > 1 region for {City}, UCI {UCI}: {e}")
    
    # Plot the original scatter data
    ax.scatter(Rs_plot_tau2Rc, Taus2Tauc, color=color, marker=CITY2MARKER[City], label=f"UCI: {UCI}")
    
    # Add a vertical line at Rc
    ax.vlines(x=1, ymin=min(Taus2Tauc), ymax=max(Taus2Tauc), linestyles='--', color='black')
    
    # Set labels and title
    ax.set_xlabel(r'$R/R_c$', fontsize=15)
    ax.set_ylabel(r'$\tau/\tau_c$', fontsize=15)
    ax.legend()
    return ax

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy import abs
from mpl_toolkits.axes_grid1.inset_locator import mark_inset  # optional for connection lines


def get_nR_R2Rc_epsilons_from_R2NtNtFit(R2NtNtFit, int_t_star,Rc):
    """
    Extracts n(R) values from R2NtNtFit for a given t_star.
    
    Args:
        R2NtNtFit (dict): Dictionary containing n(R) values.
        int_t_star (int): The time index to extract n(R) values.
        
    Returns:
        list: List of n(R) values at the specified t_star.
    """
    n_R = []
    RsOverRc= []
    epsilons = []
    for R in R2NtNtFit.keys():
        if 'n' in R2NtNtFit[R]:
            n_R.append(R2NtNtFit[R]['n'][int_t_star])
            RsOverRc.append(int(R)/int(Rc))
            epsilons.append(abs((int(R)**2 - Rc**2)/Rc**2))
    return n_R, RsOverRc, epsilons


def create_inset_plot(fig, ax_main,position="upper right"):
    """
    Create a properly sized inset axes within the main plot.
    
    Args:
        fig: The figure containing the main axes
        ax_main: The main axes where the inset should be placed
        
    Returns:
        The inset axes object
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # Create inset axes with fixed size - this avoids enormous dimensions
    ax_inset = inset_axes(ax_main, 
                        width="30%",     # Width as percentage of parent axes
                        height="30%",    # Height as percentage of parent axes
                        loc=position,
                        borderpad=1)     # Padding in points
    # Set some default properties for the inset
    ax_inset.tick_params(axis='both', labelsize=8)
    # Important: set scales early to avoid auto-scaling issues
    ax_inset.set_xscale('log')
    ax_inset.set_yscale('log')
    
    return ax_inset
def finalize_inset(ax_main, ax_inset):
    """
    Adds finishing touches to the inset plot
    """
    # Add a label to the inset if needed
    ax_inset.set_ylabel(r"$n(R,t_{\infty})$", fontsize=10)
    
    # Make the inset stand out with a border
    for spine in ax_inset.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(1)
    
    # Optional: Draw connecting lines to show the zoomed region
    # (Alternative to mark_inset which can sometimes cause issues)
    from matplotlib.patches import ConnectionPatch
    
    # Draw two connecting lines between main plot and inset
    xy_main = (1.0, 2.5)  # Point in main axes (data coordinates)
    xy_inset = (0.9, 2.2) # Corresponding point in inset (data coordinates)
    
    con1 = ConnectionPatch(xyA=xy_main, xyB=xy_inset, 
                          coordsA="data", coordsB="data",
                          axesA=ax_main, axesB=ax_inset, 
                          arrowstyle="-", linewidth=0.5, color="gray")
    ax_main.add_artist(con1)
    
    # Ensure the inset is on top of other elements
    ax_inset.set_zorder(5)

def single_plot_nr(Rs_over_Rc,
                   nR,
                   epsilons,
                   City, 
                   ax, 
                   color,
                   beta_ = None,
                   UCI=None):
    """Plot the n(R) vs R/R_c relation with an inset"""
    from numpy import array
    import numpy as np
    
    # Convert to numpy arrays for easier manipulation
    Rs_over_Rc = array(Rs_over_Rc)
    nR = array(nR)
    
    # Main plot code    
    ax.scatter(Rs_over_Rc, nR, color=color)
    
    # Create mask for region Rs_over_Rc >= 1
    mask_above_1 = Rs_over_Rc > 1.0
    
    # Get data points for Rs_over_Rc >= 1
    epsilons_above_1 = array(epsilons)[mask_above_1]
    Rs_above_1 = Rs_over_Rc[mask_above_1]
    nR_above_1 = nR[mask_above_1]
    
    # Fit and plot power law for Rs_over_Rc >= 1 (if we have at least 2 points)
    if len(Rs_above_1) >= 2:
        try:
            # For power law fitting, we use log-log linear fit: log(nR) = log(A) + beta * log(Rs_over_Rc)
            log_Rs = np.log(Rs_above_1)
            log_nR = np.log(nR_above_1)
            
            # Linear fit in log space
            coeffs = np.polyfit(log_Rs, log_nR, 1)
            beta = coeffs[0]  # Power law exponent
            log_A = coeffs[1]  # log of amplitude
            A = np.exp(log_A)  # Amplitude
            
            # Create fit line
#            Rs_fit = np.linspace(min(Rs_above_1), max(Rs_above_1), 50)
            Rs_fit = np.linspace(min(Rs_above_1), max(Rs_above_1), 50)
            
            nR_fit = A * Rs_fit**beta
            
            # Plot the power law fit
            ax.plot(Rs_fit, nR_fit, color=color, linestyle='--', linewidth=2, 
                   label='')
            
            print(f"Power law fit for {City}, UCI {UCI}: n(R) ∝ (R/Rc)^{beta:.3f}")
            
        except Exception as e:
            print(f"Error fitting power law for Rs_over_Rc >= 1 region for {City}, UCI {UCI}: {e}")
    else:
        print(f"Not enough points (only {len(Rs_above_1)}) for power law fit in Rs_over_Rc >= 1 region for {City}, UCI {UCI}")
    if beta_ is not None:
        ax.text(1.5, max(nR)/2, r"$\beta$ = " + f"{round(beta_,3)}", fontsize=12)
    # Vertical line at Rc
    ax.vlines(x=1, ymin=0, ymax=max(nR), linestyles='--', color='black')
    ax.set_xlabel(r'$R/R_c$', fontsize=15)
    ax.set_ylabel(r'$n(R,t_{\infty})$', fontsize=15)
    ax.set_xlim(0, max(Rs_over_Rc) + 0.1)
    ax.set_ylim(0, max(nR))
    
    # Add legend if we have a fit
#    if len(Rs_above_1) >= 2:
#        ax.legend()
    
    return ax



def single_plot_inset_nR(ax_inset,
                   nRfit,
                   epsilon_fit,
                   nR_region_fit,
                   City,  
                   color,
                         ):
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
    # Plot the same data in the inset, but limit axes to zoom region
    ax_inset.scatter(epsilon_fit,nR_region_fit, color=color, marker=CITY2MARKER[City], s=10)
    ax_inset.plot(epsilon_fit,nRfit, color=color, linestyle='--')
    # Set inset limits to zoom region
    ax_inset.set_xscale('log')
    ax_inset.set_yscale('log')
    ax_inset.set_xlabel(r"$\epsilon$", fontsize=15)
    ax_inset.set_xlim(0.1, 5)
    return ax_inset
import numpy as np
def single_plot_inset_nR(ax_inset, nRfit, epsilon_fit, nR_region_fit, City, color):
    """
    Plots the n(R) vs epsilon data in the inset axes.
    
    Args:
        ax_inset: The inset axes object
        nRfit: The fitted n(R) values
        epsilon_fit: The epsilon values used for fitting
        nR_region_fit: The n(R) data points in the region used for fitting
        City: The city name (used for marker style)
        color: The color to use for this data
        
    Returns:
        The inset axes object
    """
    # Convert inputs to numpy arrays if they aren't already
    epsilon_fit = np.array(epsilon_fit)
    nR_region_fit = np.array(nR_region_fit)
    nRfit = np.array(nRfit)
    
    # Filter out any NaN values or zeros that could cause log scale issues
    valid_mask = (epsilon_fit > 0) & (nR_region_fit > 0) & ~np.isnan(epsilon_fit) & ~np.isnan(nR_region_fit)
    
    # Plot the data and fit using the mask
    if sum(valid_mask) > 0:  # Only plot if we have valid data points
        ax_inset.scatter(epsilon_fit[valid_mask], nR_region_fit[valid_mask], 
                        color=color, marker=CITY2MARKER.get(City, 'o'), s=10)
        
        # Plot the fit line
        valid_fit_mask = (epsilon_fit > 0) & ~np.isnan(nRfit)
        if sum(valid_fit_mask) > 0:
            ax_inset.plot(epsilon_fit[valid_fit_mask], nRfit[valid_fit_mask], 
                        color=color, linestyle='--', linewidth=1)
    
    return ax_inset

def single_plot_beta_fit(Time2FitBeta,
                         t_star,
                         int_t_star,
                         t_vect,
                         ax):
    """
        Description:

    """   
    for t0 in Time2FitBeta.keys():
        if int_t_star == t0:
            ax.scatter(Time2FitBeta[t0]["epsilon"],Time2FitBeta[t0]["n(R,epsilon)"],marker ='+',label = f"t = {round(t_star,3)}")
            ax.plot(Time2FitBeta[t0]["epsilon"],Time2FitBeta[t0]["n(R,epsilon)_fit"], label = "")
            # ...existing code...
            ax.text(min(Time2FitBeta[t0]["epsilon"])*1.1, min(Time2FitBeta[t0]["n(R,epsilon)_fit"])*1.1, r"$\beta$ = " + f"{Time2FitBeta[t0]['Beta']}")
        else:
            is_below = t0 < int_t_star and t0 > int_t_star - 3
            is_above = t0 > int_t_star and t0 < int_t_star + 3
            is_label = is_below or is_above
            if is_label:
                ax.plot(Time2FitBeta[t0]["epsilon"],Time2FitBeta[t0]["n(R,epsilon)"],label = f"t = {round(t_vect[t0],3)}",linestyle = '--')
#    ax.plot(BestA,BestBeta,'*',label = f"Beta: {round(BestBeta,3)} A: {round(BestA,2)}")
    ax.set_xlabel(r"$\epsilon = \frac{R^{2}-R^{2}_c}{R^{2}_c}$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(min(Time2FitBeta[int_t_star]["epsilon"]),max(Time2FitBeta[int_t_star]["epsilon"]))
#    ax.set_ylim(10^(-4),3*10^(-1))
    ax.set_ylabel(r"$\beta$")
    ax.legend()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    return ax


def single_plot_error_associated_to_powerlaw_beta(Betas,
                                                  Errors,
                                                  time_,
                                                  t_star,
                                                  City,
                                                  UCI,
                                                  ax):
    """
        Description:
            - Plot the Error of the Power Law Fit
    """
    ax.scatter(time_,Errors,marker = "o",label = f"error")
    ax.scatter(time_,Betas,marker = "*",label = r"$\beta$")
    ax.vlines(x = t_star,ymin = 0,ymax = 1,color = "red",linestyles = "dashed")
    ax.legend()
    ax2 = ax.twinx()
    ax2.spines['left'].set_position(('outward', 60))
    ax2.set_xlabel("t (h)")
    ax2.set_ylabel("error")
    ax2.set_title(f"{City}, UCI: {UCI}")
    return ax


from numpy import linspace, meshgrid, full_like,array
from mpl_toolkits.mplot3d import Axes3D

def get_UCI_interval_from_UCI(UCI,UCI_intervals):
    """
        Description:
            - Get the UCI interval from the UCI
        Args:
            - UCI: float -> UCI value
            - UCI_intervals: list -> List of UCI intervals
        Returns:
            - UCI_interval: tuple -> (UCI_min,UCI_max)
    """
    for i in range(len(UCI_intervals)-1):
        if UCI >= UCI_intervals[i] and UCI < UCI_intervals[i+1]:
            return i
    return len(UCI_intervals)


def single_plot_alpha_beta_UCI(UCI_2_CriticalParams,
                               City,
                               ax,
                               colors,
                               UCI_intervals,
                               City2Marker=CITY2MARKER):
    """
        Description:
            - Plot the Alpha and Beta for the UCI
        Args:
            - UCI_2_CriticalParams: dict -> {UCI:{'alpha':alpha,'beta':beta}}
            - City: str -> City Name
            - ax: axis to plot
    """
    UCIs = [UCI for UCI in UCI_2_CriticalParams.keys() if "beta" in UCI_2_CriticalParams[UCI].keys()]
    alphas = [UCI_2_CriticalParams[UCI]['Alpha'] for UCI in UCIs if "beta" in UCI_2_CriticalParams[UCI].keys()]
    betas = [UCI_2_CriticalParams[UCI]['beta'] for UCI in UCIs if "beta" in UCI_2_CriticalParams[UCI].keys()]
    for i in range(len(betas)):
        UCI = UCIs[i]
        idx_color = get_UCI_interval_from_UCI(UCIs[i],UCI_intervals)
        color =  colors[idx_color]
        if City == "SFO":
            condition = UCI > 0.4 and UCI != 0.564 and UCI < 0.9
        elif City == "RIO":
            condition = True
        if condition:
            ax.scatter(alphas[i],betas[i],marker = City2Marker[City],color = color, label = UCI)
    ax.set_xlabel(r'$\alpha$',fontsize=15)
    ax.set_ylabel(r'$\beta$',fontsize=15)
    ax.hlines(y=0.276, xmin=min(alphas), xmax=max(alphas), color='black', linestyle='--')
    ax.hlines(y=0.578, xmin=min(alphas), xmax=max(alphas), color='black', linestyle='--')    
    ax.annotate("1DP", xy=(0.1, 0.276), xytext=(0.1, 0.3),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12, color='black')
    ax.annotate("2DP", xy=(0.1, 0.578), xytext=(0.1, 0.6),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12, color='black')
    ax.set_ylim(0)
    ax.legend(loc = "upper right")
    return ax

# ---------- ALL CITIES PLOTS ---------- #    
