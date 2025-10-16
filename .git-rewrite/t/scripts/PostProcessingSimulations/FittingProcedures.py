try:
    import pymc3 as pm
    FoundPyMC3 = True
except:
    FoundPyMC3 = False
from scipy.special import gamma
from scipy.optimize import curve_fit,minimize
from scipy import stats
from scipy.stats import powerlaw 
import numpy as np
import sys
from EfficiencyAnalysis import *
sys.path.append('~/berkeley/traffic_phase_transition/scripts/GeometrySphere')

# FUCNTIONS FOR FITTING
def powerlaw(x, amp, index):
    return amp * (np.array(x)**index)
    

def exponential(x, amp, index):
    return amp * np.exp(index*np.array(x))

def linear(x, amp,q):
    return amp * np.array(x) + q

def multilinear4variables(x, log_k,beta,gamma,d0minus1):
    '''
        N is the couples of Origin and Destination
        Fitting like Vespignani:
            Vectors\Data:
                1) log Ni -> 1 Dimensional vector of length N being the mass of the grid i.
                2) log Nj -> 1 Dimensional vector of length N being the mass of the grid j.
                3) log dij -> 1 Dimensional vector of length N being the distance between centroids of grid i and j.
            Scalar\Parameters:
                1) log_d: -> k in Ramasco Paper
                2) a: exponent mass i
                3) b: exponent mass j
                4) c: exp(-1/d0)                        
    '''
    return log_k + beta * np.log(x[0]) + gamma * np.log(x[1]) + d0minus1 * x[2] 
def objective_function_multilinear4variables(params,x,y_measured):
    if len(params)!=4:
        raise ValueError('The parameters must be an array of length 4')
    if len(x)!=3:
        raise ValueError('The x must be an array of shape (3,N)')
    if len(x[0])!=len(y_measured):
        raise ValueError('The log of the fluxes must be of the same length as the masses')
    y_guessed = multilinear4variables(x, params[0],params[1],params[2],params[3])
    return quadratic_loss_function(y_guessed,y_measured)


def lognormal(x, mean, sigma):
    return (np.exp(-(np.log(x) - mean)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))

def gamma_(x, shape, scale):
    return ((x**(shape - 1)) * np.exp(-x / scale)) / (scale**shape * gamma(shape))

def weibull(x, shape, scale):
    return (shape / scale) * (x / scale)**(shape - 1) * np.exp(-(x / scale)**shape)
# LOSS FUNCTIONS
def quadratic_loss_function(y_predict, y_measured):
    return np.sum((y_predict-y_measured)**2)

def objective_function_powerlaw(params,x,y_measured):
    if len(x)!=len(y_measured):
        raise ValueError('Power Law Loss: x and y measured do not have the same shape')
    if len(params)!=2:
        raise ValueError('Power Law Loss: the parameters are not 2 but {}'.format(len(params)))
    return quadratic_loss_function(powerlaw(x, params[0], params[1]), y_measured)

def objective_function_exponential(params,x,y_measured):
    if len(x)!=len(y_measured):
        raise ValueError('Expo Loss: x and y measured do not have the same shape')
    if len(params)!=2:
        raise ValueError('Expo Loss: the parameters are not 2 but {}'.format(len(params)))
    return quadratic_loss_function(exponential(x, params[0], params[1]), y_measured)

def objective_function_linear(params,x,y_measured):
    if len(x)!=len(y_measured):
        raise ValueError('Linear Loss: x and y measured do not have the same shape')
    if len(params)!=2:
        raise ValueError('Linear Law Loss: the parameters are not 2 but {}'.format(len(params)))
    return quadratic_loss_function(linear(x, params[0], params[1]), y_measured)

def objective_function_multilinear4variables(params,x,y_measured):
    if len(params)!=4:
        raise ValueError('The parameters must be an array of length 4')
    if len(x)!=3:
        raise ValueError('The x must be an array of shape (3,N)')
    if len(x[0])!=len(y_measured):
        raise ValueError('The log of the fluxes must be of the same length as the masses')
    y_guessed = multilinear4variables(x, params[0],params[1],params[2],params[3])
    return quadratic_loss_function(y_guessed,y_measured)
## DICTIONARY FOR LOSS FUNCTIONS
Name2Function = {'powerlaw':powerlaw,'exponential':exponential,'linear':linear,'vespignani':multilinear4variables}
Name2LossFunction = {'powerlaw':objective_function_powerlaw,'exponential':objective_function_exponential,'linear':objective_function_linear,'vespignani':objective_function_multilinear4variables}
    
@timing_decorator
def Fitting(x,y_measured,label = 'powerlaw',initial_guess = (6000,0.3),bounds = (np.array([-50,0,0,-2]),np.array([50,2,2,0])),maxfev = 50000):
    '''
        Input:
            label: 'powerlaw' or 'exponential' or 'linear'
            x: (np.array 1D) x-axis
            y_measured: (np.array 1D) y-axis
            initial_guess: (tuple 2D) parameters for fitting
        USAGE:

    '''

#    dx = np.diff(x)
    # NORMALIZE THE INTEGRAL
#    if np.sum(y_measured[1:]*dx)!=1:
    
    Z = np.sum(y_measured)
    y_measured = y_measured/Z
    print('Fitting {}'.format(label))
    result_powerlaw = minimize(Name2LossFunction[label], initial_guess, args = (x, y_measured))#,maxfev = maxfev
    optimal_params_pl = result_powerlaw.x
    fit = curve_fit(Name2Function[label], xdata = x, ydata = y_measured,p0 = list(optimal_params_pl),maxfev = maxfev) # bounds = bounds
    print(fit)
    print('{} fit: '.format(label),fit[0][0],' ',fit[0][1])
    print('Convergence fit {}: '.format(label),result_powerlaw.success)
    print('Optimal parameters: ',result_powerlaw.x)
    print('Message: ',result_powerlaw.message)
    return fit,result_powerlaw.success




## NEW FIT FOR GRAVITY MODEL
def GravityModel(Mi,Mj,Dij,k,alpha,gamma,d0minus1):
    return k*Mi**alpha*Mj**gamma* np.exp(Dij*d0minus1)
def FluxesOverProductMasses(GravityFluxes,Mi,Mj,Alpha,Beta):
    return GravityFluxes/(Mi**Alpha*Mj**Beta)


def FittingGravity(x,y_measured,initial_guess = (0,1,1,-0.001),bounds = (np.array([-50,0,0,-2]),np.array([50,2,2,0])),maxfev = 50000):
    '''
        Input:
            label: 'powerlaw' or 'exponential' or 'linear'
            x: (np.array 1D) x-axis
            y_measured: (np.array 1D) y-axis
            initial_guess: (tuple 2D) parameters for fitting
        USAGE:

    '''
    from scipy.optimize import minimize
    from scipy.optimize import curve_fit
    print("Initial guess: ",initial_guess)
    result_powerlaw = minimize(objective_function_multilinear4variables, initial_guess, args = (x, y_measured))#,maxfev = maxfev
    optimal_params_pl = result_powerlaw.x
    optimal_params_pl = initial_guess
    fit = curve_fit(multilinear4variables, xdata = x, ydata = y_measured,p0 = list(optimal_params_pl),maxfev = maxfev) # , bounds = bounds
    print(fit)
    print('Gravity fit: ',fit[0][0],' ',fit[0][1],' ',fit[0][2],' ',fit[0][3])
    return fit,result_powerlaw.success

def EstimateLogk(Tij_dist_fit_gravity):
    import polars as pl
    def Divide(z,x,y):
        return z/(x*y)
    Tij_D0_k = Tij_dist_fit_gravity.filter(pl.col('distance') == 0)
    Tij_D0_k = Tij_dist_fit_gravity.with_columns(pl.struct(["number_people","population_origin","population_destination"]).map_batches(lambda x: Divide(x.struct.field('number_people'),x.struct.field("population_origin"),x.struct.field("population_origin"))).alias("k"))
    k_av = np.mean(Tij_D0_k['k'].to_numpy())
    print("Average log_k",np.log(k_av))
    return np.log(k_av)


def EstimateD0minus1(Tij_dist_fit_gravity):
    import polars as pl
    Tij_D0_diff = Tij_dist_fit_gravity.filter(pl.col('distance') != 0)
    d0minus1_av = 1/np.mean(Tij_D0_diff['distance'].to_numpy())
    print("Average d0minus1: ",d0minus1_av)
    return d0minus1_av



from scipy.interpolate import interp1d

def smooth_function_with_zeros(y):
    """
    Smooth a noisy function by replacing zeros before the last non-zero value
    and maintaining zeros after the last zero.
    
    @params y: np.array - The input array with noisy values and zeros
    @return: np.array - The smoothed array
    """
    # Find the indices of non-zero values
    non_zero_indices = np.nonzero(y)[0]
    
    if len(non_zero_indices) == 0:
        # If there are no non-zero values, return the original array
        return y
    
    # Find the index of the last non-zero value
    last_non_zero_index = non_zero_indices[-1]
    
    # Replace zeros before the last non-zero value using interpolation
    x = np.arange(len(y))
    y_interp = y.copy()
    interp_func = interp1d(x[non_zero_indices], y[non_zero_indices], kind='linear', fill_value="extrapolate")
    y_interp[:last_non_zero_index + 1] = interp_func(x[:last_non_zero_index + 1])
    
    # Maintain zeros after the last zero
    y_interp[last_non_zero_index + 1:] = 0
    
    return y_interp

def zero_values_after_first_zero(array):
    """
    Zeros all values that appear after the first zero in the array.
    
    Parameters:
    ----------
    array : array_like
        Input array (numpy array or list)
        
    Returns:
    -------
    numpy.ndarray
        Modified array with all values after first zero set to zero
        
    Example:
    --------
    >>> zero_values_after_first_zero([1.2, 3.5, 0, 4.7, 0, 2.1])
    array([1.2, 3.5, 0., 0., 0., 0.])
    """
    # Convert input to numpy array if it's not already
    array = np.asarray(array)
    
    # Create a copy to avoid modifying the original
    result = array.copy()
    
    # Find index of first zero
    zero_indices = np.where(array == 0)[0]
    
    # If there's at least one zero in the array
    if len(zero_indices) > 0:
        first_zero_index = zero_indices[0]
        # Set all values after (and including) the first zero to zero
        result[first_zero_index:] = 0
    
    return result,first_zero_index

import matplotlib.pyplot as plt
import polars as pl
from scipy.ndimage import gaussian_filter1d


def plot_check_fit_interval(t,
                            nt,
                            n_exp,
                            n_pl,
                            t_cut,
                            n_cut,
                            init_time_hour_cut_for_fit,
                            minimum_time_with_people,
                            time_first_0,
                            best_fit,
                            t_last = 17):
    plt.plot(t,nt)
    plt.plot(t,n_exp, label='exp', color='orange', linestyle='--')
    plt.plot(t,n_pl, label='pl', color='green', linestyle='--')
    plt.scatter(t_cut,n_cut, label='cut', color='red', s=20,marker='*')
    plt.xlabel("Time (hours)")
    plt.ylabel("Fraction of People")
    plt.axvspan(init_time_hour_cut_for_fit,minimum_time_with_people, alpha=0.2, color='green')
    plt.axvspan(time_first_0, t_last, alpha=0.2, color='red', label='highlighted region')
    plt.axvline(x=time_first_0, color='r', linestyle='--')
    plt.axvline(x=init_time_hour_cut_for_fit, color='g', linestyle='--')
    plt.axvline(x=minimum_time_with_people, color='g', linestyle='--')
    plt.text(13, 0.05, best_fit, color='green', fontsize=10, ha='right')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(left=1, right=t_last)
    plt.tight_layout()
    if best_fit == 'exponential':
        plt.savefig("/data_2/alberto/SFO/Output/fit_exemplification_0.png")
    if best_fit == 'powerlaw':
        plt.savefig("/data_2/alberto/SFO/Output/fit_exemplification_1.png")
    plt.close()



def filter_t_nt_for_fit(log_x,
                        log_y,
                        x_fit,
                        y,
                        init_time_hour_cut_for_fit,
                        max_time_for_fit):
    """
        Cut the curve in the interval we consider [init_time_hour_cut_for_fit,max_time_for_fit]
        Cut non-fittable values (inf, -inf, NaN)
        NOTE: The curve we use for fit is just in the unloaded part

    """
    # Interval 2 Fit
    mask_beginning_fit = x_fit > init_time_hour_cut_for_fit
    cut_init_x_fit = x_fit[mask_beginning_fit]
    cut_init_y = y[mask_beginning_fit]
    cut_init_log_x = log_x[mask_beginning_fit]
    cut_init_log_y = log_y[mask_beginning_fit]
    # NOTE: cut the infs
    mask_is_inf = ~np.isinf(cut_init_log_y)
    cut_init_x_fit = cut_init_x_fit[mask_is_inf]
    cut_init_y = cut_init_y[mask_is_inf]
    cut_init_log_x = cut_init_log_x[mask_is_inf]
    cut_init_log_y = cut_init_log_y[mask_is_inf]
    # Remove negative infinity values
    mask_is_neg_inf = ~np.isneginf(cut_init_log_y)
    cut_init_x_fit = cut_init_x_fit[mask_is_neg_inf]
    cut_init_y = cut_init_y[mask_is_neg_inf]
    cut_init_log_x = cut_init_log_x[mask_is_neg_inf]
    cut_init_log_y = cut_init_log_y[mask_is_neg_inf]        
    # Remove NaN values
    mask_is_nan = ~np.isnan(cut_init_log_y)
    cut_init_x_fit = cut_init_x_fit[mask_is_nan]
    cut_init_y = cut_init_y[mask_is_nan]
    cut_init_log_x = cut_init_log_x[mask_is_nan]
    cut_init_log_y = cut_init_log_y[mask_is_nan]
    # Mask from above in time
    mask_upper_bound_in_time = cut_init_x_fit < max_time_for_fit
    cut_init_x_fit_masked_upper = cut_init_x_fit[mask_upper_bound_in_time]
    cut_init_y_masked_upper = cut_init_y[mask_upper_bound_in_time]
    cut_init_log_x_masked_upper = cut_init_log_x[mask_upper_bound_in_time]
    cut_init_log_y_masked_upper = cut_init_log_y[mask_upper_bound_in_time]
    return cut_init_log_x_masked_upper, cut_init_log_y_masked_upper, cut_init_x_fit_masked_upper, cut_init_y_masked_upper


@timing_decorator
def ComparePlExpoDf(DfUnload,
                    init_time_hour_cut_for_fit = 1,
                    minimum_time_with_people = 7,
                    max_time_for_fit = 7,          # NOTE: is the time upper bound that we consider for the fit.
                    maxfev = 10000):
    """
        @params:
            - minimum_time_with_people: float -> Minimum time with people in the network (Before this time I add some error to the powerlaw.)
        @Describe:
            - Compare the powerlaw and the exponential fit
        NOTE:
            1. Consider Time in terms of hours after the unloading period starts.
            2. Consider the fraction of people in the network
            3. Consider the interval for which the number of people is not 0, 
               otherwhise the log is not defined in the y-axis that is always logged for a linear fit.
        @Input:
            - DfUnload: polars.DataFrame
            Columns:
                - FractionPeople: np.array (Fraction of people in the network)
                - Time_hours: np.array (Time in hours from t_start_hour)
                - Time_hours_shifted: np.array (Time in hours after the unloading period starts) 
                - IsPeople: np.array
                - IsUnload: np.array
                - NumberPeople: np.array (Number of people, not fraction)
                - NumberPeopleFit: np.array (Number of people fitted)
                - Time: np.array (Time in seconds)
                - y_fit_exp: np.array (Fitted exponential)
                - y_fit_pl: np.array (Fitted powerlaw)
                - log_y: np.array (log of the fraction of people)
                - log_x: np.array (log of the time in hours)
                - DiffNumberPeople: np.array (Difference in the number of people)
        @Output:
            - DfUnload: polars.DataFrame
        @Describe:
            - Drops the 0s after the first 0 in the drop is found. Avoid to consider queues on few nodes.
            - Convert the time and number in log for log-log fit (pl) and x-log fit (exp).
            - Compute the fit in the time interval [init_time_hour_cut_for_fit,max_time_for_fit]
            NOTE: This last step is motivated by two facts: 
                1) the initial drop in time is too fast
                2) The best pl seems to be around 7 hours. After that there is noise due to queues in few nodes.
    """
    # Apply the Savitzky-Golay filter
#    y_load = savgol_filter(y_load, 7, 3)
    # Smoothing
    # Declare the input fit
    fraction_people = DfUnload["FractionPeople"].to_numpy()
    time_ = DfUnload["Time_hours_shifted"].to_numpy()
    x_without_load = DfUnload.filter(pl.col("IsUnload") == True)["Time_hours_shifted"].to_numpy()
#    smoothed_fraction_people = fraction_people
    # NOTE: Try to eliminate the noise to avoid to get the transtion too early
    smoothed_fraction_people,first_zero_index = zero_values_after_first_zero(fraction_people)
    DfUnload = DfUnload.with_columns(pl.Series(name="FractionPeople", values=smoothed_fraction_people))    

#    smoothed_fraction_people = smooth_function_with_zeros(fraction_people)
#    smoothed_fraction_people = gaussian_filter1d(smoothed_fraction_people, 2)
    DfUnload = DfUnload.with_columns(pl.Series(name="FractionPeople", values=smoothed_fraction_people))    
    # 
    DfUnload = DfUnload.with_columns(pl.col("FractionPeople").log().alias("log_y"),
                                     pl.col("Time_hours_shifted").log().alias("log_x"))
    
    y_load = DfUnload.filter(pl.col("IsUnload") == False)["FractionPeople"].to_numpy()

    DfUnload4Fit = DfUnload.filter(pl.col("IsPeople") == True,
                                   pl.col("IsUnload") == True)
    log_x = DfUnload4Fit["log_x"].to_numpy()
    log_y = DfUnload4Fit["log_y"].to_numpy()
    y = DfUnload4Fit["FractionPeople"].to_numpy()
    x_fit = DfUnload4Fit["Time_hours_shifted"].to_numpy()

    # NOTE: Control that we have enough time slots to fit the curve
    if len(DfUnload4Fit) < 4:
        is_fittable = False
    else:
        is_fittable = True
    if is_fittable:
        # NOTE: filter infs,nans and -infs, constrain [init_time_hour_cut_for_fit,max_time_for_fit]
        cut_init_log_x_masked_upper, cut_init_log_y_masked_upper, cut_init_x_fit_masked_upper, cut_init_y_masked_upper = filter_t_nt_for_fit(log_x,
                                                                                                                                            log_y,
                                                                                                                                            x_fit,
                                                                                                                                            y,
                                                                                                                                            init_time_hour_cut_for_fit,
                                                                                                                                            max_time_for_fit)
        # NOTE: Task 0: Compute fit: (cut_init_log_x_masked_upper, cut_init_log_y_masked_upper) -> pl
        #                            (cut_init_x_fit_masked_upper, cut_init_y_masked_upper) -> exp
        
        # NOTE: Check the cases that somehow have still 0s in y and so -inf in log
        if len(cut_init_log_x_masked_upper) < 4:
            is_fittable = False
        else:
            is_fittable = True
        if is_fittable:
            # NOTE: Compute the fit on the cutted section of the interval 
            fit_pl = curve_fit(linear, xdata = cut_init_log_x_masked_upper,
                            ydata = cut_init_log_y_masked_upper,
                            maxfev = maxfev)
            fit_exp = curve_fit(linear,
                                xdata = cut_init_x_fit_masked_upper,
                                ydata = cut_init_log_y_masked_upper,
                                maxfev = maxfev)
            # NOTE: A = exp(log(A)) -> log(A) = q   ---- alpha = index
            A_pl,alpha_pl = np.exp(fit_pl[0][1]), fit_pl[0][0]
            A0,alpha_exp = np.exp(fit_exp[0][1]), fit_exp[0][0]
            # NOTE: Compute the fitted curve needed to plot and update the DfUnload columns
            y_fit_exp = exponential(x_without_load,A0,alpha_exp)
            y_fit_pl = powerlaw(x_without_load,A_pl,alpha_pl)
            y_fit_exp = np.concatenate([y_load,y_fit_exp],axis = 0)
            y_fit_pl = np.concatenate([y_load,y_fit_pl],axis = 0)
            time_first_0 = DfUnload["Time_hours_shifted"].to_numpy()[first_zero_index]            
            # NOTE: Update DfUnload -> for return
            DfUnload = DfUnload.with_columns(pl.Series(name = "y_fit_exp",values = y_fit_exp),
                                            pl.Series(name = "y_fit_pl",values = y_fit_pl))
            # NOTE: Compute Error Exp and Error PL
            y_fit_exp_error = exponential(cut_init_x_fit_masked_upper,A0,alpha_exp)
            ErrorExp = np.sqrt(np.sum((y_fit_exp_error-cut_init_y_masked_upper)**2)/len(y))
            if len(DfUnload.filter(pl.col("IsPeople") == True,
                                pl.col("Time_hours_shifted") > minimum_time_with_people))!=0:        
                y_fit_pl_error = powerlaw(cut_init_x_fit_masked_upper,A_pl,alpha_pl)
                ErrorPL = np.sqrt(np.sum((y_fit_pl_error-cut_init_y_masked_upper)**2)/len(y))
                if ErrorPL < ErrorExp:
                    best_fit = 'powerlaw'
                else:
                    best_fit = 'exponential'
#                plot_check_fit_interval(time_,
#                                        fraction_people,
#                                        y_fit_exp,
#                                        y_fit_pl,
#                                        cut_init_x_fit_masked_upper, 
#                                        cut_init_y_masked_upper,                                        
#                                        init_time_hour_cut_for_fit,
#                                        max_time_for_fit,
#                                        time_first_0,
#                                        best_fit,
#                                        t_last = 17)

            else:
                ErrorPL = 1000000
            return DfUnload,ErrorExp,A0,alpha_exp,ErrorPL,A_pl,alpha_pl
        else:
            x_without_load = DfUnload.filter(pl.col("IsUnload") == True)["Time_hours_shifted"].to_numpy()        
            y_fit_exp = exponential(x_without_load,1,-100)
            y_fit_exp = np.concatenate([y_load,y_fit_exp],axis = 0)
            y_fit_pl = powerlaw(x_without_load,1,2)
            y_fit_pl = np.concatenate([y_load,y_fit_pl],axis = 0)

            DfUnload = DfUnload.with_columns(pl.Series(name = "y_fit_exp",values = y_fit_exp),
                                            pl.Series(name = "y_fit_pl",values = y_fit_pl))

            # NOTE: Handling the case where the roads are emptied faster then 1 hour. -> absolutely free flow
            return DfUnload,0,1,10,10000,0,0        
    else:
        x_without_load = DfUnload.filter(pl.col("IsUnload") == True)["Time_hours_shifted"].to_numpy()        
        y_fit_exp = exponential(x_without_load,1,-100)
        y_fit_exp = np.concatenate([y_load,y_fit_exp],axis = 0)
        y_fit_pl = powerlaw(x_without_load,1,2)
        y_fit_pl = np.concatenate([y_load,y_fit_pl],axis = 0)

        DfUnload = DfUnload.with_columns(pl.Series(name = "y_fit_exp",values = y_fit_exp),
                                        pl.Series(name = "y_fit_pl",values = y_fit_pl))

        # NOTE: Handling the case where the roads are emptied faster then 1 hour. -> absolutely free flow
        return DfUnload,0,1,10,10000,0,0
@timing_decorator
def ComparePlExpo(x,y,initial_guess_powerlaw = (1,-1), initial_guess_expo = (1,-1),maxfev = 10000):
    '''
        Input:
            x: (np.array 1D) x-axis
            y: (np.array 1D) y-axis
            initial_guess: (tuple 2D) parameters for fitting
        USAGE:
            ComparePlExpo(x,y,(6000,0.3))
        RETURN:
            - The error of the best fit and the parameters of the best fit
            - Parameters Fitted:
            - Boolean Value indicating if the Power Law is better than the Exponential
            - y_fit: The fitted y values
    '''
    # Fit on the log-log scale POWERLAW
    Z = np.sum(y)
    y = y/Z
    x0 = np.array([x[x_] for x_ in range(len(x)) if x[x_]>0 and y[x_]>0])
    y0 = np.array([y[x_] for x_ in range(len(x)) if x[x_]>0 and y[x_]>0])
    # Fit on the log-log scale POWERLAW
    logx = np.log(x0)
    logy = np.log(y0)
    # Adjust The Initial Guess # alpha*x + log(A) 
    log_initial_guess_pl = (initial_guess_powerlaw[1],np.log(initial_guess_powerlaw[0]))
    fit = curve_fit(linear, xdata = logx, ydata = logy,maxfev = maxfev)
    # NOTE: A = exp(log(A)) -> log(A) = q   ---- alpha = index
    A_pl = np.exp(fit[0][1])
    alpha_pl = fit[0][0]
    ## EXPO 
    log_initial_guess_expo = (initial_guess_expo[1],np.log(initial_guess_expo[0]))
    # NOTE: Just y changes
#    result_expo = minimize(objective_function_linear, log_initial_guess_expo, args = (x0, logy))
#    optimal_params_expo = result_expo.x
#    fit = curve_fit(exponential, xdata = x0, ydata = logy,p0 = list(log_initial_guess_expo),bounds = (np.array([-4,-np.inf]),np.array([0,np.inf])),maxfev = maxfev)
    fit = curve_fit(linear, xdata = x, ydata = logy,maxfev = maxfev)
    A0 = np.exp(fit[0][1])
    alpha_exp = fit[0][0]
    y_fit_exp = exponential(x0,A0,alpha_exp)
    y_fit_pl = powerlaw(x0,A_pl,alpha_pl)
    if len(y0)!=0:
        # Compare the two fits NOTE: Scale for the number of samples since the 
        ErrorExp = np.sqrt(np.sum((exponential(x0,A0,alpha_exp)-y0)**2)/len(y0))
        ErrorPL = np.sqrt(np.sum((powerlaw(x0,A_pl,alpha_pl)-y0)**2)/len(y0))
        return ErrorExp,A0,alpha_exp,Z*y_fit_exp,ErrorPL,A_pl,alpha_pl,Z*y_fit_pl
    else:
        raise ValueError(f'ComparePlExpo: x {x} and y {y} are empty')

def FittingPowerlaw(x,y,initial_guess = (1,0.5),maxfev = 10000):
#    Z = np.sum(y)
#    y = y/Z
    x0 = np.array([x[x_] for x_ in range(len(x)) if x[x_]>0 and y[x_]>0])
    y0 = np.array([y[x_] for x_ in range(len(x)) if x[x_]>0 and y[x_]>0])
    # Fit on the log-log scale POWERLAW
    if len(x0) >= 4 and len(y0) == len(x0):
        logx = np.log(x0)
        logy = np.log(y0)
        # Adjust The Initial Guess # alpha*x + log(A) 
    #    result_powerlaw = minimize(objective_function_linear, log_initial_guess_pl, args = (logx, logy))
    #    optimal_params_pl = result_powerlaw.x
        fit = curve_fit(linear, xdata = logx, ydata = logy,maxfev = maxfev)
        # NOTE: A = exp(log(A)) -> log(A) = q   ---- alpha = index
        A_pl = np.exp(fit[0][1])
        alpha_pl = fit[0][0]
        y_fit_pl = powerlaw(x,A_pl,alpha_pl)
        y_fit_pl_err = powerlaw(x0,A_pl,alpha_pl)
        # NOTE: This is done to show the error in the fit in a better way
        y0 = y0/np.sum(y0)  # Normalize y0 to have the same scale as y_fit_pl
        y_fit_pl_err = y_fit_pl_err/np.sum(y_fit_pl_err)  # Normalize y_fit_pl_err to have the same scale as y0
        ErrorPL = np.sqrt(np.sum((y_fit_pl_err-y0)**2)/len(y0))
        return y_fit_pl,ErrorPL,A_pl,alpha_pl
    else:
        print(f"FittingPowerlaw: len x0: {len(x0)} len y0: {len(y0)}")
        return None,None,None,None

class Fitter:
    def __init__(self, x, y_measured, label = 'powerlaw', initial_guess = (6000,0.3), maxfev = 10000):
        self.x = x
        self.y_measured = y_measured
        self.label = label
        self.initial_guess = initial_guess
        self.maxfev = maxfev
        self.Name2Function = {'powerlaw':powerlaw,'exponential':exponential,'linear':linear,'vespignani':multilinear4variables}
        self.Name2LossFunction = {'powerlaw':objective_function_powerlaw,'exponential':objective_function_exponential,'linear':objective_function_linear,'vespignani':objective_function_multilinear4variables}
        # Holds Informations About Success Fit
        self.Error = None
        self.fit = None
        self.success = None
        
    def Fit(self):
        Fitting(self.x,self.y_measured,label = self.label,initial_guess = self.initial_guess,maxfev = self.maxfev)
        