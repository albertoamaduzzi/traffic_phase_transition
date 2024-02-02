from scipy.optimize import curve_fit,minimize
from scipy import stats
from scipy.stats import powerlaw as pl
import numpy as np

# FUCNTIONS FOR FITTING
def powerlaw(x, amp, index):
    return amp * (np.array(x)**index)

def exponential(x, amp, index):
    return amp * np.exp(-index*np.array(x))

def quadratic_loss_function(y_predict, y_measured):
    return np.sum((y_predict-y_measured)**2)

def objective_function_powerlaw(params,x,y_measured):
    return quadratic_loss_function(powerlaw(x, params[0], params[1]), y_measured)

def objective_function_exponential(params,x,y_measured):
    return quadratic_loss_function(exponential(x, params[0], params[1]), y_measured)

def GravitationalField(mi,mj,dij,d0,k):
    '''
        Input:
            mi: (float) mass of node i
            mj: (float) mass of node j
            dij: (float) distance between node i and node j
            d0: (float) parameter for the gravitational field
            k: (float) parameter for the gravitational field
        Output:
            (float) gravitational field
    '''
    return k*mi*mj*np.exp(dij/d0)

def Fitting(x,y_measured,label = 'powerlaw',initial_guess = (6000,0.3),maxfev = 10000):
    '''
        Input:
            label: 'powerlaw' or 'exponential'
            x: (np.array 1D) x-axis
            y_measured: (np.array 1D) y-axis
            initial_guess: (tuple 2D) parameters for fitting 
    '''
    if label == 'powerlaw':
        print('Fitting powerlaw')
        result_powerlaw = minimize(objective_function_powerlaw, initial_guess, args = (x, y_measured),maxfev = maxfev)
        optimal_params_pl = result_powerlaw.x
        fit = curve_fit(powerlaw, x, y_measured,p0 = optimal_params_pl,maxfev = maxfev)
        print(fit)
        print('powerlaw fit: ',fit[0][0],' ',fit[0][1])
        print('Convergence fit powerlaw: ',result_powerlaw.success)
        print('Optimal parameters: ',result_powerlaw.x)
        print('Message: ',result_powerlaw.message)
        return fit
    elif label == 'exponential':
        print('Fitting exponential')
        result_expo = minimize(objective_function_exponential, initial_guess, args = (x, y_measured),maxfev = maxfev)
        optimal_params_expo = result_expo.x
        fitexp = curve_fit(exponential, x, y_measured,p0 = optimal_params_expo, maxfev = maxfev)
        print(fitexp)
        print('expo fit: ',fitexp[0][0],' ',fitexp[0][1])
        print('Convergence fit expo: ',result_expo.success)
        print('Optimal parameters: ',result_expo.x)
        print('Message: ',result_expo.message)
        return fitexp

    else:
        raise ValueError('label must be powerlaw or exponential')


