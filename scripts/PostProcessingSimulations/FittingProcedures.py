try:
    import pymc3 as pm
    FoundPyMC3 = True
except:
    print('PyMC3 not installed')
    FoundPyMC3 = False
from scipy.special import gamma
from scipy.optimize import curve_fit,minimize
from scipy import stats
from scipy.stats import powerlaw 
import numpy as np
import sys
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
    return log_k + beta * x[0] + gamma * x[1] + d0minus1 * x[2] 

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
    fit = curve_fit(Name2Function[label], xdata = x, ydata = y_measured,p0 = list(optimal_params_pl),bounds = bounds,maxfev = maxfev)
    print(fit)
    print('{} fit: '.format(label),fit[0][0],' ',fit[0][1])
    print('Convergence fit {}: '.format(label),result_powerlaw.success)
    print('Optimal parameters: ',result_powerlaw.x)
    print('Message: ',result_powerlaw.message)
    return fit,result_powerlaw.success

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
#    dx = np.diff(x)
#    if np.sum(y[1:]*dx)!=1:
    Z = np.sum(y)
    y = y/Z
    x0 = np.array([x[x_] for x_ in range(len(x)) if x[x_]>0 and y[x_]>0])
    y0 = np.array([y[x_] for x_ in range(len(x)) if x[x_]>0 and y[x_]>0])
    # Fit on the log-log scale POWERLAW
    logx = np.log(x0)
    logy = np.log(y0)
    # Adjust The Initial Guess # alpha*x + log(A) 
    log_initial_guess_pl = (initial_guess_powerlaw[1],np.log(initial_guess_powerlaw[0]))
#    result_powerlaw = minimize(objective_function_linear, log_initial_guess_pl, args = (logx, logy))
#    optimal_params_pl = result_powerlaw.x
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
        ErrorExp = np.sqrt(np.sum((np.log(exponential(x0,A0,alpha_exp))-np.log(y0))**2)/len(y0))
        ErrorPL = np.sqrt(np.sum((np.log(powerlaw(x0,A_pl,alpha_pl))-np.log(y0))**2)/len(y0))
        return ErrorExp,A0,alpha_exp,Z*y_fit_exp,ErrorPL,A_pl,alpha_pl,Z*y_fit_pl
    else:
        raise ValueError(f'ComparePlExpo: x {x} and y {y} are empty')

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
        