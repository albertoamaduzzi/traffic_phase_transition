import numpy as np
'''
    All the functions are built to work with numpy arrays and ndarrays
'''
def spatial_entropy(x):
    '''
        Computes the spatial entropy of a given array
    '''
    # Compute the histogram of the array
    hist = np.histogram(x, bins=range(int(np.max(x))+1))[0]
    # Compute the probability of each bin
    prob = hist/np.sum(hist)
    # Compute the spatial entropy
    return -np.sum(prob*np.log(prob))

def fractal_dimension(x):
    
