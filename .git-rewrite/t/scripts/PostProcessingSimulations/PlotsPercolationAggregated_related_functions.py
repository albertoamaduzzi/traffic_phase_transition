import numpy as np

def find_last_stable_point(v, epsilon):
    """
    Find the last index where the derivative is small before fluctuations begin.
    
    Args:
        v: Input vector (array-like)
        epsilon: Threshold for derivative comparison
        
    Returns:
        int: Index of the last element where derivative < epsilon before fluctuations,
             or 0 if something goes wrong
    """
    
    try:
        v = np.array(v)
        
        # Need at least 2 points to compute derivative
        if len(v) < 2:
            return 0
            
        # Compute derivative (difference between consecutive elements)
        derivative = np.diff(v)
        
        # Create boolean mask: True where derivative < epsilon
        is_stable = np.abs(derivative) < epsilon
        
        # Find first index where derivative >= epsilon (first fluctuation)
        fluctuation_indices = np.where(~is_stable)[0]
        
        if len(fluctuation_indices) == 0:
            # No fluctuations found, return last valid index
            return len(is_stable) - 1
            
        # Index of first fluctuation
        first_fluctuation_idx = fluctuation_indices[0]
        
        # Cut the portion after first fluctuation
        stable_portion = is_stable[:first_fluctuation_idx]
        
        if len(stable_portion) == 0:
            # Fluctuation starts immediately
            return 0
            
        # Find the last True value in the stable portion
        # This corresponds to the last point where derivative < epsilon
        last_stable_indices = np.where(stable_portion)[0]
        
        if len(last_stable_indices) == 0:
            # No stable points found before fluctuation
            return 0
            
        return last_stable_indices[-1]
        
    except Exception:
        return 0