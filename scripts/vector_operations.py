import numpy as np

def normalize_vector(x,y):
    norm = np.sqrt(x**2+y**2)
    return x/norm,y/norm

def coordinate_difference(x1,y1,x2,y2):
    return x1-x2,y1-y2

def scale(x,y,scalar):
    return x*scalar,y*scalar