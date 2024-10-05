import numpy as np
import logging
logger = logging.getLogger(__name__)

##---------------------------------- GEOMETRIC FEATURES ----------------------------------##
def ProjCoordsTangentSpace(lat,lon,lat0,lon0):
    '''
    Description:
    Projects in the tangent space of the earth in (lat0,lon0) 
    Return: 
    The projected coordinates of the lat,lon  '''
    PI = np.pi
    c_lat= 0.6*100000*(1.85533-0.006222*np.sin(lat0*PI/180))
    c_lon= c_lat*np.cos(lat0*PI/180)
    
    x = c_lon*(lon-lon0)
    y = c_lat*(lat-lat0)
    if isinstance(x,np.ndarray) or isinstance(x,np.float64):
        pass
    else:        
        x = x.to_numpy()
    if isinstance(y,np.ndarray) or isinstance(y,np.float64):
        pass
    else:
        y = y.to_numpy()
    return x,y

def ComputeAreaSquare(geometry):
    x,y = geometry.centroid.xy
    lat0 = x[0]
    lon0 = y[0]
    # Extract the coordinates from the Polygon's exterior
    latlon = np.array([[p[0],p[1]] for p in geometry.exterior.coords]).T
    lat = latlon[0]
    lon = latlon[1]
    # Ensure the last coordinate is the same as the first to close the polygon
    x,y = ProjCoordsTangentSpace(lat,lon,lat0,lon0)
    area = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)*np.sqrt((x[2] - x[1])**2 + (y[2] - y[1])**2)/1000000
    return area
