from termcolor import cprint
import os
import numpy as np
import sys
import socket
if socket.gethostname()=='artemis.ist.berkeley.edu':
    sys.path.append(os.path.join('/home/alberto/LPSim','traffic_phase_transition','scripts','PlanarGraph'))
else:
    sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','PlanarGraph'))
from global_functions import *
import logging
logger = logging.getLogger(__name__)

def SetPolygonDir(save_dir_local):
    '''
        Input:
            save_dir_local: str -> local directory to save the polygon
        Output:
            dir_polygon: str -> directory to save the polygon
    '''
    ifnotexistsmkdir(os.path.join(save_dir_local,'polygon'))
    dir_polygon = os.path.join(save_dir_local,'polygon')
    return dir_polygon

def SavePolygon(save_dir_local,polygon):
    '''
        Save the polygon
    '''
    if not os.path.isfile(os.path.join(save_dir_local,'polygon.geojson')):
        cprint('COMPUTING: {} '.format(os.path.join(save_dir_local,'polygon.geojson')),'green')
        polygon.to_file(os.path.join(save_dir_local,'polygon','polygon.geojson'), driver="GeoJSON")  
    else:
        cprint('{} ALREADY COMPUTED'.format(os.path.join(save_dir_local,'polygon','polygon.geojson')),'green')
    return polygon

def getPolygonPopulation(gdf_hexagons,gdf_polygons,city):
    '''
        Consider just the hexagons of the tiling that have population > 0
        Any time we have that a polygon is intersected by the hexagon, we add to the population column
        of the polygon the population of the hexagon times the ratio of intersection area with respect to the hexagon area
    '''
    cprint('getPolygonPopulation {}'.format(city),'green')
    if gdf_hexagons is None:
        raise ValueError('grid is None')
    elif gdf_polygons is None:
        raise ValueError('gdf_polygons is None')
    else:
        polygon_sindex = gdf_polygons.sindex
        populationpolygon = np.zeros(len(gdf_polygons))
        for idxh,hex in gdf_hexagons.loc[gdf_hexagons['population'] > 0].iterrows():
            possible_matches_index = list(polygon_sindex.intersection(hex.geometry.bounds))
            possible_matches = gdf_polygons.iloc[possible_matches_index]    
            # Filter based on actual intersection
            if len(possible_matches) > 0:
                intersecting = possible_matches[possible_matches.geometry.intersects(hex.geometry)]
                for idxint,int_ in intersecting.iterrows():
                    populationpolygon[idxint] += hex.population*int_.geometry.intersection(hex.geometry).area/hex.geometry.area        
                    if int_.geometry.intersection(hex.geometry).area/hex.geometry.area>1.01:
                        raise ValueError('Error the area of the intersection is greater than 1: ',int_.geometry.intersection(hex.geometry).area/hex.geometry.area)
            else:
                pass
            gdf_polygons['population'] = populationpolygon
    return gdf_polygons

