from termcolor import cprint
import os
import time
import geopandas as gpd
from collections import defaultdict
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','GenerationNet'))
from global_functions import *
##---------------------------------------- DIRECTORY ----------------------------------------##
def SetRingDir(save_dir_local,number_of_rings):
    '''
        Input:
            save_dir_local: str -> local directory to save the ring
            number_of_rings: int -> number of rings
        Output:
            dir_ring: str -> directory to save the ring
    '''
    ifnotexistsmkdir(os.path.join(save_dir_local,'ring'))
    ifnotexistsmkdir(os.path.join(save_dir_local,'ring',str(number_of_rings)))
    dir_ring = os.path.join(save_dir_local,'ring')
    return dir_ring

def SaveRing(save_dir_local,number_of_rings,rings):
    '''
        Save the rings
    '''
    SetRingDir(save_dir_local,number_of_rings)
    if not os.path.isfile(os.path.join(save_dir_local,str(number_of_rings),'ring','rings.geojson')):
        cprint('COMPUTING: {} '.format(os.path.join(save_dir_local,str(number_of_rings),'ring','rings.geojson')),'green')
        rings.to_file(os.path.join(save_dir_local,str(number_of_rings),'ring','rings.geojson'), driver="GeoJSON")  
    else:
        cprint('{} ALREADY COMPUTED'.format(os.path.join(save_dir_local,str(number_of_rings),'ring','rings.geojson')),'green')
    return rings

def ReadRing(save_dir_local,number_of_rings):
    '''
        Read the rings
    '''
    if os.path.isfile(os.path.join(save_dir_local,str(number_of_rings),'ring','rings.geojson')):
        cprint('{} ALREADY COMPUTED'.format(os.path.join(save_dir_local,str(number_of_rings),'ring','rings.geojson')),'green')
        rings = gpd.read_file(os.path.join(save_dir_local,str(number_of_rings),'ring','rings.geojson'))
    else:
        raise ValueError('Ring not found')
    return rings
##---------------------------------------- RING ----------------------------------------##

def GetRing(number_of_rings,
            gdf_polygons,
            crs,
            save_dir_local):
    '''
        Compute the rings of the city and the intersection with polygons
        rings: dict -> {idx:ring}
    '''
    cprint('get_rings: ' + str(number_of_rings),'blue')
    rings = defaultdict(list)
    number_of_rings = number_of_rings
    gdf_original_crs = gpd.GeoDataFrame(geometry=[gdf_polygons.geometry.unary_union.centroid], crs=crs)
    bounding_box = gdf_polygons.geometry.unary_union.bounds
    radius = max([abs(bounding_box[0] -bounding_box[2])/2,abs(bounding_box[1] - bounding_box[3])/2]) 
    radiuses = np.linspace(0,radius,number_of_rings)
    if os.path.isfile(os.path.join(save_dir_local,str(number_of_rings),'ring','rings.geojson')):
        cprint('{} ALREADY COMPUTED'.format(os.path.join(save_dir_local,str(number_of_rings),'ring','rings.geojson')),'blue')
        rings = ReadRing(save_dir_local,number_of_rings)
    else:
        for i,r in enumerate(radiuses):
            if i == 0:
                intersection_ = gdf_original_crs.buffer(r)
                rings[i] = intersection_
            else:
                intersection_ = gdf_original_crs.buffer(r).intersection(gdf_original_crs.buffer(radiuses[i-1]))
                complement = gdf_original_crs.buffer(r).difference(intersection_)
                rings[i] = complement
            rings = gpd.GeoDataFrame(geometry=pd.concat(list(rings), ignore_index=True),crs=crs)#rings.values()
            ## SAVE RINGS
    return rings
