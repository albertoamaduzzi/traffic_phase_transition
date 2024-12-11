import json
import os
from GenerateModifiedFluxesSimulation import CityName2RminRmax
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_TRAFFIC_DIR = '/home/alberto/LPSim/LivingCity/berkeley_2018'
def GenerateConfigGeometricalSettingsSpatialPartition(city,TRAFFIC_DIR,start = 7,end = 8,grid_size = 0.02,hexagon_resolution = 8,list_peaks = [1,2,3,4,5,6,8,10],covariances = [0.5,1,2,4],distribution_population_from_center = ["exponential"]):
    if city == "BOS" or city == "LIS":
        list_peaks = [1,2,3,5,10,100,1000]
    else:
        list_peaks = [1,2,3,5,10,100,1000]
    logger.info(f'Generating config for city: {city}\n')
    config = {
        'crs': 'epsg:4326',
        'city': city,
        'config_dir_local': os.path.join(TRAFFIC_DIR,'config'),
        'tiff_file_dir_local': os.path.join(TRAFFIC_DIR,'data','carto','tiff_files'),
        'shape_file_dir_local': os.path.join(TRAFFIC_DIR,'data','carto',city,'shape_files'),
        'ODfma_dir': os.path.join(TRAFFIC_DIR,'data','carto',city,'ODfma'),
        'save_dir_local': os.path.join(TRAFFIC_DIR,'data','carto',city),
        'save_dir_server': os.path.join(SERVER_TRAFFIC_DIR,city),
        'file_GraphFromPhml': os.path.join(TRAFFIC_DIR,'data','carto',city,city + '_new_tertiary_simplified.graphml'),
        'file_gdf_polygons': os.path.join(TRAFFIC_DIR,'data','carto',city,'shape_files',city + '.shp'),
        'start_group_control': start,
        'end_group_control': end,
        'grid_size': grid_size,
        'hexagon_resolution': hexagon_resolution,
        'list_peaks': list_peaks,
        'covariances': covariances,
        'distribution_population_from_center': distribution_population_from_center,
        'Rmin': CityName2RminRmax[city][0],
        'Rmax': CityName2RminRmax[city][1],
        "number_simulation_per_UCI": 15,
        "NumberUCIs": 20
    }
    return config

def SaveJsonDict(dict_,file_name):
    with open(file_name, 'w') as fp:
        json.dump(dict_, fp,indent=4)