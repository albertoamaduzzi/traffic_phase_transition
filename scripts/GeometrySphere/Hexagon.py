'''
    Description: 
        This file is used to get the hexagon tiling of the area specified by a .shp file. [BOS.shp, SFO.shp, LAX.shp, RIO.shp, LIS.shp]
        The tiling is available for the 
        1) .tiff file for the population dataset -> That is not accounting well for the population [SFO: 400K people]
        2) .gpkg file for the population dataset -> That is accounting well for the population [SFO: 12M people]
    Output:
        hexagon: gpd.GeoDataFrame -> hexagon tiling of the area with a specified resolution.
    NOTE: 
        This is very useful to get the population for the grid that will be used to define the potential in our work.
        Indeed, the grid wiil have associated the population according to intersection with hexagon and from there we fit 
        the exponentially decaying gravity model for the potential.
    

'''
from termcolor import cprint
import os
import gzip
import shutil
import sys
import socket
if socket.gethostname()=='artemis.ist.berkeley.edu':
    sys.path.append(os.path.join('/home/alberto/LPSim','traffic_phase_transition','scripts','PlanarGraph'))
else:
    sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','PlanarGraph'))
from global_functions import *
import numpy as np
from shapely.geometry import Point,Polygon
import matplotlib.pyplot as plt
import json
import networkx as nx
import haversine as hs
import shapely as shp
import h3
import shapely.geometry as sg
import rasterio
from rasterio.mask import mask
import geopandas as gpd

def SetHexagonDir(save_dir_local,resolution):
    '''
        Input:
            save_dir_local: str -> local directory to save the hexagon
            resolution: int -> resolution of the hexagon
        Output:
            dir_hexagon: str -> directory to save the hexagon
    '''
    ifnotexistsmkdir(os.path.join(save_dir_local,'hexagon'))
    ifnotexistsmkdir(os.path.join(save_dir_local,'hexagon',str(resolution)))
    dir_hexagon = os.path.join(save_dir_local,'hexagon')
    return dir_hexagon

def SaveHexagon(save_dir_local,resolution,hexagon):
    '''
        Save the hexagon
    '''
    SetHexagonDir(save_dir_local,resolution)
    if not os.path.isfile(os.path.join(save_dir_local,'hexagon',str(resolution),'hexagon.geojson')):
        cprint('COMPUTING: {} '.format(os.path.join(save_dir_local,'hexagon',str(resolution),'hexagon.geojson')),'green')
        hexagon.to_file(os.path.join(save_dir_local,'hexagon',str(resolution),'hexagon.geojson'), driver="GeoJSON")  
    else:
        cprint('{} ALREADY COMPUTED'.format(os.path.join(save_dir_local,'hexagon',str(resolution),'hexagon.geojson')),'green')
    return hexagon

def ReadHexagon(save_dir_local,resolution):
    '''
        Read the hexagon
    '''
    if os.path.isfile(os.path.join(save_dir_local,'hexagon',str(resolution),'hexagon.geojson')):
        cprint('{} ALREADY COMPUTED'.format(os.path.join(save_dir_local,'hexagon',str(resolution),'hexagon.geojson')),'green')
        hexagon = gpd.read_file(os.path.join(save_dir_local,'hexagon',str(resolution),'hexagon.geojson'))
    else:
        raise ValueError('Hexagon not found')
    return hexagon

def GetHexagon(gdf_polygons,
                tif_file,
                save_dir_local,
                CityName,
                crs = 'epsg:4326',
                resolution=8):
    '''
        This function is used to get the hexagon tiling of the area, it can be used just for US right now as we have just
        tif population file just for that
    '''
    ## READ TIF FILE
    cprint('Get Hexagon','green')
    if not os.path.exists(os.path.join(save_dir_local,'hexagon',str(resolution),'hexagon.geojson')):
        cprint('COMPUTING: {} '.format(os.path.join(save_dir_local,'hexagon',str(resolution),'hexagon.geojson')),'green')
        if 0 == 1: # CASE TIFF FILES -> NOT WORKING WELL
            hexagon_geometries, population_data_hexagons = HexagonFromTiff(tif_file,gdf_polygons,CityName,resolution)
    #            print('population_data_hexagons: ',np.shape(population_data_hexagons))
            centroid_hexagons = [h.centroid for h in hexagon_geometries]
            centroidx = [h.centroid.x for h in hexagon_geometries]
            centroidy = [h.centroid.y for h in hexagon_geometries]
            gdf_hexagons = gpd.GeoDataFrame(geometry=hexagon_geometries, data={'population':population_data_hexagons,'centroid_x':centroidx,'centroid_y':centroidy},crs = crs)
            gdf_hexagons.reset_index(inplace=True)
        else:
            gdf_hexagons,resolution = HexagonFromGpkg(tif_file,CityName,gdf_polygons,resolution=8)
    else:
        cprint('{} ALREADY COMPUTED'.format(os.path.join(save_dir_local,'hexagon',str(resolution),'hexagon.geojson')),'green')
        gdf_hexagons = ReadHexagon(save_dir_local,resolution)
    if 0 == 1: # CASE TIFF FILES -> NOT WORKING WELL
        gdf_hexagons['area'] = gdf_hexagons.to_crs({'proj':'cea'}).area / 10**6
        gdf_hexagons['density_population'] = gdf_hexagons['population']/gdf_hexagons['area']        
    else:
        pass
    return gdf_hexagons

def HexagonFromTiff(tif_file,gdf_polygons,CityName,resolution=8):
    '''
        Input:
            tif_file: str -> path to the tif file
            gdf_polygons: gpd.GeoDataFrame -> polygons of the area
            resolution: int -> resolution of the hexagon
        Output:
            hexagon_geometries: list -> list of hexagons
            population_data_hexagons: list -> list of population data
    '''
    city2file = {'SFO':'usa_ppp_2020_UNadj_constrained.tif','BOS':'usa_ppp_2020_UNadj_constrained.tif','LAX':'usa_ppp_2020_UNadj_constrained.tif','LIS':'prt_ppp_2020_UNadj_constrained.tif','RIO':'bra_ppp_2020_UNadj_constrained.tif'}
    with rasterio.open(os.path.join(tif_file,city2file[CityName])) as dataset:
        clipped_data, clipped_transform = mask(dataset, gdf_polygons.geometry, crop=True)
    ## CHANGE NULL ENTRANCIES (-99999) for US (may change for other Countries [written in United Nation page of Download])
    clipped_data = np.array(clipped_data)
#            print('resolution: ',resolution)
#            print('clipped_data: ',np.shape(clipped_data))
#            print('clipped_transform: ',np.shape(clipped_transform))
    condition = clipped_data<0
    clipped_data[condition] = 0
    # Define hexagon resolution
    bay_area_geometry = gdf_polygons.unary_union
#            print('bay_area_geometry: ',type(bay_area_geometry))
    # Convert MultiPolygon to a single Polygon
    bay_area_polygon = bay_area_geometry.convex_hull
    # Convert Polygon to a GeoJSON-like dictionary
    bay_area_geojson = sg.mapping(bay_area_polygon)
#            print('bay_area_geojson: ',type(bay_area_geojson))
    # Get hexagons within the bay area
    hexagons = h3.polyfill(bay_area_geojson, resolution, geo_json_conformant=True)
#            print('hexagons: ',type(hexagons))
    # Convert hexagons to Shapely geometries
    hexagon_geometries = [sg.Polygon(h3.h3_to_geo_boundary(h, geo_json=True)) for h in hexagons]
#            print('hexagon_geometries: ',np.shape(hexagon_geometries),' type: ',type(hexagon_geometries))

    # Aggregate population data for each hexagon
    population_data_hexagons = [aggregate_population(hexagon, clipped_transform, clipped_data) for hexagon in hexagon_geometries]
    return hexagon_geometries, population_data_hexagons



def convert_resolution(h3_idx, h3_idxresolution):
    return h3.h3_to_children(h3_idx, h3_idxresolution)
def ReadGpkg(gpkg_file_dir,CityName):
    City2FileZip = {'SFO':'kontur_population_US_20231101.gpkg.gz','BOS':'kontur_population_US_20231101.gpkg.gz','LAX':'kontur_population_US_20231101.gpkg.gz','LIS':'kontur_population_PT_20231101.gpkg.gz','RIO':'kontur_population_BR_20231101.gpkg.gz'} 
    City2FileUnZip = {'SFO':'kontur_population_US_20231101.gpkg','BOS':'kontur_population_US_20231101.gpkg','LAX':'kontur_population_US_20231101.gpkg','LIS':'kontur_population_PT_20231101.gpkg','RIO':'kontur_population_BR_20231101.gpkg'}
    if not os.path.isfile(os.path.join(gpkg_file_dir,City2FileUnZip[CityName])):
        with gzip.open(os.path.join(gpkg_file_dir,City2FileZip[CityName]), 'rb') as f_in:
            with open(os.path.join(gpkg_file_dir,City2FileUnZip[CityName]), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        pass
    USGeoDataFrame = gpd.read_file(os.path.join(gpkg_file_dir,City2FileUnZip[CityName]))
    USGeoDataFrame['geometry'] = USGeoDataFrame['h3'].apply(lambda x: Polygon(h3.h3_to_geo_boundary(x,True)))
    USGeoDataFrame.crs = "EPSG:4326"
    print(USGeoDataFrame.head())
    print(USGeoDataFrame.crs)
#    USGeoDataFrame.to_crs("EPSG:4326", inplace=True)
    return USGeoDataFrame
def ChangeResolutionExtractPopulation(population,hex_id,resolution):
    print('hex resolution: ',h3.h3_get_resolution(hex_id))
    print('resolution: ',resolution)
    polygons = Polygon(h3.h3_to_geo_boundary(hex_id, resolution))
    print('polygons: ',polygons)
    exponent = (resolution - h3.h3_get_resolution(hex_id))
    
    return polygons,list(population*7**exponent)*len(polygons)

def HexagonFromGpkg(gpkg_file_dir,CityName,gdf_polygons,resolution=8):
    USGeoDataFrame = ReadGpkg(gpkg_file_dir,CityName)
    if gdf_polygons.crs != "EPSG:4326":
        gdf_polygons.crs = "EPSG:4326"
        print('gdf_polygons: ',gdf_polygons.crs)
    gdf_hexagons = USGeoDataFrame[USGeoDataFrame.geometry.intersects(gdf_polygons.unary_union)]
#    hexagons = intersecting_hexagons.apply(lambda row: h3.polyfill(row.geometry.__geo_interface__,resolution),axis = 1).explode()
#    print(hexagons)
#    geom,population= zip(*intersecting_hexagons.apply(lambda row: ChangeResolutionExtractPopulation(row['population'],row['h3'],resolution),axis = 1).explode())
 #   gdf_hexagons = gpd.GeoDataFrame(geometry=geom, data=population, columns=['population'])
#    gdf_hexagons['population'] = intersecting_hexagons.apply(lambda row: ChangeResolutionExtractPopulation(row['population'],hex_id['h3'],resolution)) 
#    gdf_hexagons = gdf_hexagons.plot()
#    gdf_hexagons = gpd.GeoDataFrame(gdf_hexagons, geometry='geometry', crs= USGeoDataFrame.crs)    
    resolution = h3.h3_get_resolution(gdf_hexagons['h3'].tolist()[0])
    area = gdf_hexagons['geometry'].apply(lambda x: x.area)
    gdf_hexagons['area'] = area
    del area
    idx = gdf_hexagons.index
    gdf_hexagons['index'] = idx
    del idx
    centroid =gdf_hexagons['geometry'].apply(lambda x: x.centroid)
    gdf_hexagons['centroid_x'] = centroid.x
    gdf_hexagons['centroid_y'] = centroid.y
    del centroid
    density = gdf_hexagons.apply(lambda x: x['population']/x['area'],axis = 1)
    gdf_hexagons['density_population'] = density
    del density 
    return gdf_hexagons,resolution
    
def aggregate_population(hexagon, hexagon_transform, clipped_data):
    # Convert hexagon to pixel coordinates
    hexagon_coords = np.array(hexagon.exterior.xy).T
    hexagon_pixel_coords = np.array(rasterio.transform.rowcol(hexagon_transform, hexagon_coords[:, 0], hexagon_coords[:, 1]))
    # Ensure coordinates are within valid bounds
    valid_coords = (
        (0 <= np.array(hexagon_pixel_coords[0])) & (np.array(hexagon_pixel_coords[0]) < clipped_data.shape[1]) &
        (0 <= np.array(hexagon_pixel_coords[1])) & (np.array(hexagon_pixel_coords[1]) < clipped_data.shape[2])
    )

    # Filter valid coordinates
    valid_pixel_coords = hexagon_pixel_coords[:, valid_coords]

    # Extract values from clipped data
    values = clipped_data[0, valid_pixel_coords[0], valid_pixel_coords[1]]

    # Aggregate population data (example: sum)
    population_sum = np.sum(values)
    return population_sum
