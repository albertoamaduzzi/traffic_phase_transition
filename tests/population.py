import networkx as nx
import osmnx as ox
import os
import geopandas as gpd
from shapely import intersection,box,Polygon,LineString,Point
from matplotlib import pyplot as plt
import shapely as shp
import numpy as np
import sys
import pandas as pd
from shapely.geometry import Point,Polygon,LineString
from collections import defaultdict
import h3
import sys 
TRAFFIC_DIR = os.getenv('TRAFFIC_DIR')

sys.path.append(os.path.join(TRAFFIC_DIR,'scripts','PreProcessing'))
from plot import *

BO_polygon_file = '/home/aamad/Desktop/phd/berkeley/traffic_phase_transition/data/carto/BOS/shape_files/BOS.shp'
SF_polygon_file = '/home/aamad/Desktop/phd/berkeley/traffic_phase_transition/data/carto/SFO/shape_files/SFO.shp'
LAX_polygon_file = '/home/aamad/Desktop/phd/berkeley/traffic_phase_transition/data/carto/LAX/shape_files/LAX.shp'
RIO_polygon_file = '/home/aamad/Desktop/phd/berkeley/traffic_phase_transition/data/carto/RIO/shape_files/RIO.shp'
LIS_polygon_file = '/home/aamad/Desktop/phd/berkeley/traffic_phase_transition/data/carto/LIS/shape_files/LIS.shp'
BO_polygon = gpd.read_file(BO_polygon_file)
SF_polygon = gpd.read_file(SF_polygon_file)
RIO_polygon = gpd.read_file(RIO_polygon_file)
LAX_polygon = gpd.read_file(LAX_polygon_file)
LIS_polygon = gpd.read_file(LIS_polygon_file)
gpkg_file = os.path.join(TRAFFIC_DIR, 'data','tiff_files','kontur_population_20211109.gpkg')
print('Read file: ',gpkg_file)
population_gdf = gpd.read_file(gpkg_file)
print('Total number hexagons: ',len(population_gdf))
print('Total population: ',np.sum(population_gdf['population']))

namecity = ['BOS','SFO','LAX','RIO','LIS']
i = 0
for g in [BO_polygon,SF_polygon,LAX_polygon,RIO_polygon,LIS_polygon]:
    
    PopCity = population_gdf[population_gdf.geometry.apply(lambda x: x.within(g.unary_union))]
    PopCity['index'] = PopCity.index
    PopCity['area'] = PopCity.area
    print(f'population {namecity}: ',np.sum(PopCity['population']))
    plot_hexagon_tiling(PopCity,g,os.path.join(TRAFFIC_DIR,'data','carto',namecity[i],'hexagon',str(8)),str(8))
    fig,ax = plt.subplots(1,1,figsize=(12,12))
    ax.hist(PopCity['population'],bins=100)
    ax.set_xlabel('Population')
    ax.set_ylabel('Number of hexagons')
    ax.set_title(str(np.sum(PopCity['population'])))
    plt.savefig(os.path.join(TRAFFIC_DIR,'data','carto',namecity[i],'hexagon',str(8),'hist_population.png'))
    
    i+=1
i = 0
for g in [BO_polygon,SF_polygon,LAX_polygon,RIO_polygon,LIS_polygon]:
    print('city: ',namecity[i])
    bounding_box = g.geometry.unary_union.bounds
    bbox = shp.geometry.box(*bounding_box)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=population_gdf.crs)
    # Extract population data from the GeoDataFrame
    ListPopName = ['population', 'pop', 'pop_density', 'population_density']
    for popname in ListPopName:
        if popname in population_gdf.columns:
            break
    print('Columns:\n',population_gdf.columns)

    population_values = population_gdf[popname]  # Replace 'population_column' with the actual column name containing population data
    # Define the bounding box coordinates (longitude, latitude)
    min_lon, min_lat = bounding_box[0], bounding_box[1]
    max_lon, max_lat = bounding_box[2], bounding_box[3]

    # Define the desired resolution for the hexagons
    resolution = 8  # Adjust the resolution as needed

    # Generate hexagon grid cells covering the bounding box
    hexagons = h3.polyfill(
        {"type": "Polygon", "coordinates": [[[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat], [min_lon, min_lat]]]},
        res=resolution
    )
    print('Hexagon length: ',len(hexagons))
    # Create a dictionary to store population data for each hexagon
    population_dict = {hex_id: population for hex_id, population in zip(hexagons, population_values)}
    print('Population dict: ',population_dict)
    # Assign population data to the hexagon grid cells
    for hex_id in hexagons:
        population = population_dict.get(hex_id, 0)  # Get population value from dictionary, default to 0 if not found
        # Assign population value to hexagon (e.g., store in a database, export to a file, etc.)
    print(f'population {population}')
    # Optionally, create a GeoDataFrame with hexagon geometries and population data
    hexagon_geometries = [h3.h3_to_geo_boundary(hex_id) for hex_id in hexagons]
    hexagon_geometries = [np.array(coords)[:, ::-1] for coords in hexagon_geometries]  # Convert coordinates to (latitude, longitude) format
    index = list(range(len(hexagon_geometries)))
    print('Hexagon geometries: ',hexagon_geometries[:10])
    print('Length hex geom: ',len(hexagon_geometries))
    print('Population values: ',population_values[:10])
    print('Length pop_val: ',len(population_values))
    
    gdf = gpd.GeoDataFrame({
        'geometry': gpd.GeoSeries([Polygon(coords) for coords in hexagon_geometries]),
        'population': population_values,
        'index': index
    })


    # Optionally, plot the hexagons
    plot_hexagon_tiling(gdf,g,os.path.join(TRAFFIC_DIR,'data','carto',namecity[i],'hexagon',str(resolution)),resolution)
    fig,ax = plt.subplots(1,1,figsize=(12,12))
    ax.hist(population_values,bins=100)
    ax.set_xlabel('Population')
    ax.set_ylabel('Number of hexagons')
    ax.set_title(str(np.sum(population_values)))
    plt.savefig(os.path.join(TRAFFIC_DIR,'data','carto',namecity[i],'hexagon',str(resolution),'hist_population.png'))
    i+=1