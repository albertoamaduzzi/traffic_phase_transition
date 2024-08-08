from termcolor import cprint
import numpy as np
import os
import json
import geopandas as gpd
from collections import defaultdict
from shapely.geometry import Point
import time
import sys
import socket
if socket.gethostname()=='artemis.ist.berkeley.edu':
    sys.path.append(os.path.join('/home/alberto/LPSim','traffic_phase_transition','scripts','GenerationNet'))
else:
    sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','GenerationNet'))
from global_functions import *
if socket.gethostname()=='artemis.ist.berkeley.edu':
    sys.path.append(os.path.join('/home/alberto/LPSim','traffic_phase_transition','scripts','PreProcessing'))
else:
    sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','PreProcessing'))
from plot import *


## GLOBAL VARIABLES
NameCity2TractId = {'SFO':'TRACT','LAX':'external_i','LIS':'ID','RIO':'Zona','BOS':'tractid'} # Useful for th extraction of the Polygons that need to be mapped in the OD
NameCity2TiffFile = {'SFO':'usa_ppp_2020_UNadj_constrained.tif','LAX':'usa_ppp_2020_UNadj_constrained.tif','LIS':'prt_ppp_2020_UNadj_constrained.tif','RIO':'bra_ppp_2020_UNadj_constrained.tif','BOS':'usa_ppp_2020_UNadj_constrained.tif'}


def GetGeometryPopulation(gdf_hexagons,gdf_geometry,NameGeometry,NameCity):
    '''
        Input:
            gdf_hexagons: (gpd.GeoDataFrame) hexagons with population
            gdf_geometry: (gpd.GeoDataFrame) geometry to get the population from the hexagons
            NameGeometry: (str) 'polygon','grid','ring'
            NameCity: (str) name of the city
        Output:
            gdf_geometry: (gpd.GeoDataFrame) geometry with population
        Description:
            Consider just the hexagons of the tiling that have population > 0
            Any time we have that a polygon is intersected by the hexagon, we add to the population column
            of the polygon the population of the hexagon times the ratio of intersection area with respect to the hexagon area
    '''
    if gdf_hexagons is None:
        raise ValueError('gdf_hexagons is None')
    if NameGeometry == 'polygon':
        cprint('getPolygonPopulation {}'.format(NameCity),'green')
        if gdf_geometry is None:
            raise ValueError('gdf_{} is None'.format(NameGeometry))
        else:
            gdf_geometry = GetPopulation(gdf_hexagons,gdf_geometry)
            return gdf_geometry
    elif NameGeometry == 'grid':
        cprint('getGridPopulation {}'.format(NameCity),'yellow')
        if gdf_geometry is None:
            raise ValueError('gdf_{} is None'.format(NameGeometry))
        else:
            gdf_geometry = GetPopulation(gdf_hexagons,gdf_geometry)
            return gdf_geometry
    elif NameGeometry == 'ring':
        cprint('getRingPopulation {}'.format(NameCity),'blue')
        if gdf_geometry is None:
            raise ValueError('gdf_{} is None'.format(NameGeometry))
        else:
            gdf_geometry = GetPopulation(gdf_hexagons,gdf_geometry)
            return gdf_geometry
    else:
        raise ValueError('NameGeometry must be polygon, grid or ring')
    
def GetPopulation(gdf_hexagons,gdf_geometry): 
    '''
        
        Computes the population of the Geometry from the hexagons.
        Returns The Geometry gdf with the population column
    '''   
    polygon_sindex = gdf_geometry.sindex
    populationpolygon = np.zeros(len(gdf_geometry))
    for idxh,hex in gdf_hexagons.loc[gdf_hexagons['population'] > 0].iterrows():
        possible_matches_index = list(polygon_sindex.intersection(hex.geometry.bounds))
        possible_matches = gdf_geometry.iloc[possible_matches_index]    
        # Filter based on actual intersection
        if len(possible_matches) > 0:
            intersecting = possible_matches[possible_matches.geometry.intersects(hex.geometry)]
            for idxint,int_ in intersecting.iterrows():
                populationpolygon[idxint] += hex.population*int_.geometry.intersection(hex.geometry).area/hex.geometry.area        
                if int_.geometry.intersection(hex.geometry).area/hex.geometry.area>1.01:
                    raise ValueError('Error the area of the intersection is greater than 1: ',int_.geometry.intersection(hex.geometry).area/hex.geometry.area)
        else:
            pass
        gdf_geometry['population'] = populationpolygon
    return gdf_geometry

##---------------------------------------- GEOMETRY -  OD ----------------------------------------##

def Geometry2OD(gdf_geometry,
                GraphFromPhml,
                NameCity,
                GeometryName,
                save_dir_local,
                resolution):
    '''
        Given a network taken from the cartography or ours:
            Build the set of origin and destinations from the polygons that are coming from the 
            geodataframe.
        
    '''
    if GeometryName == 'polygon':
        if os.path.isfile(os.path.join(save_dir_local,GeometryName,'polygon2origindest.json')) and os.path.isfile(os.path.join(save_dir_local,GeometryName,'origindest2polygon.json')):
            cprint('{} ALREADY COMPUTED'.format(os.path.join(save_dir_local,GeometryName,'polygon2origindest.json')),'magenta')
            OD2polygon,polygon2OD = UploadMapODPol(save_dir_local,GeometryName,'origindest2polygon.json','polygon2origindest.json')
            return OD2polygon,polygon2OD,gdf_geometry
        else:
            cprint('COMPUTING {}'.format(os.path.join(save_dir_local,GeometryName,'polygon2origindest.json')),'magenta')
            key = NameCity2TractId[NameCity]
            OD2polygon,polygon2OD,gdf_geometry = MapPolygon2OD(gdf_geometry,GraphFromPhml,GeometryName,key,save_dir_local)
            return OD2polygon,polygon2OD,gdf_geometry

    elif GeometryName == 'grid':
        if os.path.isfile(os.path.join(save_dir_local,GeometryName,str(round(resolution,3)),'grid2origindest.json')) and os.path.isfile(os.path.join(save_dir_local,GeometryName,str(round(resolution,3)),'origindest2grid.json')):
            cprint('{} ALREADY COMPUTED'.format(os.path.join(save_dir_local,GeometryName,str(round(resolution,3)),'grid2origindest.json')),'yellow')
            OD2Geom,Geom2OD = UploadMapODGeom(save_dir_local,GeometryName,'origindest2grid.json','grid2origindest.json',round(resolution,3))
            return OD2Geom,Geom2OD,gdf_geometry
        else:
            cprint('COMPUTING {}'.format(os.path.join(save_dir_local,GeometryName,str(round(resolution,3)),'grid2origindest.json')),'yellow')
            OD2Geom,Geom2OD,gdf_geometry = Map2Geom2OD(gdf_geometry,GraphFromPhml,GeometryName,save_dir_local,round(resolution,3))
            return OD2Geom,Geom2OD,gdf_geometry
    elif GeometryName == 'ring':
        if os.path.isfile(os.path.join(save_dir_local,GeometryName,str(round(resolution,3)),'ring2origindest.json')) and os.path.isfile(os.path.join(save_dir_local,GeometryName,str(round(resolution,3)),'origindest2ring.json')):
            cprint('{} ALREADY COMPUTED'.format(os.path.join(save_dir_local,GeometryName,str(round(resolution,3)),'ring2origindest.json')),'blue')
            OD2Geom,Geom2OD = UploadMapODGeom(save_dir_local,GeometryName,'origindest2ring.json','ring2origindest.json',round(resolution,3))
            return OD2Geom,Geom2OD,gdf_geometry
        else:
            cprint('COMPUTING {}'.format(os.path.join(save_dir_local,GeometryName,str(round(resolution,3)),'ring2origindest.json')),'blue')
            OD2Geom,Geom2OD,gdf_geometry = Map2Geom2OD(gdf_geometry,GraphFromPhml,GeometryName,save_dir_local,round(resolution,3))
            return OD2Geom,Geom2OD,gdf_geometry
    elif GeometryName == 'hexagon':
        if os.path.isfile(os.path.join(save_dir_local,GeometryName,str(resolution),'hexagon2origindest.json')) and os.path.isfile(os.path.join(save_dir_local,GeometryName,str(resolution),'origindest2hexagon.json')):
            cprint('{} ALREADY COMPUTED'.format(os.path.join(save_dir_local,GeometryName,str(resolution),'hexagon2origindest.json')),'red')
            OD2Geom,Geom2OD = UploadMapODGeom(save_dir_local,GeometryName,'origindest2hexagon.json','hexagon2origindest.json',resolution)
            return OD2Geom,Geom2OD,gdf_geometry
        else:
            cprint('COMPUTING {}'.format(os.path.join(save_dir_local,GeometryName,str(resolution),'hexagon2origindest.json')),'red')
            OD2Geom,Geom2OD,gdf_geometry = Map2Geom2OD(gdf_geometry,GraphFromPhml,GeometryName,save_dir_local,resolution)
            return OD2Geom,Geom2OD,gdf_geometry
    else:
        raise ValueError('GeometryName must be polygon, grid, ring or hexagon')

##---------------------------------------- UPLOAD GEOMETRY - OD ----------------------------------------##

def UploadMapODPol(save_dir_local,GeometryName,NameOD2Pol,NamePol2OD):
    '''
        Upload the files to the server
    '''
    with open(os.path.join(save_dir_local,GeometryName,NameOD2Pol),'r') as f:
        OD2polygon =json.load(f)
    with open(os.path.join(save_dir_local,GeometryName,NamePol2OD),'r') as f:
        polygon2OD = json.load(f)
    return OD2polygon,polygon2OD

def UploadMapODGeom(save_dir_local,GeometryName,NameOD2Geom,NameGeom2OD,resolution):
    '''
        Upload the files to the server
    '''
    with open(os.path.join(save_dir_local,GeometryName,str(resolution),NameOD2Geom),'r') as f:
        OD2Geom = json.load(f)
    with open(os.path.join(save_dir_local,GeometryName,str(resolution),NameGeom2OD),'r') as f:
        Geom2OD = json.load(f)
    return OD2Geom,Geom2OD


def WriteMapODPol(save_dir_local,GeometryName,NameOD2Pol,NamePol2OD,OD2polygon,polygon2OD):
    '''
        Write the files Locally
    '''
    with open(os.path.join(save_dir_local,GeometryName,NameOD2Pol),'w') as f:
        json.dump(OD2polygon,f,indent=4)
    with open(os.path.join(save_dir_local,GeometryName,NamePol2OD),'w') as f:
        json.dump(polygon2OD,f,indent=4)

def WriteMapODGeom(save_dir_local,GeometryName,NameOD2Geom,NameGeom2OD,OD2Geom,Geom2OD,resolution):
    '''
        Write the files Locally
    '''
    with open(os.path.join(save_dir_local,GeometryName,str(resolution),NameOD2Geom),'w') as f:
        json.dump(OD2Geom,f,indent=4)
    with open(os.path.join(save_dir_local,GeometryName,str(resolution),NameGeom2OD),'w') as f:
        json.dump(Geom2OD,f,indent=4)
##---------------------------------------- MAP POLYGON 2 OD if NOT UPLOAD ----------------------------------------##

def MapPolygon2OD(gdf_geometry,
                   GraphFromPhml,
                   GeometryName,
                   key,
                   save_dir_local):
    '''
        Given a network taken from the cartography or ours:
            Build the set of origin and destinations from the polygons that are coming from the 
            geodataframe.
        
    '''
    Geom2OD = defaultdict(list)
    OD2Geom = defaultdict(list)
    for node in GraphFromPhml.nodes():
        containing_polygon = gdf_geometry.geometry.apply(lambda x: Point(GraphFromPhml.nodes[node]['x'],GraphFromPhml.nodes[node]['y']).within(x))
        idx_containing_polygon = gdf_geometry[containing_polygon].index
        tract_id_polygon = gdf_geometry.loc[idx_containing_polygon][key]
        if len(tract_id_polygon)==1:
            try:
                tract_id_polygon = int(tract_id_polygon.tolist()[1])
            except IndexError:
                tract_id_polygon = int(tract_id_polygon.tolist()[0])
            Geom2OD[int(tract_id_polygon)].append(node)
            OD2Geom[node] = int(tract_id_polygon)
        elif len(tract_id_polygon)>1:
            print('tract_id: ',tract_id_polygon)
            print('more than one tract id: THIS IS STRANGE')
        else:
            pass
    WriteMapODPol(save_dir_local,GeometryName,'origindest2polygon.json','polygon2origindest.json',OD2Geom,Geom2OD)
    return OD2Geom,Geom2OD

def Map2Geom2OD(gdf_geometry,
                GraphFromPhml,
                GeometryName,
                save_dir_local,
                resolution):
    '''
       Description:
            NOTE: Considers just those grids that have road network -> NOTE: This is Important to understand, since, when create OD, there are grids that are populated, but not connected to the road network
    '''
    Geom2OD = defaultdict(list)
    OD2Geom = defaultdict(list)
    number_nodes_outside_box = 0
    numbr_nodes_inside_box = 0
    percentage_nodes_outside_box = 0
    print('{} \nnumber nodes to control: '.format(save_dir_local),len(GraphFromPhml.nodes))
    # Loop over the nodes of the graph
    for node in GraphFromPhml.nodes():
        # Check if the node is inside the geometry
        containing_geom = gdf_geometry.geometry.apply(lambda x: Point(GraphFromPhml.nodes[node]['x'],GraphFromPhml.nodes[node]['y']).within(x))
        # Create the Index Column that goes from 0 to len(gdf_geometry)
        gdf_geometry['index'] = gdf_geometry.index
        idx_containing_geom = gdf_geometry[containing_geom].index
        tract_id_geom = gdf_geometry.loc[idx_containing_geom]['index']
        if len(tract_id_geom)==1:
            try:
                tract_id_geom = int(tract_id_geom.tolist()[1])
            except IndexError:
                tract_id_geom = int(tract_id_geom.tolist()[0])
            Geom2OD[int(tract_id_geom)].append(node)
            OD2Geom[node] = int(tract_id_geom)
            numbr_nodes_inside_box += 1
            if (number_nodes_outside_box + numbr_nodes_inside_box) % 100 == 0:
                percentage_nodes_outside_box = number_nodes_outside_box/(number_nodes_outside_box+numbr_nodes_inside_box)
                print('{}\nnumber_nodes_outside_box: '.format(save_dir_local),number_nodes_outside_box)
                print('{}\nnumber_nodes_inside_box: '.format(save_dir_local),numbr_nodes_inside_box)
                print('{}\npercentage_nodes_outside_box: '.format(save_dir_local),percentage_nodes_outside_box)

        elif len(tract_id_geom)>1:
            print('node: ',GraphFromPhml.nodes[node]['x'],GraphFromPhml.nodes[node]['y'])
            print(len([i for i in containing_geom if i==True]))
            print('containing_geom')
            print('tract_id: ',tract_id_geom[0])
            print('tract_id: ',tract_id_geom[1])
            raise ValueError('more than one tract id: THIS IS STRANGE')
        else:
            number_nodes_outside_box += 1
            if (number_nodes_outside_box + numbr_nodes_inside_box) % 100 == 0:
                percentage_nodes_outside_box = number_nodes_outside_box/(number_nodes_outside_box+numbr_nodes_inside_box)
                print('{}\nnumber_nodes_outside_box: '.format(save_dir_local),number_nodes_outside_box)
                print('{}\nnumber_nodes_inside_box: '.format(save_dir_local),numbr_nodes_inside_box)
                print('{}\npercentage_nodes_outside_box: '.format(save_dir_local),percentage_nodes_outside_box)

#            print('Point {} outside geometry: '.format(Point(GraphFromPhml.nodes[node]['x'],GraphFromPhml.nodes[node]['y'])))
#            print('Point {} outside geometry: '.format(Point(GraphFromPhml.nodes[node]['y'],GraphFromPhml.nodes[node]['x'])))
            pass
    percentage_nodes_outside_box = number_nodes_outside_box/(number_nodes_outside_box+numbr_nodes_inside_box)
    if percentage_nodes_outside_box > 0.4:
        raise ValueError('More than 10% of the nodes are outside the geometry')
    # Generate the Column That Tells If Roads Are Present In The Grid
    gdf_geometry["with_roads"] = gdf_geometry.apply(lambda x: x['index'] in list(Geom2OD.keys()))
    WriteMapODGeom(save_dir_local,GeometryName,'origindest2'+GeometryName+'.json',GeometryName+'2origindest.json',OD2Geom,Geom2OD,resolution)
    return OD2Geom,Geom2OD,gdf_geometry







