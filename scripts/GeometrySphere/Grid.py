'''
    The output of this script are:
        - grid: GeoDataFrame -> grid of points of size grid_size
        - lattice: graph -> graph object of the lattice
        - direction_distance_matrix: DataFrame -> [i,j,dir_vector,distance] -> The index is index
        TO BE USED FOR THE COMPUTATION OF THE GRADIENT AND THE CURL
        - gridIdx2dest: dict -> {(i,j): number_people}
        - gridIdx2ij: dict -> {index: (i,j)}

        In general I will have that the dataframe containing origin and destination is associated to the unique integer for the grid.
        Then I wiil need yto use gridIdx2ij to obtain the position that I am going to use to compute the gradient and the curl. 
'''

from termcolor import cprint
import sys
import os
import time
import geopandas as gpd
import numpy as np
import shapely as shp
from GeometrySphere import ComputeAreaSquare
import networkx as nx
from PolygonSettings import *
import haversine as hs
from collections import defaultdict
import pandas as pd
sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','GenerationNet'))
from global_functions import *
from GeometrySphere import ProjCoordsTangentSpace

##---------------------------------------- DIRECTORY ----------------------------------------##

def SetGridDir(save_dir_local,grid_size):
    '''
        Input:
            save_dir_local: str -> local directory to save the grid
            grid_size: float -> size of the grid
        Output:
            dir_grid: str -> directory to save the grid
    '''
    ifnotexistsmkdir(os.path.join(save_dir_local,'grid'))
    ifnotexistsmkdir(os.path.join(save_dir_local,'grid',str(round(grid_size,3))))
    dir_grid = os.path.join(save_dir_local,'grid')
    return dir_grid

def SaveGrid(save_dir_local,grid_size,grid):
    '''
        Save the grid
    '''
    SetGridDir(save_dir_local,grid_size)
    if not os.path.isfile(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"grid.geojson")):
        cprint('COMPUTING: {} '.format(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"grid.geojson")),'yellow')
        grid.to_file(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"grid.geojson"), driver="GeoJSON")  
    else:
        cprint('{} ALREADY COMPUTED'.format(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"grid.geojson")),'yellow')
    return grid

def SaveLattice(save_dir_local,grid_size,lattice):
    '''
        Save the lattice
    '''
    SetGridDir(save_dir_local,grid_size)
    if not os.path.isfile(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"centroid_lattice.graphml")):
        cprint('COMPUTING: {} '.format(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"centroid_lattice.graphml")),'yellow')
        nx.write_graphml(lattice, os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"centroid_lattice.graphml"))  
    else:
        cprint('{} ALREADY COMPUTED'.format(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"centroid_lattice.graphml")),'yellow')
    return lattice

def SaveDirectionDistanceMatrix(save_dir_local,grid_size,df_direction_distance_matrix):
    '''
        Save the direction matrix
    '''
    SetGridDir(save_dir_local,grid_size)
    if not os.path.isfile(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"direction_distance_matrix.csv")):
        cprint('COMPUTING: {} '.format(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"direction_distance_matrix.csv")),'yellow')
        df_direction_distance_matrix.to_csv(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"direction_distance_matrix.csv"))
    else:
        cprint('{} ALREADY COMPUTED'.format(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"direction_distance_matrix.csv")),'yellow')
    return df_direction_distance_matrix


##---------------------------------------- GRID ----------------------------------------##


def GetGrid(grid_size,
            bounding_box,
            crs,
            save_dir_local):
    '''
        Input:
            grid_size: float -> size of the grid
            save_dir_local: str -> local directory to save the grid
            save_dir_server: str -> server directory to save the grid
            Files2Upload: dict -> dictionary to upload the files
        Output:

        centroid: Point -> centroid of the city
        bounding_box: tuple -> (minx,miny,maxx,maxy)
        grid: GeoDataFrame -> grid of points of size grid_size
        In this way grid is ready to be used as the matrix representation of the city and the gradient and the curl defined on it.
        From now on I will have that the lattice is associated to the centroid grid.
        Usage:
            grid and lattice are together containing spatial and network information
    '''
    cprint('Initialize Grid: ' + str(round(grid_size,3)),'yellow')
    dir_grid = SetGridDir(save_dir_local,grid_size)
    if os.path.isfile(os.path.join(dir_grid,str(round(grid_size,3)),"grid.geojson")):
        cprint('ALREADY COMPUTED'.format(os.path.join(dir_grid,str(round(grid_size,3)),"grid.geojson")),'yellow')
        grid = gpd.read_file(os.path.join(dir_grid,str(round(grid_size,3)),"grid.geojson"))
        bbox = shp.geometry.box(*bounding_box)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=crs)
        x = np.arange(bounding_box[0], bounding_box[2], grid_size)
        y = np.arange(bounding_box[1], bounding_box[3], grid_size)

    else:
        cprint('COMPUTING {}'.format(os.path.join(dir_grid,str(round(grid_size,3)),"grid.geojson")),'green')
        bbox = shp.geometry.box(*bounding_box)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=crs)
        x = np.arange(bounding_box[0], bounding_box[2], grid_size)
        y = np.arange(bounding_box[1], bounding_box[3], grid_size)
#        grid_points = gpd.GeoDataFrame(geometry=[shp.geometry.box(xi, yi,maxx = max(x),maxy = max(y)) for xi in x for yi in y], crs=crs)
        grid_points = gpd.GeoDataFrame(geometry=[shp.geometry.box(xi, yi, xi + grid_size, yi + grid_size) for xi in x for yi in y], crs=crs)
        ij = [[i,j] for i in range(len(x)) for j in range(len(y))]
        grid_points['i'] = np.array(ij)[:,0]
        grid_points['j'] = np.array(ij)[:,1]
        # Clip the grid to the bounding box
        grid = gpd.overlay(grid_points, bbox_gdf, how='intersection')
        grid['centroidx'] = grid.geometry.centroid.x
        grid['centroidy'] = grid.geometry.centroid.y                
        grid['area'] = grid['geometry'].apply(ComputeAreaSquare)
        grid['index'] = grid.index
    return grid
## Direction matrix
def GetDirectionMatrix(save_dir_local,grid_size):
    if os.path.isfile(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"direction_distance_matrix.csv")):
        direction_distence_df = pd.read_csv(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"direction_distance_matrix.csv"))   
        return direction_distence_df,True
    else:
        return None,False


def ComputeDirectionMatrix(grid):
    '''
        Input:
            grid: GeoDataFrame -> grid of points
        Output:
            direction_matrix: dict -> {(idxj,idxj): [xj-xi/norm(xj-xi,yj-yi),yj-yi/norm(xj-xi,yj-yi)]}
        NOTE: The index of the grid is the index of the grid. Not the couple (i,j) that is useful for the definition of the gradient.
    '''
    direction_matrix = {i: {j: np.array([grid.iloc[j]['centroidx']-grid.iloc[i]['centroidx'],grid.iloc[j]['centroidy']-grid.iloc[i]['centroidy']])/np.linalg.norm(np.array([grid.iloc[j]['centroidx']-grid.iloc[i]['centroidx'],grid.iloc[j]['centroidy']-grid.iloc[i]['centroidy']])) for j in range(len(grid))} for i in range(len(grid))}
    distance_matrix = {i: {j: hs.haversine((grid.iloc[i]['centroidy'],grid.iloc[i]['centroidx']),(grid.iloc[j]['centroidy'],grid.iloc[j]['centroidx'])) for j in range(len(grid))} for i in range(len(grid))}
    return direction_matrix,distance_matrix 
'''
def DirectionDistance2Df(direction_matrix,distance_matrix):
    rows = []
    columns = ['i', 'j', 'dir_vector', 'distance']

    # Iterate over the direction and distance matrices to construct DataFrame rows
    for i, dir_row in direction_matrix.items():
        for j, dir_vector in dir_row.items():
            distance = distance_matrix[i][j]
            rows.append([i, j, dir_vector, distance])
    # Create DataFrame
    df = pd.DataFrame(rows, columns=columns)
    return df
'''
def DirectionDistance2Df(direction_matrix,distance_matrix):
    rows = []
    columns = ['i', 'j', 'dir_vector', 'distance']

    # Iterate over the direction and distance matrices to construct DataFrame rows
    for i, dir_row in direction_matrix.items():
        for j, dir_vector in dir_row.items():
            distance = distance_matrix[i][j]
            rows.append([i, j, dir_vector, distance])
    # Create DataFrame
    df = pd.DataFrame(rows, columns=columns)
    return df


    # LATTICE

def GetLattice(grid,
               grid_size,
               bounding_box,
               save_dir_local):
    '''
        Output:
            lattice: graph -> graph object of the lattice
        Description:
            This function is used to get the lattice of the city, it is a graph object that contains the nodes and the edges of the city.
            It is used to compute the gradient and the curl of the city.
    '''
    dir_grid = SetGridDir(save_dir_local,grid_size)
    ## BUILD GRAPH OBJECT GRID
    cprint('Get Lattice','yellow')
    x = np.arange(bounding_box[0], bounding_box[2], grid_size)
    y = np.arange(bounding_box[1], bounding_box[3], grid_size)
    if os.path.isfile(os.path.join(dir_grid,str(round(grid_size,3)),"centroid_lattice.graphml")):
        cprint('{} ALREADY COMPUTED'.format(os.path.join(dir_grid,str(round(grid_size,3)),"centroid_lattice.graphml")),'yellow')
        lattice = nx.read_graphml(os.path.join(dir_grid,str(round(grid_size,3)),"centroid_lattice.graphml"))
        return lattice
    else:
        cprint('COMPUTING {}'.format(os.path.join(dir_grid,str(round(grid_size,3)),"centroid_lattice.graphml")),'yellow')
        lattice = nx.grid_2d_graph(len(x),len(y))
        node_positions = {(row['i'],row['j']): {'x': row['centroidx'],'y':row['centroidy']} for idx, row in grid.iterrows()}
        # Add position attributes to nodes
        nx.set_node_attributes(lattice, node_positions)
        for edge in lattice.edges():
            try:
                dx,dy = ProjCoordsTangentSpace(lattice.nodes[edge[1]]['x'],lattice.nodes[edge[1]]['y'],lattice.nodes[edge[0]]['x'],lattice.nodes[edge[0]]['y'])
                lattice[edge[0]][edge[1]]['dx'] = dx
                lattice[edge[0]][edge[1]]['dy'] = dy  
                lattice[edge[0]][edge[1]]['distance'] = hs.haversine((lattice.nodes[edge[0]]['y'],lattice.nodes[edge[0]]['x']),(lattice.nodes[edge[1]]['y'],lattice.nodes[edge[1]]['x']))
                lattice[edge[0]][edge[1]]['angle'] = np.arctan2(lattice[edge[0]][edge[1]]['dy'],lattice[edge[0]][edge[1]]['dx'])
                if not np.isnan(1/dx):
                    lattice[edge[0]][edge[1]]['d/dx'] = 1/dx  
                else:
                    lattice[edge[0]][edge[1]]['d/dx'] = np.inf
                if not np.isnan(1/dy):
                    lattice[edge[0]][edge[1]]['d/dy'] = 1/dy
                else:
                    lattice[edge[0]][edge[1]]['d/dy'] = np.inf
            except KeyError:
                pass
        ## SAVE GRID AND LATTICE
        return lattice
    
def GridIdx2OD(grid):
    '''
        Saves the origin destination in terms of the index column of the grid
    '''
    gridIdx2dest = defaultdict(int)
    for o in grid['index'].tolist():
        for d in grid['index'].tolist():
            gridIdx2dest[(o,d)] = 0    
    return gridIdx2dest

def ODGrid(gridIdx2dest,
           gridIdx2ij):
    '''
        Input:
            gridIdx2dest: dict -> {(i,j): number_people}
            gridIdx2ij: dict -> {index: (i,j)}
        Output:
    '''    
    orig = []
    dest = []
    number_people = []
    idxorig = []
    idxdest = []
    for k in gridIdx2dest.keys():
        orig.append(k[0])
        dest.append(k[1])
        number_people.append(gridIdx2dest[k])
        idxorig.append(gridIdx2ij[k[0]])
        idxdest.append(gridIdx2ij[k[1]])
    df = pd.DataFrame({'origin':orig,'destination':dest,'number_people':number_people,'(i,j)O':idxorig,'(i,j)D':idxdest})
    return df





























