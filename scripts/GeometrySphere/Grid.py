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
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from GeometrySphere import ComputeAreaSquare
import networkx as nx
from PolygonSettings import *
import haversine as hs
from collections import defaultdict
import pandas as pd
import socket
if socket.gethostname()=='artemis.ist.berkeley.edu':
    sys.path.append(os.path.join('/home/alberto/LPSim','traffic_phase_transition','scripts','GenerationNet'))
else:
    sys.path.append(os.path.join(os.getenv('TRAFFIC_DIR'),'scripts','GenerationNet'))
from global_functions import *
from GeometrySphere import ProjCoordsTangentSpace
import logging
logger = logging.getLogger(__name__)

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
    logger.info('SAVING: {} '.format(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"grid.geojson")))
    grid.to_file(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"grid.geojson"), driver="GeoJSON")  
    return grid

def SaveLattice(save_dir_local,grid_size,lattice):
    '''
        Save the lattice
    '''
    SetGridDir(save_dir_local,grid_size)
    if not os.path.isfile(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"centroid_lattice.graphml")):
        logger.info('SAVING: {} '.format(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"centroid_lattice.graphml")))
        nx.write_graphml(lattice, os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"centroid_lattice.graphml"))  
    else:
        pass
    return lattice

def SaveDirectionDistanceMatrix(save_dir_local,grid_size,df_direction_distance_matrix):
    '''
        Save the direction matrix
    '''
    SetGridDir(save_dir_local,grid_size)
    if not os.path.isfile(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"direction_distance_matrix.csv")):
        logger.info('SAVING: {} '.format(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"direction_distance_matrix.csv")))
        df_direction_distance_matrix.to_csv(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"direction_distance_matrix.csv"))
    else:
        logger.info("Direction Matrix already exists in {}".format(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"direction_distance_matrix.csv")))
        pass
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
    dir_grid = SetGridDir(save_dir_local,grid_size)
    if os.path.isfile(os.path.join(dir_grid,str(round(grid_size,3)),"grid.geojson")):
        logger.info(f"Uploading grid from file {os.path.join(dir_grid,str(round(grid_size,3)),"grid.geojson")} ...")
        grid = gpd.read_file(os.path.join(dir_grid,str(round(grid_size,3)),"grid.geojson"))
        bbox = shp.geometry.box(*bounding_box)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=crs)
        x = np.arange(bounding_box[0], bounding_box[2], grid_size)
        y = np.arange(bounding_box[1], bounding_box[3], grid_size)

    else:
        logger.info(f"Computing grid with size {grid_size}, to be saved in {os.path.join(dir_grid,str(round(grid_size,3)),"grid.geojson")}...")
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
def ObtainDirectionMatrix(grid,save_dir_local,grid_size):
    """
        Input:
            grid: GeoDataFrame -> grid of points
            save_dir_local: str -> local directory to save the grid
            grid_size: float -> size of the grid
        Output:
            direction_distance_df: DataFrame -> [i,j,dir_vector,distance]
        Description:
            This function is used to obtain the direction matrix and the distance matrix of the grid.
            The direction matrix is a dictionary that contains the direction vector between two points of the grid.
            The distance matrix is a dictionary that contains the distance between two points of the grid.
            The direction_distance_df is a DataFrame that contains the information of the direction matrix and the distance matrix.
            The index of the grid is the index of the grid. Not the couple (i,j) that is useful for the definition of the gradient.
    """
    direction_distance_df,IsComputed = GetDirectionMatrix(save_dir_local,grid_size)
    if IsComputed:
        return direction_distance_df
    else:
        direction_matrix,distance_matrix = ComputeDirectionMatrix(grid)
        direction_distance_df = DirectionDistance2Df(direction_matrix,distance_matrix)
        SaveDirectionDistanceMatrix(save_dir_local,grid_size,direction_distance_df)
        return direction_distance_df
    
def GetDirectionMatrix(save_dir_local,grid_size):
    if os.path.isfile(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),"direction_distance_matrix.csv")):
        logger.info(f"Uploading direction matrix from: {os.path.join(save_dir_local,'grid',str(round(grid_size,3)),'direction_distance_matrix.csv')} ...")
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
    logger.info("Computing Direction Matrix ...")
    t0 = time.time()
    direction_matrix = {i: {j: np.array([grid.iloc[j]['centroidx']-grid.iloc[i]['centroidx'],grid.iloc[j]['centroidy']-grid.iloc[i]['centroidy']])/np.linalg.norm(np.array([grid.iloc[j]['centroidx']-grid.iloc[i]['centroidx'],grid.iloc[j]['centroidy']-grid.iloc[i]['centroidy']])) for j in range(len(grid))} for i in range(len(grid))}
    t1 = time.time()
    logger.info("Time to compute Direction Matrix: {}".format(t1-t0))
    logger.info("Size direction Matrix: {}".format(len(direction_matrix)))
    t0 = time.time()
    distance_matrix = {i: {j: hs.haversine((grid.iloc[i]['centroidy'],grid.iloc[i]['centroidx']),(grid.iloc[j]['centroidy'],grid.iloc[j]['centroidx'])) for j in range(len(grid))} for i in range(len(grid))}
    t1 = time.time()
    logger.info("Time to compute Distance Matrix: {}".format(t1-t0))
    logger.info("Size distance Matrix: {}".format(len(distance_matrix)))
    return direction_matrix,distance_matrix 

def DirectionDistance2Df(direction_matrix,distance_matrix,verbose = True):
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
    x = np.arange(bounding_box[0], bounding_box[2], grid_size)
    y = np.arange(bounding_box[1], bounding_box[3], grid_size)
    if os.path.isfile(os.path.join(dir_grid,str(round(grid_size,3)),"centroid_lattice.graphml")):
        logger.info(f"Uploading lattice from: {os.path.join(dir_grid,str(round(grid_size,3)),"centroid_lattice.graphml")}...")
        lattice = nx.read_graphml(os.path.join(dir_grid,str(round(grid_size,3)),"centroid_lattice.graphml"))
        return lattice
    else:
        logger.info(f"Computing lattice to be stored in: {os.path.join(dir_grid,str(round(grid_size,3)),"centroid_lattice.graphml")}...")
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
    logger.info("Init GridIdx2OD: {(Ogrid, Dgrid): Flux = 0} ...")
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
             ODGrid DataFrame: [origin,destination,number_people,(i,j)O,(i,j)D]...
                    Output:
    ''' 
    logger.info("Computing Fluxes in the Grid ...")   
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

##---------------------- INTERIOR AND BOUNDARY ----------------------
def PlotBoundaryLines(boundary_lines,city):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i,boundary_line in enumerate(boundary_lines):
        x, y = boundary_line.xy
        ax.plot(x, y, color=colors[i], alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2,label = f"Boundary {i}")
    ax.set_title(f"{city} Boundary Lines")
    ax.legend()
    plt.savefig(os.path.join(os.environ["TRAFFIC_DIR"],f"{city}_Lines.png"))

def PlotBoundariesHull(grid,city):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    grid.plot(column = 'position',legend = True,ax = ax)
    ax.set_title(f"{city} Position to Boundaries")
    ax.legend()
    plt.savefig(os.path.join(os.environ["TRAFFIC_DIR"],f"{city}_position.png"))
    plt.close()
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    grid.plot(column = 'relation_to_line',legend = True,ax = ax)
    ax.set_title(f"{city} Boundaries")
    ax.legend()
    plt.savefig(os.path.join(os.environ["TRAFFIC_DIR"],f"{city}_Boundaries.png"))
    plt.close()

def GetBoundariesInterior(grid,gdf_polygons,city):
    """
        Input:
            grid: GeoDataFrame -> grid of points
            SFO_obj: object -> object containing the boundaries of the city
        Output:
            grid: GeoDataFrame -> grid of points with the position and relation to the line
        Description:
            This function is used to get the position of the grid with respect to the boundaries of
            the city. The position can be inside, outside or edge. The relation to the line can be edge or not_edge.
            If There are connected Components Then Something Must Be Done.
    """
    crs_grid = grid.crs
    crs_gdf_polygons = gdf_polygons.crs
    if crs_grid != crs_gdf_polygons:
        grid = grid.to_crs(crs_gdf_polygons)    
    boundary = gpd.overlay(gdf_polygons, gdf_polygons, how='union',keep_geom_type=False).unary_union
    convex_hull = gdf_polygons.unary_union.convex_hull

    # CREATE BOUNDARY LINE
    if isinstance(boundary, Polygon):
        logger.info(f"Boundaries and Interior Polygon Case for {city} ...")
        boundary_line = LineString(boundary.exterior.coords)
#        grid['relation_to_line'] = grid.geometry.apply(lambda poly: 'edge' if boundary_line.crosses(poly) else 'not_edge')
        grid['relation_to_line'] = grid.geometry.apply(lambda poly: 'edge' if convex_hull.boundary.crosses(poly) else 'not_edge')        
#        grid['position'] = grid.geometry.apply(lambda x: 'inside' if x.within(boundary) else ('edge' if x.touches(boundary) else 'outside'))
        grid['position'] = grid.apply(lambda x: 'inside' if (x["geometry"].within(boundary) or x["geometry"].intersects(boundary_line)) else 'outside',axis = 1)
        PlotBoundaryLines([boundary_line],city)
        PlotBoundariesHull(grid,city)
    elif isinstance(boundary, MultiPolygon):
        logger.info(f"Boundaries and Interior MultiPolygon Case for {city} ...")
        grid['relation_to_line'] = grid.geometry.apply(lambda poly: 'edge' if convex_hull.boundary.crosses(poly) else 'not_edge')        
        grid['position'] = grid.apply(lambda x: 'inside' if (x["geometry"].within(boundary) or x["geometry"].intersects(convex_hull)) else 'outside',axis = 1)
        PlotBoundariesHull(grid,city)
#        boundary_lines = [LineString(polygon.exterior.coords) for polygon in boundary.geoms]
#        boundaries = [polygon for polygon in boundary.geoms]
#        PlotBoundaryLines(boundary_lines,city)
#        for i,boundary_line in enumerate(boundary_lines):
#            grid[f'position_{i}'] = grid.geometry.apply(lambda x: 'inside' if x.within(boundary) else ('edge' if x.touches(boundary) else 'outside'))            
#            grid[f'relation_to_line_{i}'] = grid.geometry.apply(lambda poly: 'edge' if boundary_line.crosses(poly) else 'not_edge')
#            grid[f'position_{i}'] = grid.apply(lambda x: 'inside' if (x["geometry"].within(boundaries[i]) or x["geometry"].intersects(boundary_line)) else 'outside',axis = 1)
#            logger.info(f"Is Inside in Boundary {i}: {len([True for x in grid[f'position_{i}'] if x == 'inside'])} ...")
#        InsideEdge2Bool = {"inside":True,"outside":False,"edge":True,"not_edge":False}    
#        Bool2Edge = {True:"edge",False:"not_edge"}
#        Bool2Inside = {True:"inside",False:"outside"}    
#        for i in range(len(boundary_lines)-1):
#            if i == 0:
#                new_col = np.logical_or(grid[f'position_{i}'].apply(lambda x: InsideEdge2Bool[x]),grid[f'position_{i+1}'].apply(lambda x: InsideEdge2Bool[x]))
#                new_col1 = np.logical_or(grid[f'relation_to_line_{i}'].apply(lambda x: InsideEdge2Bool[x]),grid[f'relation_to_line_{i+1}'].apply(lambda x: InsideEdge2Bool[x]))
#            else:
#                new_col = np.logical_or(new_col,grid[f'position_{i}'].apply(lambda x: InsideEdge2Bool[x]))
#                new_col1 = np.logical_or(new_col1,grid[f'relation_to_line_{i}'].apply(lambda x: InsideEdge2Bool[x]))
#        grid['position'] = [Bool2Inside[bv] for bv in new_col] 
#        grid['relation_to_line'] = [Bool2Edge[bv] for bv in new_col1]
#        for i in range(len(boundary_lines)):
#            grid.drop(columns=[f'position_{i}', f'relation_to_line_{i}'], inplace=True)    
    grid = grid.to_crs(crs_grid)            
    logger.info(f"{city} Grid.columns\n {grid.columns}")
    return grid


def GetLargestConnectedComponentPolygons(gdf):
    """
        Input:
            gdf: GeoDataFrame -> GeoDataFrame containing polygons
        Output:
            largest_component: Polygon -> the largest connected component
        Description:
            This function is used to identify the largest connected component of a GeoDataFrame containing polygons.
            The function also prints the number of connected components
    """
    # Perform unary union to merge all geometries
    merged_geometry = shp.ops.unary_union(gdf.geometry)
    
    # Identify connected components
    if isinstance(merged_geometry, MultiPolygon):
        connected_components = list(merged_geometry.geoms)
    else:
        connected_components = [merged_geometry]
    
    # Calculate the area of each connected component and select the largest
    largest_component = max(connected_components, key=lambda geom: geom.area)
    
    # Check if there are more than one connected components
    num_connected_components = len(connected_components)
    if num_connected_components > 1:
        print(f"There are {num_connected_components} connected components.")
    else:
        print("There is only one connected component.")
    
    return largest_component
























