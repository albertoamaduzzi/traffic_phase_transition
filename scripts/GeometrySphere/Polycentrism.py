from tqdm import tqdm
import numpy as np
import numba
import pandas as pd

##--------------------------------------- GEOMETRIC FACTS ---------------------------------------##

def FilterPopulation(grid):
    '''
        Filters population and add reshape population.
        The indices are a subsample of 0,...,Ngridx*Ngridy -> 0 is not guaranteed (and most probably) to be present in this subset
    '''
    assert 'population' in grid.columns, 'The grid does not have a population column'
    assert 'centroidx' in grid.columns, 'The grid does not have a centroidx column'
    assert 'centroidy' in grid.columns, 'The grid does not have a centroidy column'
    grid['reshaped_population']= np.zeros(len(grid))
    filtered_grid = grid[grid['population'] > 0]

    # Extract the population, centroidx, and centroidy values
    population = filtered_grid['population'].values
    centroidx = filtered_grid['centroidx'].values
    centroidy = filtered_grid['centroidy'].values
    index = filtered_grid.index
    return population,centroidx,centroidy,index,filtered_grid

def ExtractCenterByPopulation(grid):
    assert 'population' in grid.columns
    population = grid['population'].copy()
    center_idx = np.argmax(population)
    coords_center = np.array([grid['centroidx'][center_idx],grid['centroidy'][center_idx]]) 
    return coords_center

def ComputeCM(grid,coords_center):
    '''
        This function computes the center of mass of the map
    '''
    return np.mean(grid['population'].to_numpy()[:,np.newaxis]*(np.column_stack((grid['centroidx'].values, grid['centroidy'].values)) - np.tile(coords_center,(len(grid),1))),axis = 0)


def polar_coordinates(point, center, r_step, theta_step):
    # Calculate r    
    point = np.array(point.coords)[0]
    y = point[1] - center[1]#ProjCoordsTangentSpace(center[0],center[1],point[0],point[1])
    x = point[0] - center[0]
    r = np.sqrt(x**2 + y**2)/1000
    # Calculate θ
    theta = np.arctan(y/x)
    
    # Adjust θ to be positive
#    theta = (theta + 2 * np.pi) % (2 * np.pi)
    # Round r and θ to the nearest steps
#    r = round(r / r_step) * r_step
#    theta = round(theta / theta_step) * theta_step
    return r, theta




##--------------------------------------- UCI POLYCENTRISM ---------------------------------------##
def GetIndexEdgePolygon(grid):
    IndexEdge = []
    for i in range(grid.shape[0]):
        if grid.iloc[i]['relation_to_line']=='edge':
            IndexEdge.append(i)
    return IndexEdge

def PrepareJitCompiledComputeV(df_distance,IndexEdge,SumPot,NumGridEdge,PotentialDataframe,case = 'Vmax'):
    '''
        This function prepares the data for the jit compiled function ComputeV
    '''
    if case == 'Vmax':
        maski = [i in IndexEdge for i in df_distance['i']]
        dd = df_distance.loc[maski]
        maskj = [j in IndexEdge for j in dd['j']]
        dd = dd.loc[maskj]        
        result_vector = np.ones(len(dd))*SumPot/NumGridEdge
    else:
        maski = [i in IndexEdge for i in df_distance['i']]
        dd = df_distance.loc[maski]
        maskj = [j in IndexEdge for j in dd['j']]
        dd = dd.loc[maskj]
        PD = PotentialDataframe.loc[PotentialDataframe['index'].isin(IndexEdge)]
        result_vector = []
        for _, value in enumerate(PD['V_out'].values):
            result_vector.extend([value] * (len(PD)))

    return dd['distance'].to_numpy(dtype = np.float32),np.array(result_vector).astype(np.float32)
@numba.jit(parallel = True)
def ComputeJitV(distance_vector,potential_vector):
    if len(distance_vector) != len(potential_vector):
        print('distance_vector:',len(distance_vector))
        print('potential_vector:',len(potential_vector))
        raise ValueError('distance_vector and potential_vector must have the same length')
    return np.sum(distance_vector*potential_vector)    

def ComputePI(V,MaxV):
    return 1 - V/MaxV

