from tqdm import tqdm
import numpy as np
import numba
import pandas as pd
from numba import prange
from shapely.geometry import Point
import os
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

def ComputeCM(grid,coords_center):
    '''
        This function computes the center of mass of the map
    '''
    return np.mean(grid['population'].to_numpy()[:,np.newaxis]*(np.column_stack((grid['centroidx'].values, grid['centroidy'].values)) - np.tile(coords_center,(len(grid),1))),axis = 0)


def polar_coordinates(point, center):
    # Calculate r    
    if isinstance(point,Point):
        point = np.array(point.coords)[0]
    else:
        pass
    y = point[1] - center[1]#ProjCoordsTangentSpace(center[0],center[1],point[0],point[1])
    x = point[0] - center[0]
    r = np.sqrt(x**2 + y**2)/1000
    theta = np.arctan(y/x)
    return r, theta


def ExtractCenterByPopulation(grid,verbose = False):
    '''
        This code defines a function ExtractCenterByPopulation that takes a pandas DataFrame grid as input.
        It asserts that the DataFrame has a column named 'population'.
        It then creates a copy of the 'population' column and finds the index of the maximum value in it. 
        It uses this index to extract the corresponding values from the 'centroidx' and 'centroidy' columns, 
        and returns them as a numpy array coords_center along with the index center_idx.
    '''
    assert 'population' in grid.columns
    population = grid['population'].copy()
    center_idx = np.argmax(population)
    coords_center = np.array([grid['centroidx'][center_idx],grid['centroidy'][center_idx]]) 
    if verbose:
        print("++++++++ Extract Center By Population ++++++++")
        print("Grid with Highest Population: ",center_idx)
        print('Center coords: ',coords_center)
    return coords_center,center_idx

def ExtractKeyFromValue(dict_,value):
    for key, val in dict_.items():
        if val == value:
            return key




##--------------------------------------- UCI POLYCENTRISM ---------------------------------------##
def GetIndexEdgePolygon(grid):
    IndexEdge = []
    for i in range(grid.shape[0]):
        if grid.iloc[i]['relation_to_line']=='edge':
            IndexEdge.append(i)
    return IndexEdge



def PrepareJitCompiledComputeV(df_distance,IndexEdge,SumPot,NumGridEdge,PotentialDataframe,case = 'Vmax',verbose = False):
    '''
        Input:
            1) df_distance: pd.DataFrame ['distance','direction vector','i','j'] -> NOTE: Stores the matrix in order in one dimension 'j' increases faster than 'i' index
            2) IndexEdge: Index of the grid for which extracting distances and potential
            3) SumPot: float [Total Sum of the Potential over the grid]
            4) NumGridEdge: int
            5) PotentialDataframe: pd.DataFrame ['index','V_out'] -> Potential values for the grid
        Output:
            1) distance_filtered: np.array distance_vector between ordered couples i,j in IndexEdge [For each couple of grids for which the potential != 0]
            2) PD['V_out']: np.array V_ij [For each couple of grids for which the potential != 0]
        NOTE:
            Usage: Compute the input for ComputeJitV
    '''
    if case == 'Vmax':
        maski = [i in IndexEdge for i in df_distance['i']]
        dd = df_distance.loc[maski]
        maskj = [j in IndexEdge for j in dd['j']]
        dd = dd.loc[maskj]        
        result_vector = np.ones(NumGridEdge)*(SumPot/NumGridEdge)**2
        return np.array(result_vector).astype(np.float32),dd['distance'].to_numpy(dtype = np.float32)
    else:
        maski = [i in IndexEdge for i in df_distance['i']]
        dd = df_distance.loc[maski]
        maskj = [j in IndexEdge for j in dd['j']]
        dd = dd.loc[maskj]
        PD = PotentialDataframe.loc[PotentialDataframe['index'].isin(IndexEdge)]
#        result_vector = []
#        for i, value in enumerate(PD['V_out'].values):
#            for j,value2 in enumerate(PD['V_out'].values):
#                if i <=j:
#                    result_vector.append(value*value2)
#                    if verbose:
#                        print('i: ',i,'\tj: ',j,'\tPD[i]: ',value,'\tPD[j]: ',value2)
#                        print('index i: ',PD['index'].values[i],'\tindex j: ',PD['index'].values[j])

#        for _, value in enumerate(PD['V_out'].values):
#            result_vector.extend([value] * (len(PD)))
        return np.array(PD['V_out'].values).astype(np.float32),dd['distance'].to_numpy(dtype = np.float32)

@numba.jit(['(float32[:], float32[:])'],parallel = True)
def ComputeJitV(Filtered_Potential,Filtered_Distance):
    '''
        Input:
            Filtered_Potential: array of potential values [Pot_O,...,Pot_(Ngrids with non 0 potential)]
            Filtered_Distance: array of distances
        Output:
            V_in_PI: Average of Pot_i * Pot_j * Dist_ij.
            NOTE: I am putting the renormalization with the number of couples.
    '''
    V_in_PI = 0
    Filtered_Potential = Filtered_Potential/np.sum(Filtered_Potential)
    index_distance = 0
    for i in prange(len(Filtered_Potential)):
        for j in prange(len(Filtered_Potential)):
            V_in_PI += Filtered_Potential[i]*Filtered_Potential[j]*Filtered_Distance[index_distance]
            index_distance += 1
    V_in_PI = V_in_PI/len(Filtered_Potential)**2
    return V_in_PI    


#@numba.jit(parallel = True)
#def ComputeJitV(distance_vector,potential_vector):
#    if len(distance_vector) != len(potential_vector):
#        print('distance_vector:',len(distance_vector))
#        print('potential_vector:',len(potential_vector))
#        raise ValueError('distance_vector and potential_vector must have the same length')
#    return np.sum(distance_vector*potential_vector)    

def ComputePI(V,MaxV,verbose=True):
    if verbose:
        print('PI: ',1-V/MaxV)
    return 1 - V/MaxV

def LaunchComputationPI(df_distance,grid,SumPot,NumGridEdge,PotentialDataframe,verbose = True):
    '''
        Computes maximum value for the PI -> Vmax (considering just the edges of the cartography)
        Computes the value for the PI -> V (considering all the points of the cartography with
                                            1) grid['population']>0
                                            2) PotentialDataframe['V_out']>0)
        Returns the PI
    '''
    Filtered_Potential,Filtered_Distance = PrepareJitCompiledComputeV(df_distance,GetIndexEdgePolygon(grid),SumPot,NumGridEdge,PotentialDataframe,case = 'Vmax')
    if verbose:
        print('Filtered Potential: ',len(Filtered_Potential))
        print('Filtered Distance: ',len(Filtered_Distance))
        print('N**2: ',len(Filtered_Potential)*(len(Filtered_Potential)))
    Vmax = ComputeJitV(Filtered_Potential,Filtered_Distance)/2
    PotentialDataframeMass = PotentialDataframe.loc[grid['population']>0]
    Filtered_Potential,Filtered_Distance = PrepareJitCompiledComputeV(df_distance,PotentialDataframeMass.loc[PotentialDataframeMass['V_out']>0]['index'].values,SumPot,NumGridEdge,PotentialDataframe,case = 'V')
    V = ComputeJitV(Filtered_Potential,Filtered_Distance)/2
    if verbose:
        print('Vmax: ',Vmax,'V: ',V)
    return ComputePI(V,Vmax)
    
def LorenzCenters(potential,verbose =True):
    '''
        Input:
            Potential from grid.
        This function computes the indices of the centers in the linearized grid.
        We are using here the index column and not the double index.
    '''
    # Step 1: Sort the potential and compute the sorting map
    sorted_indices = np.argsort(potential)
    # Step 2: Compute the cumulative distribution
    sorted_potential = potential[sorted_indices]
    cumulative = np.cumsum(sorted_potential)
    # Step 3: Determine the angle and delta index
    angle = cumulative[-1] - cumulative[-2]
#    print('angle: ',angle)
    Fstar = int(len(cumulative) +1 -cumulative[-1]/angle)
    # Step 4: Retrieve the indices based on the delta index and mapping
    result_indices = [sorted_indices[-i] for i in range(len(cumulative) - int(Fstar))]
    cumulative = cumulative/np.sum(sorted_potential)
    if verbose:
        print("*********** LORENZ CURVE ************")
        print('cumulative: ',cumulative)
        print('Fstar: ',Fstar)
        print('index: ',int(Fstar*len(cumulative)))
        print("*************************************")
    return result_indices,angle,cumulative,Fstar

def ComputeUCI(grid,PotentialDataframe,df_distance,verbose = True):
    '''
        Input:
            InfoConfigurationPolicentricity: dictionary {'grid': geopandas grid, 'potential': potential dataframe}
            num_peaks: int -> number of peaks (centers)
        Description:
            Compute the UCI for the given number of centers.
            NOTE: 
                The UCI is computed just on the fraction of Cells that are inside the geometry.
                In particular the Lorenz Centers.
        
    '''
    SumPot = PotentialDataframe['V_out'].sum()
    NumGridEdge = grid[grid['relation_to_line']=='edge'].shape[0]
    PI = LaunchComputationPI(df_distance,grid,SumPot,NumGridEdge,PotentialDataframe)
    MaskOutside = [True if (row['position'] == 'outside' or row['position'] == 'edge') else False for i,row in grid.iterrows()]
    PotentialFiltered = [PotentialDataframe.iloc[i]['V_out'] if MaskOutside[i] else 0 for i in range(len(MaskOutside))] 
    result_indices,angle,cumulative,Fstar = LorenzCenters(np.array(PotentialFiltered))
    LC = Fstar/len(cumulative)
    UCI = PI*LC
    if verbose:
        print('*********** COMPUTE UCI ************')
        print('Sum Potential: ',SumPot)
        print('Number of Edges boundary: ',NumGridEdge)
        print('LC: ',LC,'PI: ',PI,'UCI: ',UCI)
        print('*********** END UCI ************')
    return PI,LC,UCI,result_indices,angle,cumulative,Fstar

##------------------------------- CONFIGURATIONS -----------------------------------##    

def InitConfigurationPolicentricity(num_peaks,InfoConfigurationPolicentricity,grid,Tij):
    InfoConfigurationPolicentricity[num_peaks]['grid'] = grid.copy()
    InfoConfigurationPolicentricity[num_peaks]['Tij'] = Tij.copy()
    return InfoConfigurationPolicentricity

def StoreConfigurationsPolicentricity(new_population, Modified_Fluxes,New_Vector_Field,New_Potential_Dataframe,num_peaks,InfoConfigurationPolicentricity):
    InfoConfigurationPolicentricity[num_peaks]['grid']['population'] = new_population
    InfoConfigurationPolicentricity[num_peaks]['Tij']['number_people'] = Modified_Fluxes    #Modified_Fluxes_ipfn[0][0]        
    InfoConfigurationPolicentricity[num_peaks]['vector_field'] = New_Vector_Field
    InfoConfigurationPolicentricity[num_peaks]['potential'] = New_Potential_Dataframe
    return InfoConfigurationPolicentricity
 



def GetDirGrid(TRAFFIC_DIR,name,grid_size,num_peaks,cov,distribution_type,UCI):
    dir_grid = os.path.join(TRAFFIC_DIR,'data','carto',name,'OD')
    if not os.path.exists(dir_grid):
        os.mkdir(dir_grid)
    dir_grid = os.path.join(dir_grid,str(grid_size))
    if not os.path.exists(dir_grid):
        os.mkdir(dir_grid)
    dir_grid = os.path.join(dir_grid,str(num_peaks))
    if not os.path.exists(dir_grid):
        os.mkdir(dir_grid)
    dir_grid = os.path.join(dir_grid,str(cov))
    if not os.path.exists(dir_grid):
        os.mkdir(dir_grid)
    dir_grid = os.path.join(dir_grid,distribution_type)
    if not os.path.exists(dir_grid):
        os.mkdir(dir_grid)
    dir_grid = os.path.join(dir_grid,'UCI_{}'.format(round(UCI,3)))
    if not os.path.exists(dir_grid):
        os.mkdir(dir_grid)
    return dir_grid
