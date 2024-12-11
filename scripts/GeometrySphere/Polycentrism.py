from tqdm import tqdm
import numpy as np
import numba
import pandas as pd
from numba import prange
from shapely.geometry import Point
import os
import logging

# QuadProg
from cvxopt import matrix, solvers
logger = logging.getLogger(__name__)

##--------------------------------------- GEOMETRIC FACTS ---------------------------------------##


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


#### FILTERS ####
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

def GetIndicesEdgeFromGrid(grid):
    """
        @params grid: pd.DataFrame -> ['index','population','relation_to_line']
        @description: Get the indices of the grid that are edges
    """
    assert 'relation_to_line' in grid.columns, 'GetIndicesEdgeFromGrid: grid must have a relation_to_line column in: {}'.format(grid.columns)
    IndexEdge = []
    for i in range(grid.shape[0]):
        if grid.iloc[i]['relation_to_line']=='edge':
            IndexEdge.append(i)
    return IndexEdge

def GetIndicesInsideFromGrid(grid):
    """
        @params grid: pd.DataFrame -> ['index','population','relation_to_line']
        @description: Get the indices of the grid that are inside
    """
    assert "position" in grid.columns, 'GetIndicesInsideFromGrid: grid must have a position column in: {}'.format(grid.columns)
    assert "population" in grid.columns, 'GetIndicesInsideFromGrid: grid must have a population column in: {}'.format(grid.columns)
    IndicesInside = [grid.loc[i]["index"] for i,row in grid.iterrows() if (row['position'] == 'inside' and row["population"] > 0)]
    assert len(IndicesInside)!=0, 'GetIndicesInsideFromGrid: The IndicesInside must not be empty'
    return IndicesInside



def MaskDistanceGivenIndices(df_distance,Indices):
    """
        @params df_distance: pd.DataFrame -> ['distance','direction vector','i','j']
        @params Indices: list -> [i1,i2,...,iN] indices of the grid for which extracting distances
    """
    maski = [i in Indices for i in df_distance['i']]
    dd = df_distance.loc[maski]
    maskj = [j in Indices for j in dd['j']]
    dd = dd.loc[maskj]
    assert np.array_equal(dd["i"].unique(), dd["j"].unique()), 'The indices must be the same'
    assert np.array_equal(dd["i"].unique(), Indices), 'The indices must be the same'    # Number of squares in the grid that are edges
    return dd['distance'].to_numpy(dtype = np.float64)

def MaskDistanceGridGivenIndices(df_distance,Indices,DfGridOrPot,case):
    """
        @params df_distance: pd.DataFrame -> ['distance','direction vector','i','j']
        @params Indices: list -> [i1,i2,...,iN] indices of the grid for which extracting distances
        @params Grid: pd.DataFrame -> ['index','population']
    """
    assert len(df_distance)!=0, 'The df_distance must not be empty'
    assert len(Indices)!=0, 'The Indices must not be empty'
    assert df_distance is not None, 'The df_distance must be provided'
    assert DfGridOrPot is not None, 'The Grid or Potential Dataframe must be provided'
    logger.info(f"MaskDistanceGridGivenIndices: #Indices: {len(Indices)}")
    D = MaskDistanceGivenIndices(df_distance,Indices)
    if case == 'Mass':
        assert 'population' in DfGridOrPot.columns, f'The Grid Dataframe must have a population in: {DfGridOrPot.columns}'
        S = np.array(DfGridOrPot.loc[DfGridOrPot['index'].isin(Indices)]['population'].values).astype(np.float64)
    if case == "Potential":
        assert 'V_out' in DfGridOrPot.columns, f'The Potential Dataframe must have a V_out in: {DfGridOrPot.columns}'
        S = np.array(DfGridOrPot.loc[DfGridOrPot['index'].isin(Indices)]['V_out'].values).astype(np.float64)
    return S,D

##--------------------------------------- UCI POLYCENTRISM ---------------------------------------##

def ComputeVmaxUCI(df_distance,Indices,method = 'Pereira',case = 'Potential'):
    """
        @params df_distance_inside: pd.DataFrame ['distance','direction vector','i','j'] with a mask just in the inside of the polygon-> NOTE: Stores the matrix in order in one dimension 'j' increases faster than 'i' index
        @params method: str -> 'Pereira' or 'CVXOPT' [Method to compute the Vmax]
        @params case: str -> 'Potential' or 'Mass' [Case to compute the Vmax]
        @description: Compute the Vmax for the UCI
        It is a quadratic programming problem that minimizes - \sum_{ij} D_ij x_i x_j subject to \sum_i x_i = 1
        NOTE: Usage:
            - df_distance is df_distance_edge in case of Pereira. Already Filtered
            - df_distance is df_distance_edge in case of CVXOPT. Will Look For Configurations that are in the whole inside of the polygon
        NOTE:
            The Smax is normalized \sum_i Smax_i = 1
    """
    if method == 'Pereira':
        DVecMax = MaskDistanceGivenIndices(df_distance,Indices)
        NumGridEdge = len(Indices)
        Smax = np.ones(NumGridEdge)*(1/NumGridEdge)#**2
        assert np.array_equal(df_distance["i"].unique(), df_distance["j"].unique()), 'The indices must be the same'
        return np.array(Smax).astype(np.float64),DVecMax
    if method == 'CVXOPT':
        d = int(np.sqrt(len(df_distance)))
        if d * d != len(df_distance):
            raise ValueError("The length of df_distance_inside is not a perfect square.")
        df_distance = df_distance["distance"].fillna(0)
        D = df_distance.to_numpy().reshape((d,d))    
        # Convert D to a CVXOPT matrix Minimize: xPx + qx
        P = matrix(-D)
        q = matrix(np.zeros((d,1)))

        # Equality constraint: sum(x) = 1
        A = matrix(np.ones((1,d)),(1,d))
        b = matrix(1.0)
        if case == "Potential":
            # No inequality constraints
            G = matrix(np.zeros((d, d)))
            h = matrix(np.zeros((d,1)))
            dims = {'l': 0, 'q': [], 's': []}
        if case == "Mass":
            # Inequality constraints: x_i >= 0
            G = matrix(-np.identity(d))
            h = matrix(np.zeros((d,1)))
            dims = {'l': G.size[0], 'q': [], 's': []}

        sol = solvers.qp(P, q, G = None, h = None, dims= dims,A = A, b = b)
        # Extract the solution
        Smaxi = np.array(sol['x'])    
        MaskIdx = [True if i in Smax[i]!=0 and Smax[j]!=0 else False for i in range(len(Smax)) for j in range(len(Smax))]
        DVecMax = df_distance.loc[MaskIdx]
    return Smaxi,DVecMax


####### PI #######
@numba.jit(['(float64[:], float64[:])'],parallel = True)
def ComputeJitV(Filtered_Potential_Normalized,Filtered_Distance):
    '''
        Input:
            Filtered_Potential: array of potential values [Pot_O,...,Pot_(Ngrids with non 0 potential)]
            Filtered_Distance: array of distances
        Output:
            V_in_PI: Average of Pot_i * Pot_j * Dist_ij.
            NOTE: I am putting the renormalization with the number of couples.
    '''
    V_in_PI = 0
#    Filtered_Potential = Filtered_Potential/np.sum(Filtered_Potential)    
    index_distance = 0
    for i in prange(len(Filtered_Potential_Normalized)):
        for j in prange(len(Filtered_Potential_Normalized)):
            # Remove Noise
            if Filtered_Potential_Normalized[i] < 1e-7 or Filtered_Potential_Normalized[j] < 1e-7:
                pass
            else:
                V_in_PI += np.abs(Filtered_Potential_Normalized[i])*np.abs(Filtered_Potential_Normalized[j])*Filtered_Distance[index_distance]        # Si*Sj*Dij
            index_distance += 1
#    V_in_PI = V_in_PI/len(Filtered_Potential_Normalized)**2
    return V_in_PI    

def CheckRightMappingComputationV(Df_Filtered_Potential_Normalized,Df_Filtered_Distance):
    """
        @params Df_Filtered_Potential_Normalized: pd.DataFrame ['index','V_out'] -> Potential values for the grid
        @params Df_Filtered_Distance: pd.DataFrame ['distance','direction vector','i','j'] -> NOTE: Stores the matrix in order in one dimension 'j' increases faster than 'i' index
        @description: Check the right mapping between the potential and the distance
        Controls that the indices of the filtered potential and distance are the same. So that ComputeJitV is correct.
    """
    index_distance = 0
    for i in range(len(Df_Filtered_Potential_Normalized)):
        for j in range(len(Df_Filtered_Potential_Normalized)):
            if i != j:
                Pidx = Df_Filtered_Potential_Normalized.iloc[i]["index"]
                Pjdx = Df_Filtered_Potential_Normalized.iloc[j]["index"]
                Didx = Df_Filtered_Distance.iloc[index_distance]["i"]
                Djdx = Df_Filtered_Distance.iloc[index_distance]["j"]
                assert Df_Filtered_Potential_Normalized.iloc[i]["index"] == Df_Filtered_Distance.iloc[index_distance]["i"], f'i: {i}, P.index: {Pidx} D.index: {Didx} The indices must be the same'
                assert Df_Filtered_Potential_Normalized.iloc[j]["index"] == Df_Filtered_Distance.iloc[index_distance]["j"], f'j: {j}, P.index: {Pjdx} D.index: {Djdx} The indices must be the same'
            index_distance += 1

def ComputePI(V,MaxV,verbose=True):
    if verbose:
        print('PI: ',1-V/MaxV)
    return 1 - V/MaxV

#### LC ####    
def LorenzCenters(potential,verbose =True):
    '''
        Input:
            Potential from grid.
        This function computes the indices of the centers in the linearized grid.
        We are using here the index column and not the double index.
        NOTE: F* is the number of non-centers. Similar 1 if monocentric, 0 if polycentric.
    '''
    min_pot = min(potential)
    # Shift the potential to be positive
    if min_pot < 0:
        potential = potential - min_pot
    else:
        pass
    x = np.arange(len(potential))
    dx = x[1] -x[0]
    # Step 1: Sort the potential and compute the sorting map
    sorted_indices = np.argsort(potential)
    # Step 2: Compute the cumulative distribution
    sorted_potential = potential[sorted_indices]
    cumulative = np.cumsum(sorted_potential)
    # Normalize the cumulative distribution (to make angle dy/dx) (dx = 1)
    cumulative_norm = cumulative/np.sum(sorted_potential)
    # Step 3: Determine the angle and delta index
    dy_over_dx = (cumulative_norm[-1] - cumulative_norm[-2])/dx
#    print('angle: ',angle)
    max_idx = len(cumulative_norm) +1
    y_max = cumulative_norm[-1]
    # NOTE: Count of Non-Centers
    Fstar = int(max_idx -y_max/dy_over_dx)
    assert Fstar <= len(cumulative), 'Fstar must be less than the length of the cumulative distribution'
    assert Fstar >= 0, 'Fstar must be greater than 0'
    # Step 4: Retrieve the indices based on the delta index and mapping
    result_indices = [sorted_indices[-i] for i in range(len(cumulative) - int(Fstar))]
    logger.info(f'Fstar: {Fstar}')
    return result_indices,dy_over_dx,cumulative,Fstar


def ComputeUCICase(df_distance,DfGridOrPot,IndicesInside,Vmax,case = 'Potential'):
    """
        @params df_distance: pd.DataFrame ['distance','direction vector','i','j'] -> NOTE: Stores the matrix in order in one dimension 'j' increases faster than 'i' index
        @params DfGridOrPot: pd.DataFrame ['index','population'] or ['index','V_out'] -> Potential values for the grid
        @params IndicesInside: Index of the grid for which extracting distances and potential
        @params Vmax: float [Maximum V]
        @params case: str -> 'Potential' or 'Mass' [Case to compute the UCI]
        @description: Compute the UCI for the given number of centers.
    """
    if case == 'Potential':
        # POTENTIAL
        logger.info("Compute V for Potential...")
        Si,Dij = MaskDistanceGridGivenIndices(df_distance,IndicesInside,DfGridOrPot,case = 'Potential')
        Si_Normalized = Si/np.sum(Si)
        assert np.sum(Si_Normalized)-1 < 0.000005, 'The sum of the potential on the edge must be 1'
        V = ComputeJitV(Si_Normalized,Dij)/2
        logger.info(f'Potential: #Inside: {len(Si)},#Inside: {len(Dij)},Vmax: {Vmax}, V: {V}')
        assert V <= Vmax, f'V must be less than Vmax: {Vmax}, V: {V}'
        PI = ComputePI(V,Vmax)
        result_indices,angle,cumulative,Fstar = LorenzCenters(np.array(Si_Normalized))
        LC = Fstar/len(cumulative)
        UCI = PI*LC
        logger.info(f"UCI Pot: LC: {LC}, PI: {PI}, UCI: {round(UCI,3)}")    
        return PI,LC,UCI,result_indices,angle,cumulative,Fstar
    if case == 'Mass':
        logger.info("Compute V for Mass...")
        SiMass,DijMass = MaskDistanceGridGivenIndices(df_distance,IndicesInside,DfGridOrPot,case = 'Mass')
        SiMass_Normalized = SiMass/np.sum(SiMass)
        assert np.sum(SiMass_Normalized)-1 < 0.000005, 'The sum of the potential on the edge must be 1'
        VMass = ComputeJitV(SiMass_Normalized,DijMass)/2
        logger.info(f'Mass: #Inside: {len(SiMass)},#Inside: {len(DijMass)},Vmax: {Vmax}, VMass: {VMass}')
        assert VMass <= Vmax, f'VMass must be less than Vmax: {Vmax}, V: {VMass}'
        PIM = ComputePI(VMass,Vmax)
        result_indicesM,angleM,cumulativeM,FstarM = LorenzCenters(np.array(SiMass_Normalized))
        LCM = FstarM/len(cumulativeM)
        UCIM = PIM*LCM
        logger.info(f"UCI Mass: LC: {LCM}, PI: {PIM}, UCI: {round(UCIM,3)}")
        return PIM,LCM,UCIM,result_indicesM,angleM,cumulativeM,FstarM


def ComputeUCI(df_distance,grid,Potentialdf,IndicesEdge,IndicesInside,Smax_i,Smax_i_Mass,Dmax_ij,Dmax_ij_Mass):
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
    # Filter The Distance Just For the Inside Case
    assert len(df_distance) == len(grid)**2, f'The number of distances {len(df_distance)} must be the same of square number of grids {len(grid)**2}, {len(grid)}'
    assert np.array_equal(df_distance["i"].unique(), df_distance["j"].unique()), 'The indices must be the same'
    assert np.array_equal(df_distance["i"].unique(), grid['index'].unique()), 'The indices must be the same'    # Number of squares in the grid that are edges
    assert np.array_equal(Potentialdf["index"].unique(), grid['index'].unique()), 'The indices must be the same'    # Number of squares in the grid that are edges
    # Compute The Indices Of The Edge If Not Provided
    if IndicesEdge is None:
        IndicesEdge = GetIndicesEdgeFromGrid(grid)
    if IndicesInside is None:
        assert 'position' in grid.columns, f'Computing IndicesInside: The grid must have a position column in: {grid.columns}'
        IndicesInside = GetIndicesInsideFromGrid(grid)   
    # Check Smax_i, Smax_i_Mass are not None
    if Smax_i is None:
        Smax_i,Dmax_ij = ComputeVmaxUCI(df_distance,IndicesEdge,method = 'Pereira',case = 'Potential')
    else:
        Dmax_ij = np.array(Dmax_ij).astype(np.float64)
        Smax_i = np.array(Smax_i).astype(np.float64)
    if Smax_i_Mass is None:
        Smax_i_Mass,Dmax_ij_Mass = ComputeVmaxUCI(df_distance,IndicesEdge,method = 'Pereira',case = 'Mass')
    else:
        Dmax_ij_Mass = np.array(Dmax_ij_Mass).astype(np.float64)
        Smax_i_Mass = np.array(Smax_i_Mass).astype(np.float64)
    logger.info("#Edges: {}, #Inside: {}".format(len(IndicesEdge),len(IndicesInside)))

    # Compute V For Internal Area
    GridInside = grid.loc[IndicesInside]
#    assert len(GridInside) == len(IndicesInside), f'ComputeUCI:The GridInside must have the same length {len(GridInside)} of the IndicesInside: {len(IndicesInside)}'
    # Plot Where We Compute the UCI
    PlotGridUsedComputationUCI(GridInside)   
    PlotGridUsedComputationUCIEdges(GridInside)
    
    # POTENTIAL
    assert len(Smax_i)**2 == len(Dmax_ij), f'ComputeUCI: The Smax_i must have the same length {len(Smax_i)**2} of Dmax_ij {len(Dmax_ij)}'
    Vmax = ComputeJitV(Smax_i,Dmax_ij)/2
    PI,LC,UCI,result_indices,angle,cumulative,Fstar = ComputeUCICase(df_distance,Potentialdf,IndicesInside,Vmax,case = 'Potential')
    
    # MASS
    assert len(Smax_i_Mass)**2 == len(Dmax_ij_Mass), f'ComputeUCI: The Smax_i_Mass must have the same length {len(Smax_i_Mass)**2} of Dmax_ij_Mass {len(Dmax_ij_Mass)}'
    VMassmax = ComputeJitV(Smax_i_Mass,Dmax_ij_Mass)/2
    PIM,LCM,UCIM,result_indicesM,angleM,cumulativeM,FstarM = ComputeUCICase(df_distance,grid,IndicesInside,VMassmax,case = 'Mass')

    return PI,LC,UCI,result_indices,cumulative,Fstar,PIM,LCM,UCIM,result_indicesM,angleM,cumulativeM,FstarM


##----------------- UCI MASS ---------------------------------##
def FilterDistancelGrid(grid,df_distance):
    IndicesEdge = GetIndicesEdgeFromGrid(grid)
    # Mask For Inside Grids That have population
    IndicesInside = [grid.loc[i]["index"] for i,row in grid.iterrows() if (row['position'] == 'inside' and row["population"] > 0)]
    ## MASK GRID ##
    GridInside = grid.loc[IndicesInside]
    SumMass = np.sum(GridInside['population'])
    df_distance_inside = df_distance.loc[df_distance['i'].isin(IndicesInside)]
    df_distance_inside = df_distance_inside.loc[df_distance_inside['j'].isin(IndicesInside)]
    return SumMass,df_distance_inside,GridInside,IndicesEdge,IndicesInside



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


##--------------------- PLOTTING ---------------------------------##
def PlotGridUsedComputationUCI(grid):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    grid.plot(ax = ax,column = 'position',legend = True)
    plt.savefig(os.path.join(os.environ['TRAFFIC_DIR'],'grid_used_UCI.png'))
    plt.close()

def PlotGridUsedComputationUCIEdges(grid):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    grid.plot(ax = ax,column = 'position',legend = True)
    plt.savefig(os.path.join(os.environ['TRAFFIC_DIR'],'grid_edges_used_UCI.png'))
    plt.close()
