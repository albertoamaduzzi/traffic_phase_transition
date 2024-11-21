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

def ComputeVmaxUCI(df_distance_inside):
    """
        @params df_distance_inside: pd.DataFrame ['distance','direction vector','i','j'] with a mask just in the inside of the polygon-> NOTE: Stores the matrix in order in one dimension 'j' increases faster than 'i' index
        @description: Compute the Vmax for the UCI
        It is a quadratic programming problem that minimizes - \sum_{ij} D_ij x_i x_j subject to \sum_i x_i = 1
    """

    d = int(np.sqrt(len(df_distance_inside)))
    if d * d != len(df_distance_inside):
        raise ValueError("The length of df_distance_inside is not a perfect square.")
    df_distance_inside = df_distance_inside["distance"].fillna(0)
    D = df_distance_inside.to_numpy().reshape((d,d))    
    # Convert D to a CVXOPT matrix Minimize: xPx + qx
    P = matrix(-D)
    q = matrix(np.zeros((d,1)))

    # Equality constraint: sum(x) = 1
    A = matrix(np.ones((1,d)),(1,d))
    b = matrix(1.0)
    # No inequality constraints
    G = matrix(np.zeros((d, d)))
    h = matrix(np.zeros((d,1)))
    logger.info(f'Rank(P): {np.linalg.matrix_rank(P)}, Rank(A): {np.linalg.matrix_rank(A)}, Rank(G): {np.linalg.matrix_rank(G)}')
    dims = {'l': 0, 'q': [], 's': []}
    if np.linalg.matrix_rank(np.vstack([P, A, G])) < P.size[1]:
        raise ValueError("Rank([P; A; G]) < n")
    
    # Check the rank of A and [P; A; G]
    # Solve the quadratic programming problem
    sol = solvers.qp(P, q, G = None, h = None, dims= dims,A = A, b = b)
    # Extract the solution
    Smaxi = np.array(sol['x'])    
    return Smaxi

def ComputeVMaxUCIMass(df_distance_inside):
    """
        @params df_distance_inside: pd.DataFrame ['distance','direction vector','i','j'] with a mask just in the inside of the polygon-> NOTE: Stores the matrix in order in one dimension 'j' increases faster than 'i' index
        @description: Compute the Vmax for the UCI
        It is a quadratic programming problem that minimizes - \sum_{ij} D_ij x_i x_j subject to \sum_i x_i = 1 and x_i >= 0

    """
    d = int(np.sqrt(len(df_distance_inside)))
    if d * d != len(df_distance_inside):
        raise ValueError("The length of df_distance_inside is not a perfect square.")
    df_distance_inside = df_distance_inside["distance"].fillna(0)
    D = df_distance_inside.to_numpy().reshape((d,d))    
    # Convert D to a CVXOPT matrix
    P = matrix(-D)
    q = matrix(np.zeros((d,1)))

    # Equality constraint: sum(x) = 1 (Repeated d times to circumvent an error on the matrix rank given by the solver)
    A = matrix(np.ones((1,d)),(1,d))
    b = matrix(1.0)

    # Inequality constraints: x_i >= 0
    G = matrix(-np.identity(d))
    h = matrix(np.zeros((d,1)))
    if np.linalg.matrix_rank(np.vstack([P, A, G])) < P.size[1]:
        raise ValueError("Rank([P; A; G]) < n")
    dims = {'l': G.size[0], 'q': [], 's': []}
    # Solve the quadratic programming problem
    sol = solvers.qp(P, q, G = G, h = h,dims = dims, A = A, b = b)
    # Extract the solution
    Smaxi = np.array(sol['x'])    
    return Smaxi

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
        Smax = np.ones(NumGridEdge)*(1/NumGridEdge)#**2
        assert np.array_equal(dd["i"].unique(), dd["j"].unique()), 'The indices must be the same'
        return np.array(Smax).astype(np.float64),dd['distance'].to_numpy(dtype = np.float64)
    else:
        maski = [i in IndexEdge for i in df_distance['i']]
        dd = df_distance.loc[maski]
        maskj = [j in IndexEdge for j in dd['j']]
        dd = dd.loc[maskj]
        PD = PotentialDataframe.loc[PotentialDataframe['index'].isin(IndexEdge)]
        assert np.array_equal(dd["i"].unique(), dd["j"].unique()), 'The indices must be the same'
        assert np.array_equal(dd["i"].unique(), PD['index'].unique()), 'The indices must be the same'    # Number of squares in the grid that are edges
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
        return np.array(PD['V_out'].values).astype(np.float64),dd['distance'].to_numpy(dtype = np.float64)

def PrepareJitCompiledComputeMass(df_distance,IndexEdge,SumPot,NumGridEdge,Grid,case = 'Vmax',verbose = False):
    '''
        Input:
            1) df_distance: pd.DataFrame ['distance','direction vector','i','j'] -> NOTE: Stores the matrix in order in one dimension 'j' increases faster than 'i' index
            2) IndexEdge: Index of the grid for which extracting distances and potential
            3) SumPot: float [Total Sum of the Potential over the grid]
            4) NumGridEdge: int
            5) Grid: pd.DataFrame ['index','population'] -> Potential values for the grid
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
        Smax = np.ones(NumGridEdge)*(1/NumGridEdge)#**2
        assert np.array_equal(dd["i"].unique(), dd["j"].unique()), 'The indices must be the same'
        return np.array(Smax).astype(np.float64),dd['distance'].to_numpy(dtype = np.float64)
    else:
        maski = [i in IndexEdge for i in df_distance['i']]
        dd = df_distance.loc[maski]
        maskj = [j in IndexEdge for j in dd['j']]
        dd = dd.loc[maskj]
        Grid = Grid.loc[Grid['index'].isin(IndexEdge)]
        assert np.array_equal(dd["i"].unique(), dd["j"].unique()), 'The indices must be the same'
        assert np.array_equal(dd["i"].unique(), Grid['index'].unique()), 'The indices must be the same'    # Number of squares in the grid that are edges
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
        return np.array(Grid['population'].values).astype(np.float64),dd['distance'].to_numpy(dtype = np.float64)



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
            V_in_PI += np.abs(Filtered_Potential_Normalized[i])*np.abs(Filtered_Potential_Normalized[j])*Filtered_Distance[index_distance]        # Si*Sj*Dij
            index_distance += 1
#    V_in_PI = V_in_PI/len(Filtered_Potential_Normalized)**2
    return V_in_PI    

def CheckRightMappingComputationV(Df_Filtered_Potential_Normalized,Df_Filtered_Distance):
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

def FilterDistancePotentialGrid(df_distance,grid,PotentialDataframe):
    IndicesEdge = GetIndexEdgePolygon(grid)
    # Mask For Inside Grids That have population
    IndicesInside = [grid.loc[i]["index"] for i,row in grid.iterrows() if (row['position'] == 'inside' and row["population"] > 0)]
    ## MASK GRID ##
    GridInside = grid.loc[IndicesInside]
    # Total Potential Copmuted for the generated OD
#    SumPot = PotentialDataframe['V_out'].sum()
    ## MASK POTENTIAL ##
    PotentialInside = PotentialDataframe.loc[PotentialDataframe['index'].isin(IndicesInside)]
    SumPot = np.sum(np.abs(PotentialInside['V_out']))
    ## MASK DISTANCE MATRIX ##
    df_distance_inside = df_distance.loc[df_distance['i'].isin(IndicesInside)]
    df_distance_inside = df_distance_inside.loc[df_distance_inside['j'].isin(IndicesInside)]
    return df_distance_inside,GridInside,PotentialInside,SumPot,IndicesEdge,IndicesInside

#def FillDiiWithZero(df_distance):
#    """
#        Fill the diagonal of the distance matrix with 0.
#    """
#    df_distance['distance'] = df_distance.apply(lambda row: 0 if row['i'] == row['j'] else row['distance'], axis=1)
#    return df_distance

    
def ComputeUCI(df_distance_inside,GridInside,PotentialInside,SumPot,IndicesEdge,IndicesInside,grid,df_distance,Smax_i,Smax_i_Mass):
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
    logger.info("Extract: Edges, Inside Indices, corresponding Potential, Grid and Distances ...")
#    df_distance_inside = FillDiiWithZero(df_distance_inside)
    PlotGridUsedComputationUCI(GridInside)   
    PlotGridUsedComputationUCIEdges(grid.loc[grid["index"].isin(IndicesEdge)]) 
    # Filter The Distance Just For the Inside Case
    assert len(df_distance_inside) == len(GridInside)**2, f'The number of distances {len(df_distance_inside)} must be the same of square number of grids {len(GridInside)**2}, {len(GridInside)}'
    assert np.array_equal(df_distance_inside["i"].unique(), df_distance_inside["j"].unique()), 'The indices must be the same'
    assert np.array_equal(df_distance_inside["i"].unique(), GridInside['index'].unique()), 'The indices must be the same'    # Number of squares in the grid that are edges
    assert np.array_equal(PotentialInside["index"].unique(), GridInside['index'].unique()), 'The indices must be the same'    # Number of squares in the grid that are edges
    NumGridEdge = grid[grid['relation_to_line']=='edge'].shape[0]
#    CheckRightMappingComputationV(PotentialInside,df_distance_inside)
#    PI = LaunchComputationPI(df_distance,grid,SumPot,NumGridEdge,PotentialDataframe)
#    MaskOutside = [True if (row['position'] == 'outside' or row['position'] == 'edge') else False for i,row in grid.iterrows()]
#    PotentialFiltered = [PotentialDataframe.iloc[i]['V_out'] if not MaskOutside[i] else 0 for i in range(len(MaskOutside))] 
#    result_indices,angle,cumulative,Fstar = LorenzCenters(np.array(PotentialFiltered))
    # Compute Vmax 
    # Smax_i: (Vtot/N_edges,...,Vtot/N_edges)
    # Dmax_ij: (distance between the edges)    
    logger.info("Compute Vmax ...")
    _,Dmax_ij = PrepareJitCompiledComputeV(df_distance,IndicesEdge,SumPot,NumGridEdge,PotentialInside,case = 'Vmax')
#    Smax_i = np.abs(Smax_i)
    assert np.sum(Smax_i)-1 < 0.000005, 'The sum of the potential on the edge must be 1'
    Vmax = ComputeJitV(Smax_i,Dmax_ij)/2
    # Compute UCI per Mass
    # Copmute V
    # Si: Potential values for the grid with population > 0 and Potential > 0 [i is the index of the grid]
    # D_ij: Distance between the grids with population > 0 and Potential > 0
    logger.info("Compute V ...")
    Si,D_ij = PrepareJitCompiledComputeV(df_distance_inside,IndicesInside,SumPot,NumGridEdge,PotentialInside,case = 'V')
#    Si = np.array([i if i>0 else 0 for i in Si])
#    Si = np.abs(Si)
    Si_Normalized = Si/np.sum(Si)
    assert np.sum(Si_Normalized)-1 < 0.000005, 'The sum of the potential on the edge must be 1'
    V = ComputeJitV(Si_Normalized,D_ij)/2
    # Compute UCI Mass
    logger.info(f'Potential, #edges: {len(Smax_i)}, #Inside: {len(Si)}')
    logger.info(f'Potential, #Distances edges: {len(Dmax_ij)}, #Inside: {len(D_ij)}')
    logger.info(f'Potential, Vmax: {Vmax}, V: {V}')
    assert V <= Vmax, 'V must be less than Vmax'
    PI = ComputePI(V,Vmax)
    result_indices,angle,cumulative,Fstar = LorenzCenters(np.array(PotentialInside['V_out']))
    LC = Fstar/len(cumulative)
    UCI = PI*LC
    # Compute UCI Mass
    PIM,LCM,UCIM,result_indicesM,angleM,cumulativeM,FstarM = ComputeUCIMAss(df_distance,IndicesEdge,np.sum(GridInside["population"]),NumGridEdge,GridInside,df_distance_inside,IndicesInside,Smax_i_Mass)
    logger.info(f'Mass, #edges: {len(Smax_i)}, #Inside: {len(Si)}')
    logger.info(f'Mass, #Distances edges: {len(Dmax_ij)}, #Inside: {len(D_ij)}')
    logger.info(f'Mass, Vmax: {Vmax}, V: {V}')
    logger.info(f"UCI Pot: LC: {LC}, PI: {PI}, UCI: {round(UCI,3)}")
    logger.info(f"UCI Mass: LC: {LCM}, PI: {PIM}, UCI: {round(UCIM,3)}")
    return PI,LC,UCI,result_indices,cumulative,Fstar,PIM,LCM,UCIM,result_indicesM,angleM,cumulativeM,FstarM

def ComputeUCIMAss(df_distance,IndicesEdge,SumPot,NumGridEdge,GridInside,df_distance_inside,IndicesInside,Smax_i_Mass):
    Sma,Dmax_ij = PrepareJitCompiledComputeMass(df_distance,IndicesEdge,SumPot,NumGridEdge,GridInside,case = 'Vmax',verbose = False)    
    logger.info("Compute UCI per mass ...")
    assert np.sum(Smax_i_Mass)-1 < 0.000005, 'The sum of the mass normalized on the edge must be 1'
    VMmax = ComputeJitV(Smax_i_Mass,Dmax_ij)/2
    SMi,D_ij = PrepareJitCompiledComputeMass(df_distance_inside,IndicesInside,SumPot,NumGridEdge,GridInside,case = 'V',verbose = False)
    SMi = SMi/np.sum(SMi)
    assert np.sum(SMi)-1 < 0.000005, 'The sum of the mass normalized on the inside must be 1'
    VM = ComputeJitV(SMi,D_ij)/2
    PI = ComputePI(VM,VMmax)
    result_indices,angle,cumulative,Fstar = LorenzCenters(np.array(GridInside['population']))
    LC = Fstar/len(cumulative)
    UCI = PI*LC
    return PI,LC,UCI,result_indices,angle,cumulative,Fstar


##----------------- UCI MASS ---------------------------------##
def FilterDistancelGrid(grid,df_distance):
    IndicesEdge = GetIndexEdgePolygon(grid)
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
