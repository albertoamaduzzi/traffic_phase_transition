import numpy as np
import numba
import pandas as pd
from numba import prange
from shapely.geometry import Point
import os
import logging
import polars as pl
from ModifyPotential import *

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
            Pidx = Df_Filtered_Potential_Normalized.iloc[i]["index"]
            Pjdx = Df_Filtered_Potential_Normalized.iloc[j]["index"]
            Didx = Df_Filtered_Distance.iloc[index_distance]["i"]
            Djdx = Df_Filtered_Distance.iloc[index_distance]["j"]
            print(f'i: {i}, P.index: {Pidx} D.index: {Didx} The indices must be the same')
            print(f'j: {j}, P.index: {Pjdx} D.index: {Djdx} The indices must be the same')
            assert Df_Filtered_Potential_Normalized.iloc[i]["index"] == Df_Filtered_Distance.iloc[index_distance]["i"], f'i: {i}, P.index: {Pidx} D.index: {Didx} The indices must be the same'
            assert Df_Filtered_Potential_Normalized.iloc[j]["index"] == Df_Filtered_Distance.iloc[index_distance]["j"], f'j: {j}, P.index: {Pjdx} D.index: {Djdx} The indices must be the same'
            index_distance += 1

def ComputePI(V,MaxV,verbose=True):
    if verbose:
        print('PI: ',1-V/MaxV)
    return 1 - V/MaxV


### POLAR VERSION PI ####
def ComputeSumDijViVj(dij,vi,vj):
    """
        @params dij: float -> Distance between two points
        @params vi: float -> Potential value at point i
        @params vj: float -> Potential value at point j
        @description: Compute the sum of the product of the distance and the potential values
    """
    return dij*vi*vj


def ComputePIPotential(grid,distance_matrix,PotentialDf,column_PI = "V_out"):
    """
        @params grid: pd.DataFrame -> Grid with the potential values
        @params distance_matrix: pd.DataFrame -> Distance matrix between the grids
        @params PotentialDf: pd.DataFrame -> Potential values
    """
    g = grid[["index","relation_to_line","position"]]
    if isinstance(grid,pd.DataFrame):
        g = pl.DataFrame(g)
    if isinstance(distance_matrix,pd.DataFrame):
        distance_matrix = pl.DataFrame(distance_matrix)
    if isinstance(grid,gpd.GeoDataFrame):
        g = pl.DataFrame(g)
    if isinstance(PotentialDf,pd.DataFrame):
        PotentialDf = PotentialDf[[column_PI,"index"]]
        PotentialDf = pl.DataFrame(PotentialDf)
    # Compute The Fraction
    g = g.join(PotentialDf, left_on="index", right_on="index", how="inner")
    g = g.with_columns((pl.col(column_PI)/pl.col(column_PI).sum()).alias(f"fraction_{column_PI}"))
    g = g[[column_PI,"index",f"fraction_{column_PI}","relation_to_line","position"]]
    dm = distance_matrix.join(g, left_on="i", right_on="index", how="inner",suffix = "_i")
    dm = dm.join(g, left_on="j", right_on="index", how="inner",suffix = "_j")
    dm = dm.with_columns(pl.struct(["distance",f"fraction_{column_PI}",f"fraction_{column_PI}_j"]).map_batches(lambda x: ComputeSumDijViVj(x.struct.field('distance'),x.struct.field(f"fraction_{column_PI}"),x.struct.field(f"fraction_{column_PI}_j"))).alias("sum_dij_vivj"))
    # COMPUTE THE MAX VIA COMPUTATION EDGES
    ConditionEdge = (pl.col("relation_to_line") == "edge") & (pl.col("relation_to_line_j") == "edge")
    NumberEdges = np.sqrt(len(dm.filter(ConditionEdge)))
    dm = dm.with_columns(pl.when(ConditionEdge).then(1/NumberEdges).otherwise(0).alias(f"fraction_{column_PI}_edge_i"))
    dm = dm.with_columns(pl.when(ConditionEdge).then(1/NumberEdges).otherwise(0).alias(f"fraction_{column_PI}_edge_j"))
    dm = dm.with_columns(pl.struct(["distance",f"fraction_{column_PI}_edge_i",f"fraction_{column_PI}_edge_j"]).map_batches(lambda x: ComputeSumDijViVj(x.struct.field('distance'),x.struct.field(f'fraction_{column_PI}_edge_i'),x.struct.field(f'fraction_{column_PI}_edge_j'))).alias("sum_dij_vivj_edge"))
    V = dm.filter(pl.col("position") == "inside",
                pl.col("position_j")== "inside").select(pl.col("sum_dij_vivj").sum())
    Vmax = dm.filter(pl.col("relation_to_line") == "edge",
                pl.col("relation_to_line_j") == "edge").select(pl.col("sum_dij_vivj_edge").sum())
    print("PI Potential: ",1 - V["sum_dij_vivj"][0]/Vmax["sum_dij_vivj_edge"][0])
    dm.filter(pl.col("relation_to_line") == "edge",
            pl.col("relation_to_line_j") == "edge")#.select(pl.col("sum_dij_vivj").sum())
    PI = 1 - V["sum_dij_vivj"][0]/Vmax["sum_dij_vivj_edge"][0]
    return PI,g[f"fraction_{column_PI}"].to_numpy()

def ComputePIMass(grid,distance_matrix,column_PI = "population"):
    """
        @params grid: pd.DataFrame -> Grid with the potential values
        @params distance_matrix: pd.DataFrame -> Distance matrix between the grids
    """
    g = grid[[column_PI,"index","relation_to_line","position"]]
    if isinstance(grid,pd.DataFrame):
        g = pl.DataFrame(g)
    if isinstance(distance_matrix,pd.DataFrame):
        distance_matrix = pl.DataFrame(distance_matrix)
    if isinstance(grid,gpd.GeoDataFrame):
        g = pl.DataFrame(g)
    if isinstance(grid,np.ndarray):
        g = pl.DataFrame(g)
    # Compute The Fraction 
    g = g.with_columns((pl.col(column_PI)/pl.col(column_PI).sum()).alias(f"fraction_{column_PI}"))
    g = g[[column_PI,"index",f"fraction_{column_PI}","relation_to_line","position"]]
    dm = distance_matrix.join(g, left_on="i", right_on="index", how="inner",suffix = "_i")
    dm = dm.join(g, left_on="j", right_on="index", how="inner",suffix = "_j")
    dm = dm.with_columns(pl.struct(["distance",f"fraction_{column_PI}",f"fraction_{column_PI}_j"]).map_batches(lambda x: ComputeSumDijViVj(x.struct.field('distance'),x.struct.field(f"fraction_{column_PI}"),x.struct.field(f"fraction_{column_PI}_j"))).alias("sum_dij_vivj"))
    # COMPUTE THE MAX VIA COMPUTATION EDGES
    ConditionEdge = (pl.col("relation_to_line") == "edge") & (pl.col("relation_to_line_j") == "edge")
    NumberEdges = np.sqrt(len(dm.filter(ConditionEdge)))
    dm = dm.with_columns(pl.when(ConditionEdge).then(1/NumberEdges).otherwise(0).alias(f"fraction_{column_PI}_edge_i"))
    dm = dm.with_columns(pl.when(ConditionEdge).then(1/NumberEdges).otherwise(0).alias(f"fraction_{column_PI}_edge_j"))
    dm = dm.with_columns(pl.struct(["distance",f"fraction_{column_PI}_edge_i",f"fraction_{column_PI}_edge_j"]).map_batches(lambda x: ComputeSumDijViVj(x.struct.field('distance'),x.struct.field(f'fraction_{column_PI}_edge_i'),x.struct.field(f'fraction_{column_PI}_edge_j'))).alias("sum_dij_vivj_edge"))
    V = dm.filter(pl.col("position") == "inside",
                pl.col("position_j")== "inside").select(pl.col("sum_dij_vivj").sum())
    Vmax = dm.filter(pl.col("relation_to_line") == "edge",
                pl.col("relation_to_line_j") == "edge").select(pl.col("sum_dij_vivj_edge").sum())
    print("PI Mass: ",1 - V["sum_dij_vivj"][0]/Vmax["sum_dij_vivj_edge"][0])
    dm.filter(pl.col("relation_to_line") == "edge",
            pl.col("relation_to_line_j") == "edge")#.select(pl.col("sum_dij_vivj").sum())
    PI = 1 - V["sum_dij_vivj"][0]/Vmax["sum_dij_vivj_edge"][0]
    return PI,g[f"fraction_{column_PI}"].to_numpy() 



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


def _ComputeUCICase(df_distance,DfGridOrPot,IndicesInside,Vmax,case = 'Potential'):
    """
        @params df_distance: pd.DataFrame ['distance','direction vector','i','j'] -> NOTE: Stores the matrix in order in one dimension 'j' increases faster than 'i' index
        @params DfGridOrPot: pd.DataFrame ['index','population'] or ['index','V_out'] -> Potential values for the grid
        @params IndicesInside: Index of the grid for which extracting distances and potential
        @params Vmax: float [Maximum V]
        @params case: str -> 'Potential' or 'Mass' [Case to compute the UCI]
        @description: Compute the UCI for the given number of centers.
    """
    assert len(df_distance) == len(DfGridOrPot)**2, f'The number of distances {len(df_distance)} must be the same of square number of grids {len(DfGridOrPot)**2}, {len(DfGridOrPot)}'
    assert np.array_equal(df_distance["i"].unique(), df_distance["j"].unique()), 'The indices must be the same'
    assert np.array_equal(df_distance["i"].unique(), DfGridOrPot['index'].unique()), 'The indices must be the same'    # Number of squares in the grid that are edges
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
        GridInside = DfGridOrPot.loc[IndicesInside]
        PlotGridUsedComputationUCI(GridInside)   
        PlotGridUsedComputationUCIEdges(GridInside)

        logger.info(f"UCI Mass: LC: {LCM}, PI: {PIM}, UCI: {round(UCIM,3)}")
        return PIM,LCM,UCIM,result_indicesM,angleM,cumulativeM,FstarM


def ComputeUCICase(df_distance,Grid,DfPotential,IndicesInside,case = 'Potential'):
    """
        @params df_distance: pd.DataFrame ['distance','direction vector','i','j'] -> NOTE: Stores the matrix in order in one dimension 'j' increases faster than 'i' index
        @params Grid,DfPotential: pd.DataFrame ['index','population'] or ['index','V_out'] -> Potential values for the grid
        @params IndicesInside: Index of the grid for which extracting distances and potential
        @params case: str -> 'Potential' or 'Mass' [Case to compute the UCI]
        @description: Compute the UCI for the given number of centers.
    """
    if case == 'Potential':
        # POTENTIAL
        logger.info("Compute UCI for Potential...")
        PI,Si_Normalized = ComputePIPotential(Grid,df_distance,DfPotential)
        result_indices,angle,cumulative,Fstar = LorenzCenters(np.array(Si_Normalized))
        LC = Fstar/len(cumulative)
        UCI = PI*LC
        logger.info(f"UCI Pot: LC: {LC}, PI: {PI}, UCI: {round(UCI,3)}")    
        return PI,LC,UCI,result_indices,angle,cumulative,Fstar
    if case == 'Mass':
        logger.info("Compute UCI for Mass...")
        PIM,SiMass_Normalized = ComputePIMass(Grid,df_distance)
        result_indicesM,angleM,cumulativeM,FstarM = LorenzCenters(np.array(SiMass_Normalized))
        LCM = FstarM/len(cumulativeM)
        UCIM = PIM*LCM
        GridInside = Grid.loc[IndicesInside]
        PlotGridUsedComputationUCI(GridInside)   
        PlotGridUsedComputationUCIEdges(GridInside)

        logger.info(f"UCI Mass: LC: {LCM}, PI: {PIM}, UCI: {round(UCIM,3)}")
        return PIM,LCM,UCIM,result_indicesM,angleM,cumulativeM,FstarM

def ComputeUCIPidMass(Grid,df_distance,IndicesInside):
        logger.info("Compute UCI for Mass...")
        PIM,SiMass_Normalized = ComputePIMass(Grid,df_distance)
        result_indicesM,angleM,cumulativeM,FstarM = LorenzCenters(np.array(SiMass_Normalized))
        LCM = FstarM/len(cumulativeM)
        UCIM = PIM*LCM
        GridInside = Grid.loc[IndicesInside]
        PlotGridUsedComputationUCI(GridInside)   
        PlotGridUsedComputationUCIEdges(GridInside)
        return PIM,LCM,UCIM,result_indicesM,angleM,cumulativeM,FstarM

def GenerateRandomPopulationPid(Grid,df_distance,IndicesInside,Smax_i_Mass,Dmax_ij_Mass,UCIsInterval,UCIInterval2UCI,LockUCIInterval2UCI,AcceptedConfigurations,LockAcceptedConfigurations,FlagEnd,LockFlagEnd,cov,distribution,num_peaks,total_population,SaveDir):
    """
        @params Grid: pd.DataFrame -> ['index','population']
    """
    from numba import set_num_threads
    from ModifyPotential import GenerateRandomPopulation
    set_num_threads(10)
    while not FlagEnd.value:
        InfoCenters = {'center_settings': {"type":distribution},
                            'covariance_settings':{
                                "covariances":{"cvx":cov,"cvy":cov},
                            "Isotropic": True,
                            "Random": False}}
        new_population,index_centers = GenerateRandomPopulation(Grid,num_peaks,total_population,InfoCenters)
        Grid['population'] = new_population        
        PIM,LCM,UCIM,result_indicesM,angleM,cumulativeM,FstarM = ComputeUCIPidMass(Grid,df_distance,IndicesInside)
        logger.info(f"UCI Mass: LC: {LCM}, PI: {PIM}, UCI: {round(UCIM,3)}, Cov: {cov}, NumPeaks: {num_peaks}")

        for i,UCI in enumerate(UCIsInterval):
            if UCIM > UCI and UCIM <= UCIsInterval[i+1]:
                with LockUCIInterval2UCI:
                    if len(UCIInterval2UCI[round(UCI,3)]) <= 4:
                        UCIInterval2UCI[round(UCI,3)].append(round(UCIM,3))
                        Grid.to_parquet(os.path.join(SaveDir,f'Grid_{round(UCIM,3)}.parquet'),index = False)
                        with LockAcceptedConfigurations:
                            if round(UCI, 3) not in AcceptedConfigurations:
                                AcceptedConfigurations[round(UCI, 3)] = []
                            AcceptedConfigurations[round(UCI,3)].append({'UCI':UCIM,"cov":cov,"distribution":distribution,"num_peaks":num_peaks,"PI":PIM,"LC":LCM,"Fstar":FstarM})
            with LockFlagEnd:
                FlagEnd.value = CheckDictionaryFull(UCIInterval2UCI,4)
                if FlagEnd.value:
                    break
    return UCIInterval2UCI,AcceptedConfigurations

def GenerateRandomPopulationAndComputeUCI(Covariances,Distributions,ListPeaks,Grid,total_population,df_distance,IndicesInside,Smax_i_Mass,Dmax_ij_Mass,UCIInterval2UCI,AcceptedConfigurations,SaveDir):
    """
        @params cov: Covariance that sets the width of population
        @params distribution: [exponential,gaussian]
        @params num_peaks: Number of peaks in the population
        Generate Random Population and Compute UCI
        The goal of this function in terms of the general context is to provid the computation of the new grids,
        from which, sure, that we have all the needed UCIs we can procede computing 
        Tij and Potential and whatever.
        The idea is to have a control on the UCIs that we are computing.
    """
    from collections import defaultdict
    from multiprocessing import Manager,Process,log_to_stderr
    # Control How Many UCIs are there
    UCIsInterval = np.linspace(0,1,11)
    if UCIInterval2UCI is None:
        UCIInterval2UCI = {round(UCIInterval,3):[] for UCIInterval in UCIsInterval}
    if AcceptedConfigurations is None:
        AcceptedConfigurations = defaultdict(list)
    # Check That The computations are not completed: If the dictionary is full
    # The dictionary is full only if each interval has 4 values
    Break = CheckDictionaryFull(UCIInterval2UCI,4)
    if Break:
        return UCIInterval2UCI,AcceptedConfigurations
    log_to_stderr()
    # INITIALIZE PARALLELISM
    manager = Manager()
    # UCIInterval2UCI
    UCIInterval2UCI = manager.dict(UCIInterval2UCI)
    LockUCIInterval2UCI = manager.Lock()
    # AcceptedConfigurations
    AcceptedConfigurations = manager.dict(AcceptedConfigurations)
    LockAcceptedConfigurations = manager.Lock()
    for UCI in UCIsInterval:
        UCIInterval2UCI[round(UCI, 3)] = manager.list(UCIInterval2UCI[round(UCI, 3)])
        AcceptedConfigurations[round(UCI, 3)] = manager.list(AcceptedConfigurations[round(UCI, 3)])    
    # Flag End
    FlagEnd = manager.Value('b',False)
    LockFlagEnd = manager.Lock()
    processes = []
    NumberGrids = len(Grid)
    ListPeaks = [1,2,3,4,5,NumberGrids,NumberGrids/2,NumberGrids/3]
    for cov in Covariances:
        for distribution in Distributions:
            for num_peaks in ListPeaks:
                # Compue in Parallel all the possible new configurations of the population
                p = Process(target=GenerateRandomPopulationPid, args=(Grid,df_distance,IndicesInside,Smax_i_Mass,Dmax_ij_Mass,UCIsInterval,UCIInterval2UCI,LockUCIInterval2UCI,AcceptedConfigurations,LockAcceptedConfigurations,FlagEnd,LockFlagEnd,cov,distribution,num_peaks,total_population,SaveDir))
                processes.append(p)
                p.start()

    for p in processes:
        p.join()
    return UCIInterval2UCI,AcceptedConfigurations




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
 

def CheckDictionaryFull(d,limit_length):
    """
        @params d: dict -> {UCI:[values]}
        Check if the dictionary is full, that is, each key contains limit_length values
        Avoid to check the value 1.0 as there is just one configuration that may not be matched.
    """
    for key in d.keys():
        if key !=1.:
            if len(d[key]) < limit_length:
                return False
    return True
def FillDictionaryIntervalWithValues(d,values,limit_length = 4):
    """
        @params d: dict -> {UCI:[values]}
        @params values: list -> [value1,value2,...,valueN]
        Append the values to the values of the dictionary up until limit_length is reached
        NOTE: Used in GenerateRandomPopulationAndComputeUCI, to have control on the UCI produced
    """
    for value in values:
        for i,UCI in enumerate(list(d.keys())):
            if i < len(d.keys())-1:
                if value < d[i + 1] and value >= d[i]:
                    if len(d[UCI]) < limit_length:
                        d[UCI].append(value)
        if CheckDictionaryFull(d,limit_length):
            break
    return d

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
