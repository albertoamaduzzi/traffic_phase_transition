import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
import json
import os
import sys
import time
import matplotlib.pyplot as plt
import numba
import logging
import pandas as pd
import ast
import math
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append('~/berkeley/traffic_phase_transition/scripts')
from FittingProcedures import Fitting

###################################################################################################################
###############################         FITTING PROCEDURE           ###############################################
###################################################################################################################

#### STEP1: Grid, Df_distance, T -> 1D array [Done For Optimization in Numba]
def DistanceDf2Matrix(df_distance):
    '''
        Input:
            df_distance: (pd.DataFrame) [i,j,distance]
        Output:
            distance_matrix: (np.array 2D) [i,j] -> distance
    '''
    pivot_df = df_distance.pivot(index='i', columns='j', values='distance')

    # Fill missing values with 0
    pivot_df = pivot_df.fillna(0)

    # Convert to numpy array
    distance_matrix = pivot_df.to_numpy(dtype = np.float32)
    return distance_matrix

def Grid2Arrays(grid):
    gridIdx = grid['index'].to_numpy(dtype = np.int32)
    gridPopulation = grid['population'].round().astype(dtype = np.int32).to_numpy(dtype = np.int32)
    return gridIdx,gridPopulation

def T2Arrays(T):
    Vnpeople = T['number_people'].round().astype(dtype = np.int32).to_numpy(dtype = np.int32)
    Vorigins = T['origin'].to_numpy(dtype = np.int32)
    Vdestinations = T['destination'].to_numpy(dtype = np.int32)
    return Vnpeople,Vorigins,Vdestinations

## SUBSAMPLING

######
######         NOTE: Vorigins and Vdestinations are the Indices shared by T,Grid,Df_distance np.where on these in the OPTIMIZED CODE
######
#------------------------------------------------- FLUXES  -------------------------------------------------#
@numba.njit(['(int32[:], int32[:], int32[:])'],parallel=True)
def SubsampleFluxesByPop(Vnpeople, Vorigins, Vdestinations):
    '''
        Input:
            Vnpeople: (np.array 1D) -> number of people
            Vorigins: (np.array 1D) -> origin
            Vdestinations: (np.array 1D) -> destination
            chunk_size: (int) -> Number of combinations to process in each chunk
        Output:
            Vnpeople: (np.array 1D) -> number of people
            Vorigins: (np.array 1D) -> origin
            Vdestinations: (np.array 1D) -> destination
    '''
    ## Create subsample of OD pairs for which the flux is bigger then 0
    indicesT2consider = np.where(Vnpeople > 0)
    ## Chose subsample fluxes,O,D
    Vnpeople = Vnpeople[indicesT2consider]
    Vorigins = Vorigins[indicesT2consider]
    Vdestinations = Vdestinations[indicesT2consider]
    return Vnpeople, Vorigins, Vdestinations


@numba.njit(['(int32[:], int32[:], int32[:],int32)'],parallel=True)
def SubsampleByCellFluxOrigin(Vnpeople, Vorigins, Vdestinations,cell_x):
    '''
        Input:
            Vnpeople: (np.array 1D) -> number of people
            Vorigins: (np.array 1D) -> origin
            Vdestinations: (np.array 1D) -> destination
            cell_x: (int) -> Cell of Origin
        Output:
            Vnpeople: (np.array 1D) -> number of people
            Vorigins: (np.array 1D) -> origin
            Vdestinations: (np.array 1D) -> destination
            IndicesXFlux: Indices of choice -> Never Used: USELESS!!!!!
        NOTE: The output is a vector
    '''
    IndicesXFlux = np.where(Vorigins == cell_x)
    VnpeopleX = Vnpeople[IndicesXFlux]
    VoriginsX = Vorigins[IndicesXFlux]
    VdestinationsX = Vdestinations[IndicesXFlux]
    return IndicesXFlux,VnpeopleX, VoriginsX, VdestinationsX

@numba.njit(['(int32[:], int32[:], int32[:],int32)'],parallel=True)
def SubsampleByCellFluxDestination(Vnpeople, Vorigins, Vdestinations,cell_i):

    '''
        Input:
            Vnpeople: (np.array 1D) -> number of people
            Vorigins: (np.array 1D) -> origin
            Vdestinations: (np.array 1D) -> destination
            cell_i: (int) -> Cell of Origin
        Output:
            Vnpeople: int -> number of people
            Vorigins: int -> origin
            Vdestinations: int -> destination
            NOTE: IndicesXFlux: Indices of choice LOCAL to FLUX-> Never Used: USELESS!!!!!
        NOTE: The output is a float
        
    '''
    IndicesXFlux = np.where(Vdestinations == cell_i)
    VnpeopleX = Vnpeople[IndicesXFlux]
    VoriginsX = Vorigins[IndicesXFlux]
    VdestinationsX = Vdestinations[IndicesXFlux]
    return IndicesXFlux,VnpeopleX, VoriginsX, VdestinationsX

#------------------------------------------------- GRID  -------------------------------------------------#

@numba.njit(['(int32[:], int32[:],int32)'],parallel=True)
def SubSampleGridByCell(VgridIdx, VgridPopulation,cell_x):
    '''
        Input:
            VgridIdx: (np.array 1D) -> index of the grid
            VgridPopulation: (np.array 1D) -> population of the grid
            cell_x: (int) -> index of the cell
        Output:
            VgridIdxX: (np.array 1D) -> index of the grid
            VgridPopulationX: (np.array 1D) -> population of the grid
        NOTE: Same story of the Flux Vector, we have that the value of cell_x is shared with fluxes.
        NOTE: IndicesXGrid never used as it is local for the Grid: USELESS!!!!
    '''
    IndicesXGrid = np.where(VgridIdx == cell_x)
    VgridIdxX = VgridIdx[IndicesXGrid]
    VgridPopulationX = VgridPopulation[IndicesXGrid]
    return IndicesXGrid,VgridIdxX,VgridPopulationX


#####
#####           NOTE: EXTENSIVE USAGE OF MEMORY. OPTIMIZATION NEEDED
#####

#------------------------------------------------- GRAVITATION LAW FITTING  -------------------------------------------------#
#@numba.njit#('int32(int32)',noptyhon=True)
@numba.njit(['int32(int32[:], int32[:], int32[:])'])
def ComputeMaxSized0(Vnpeople, Vorigins, Vdestinations):
    max_size = 0
    for cell_x in Vorigins:
        # Get Fluxes from cell_x
        _,VnpeopleX, VoriginsX, VdestinationsX = SubsampleByCellFluxOrigin(Vnpeople, Vorigins, Vdestinations,cell_x)
        max_size += len(VnpeopleX)*(len(VnpeopleX)-1)/2
        for cell_i in VdestinationsX:
            if cell_x < cell_i:
                _,_, _, _ = SubsampleByCellFluxDestination(VnpeopleX, VoriginsX, VdestinationsX,cell_i)
    return max_size

## POTENTIAL FITTING
@numba.njit(['(int32[:], int32[:], int32[:],int32[:],int32[:],float32[:,:])'],parallel=True)
def d0PotentialFitOptimized(Vnpeople, Vorigins, Vdestinations, VgridIdx, VgridPopulation, DistanceMatrix): # debug = False
#    if debug:
#        count = 0
#        NCycleControl = 1000
    count = 0 
    Vnpeople, Vorigins, Vdestinations = SubsampleFluxesByPop(Vnpeople, Vorigins, Vdestinations)
#    t0 = time.time()
    max_size = ComputeMaxSized0(Vnpeople, Vorigins, Vdestinations)
#    t1 = time.time()
#    print('time spent to compute max size: ',t1-t0)
#    print('max_size: ',max_size)
    d0s = np.zeros(max_size,dtype = np.float32)
    for cell_x in Vorigins:
        # Get Fluxes from cell_x
        _,VnpeopleX, VoriginsX, VdestinationsX = SubsampleByCellFluxOrigin(Vnpeople, Vorigins, Vdestinations,cell_x)
        # Get Row of the grid
#        if debug:
#            DebuggingGetd0Cellx(count,NCycleControl,VnpeopleX,VoriginsX,VdestinationsX,cell_x)
#        count += 1
#        if debug:
#            count_i = 0
        for cell_i in VdestinationsX:
            if cell_x < cell_i:
                _,VnpeopleXi, VoriginsXi, VdestinationsXi = SubsampleByCellFluxDestination(VnpeopleX, VoriginsX, VdestinationsX,cell_i)
                _,VgridIdxXi,VgridPopulationXi = SubSampleGridByCell(VgridIdx, VgridPopulation,cell_i)
#                if debug:
#                    DebuggingGetd0Celli(count_i,NCycleControl,VnpeopleXi,VoriginsXi,VdestinationsXi,cell_i)
#                    count_i += 1
#                if debug:
#                    count_j = 0
                for cell_j in VdestinationsX:
                    if cell_i < cell_j:
                        _,VnpeopleXj, VoriginsXj, VdestinationsXj = SubsampleByCellFluxDestination(VnpeopleX, VoriginsX, VdestinationsX,cell_j)
                        _,_,VgridPopulationXj = SubSampleGridByCell(VgridIdx, VgridPopulation,cell_j)
 #                       if debug:
 #                           DebuggingGetd0Celli(count_j,NCycleControl,VnpeopleXj,VoriginsXj,VdestinationsXj,cell_j)
#                           count_j += 1
                        if len(VnpeopleXi)==1 and len(VnpeopleXj)==1:
                            idx_i0 = VoriginsXi[0]
                            idx_i1 = VdestinationsXi[0]
                            dxi = DistanceMatrix[idx_i0][idx_i1]
                            idx_j0 = VoriginsXj[0]
                            idx_j1 = VdestinationsXj[0]
                            dxj = DistanceMatrix[idx_j0][idx_j1]
                            Txi = VnpeopleXi[0]
                            Txj = VnpeopleXj[0]
                            mi = VgridPopulationXi[0]
                            mj = VgridPopulationXj[0]

#                            if np.isnan(np.log(Txi * mi / (Txj * mj))):
#                                print('Invalid: ',' Tx{0}: {1} Tx{2} {3} mi: {4},mj: {5}'.format(cell_i,Txi,cell_j,Txj,mi,mj))
#                            else:
#                                print('Valid: ',' Tx{0}: {1} Tx{2} {3} mi: {4},mj: {5}, idxj {6} idxi {7}'.format(cell_i,Txi,cell_j,Txj,mi,mj,VgridIdxXj[0],VgridIdxXi[0]))
                            if not np.isnan(np.log(Txi * mi / (Txj * mj))):
                                if (-np.log(Txi * mi / (Txj * mj)))!=0:
                                    d0s[count] = np.abs(dxi - dxj) / (-np.log(Txi * mi / (Txj * mj)))
                                else:
                                    pass

                            else:
                                pass
                        else:
                            raise ValueError('More than 1 flux')
                        count += 1
    idx = np.where(d0s != 0)
    median = np.median(d0s[idx])
    return median,d0s

@numba.njit(['int32(int32[:], int32[:], int32[:])'])
def ComputeMaxSizek(Vnpeople, Vorigins, Vdestinations):
    '''
        Input:
            Vnpeople: np.array(int32) [number of people exchanged]  example: [1,1,5,...,32,12,1,...]
            Vorigins: np.array(int32) [indices origin]              example: [32,32,32,...,34,34,34,34,...] with repetitions as it is extended for each dest
            Vdestination: np.array(int32) [indices destination]     example: [34,42,56,...,97,108,115,122,...]
        TOUSE in:
            GetkPotential
        Return:
            max_size -> int (number of couples ij for which there is a flux !=0)
        NOTE: It may be useless as it should be len(Vorigins)
    '''
    max_size = 0
    for cell_i in Vorigins:
        # Get Fluxes from cell_x
        _,VnpeopleX, _, _ = SubsampleByCellFluxOrigin(Vnpeople, Vorigins, Vdestinations,cell_i)
        max_size += len(VnpeopleX)
    return max_size

@numba.njit(['(int32[:], int32[:], int32[:],int32[:],int32[:],float32[:,:],float32)'],parallel=True)
def GetEstimationFluxesVector(Vnpeople, Vorigins, Vdestinations, VgridIdx, VgridPopulation, DistanceMatrix,d0):
    '''
        Input:
            Vnpeople: 
        Function to get the parameters of the potential
        cell_i is the origin
        cell_j is the destination
    '''
    if 0 in Vnpeople:
        raise ValueError('You forgot to filter the fluxes')
    count = 0 
    max_size = ComputeMaxSizek(Vnpeople, Vorigins, Vdestinations)
    EstimateFluxesScaled = np.zeros(max_size,dtype = np.float32)
    Fluxes = np.zeros(max_size,dtype = np.float32)
    Massi = np.zeros(max_size,dtype = np.float32)
    Massj = np.zeros(max_size,dtype = np.float32)
    DistanceVector = np.zeros(max_size,dtype = np.float32)
    ErrorFluxes = np.zeros(max_size,dtype = np.float32)
    ErrorEsteem = np.zeros(max_size,dtype = np.float32)
    ErrorDist = np.zeros(max_size,dtype = np.float32)
    maxVnpeople = max(Vnpeople)
    for cell_i in Vorigins:
        # Get Fluxes from cell_i
        _,VnpeopleI, VoriginsI, VdestinationsI = SubsampleByCellFluxOrigin(Vnpeople, Vorigins, Vdestinations,cell_i)
        _,_,VgridPopulationI = SubSampleGridByCell(VgridIdx, VgridPopulation,cell_i)
        # Get Row of the grid
        for cell_j in VdestinationsI:
            if cell_i < cell_j:
                _,VnpeopleIJ, _, VdestinationsIJ = SubsampleByCellFluxDestination(VnpeopleI, VoriginsI, VdestinationsI,cell_j)
                _,_,VgridPopulationJ = SubSampleGridByCell(VgridIdx, VgridPopulation,cell_j)
#                       if debug:
#                           DebuggingGetd0Celli(count_j,NCycleControl,VnpeopleXj,VoriginsXj,VdestinationsXj,cell_j)
#                           count_j += 1
                if len(VnpeopleIJ)==1:
                    idx_i0 = VoriginsI[0]
                    idx_i1 = VdestinationsIJ[0]
                    dij = DistanceMatrix[idx_i0][idx_i1]
                    Tij = VnpeopleIJ[0]
                    mi = VgridPopulationI[0]
                    mj = VgridPopulationJ[0]
                    Esteem = mi*mj*np.exp(-dij/d0)
                    if not np.isnan(Esteem):
                        if Esteem < 10*maxVnpeople:
                            EstimateFluxesScaled[count] = Esteem
                            Fluxes[count] = Tij
                            DistanceVector[count] = dij
                            Massi[count] = mi
                            Massj[count] = mj
                        else:
                            ErrorEsteem[count] = Esteem
                            ErrorFluxes[count] = Tij
                            ErrorDist[count] = dij

                    else:
                        pass
                else:
                    raise ValueError('More than 1 flux')
                count += 1
    idx = np.where(EstimateFluxesScaled != 0)
    EstimateFluxesScaled = EstimateFluxesScaled[idx]
    idx = np.where(Fluxes != 0)
    Fluxes = Fluxes[idx]
    idx = np.where(Massi != 0)
    Massi = Massi[idx]
    idx = np.where(Massj != 0)
    Massj = Massj[idx]
    idx = np.where(DistanceVector != 0)
    DistanceVector = DistanceVector[idx]
    return EstimateFluxesScaled,Fluxes,DistanceVector,ErrorEsteem,ErrorFluxes,ErrorDist,Massi,Massj

def GetkPotential(EstimateFluxesScaled,Fluxes,d0,save_dir,initial_guess = [10**(-6),0]):    
    if not os.path.isfile(os.path.join(save_dir,'FitFluxesParameters.json')):        
        k,q = Fitting(np.array(EstimateFluxesScaled),np.array(Fluxes),label = 'linear',initial_guess = initial_guess ,maxfev = 10000) #max(Fluxes)/max(EstimateFluxesScaled)
        with open(os.path.join(save_dir,'FitFluxesParameters.json'),'w') as f:
            json.dump({'d0':float(d0),'k':k[0],'q': k[1]},f)
    else:
        d0,k,q = json.load(open(os.path.join(save_dir,'FitFluxesParameters.json'),'r'))
    return d0,k,q

def PlotDistanceFluxes(EstimateFluxesScaled,Fluxes,DistanceVector,title):
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    n,bins = np.histogram(DistanceVector,bins = 50)
    AvgFluxGivenR =[np.mean(Fluxes[np.where(DistanceVector>bins[i]) and DistanceVector < bins[i+1]]) for i in range(len(bins)-1)]
    AvgEsteemGivenR =[np.mean(EstimateFluxesScaled[np.where(DistanceVector>bins[i]) and DistanceVector < bins[i+1]]) for i in range(len(bins)-1)]
    ax.scatter(bins[:-1],AvgFluxGivenR)
    ax.plot(bins[:-1],AvgEsteemGivenR)
    ax.set_xlabel('R(km)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend(['Fluxes','Gravity'])
    plt.show()

#------------------------------------------------- GRAVITATION VESPIGNANI FITTING  -------------------------------------------------#

#####
#####               NOTE: Equivalent to GetEstimationFluxes
#####
    

@numba.njit(['(int32[:], int32[:], int32[:],int32[:],int32[:],float32[:,:])'],parallel=True)
def PrepareVespignani(Vnpeople, Vorigins, Vdestinations, VgridIdx, VgridPopulation, DistanceMatrix):
    '''
        
        Returns:
            4 1D vectors:
                1) Massi
                2) Massj
                3) Dij
                4) Wij
        TO GIVE to Fitting(...,'vespignani')
    '''
    if 0 in Vnpeople:
        raise ValueError('You forgot to filter the fluxes')
    count = 0 
    max_size = ComputeMaxSizek(Vnpeople, Vorigins, Vdestinations)
    Fluxes = np.zeros(max_size,dtype = np.float32)
    Massi = np.zeros(max_size,dtype = np.float32)
    Massj = np.zeros(max_size,dtype = np.float32)
    DistanceVector = np.zeros(max_size,dtype = np.float32)
    for cell_i in Vorigins:
        # Get Fluxes from cell_i
        _,VnpeopleI, VoriginsI, VdestinationsI = SubsampleByCellFluxOrigin(Vnpeople, Vorigins, Vdestinations,cell_i)
        _,_,VgridPopulationI = SubSampleGridByCell(VgridIdx, VgridPopulation,cell_i)
        # Get Row of the grid
        for cell_j in VdestinationsI:
            if cell_i < cell_j:
                _,VnpeopleIJ, _, VdestinationsIJ = SubsampleByCellFluxDestination(VnpeopleI, VoriginsI, VdestinationsI,cell_j)
                _,_,VgridPopulationJ = SubSampleGridByCell(VgridIdx, VgridPopulation,cell_j)
#                       if debug:
#                           DebuggingGetd0Celli(count_j,NCycleControl,VnpeopleXj,VoriginsXj,VdestinationsXj,cell_j)
#                           count_j += 1
                if len(VnpeopleIJ)==1:
                    idx_i0 = VoriginsI[0]
                    idx_i1 = VdestinationsIJ[0]
                    dij = DistanceMatrix[idx_i0][idx_i1]
                    Tij = VnpeopleIJ[0]
                    mi = VgridPopulationI[0]
                    mj = VgridPopulationJ[0]
                    Fluxes[count] = Tij
                    DistanceVector[count] = dij
                    Massi[count] = mi
                    Massj[count] = mj
                else:
                    raise ValueError('More than 1 flux')
                count += 1
    idx = np.where(Fluxes != 0)
    Fluxes = Fluxes[idx]
#    idx = np.where(Massi != 0)
    Massi = Massi[idx]
#    idx = np.where(Massj != 0)
    Massj = Massj[idx]
#    idx = np.where(Fluxes != 0)
    DistanceVector = DistanceVector[idx]
    VespignaniVector = [Massi,Massj,DistanceVector]
    return VespignaniVector,Fluxes


def PlotVespignaniFit(EstimateFluxesScaled,Fluxes,DistanceVector,Massi,Massj):
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    n,bins = np.histogram(DistanceVector,bins = 50)
    AvgFluxGivenR =[Fluxes[np.where(DistanceVector>bins[i]) and DistanceVector < bins[i+1]]/(Massi[np.where(DistanceVector>bins[i]) and DistanceVector < bins[i+1]]*Massj[np.where(DistanceVector>bins[i]) and DistanceVector < bins[i+1]]) for i in range(len(bins)-1)]
    ax.boxplot(bins[:-1],AvgFluxGivenR)
    ax.set_xlabel('R(km)')
    ax.set_ylabel('W/(mi*mj)')
    ax.legend(['Fluxes','Gravity'])
    plt.show()



# END FITTING PROCEDURE

def GetCommunicationLevelsAmongGrids(Tij):
    subsetflux = Tij[Tij['number_people'] > 0]
    AdjacencyColumnSumDistribution = []
    AdjacencyColumnSumDistributionPWeightedPeople = []
    for _,subsetorigin in subsetflux.groupby('origin'):
        AdjacencyColumnSumDistributionPWeightedPeople.append(subsetorigin['number_people'].sum())
        AdjacencyColumnSumDistribution.append(len(subsetorigin))
    hist, xedges, yedges = np.histogram2d(AdjacencyColumnSumDistribution, AdjacencyColumnSumDistributionPWeightedPeople, bins=20, density=True)
    # Plot the 2D histogram
    plt.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='viridis')
    plt.colorbar(label='Probability Density')
    plt.xlabel('Number Connections By Flux')
    plt.ylabel('Total Number Commuting People')
    plt.title('Joint Probability Distribution')
    plt.show()
    

###################################################################################################################
###############################         VECTOR FIELD AND POTENTIAL           ######################################
###################################################################################################################

#------------------------------------------ VECTOR FIELD --------------------------------------------------------#
    
#####
#####                   NOTE: Usage ->  VectorField = GetVectorField(Tij,df_distance)
#####                                   VectorFieldDir = os.path.join(TRAFFIC_DIR,'data','carto',name,'grid',str(grid_size))
#####                                   SaveVectorField(VectorField,VectorFieldDir)

def parse_dir_vector(vector_string):
    if vector_string== '[nan,nan]' or vector_string== '[nan nan]':
        vector_array = np.array([0,0])
    # Split the string representation of the vector
    else:
        vector_parts = vector_string.strip('[]').split()
        # Convert each part to a float or np.nan if it's 'nan'
        vector_array = np.array([float(part) if part != 'nan' else np.nan for part in vector_parts])
    return vector_array

def GetVectorField(Tij,df_distance):
    Tij['vector_flux'] = df_distance['dir_vector'].apply(lambda x: parse_dir_vector(x) ) * Tij['number_people']

    # Create VectorField DataFrame
    VectorField = pd.DataFrame(index=Tij['(i,j)D'].unique(), columns=['(i,j)', 'Ti', 'Tj'])
    Tj_values = Tij.groupby('(i,j)D')['vector_flux'].sum()
    VectorField['Tj'] = Tj_values

    # Calculate 'Ti' values
    Ti_values = Tij.groupby('(i,j)O')['vector_flux'].sum()
    VectorField['Ti'] = Ti_values
    VectorField['index'] = VectorField.index
    VectorField['(i,j)'] = VectorField['index']
    VectorField['index'] = VectorField.index
    VectorField.reset_index(inplace=True)
    return VectorField

def SaveVectorField(VectorField,save_dir):
    VectorField.to_csv(os.path.join(save_dir,'VectorField.csv'))

def GetSavedVectorFieldDF(save_dir):
    return pd.read_csv(os.path.join(save_dir,'VectorField.csv'))

#------------------------------------------ POTENTIAL ----------------------------------------------------------#
#####
#####                   NOTE: Usage ->  lattice = GetPotentialLattice(lattice,VectorField)
#####                                   PotentialDataframe = ConvertLattice2PotentialDataframe(lattice)
#####                                   SavePotentialDataframe(PotentialDataframe,dir_grid)

def GetPotentialLattice(lattice,VectorField):
    '''
        Input: 
            lattice -> without ['V_in','V_out']
            VectorField: Dataframe [index,(i,j),Ti,Tj]
        Output:
            lattice with:
                'V_in' potential for the incoming fluxes
                'V_out' potential for the outgoing fluxes
                'rotor_z_in': Is the rotor at the point (i,j) for the ingoing flux. (Tj) sum over i. So I look at a source and I say that the field
                              is the ingoing flux. This is strange as it does not give any information about where to go to find the sink.
                'rotor_z_out': Is the rotor at the point (i,j) for the ingoing flux. (Ti) sum over j. So I look at a source and I say that the field
                               is the outgoing flux. In this way I am considering the analogue case to the google algorithm for
                               page rank as I am at a random point and the field points at the direction with smaller potential, the sink, that is 
                               the higher rank of importance.
                    
        Describe:
            Output = Input for ConvertLattice2PotentialDataframe
    '''
    nx.set_node_attributes(lattice, 0, 'V_in')
    nx.set_node_attributes(lattice, 0, 'V_out')
    nx.set_node_attributes(lattice, 0, 'index')
    nx.set_node_attributes(lattice, 0, 'rotor_z_out')
    nx.set_node_attributes(lattice, 0, 'rotor_z_in')
    max_i = max(ast.literal_eval(node_str)[0] for node_str in lattice.nodes)
    max_j = max(ast.literal_eval(node_str)[1] for node_str in lattice.nodes)
    for node_str in lattice.nodes:    
        ij = ast.literal_eval(node_str)
        i = ij[0]
        j = ij[1]
        lattice.nodes[node_str]['V_in'] = 0
        lattice.nodes[node_str]['V_out'] = 0

    for edge in lattice.edges(data=True):
        # Extract the indices of the nodes
        node_index_1, node_index_2 = edge[:2]    
        VectorField.index = VectorField['(i,j)']
        # Compute the value of V for the edge using the formula
        # NOTE: The formula seems to have dx multiplying the y component of the vector field (it is not the case as (0,0),(1,0) is moving in x)
        # NOTE: I computed dx thinking that (1,0) being the y !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  dx is dy
        # NOTE: CHECK df_distance
        dxTjx = lattice[node_index_1][node_index_2]['dy'] * VectorField.loc[node_index_1, 'Tj'][0]
        dyTjy = lattice[node_index_1][node_index_2]['dx'] * VectorField.loc[node_index_1, 'Tj'][1]
        node_Vin = lattice.nodes[node_index_1]['V_in'] + dxTjx  + dyTjy  
        node_Vout = lattice.nodes[node_index_1]['V_out'] + lattice[node_index_1][node_index_2]['dx'] * VectorField.loc[node_index_1, 'Ti'][1]  + lattice[node_index_1][node_index_2]['dy'] * VectorField.loc[node_index_1, 'Ti'][0]      
        # ROTOR NOTE: AGAIN[d/dx is d/dy] NETWORKX USES THIS CONVENTION
        if not math.isinf(lattice[node_index_1][node_index_2]['d/dy']):
            ddxTjy = lattice[node_index_1][node_index_2]['d/dy'] * VectorField.loc[node_index_1, 'Tj'][1]
        else:
            ddxTjy = 0
        if not math.isinf(lattice[node_index_1][node_index_2]['d/dx']):
            ddyTjx = lattice[node_index_1][node_index_2]['d/dx'] * VectorField.loc[node_index_1, 'Tj'][0]  
        else:
            ddyTjx = 0
        rotor_z_in = ddxTjy  - ddyTjx
        if not math.isinf(lattice[node_index_1][node_index_2]['d/dy']):    
            ddxTiy = lattice[node_index_1][node_index_2]['d/dy'] * VectorField.loc[node_index_1, 'Ti'][1]
        else:
            ddxTiy = 0
        if not math.isinf(lattice[node_index_1][node_index_2]['d/dx']):
            ddyTix = lattice[node_index_1][node_index_2]['d/dx'] * VectorField.loc[node_index_1, 'Ti'][0]  
        else:
            ddyTix = 0
        rotor_z_out = ddxTiy  - ddyTix
        lattice.nodes[node_index_2]['V_in'] = node_Vin
        lattice.nodes[node_index_2]['V_out'] = node_Vout
        lattice.nodes[node_index_2]['index'] = VectorField.loc[node_index_1, 'index']
        lattice.nodes[node_index_2]['rotor_z_in'] = rotor_z_in
        lattice.nodes[node_index_2]['rotor_z_out'] = rotor_z_out
    return lattice

def SmoothPotential(lattice):
    # Smooth V_in and V_out by taking the average over all the neighbors
    for node_str in lattice.nodes:
        neighbors = list(lattice.neighbors(node_str))
        num_neighbors = len(neighbors)
        
        # Calculate the average of V_in and V_out for the neighbors
        avg_Vin = sum(lattice.nodes[neighbor]['V_in'] for neighbor in neighbors) / num_neighbors
        avg_Vout = sum(lattice.nodes[neighbor]['V_out'] for neighbor in neighbors) / num_neighbors
        
        # Assign the average values to the current node
        lattice.nodes[node_str]['V_in'] = avg_Vin
        lattice.nodes[node_str]['V_out'] = avg_Vout    
    return lattice

def ConvertLattice2PotentialDataframe(lattice):
    '''
        Input: 
            Lattice with potential
        Output:
            Dataframe with:
                V_in, V_out, centroid (x,y), index, node_id(i,j)
        Usage:
            3D plot for Potential and Lorenz Curve.
    '''
    data_ = []
    for node,data in lattice.nodes(data=True):
        # Extract the indices of the nodes
        ij = ast.literal_eval(node)    
        node_id = (ij[0],ij[1])
        # Compute the value of V_in for the edge
        node_Vin = lattice.nodes[node]['V_in']  
        
        # Compute the value of V_out for the edge
        node_Vout = lattice.nodes[node]['V_out']
        
        x = lattice.nodes[node]['x']
        y = lattice.nodes[node]['y']
        index_ = lattice.nodes[node]['index']
        rotor_z_in = lattice.nodes[node]['rotor_z_in']
        rotor_z_out = lattice.nodes[node]['rotor_z_out']
        # Save the information to the list
        data_.append({'V_in': node_Vin, 'V_out': node_Vout,'index': index_ ,'node_id': node_id,'x':x,'y':y,'rotor_z_in':rotor_z_in,'rotor_z_out':rotor_z_out})
        
        # Create a DataFrame from the list
        PotentialDataframe = pd.DataFrame(data_)
        PotentialDataframe['index'] = PotentialDataframe.index
        # Format the 'node_id' column using ast.literal_eval
#        PotentialDataframe['node_id'] = PotentialDataframe['node_id'].apply(ast.literal_eval)
    return PotentialDataframe

def CompletePotentialDataFrame(VectorField,grid,PotentialDataframe):
    PotentialDataframe['Ti'] = VectorField['Ti']
    PotentialDataframe['population'] = grid['population']    
    return PotentialDataframe

def SavePotentialDataframe(PotentialDataFrame,save_dir):
    PotentialDataFrame.to_csv(os.path.join(save_dir,'PotentialDataFrame.csv'))    

def GetSavedPotentialDF(save_dir):
    return pd.read_csv(os.path.join(save_dir,'PotentialDataFrame.csv'))



# ------------------------------------------------- DEBUGGING -------------------------------------------------#

def DebuggingGetd0Cellx(count,NCycleControl,VnpeopleX,VoriginsX,VdestinationsX,cell_x):
    if count % NCycleControl == 0:
        logging.info(f'cell_x: {cell_x}')
        logging.info(f'Number Destination from {cell_x}: {len(VnpeopleX)}')
        if len(VnpeopleX) < 30:
            logging.info(f'Vector of people:\n{VnpeopleX}')
            logging.info(f'Indices Origin:\n{VoriginsX}')
            logging.info(f'Indices Destination:\n{VdestinationsX}')


def DebuggingGetd0Celli(count_i,NCycleControl,VnpeopleXi,VoriginsXi,VdestinationsXi,cell_i,cell_x):
    if count_i % NCycleControl == 0:
        logging.info(f'cell_i: {cell_i}')
        logging.info(f'Number Destination xi: {cell_x}-{cell_i}: {len(VnpeopleXi)}')
        if len(VnpeopleXi) < 30:
            logging.info(f'Vector of people:\n{VnpeopleXi}')
            logging.info(f'Indices Origin:\n{VoriginsXi}')
            logging.info(f'Indices Destination:\n{VdestinationsXi}')
            count_i += 1



##------------------------------------------------ RANDOM FUNCTIONS ----------------------------------------------#
def filter_within_percentage(arr, lower_percentile, upper_percentile):
    # Calculate the lower and upper percentile values
    lower_val = np.percentile(arr, lower_percentile)
    upper_val = np.percentile(arr, upper_percentile)
    
    # Filter values within the specified percentile range
    filtered_values = arr[(arr >= lower_val) & (arr <= upper_val)]
    
    return filtered_values



##-------------------------------------------- PLOTS ----------------------------------------------------------#

def PlotVFPotMass(grid,SFO_obj,PotentialDataframe,VectorField,dir_grid,label_potential = 'V_out',label_fluxes = 'Ti',plot_mass = True,verbose = False):
    '''
        NOTE:
            label_potential:    V_in, V_out
            label_fluxes:       Tj  , Ti
        USAGE:
            PlotVFPotMass(grid,SFO_obj,PotentialDataframe,VectorField,label_potential = 'V_out',label_fluxes = 'Ti')
            PlotVFPotMass(grid,SFO_obj,PotentialDataframe,VectorField,label_potential = 'population',label_fluxes = 'Ti')

    '''
    labelf2title = {'Tj': 'Incoming','Ti':'Outgoing'}
    label2save = {'V_in':'Potential','V_out':'Potential','population':'Mass'}
    fig, ax = plt.subplots(figsize=(15, 15))
    centroid_coords = np.array([grid['centroidx'].to_numpy(),grid['centroidy'].to_numpy()])
    centroid_coords = centroid_coords.T
    #grav_vector_field = gravitational_field(fluxes_matrix,normalized_vectors,nv)
    SFO_obj.gdf_polygons.plot(ax=ax, color='white', edgecolor='black')
    if label_potential in PotentialDataframe.columns: 
        grid[label_potential] = PotentialDataframe[label_potential]
    elif label_potential in grid.columns:
        pass
    else:
        raise KeyError(label_potential,' Neither in grid nor potential columns: ',grid.columns,PotentialDataframe.columns)
    if plot_mass:
        grid_plot = grid.plot(ax=ax, column = label_potential, cmap = 'Greys',edgecolor='black', alpha=0.3)
        grid_cbar = plt.colorbar(grid_plot.get_children()[1], ax=ax)
        grid_cbar.set_label('{}'.format(label_potential), rotation=270, labelpad=15)
    else:
        pass
    VF = np.stack(VectorField[label_fluxes].to_numpy(dtype = np.ndarray))
    VF_norm = np.linalg.norm(VF, axis=1)
    VF_normalized = np.stack(np.array([VF[i] / VF_norm[i] if VF_norm[i] !=0 else [0, 0] for i in range(len(VF_norm))]))
    mask = [True if VF_norm[i]!=0 else False for i in range(len(VF_norm))]
    quiver_plot = ax.quiver(centroid_coords[mask,0], centroid_coords[mask,1], VF_normalized[mask,0], VF_normalized[mask,1],
            VF_norm[mask], cmap='inferno_r', angles='xy', scale_units='xy', scale=60, width=0.005,headwidth=1, headlength=3)
#    quiver_plot = ax.quiver(centroid_coords[:,0], centroid_coords[:,1], VF_normalized[:,0], VF_normalized[:,1],
#            VF_norm, cmap='inferno', angles='xy', scale_units='xy', scale=50, width=0.005,headwidth=1, headlength=2)
    quiver_cbar = plt.colorbar(quiver_plot, ax=ax)
    quiver_cbar.set_label('Normalized Vector Magnitude', rotation=270, labelpad=15)
    ax.set_title('Vector Field {} Fluxes'.format(labelf2title[label_fluxes]))
    plt.savefig(os.path.join(dir_grid,'{0}Flux{1}.png'.format(labelf2title[label_fluxes],label2save[label_potential])),dpi = 200)
    if verbose:
        plt.show()