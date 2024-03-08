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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append('~/berkeley/traffic_phase_transition/scripts')
from FittingProcedures import Fitting
def OD2Gradient(df_grid,grid,lattice):
    df_grid.groupby('origin')['destination'].unique().apply(lambda x: x[0]).to_dict()


## FITTING THE POTENTIAL
'''
def d0PotentialFit(grid,df_distance,T):
    
        Input:
            grid: output of GetGrid
            df_distance: output of distance_matrix
            T is the output of OD2Grid
        Output:
            d0: float -> The value of d0
    
    d0s = []
    for _,cell_x in grid.iterrows():
        for _,cell_i in grid.iterrows():
            for _,cell_j in grid.iterrows():
                if cell_i['index']!=cell_j['index'] and cell_i['index']!=cell_x['index'] and cell_j['index']!=cell_x['index']:
                    dxi = df_distance['distance'].loc[df_distance['i'] == cell_i['index']].loc[df_distance['j'] == cell_x['index']]
                    dxj = df_distance['distance'].loc[df_distance['i'] == cell_j['index']].loc[df_distance['j'] == cell_x['index']]
                    mi = cell_i['population']
                    mj = cell_j['population']
                    Txi = T['number_people'].loc[T['origin'] == cell_x['index']].loc[T['destination'] == cell_i['index']]
                    Txj = T['number_people'].loc[T['origin'] == cell_x['index']].loc[T['destination'] == cell_j['index']]
                    d0s.append(dxi-dxj/np.log(Txi*mi/(Txj*mj)))
    return np.median(d0s)
'''
def chunked_combinations(subsetflux, x, chunk_size=1000):
    combinations = ((i, j) for i in subsetflux['destination'] for j in subsetflux['destination'] if i != j != x)
    chunk = []
    for combination in combinations:
        chunk.append(combination)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

#### CONVERSION DATA:
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
            chunk_size: (int) -> Number of combinations to process in each chunk
        Output:
            Vnpeople: (np.array 1D) -> number of people
            Vorigins: (np.array 1D) -> origin
            Vdestinations: (np.array 1D) -> destination
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
            chunk_size: (int) -> Number of combinations to process in each chunk
        Output:
            Vnpeople: (np.array 1D) -> number of people
            Vorigins: (np.array 1D) -> origin
            Vdestinations: (np.array 1D) -> destination
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
    '''
    IndicesXGrid = np.where(VgridIdx == cell_x)
    VgridIdxX = VgridIdx[IndicesXGrid]
    VgridPopulationX = VgridPopulation[IndicesXGrid]
    return IndicesXGrid,VgridIdxX,VgridPopulationX

#------------------------------------------------- POTENTIAL  -------------------------------------------------#
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

## PREPARE VESPIGNANI
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
    idx = np.where(Massi != 0)
    Massi = Massi[idx]
    idx = np.where(Massj != 0)
    Massj = Massj[idx]
    idx = np.where(DistanceVector != 0)
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
    
def VectorField(lattice,df_distance,Tij):  
    TijDirection = []  
    for i in df_distance['i'].tolist():
        uix = df_distance.loc[df_distance['i'] == i]
        Tix = Tij.loc[Tij['i'] == i]
        Tiui = []
        for j in uix['j'].tolist():
            Tiui.append(Tix.loc[Tix['j'] == j]*np.array(uix['dir_vector'].loc[uix['j'] == j]))
        Tiui = np.sum(Tiui)
        TijDirection.append(Tiui)
    return TijDirection        



def GetPotential(TijDirection,lattice):
    '''TODO: Implement this function'''
    Vi = []
    for node in lattice.nodes(data=True):
        for edge in lattice.edges():
            np.array(lattice[edge[0]][edge[1]]['dx']*TijDirection[edge[0]][0])-np.array(lattice.nodes[edge[0]]['x'])
    
def Potential(grid,df_distance,Tij,lattice,save_dir):
    for i in range(len(grid)):
        for j in range(len(grid)):
            if i < j:
                dij = df_distance.loc[df_distance['i'] == i].loc[df_distance['j'] == j]['distance'].values[0]
                if np.isnan():
                    mi = grid[i]['population']
                    mj = grid[j]['population']
                    Tij['potential'].loc[Tij['i'] == i].loc[Tij['j'] == j] = k*mi*mj*np.exp(dij/d0)

def OD2matrix(DfGrid,direction_matrix,gridIdx2ij,lattice,grid):
    '''
        Create: 
            1) 02mapidxorigin {0:origin1,1:origin2,...}
            2) 02mapidxdestination {0:destination1,1:destination2,...}
            3) mapidx2origin {origin1:0,origin2:1,...}
            4) mapidx2destination {destination1:0,destination2:1,...}
            origin_i,destination_i are the nominative of self.gdf_polygons
    '''
    origins = np.unique(DfGrid['origin'].to_numpy())
    destinations = np.unique(DfGrid['destination'].to_numpy())
    mapOD = DfGrid.groupby('origin')['destination'].unique().apply(lambda x: x[0]).to_dict()
    zero2mapidxorigin = {i:idx for idx,i in enumerate(origins)}
    zero2mapidxdestination = {i:idx for idx,i in enumerate(destinations)}
    mapidx2origin = {idx:i for idx,i in enumerate(origins)}
    mapidx2destination = {idx:i for idx,i in enumerate(destinations)}
    dimensionality_matrix = (len(origins),len(destinations))
    matrix_OD = np.zeros(dimensionality_matrix)
    trips_OD = DfGrid.groupby(['origin']).count()['destination'].to_numpy(dtype= int)
    for idx in range(len(origins)):
        matrix_OD[zero2mapidxorigin[origins[idx]],zero2mapidxdestination[mapOD[origins[idx]]]] = trips_OD[idx]
    matrix_out = np.sum(matrix_OD,axis=1)
    matrix_in = np.sum(matrix_OD,axis=0)
    return matrix_OD,matrix_out,matrix_in,zero2mapidxorigin,zero2mapidxdestination,mapidx2origin,mapidx2destination



def GradientFromPotential(grid,
                          lattice):
    '''
        Input:
            dfOD: (pd.DataFrame) [i,j,Flux,'(i,j)O','(i,j)D']
            LatticeGrid: (nx.Graph) LatticeGrid NODES: [(i,j),'centroidx','centroidy'] EDGES: ['d\dx','d\dy','dx','dy']
        Create the vector field from the potential
    '''
    point_coords = np.array([np.array([lattice.nodes[node]['x'],lattice.nodes[node]['y']]) for node in lattice.nodes() if 'x' in lattice.nodes[node].keys() and 'y' in lattice.nodes[node].keys()]) #SHAPE (ncenters,2)
    points_3d = point_coords[:, np.newaxis, :]
    Directions = point_coords - points_3d #SHAPE (ncenters,ncenters,2)`
    Dij = distance_matrix(point_coords,point_coords) #SHAPE (ncenters,ncenters)
#    Dij =  np.linalg.norm(Directions, axis=2)[:, :, np.newaxis] #SHAPE (ncenters,ncenters,1)
#    Dij = np.array([np.array([Dij[i][j][0] if Dij[i][j][0] != 0 else 1 for j in range(len(Dij[i]))]) for i in range(len(Dij))]) #SHAPE (ncenters,ncenters)

    eij = [[Directions[i][j] / Dij[i][j] for j in range(len(Directions[i]))] for i in range(len(Directions))] #SHAPE (ncenters,ncenters,2)
    not_yet_flux_vectors = np.array([np.array([eij[i][j]*fluxes_matrix[i][j] for j in range(len(eij[0]))]) for i in range(len(eij))])
    flux_vectors = np.sum(not_yet_flux_vectors,axis=1) #SHAPE (ncenters,2)
    mass = np.sum(matrix_out)


def Lorenz(DensityVector):
    '''
        Lorenz curve is the comulative probability distribution function of the density (In our case potential).
        Output:
            CDF: vector(float) -> Increases
            SortedByDensity2PrimitiveIndex: dict -> {0: idx DensityVector that is smallest,....,n: Idx DensityVector that is biggest}
    '''
    Dv = np.argsort(DensityVector)
    SortedByDensity2PrimitiveIndex = {idx: Dv[idx] for idx in range(len(Dv))}
    Z = np.sum(Dv)
    CDF = np.cumsum(DensityVector)/Z
    return CDF,SortedByDensity2PrimitiveIndex

    





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
