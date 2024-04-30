import numpy as np
import sys
import matplotlib.pyplot as plt
import numba
import logging
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
##------------------------------ GRAVITATIONAL MODEL MODIFIED ------------------------------##
@numba.njit(['int32(int32[:], int32[:], int32[:])'])
def ComputeMaxSizek(Vnpeople, Vorigins, Vdestinations):
    '''
        Input:
            Vnpeople: np.array(int32) [number of people exchanged]  example: [1,1,5,...,32,12,1,...]
            Vorigins: np.array(int32) [indices origin]              example: [32,32,32,...,34,34,34,34,...] with repetitions as it is extended for each dest
            Vdestination: np.array(int32) [indices destination]     example: [34,42,56,...,97,108,115,122,...]
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
    OriginDestination = np.zeros((max_size,2),dtype = np.float32)
    for cell_i in Vorigins:
        # Get Fluxes from cell_i
        _,VnpeopleI, VoriginsI, VdestinationsI = SubsampleByCellFluxOrigin(Vnpeople, Vorigins, Vdestinations,cell_i)
        _,_,VgridPopulationI = SubSampleGridByCell(VgridIdx, VgridPopulation,cell_i)
        # Get Row of the grid
        for cell_j in VdestinationsI:
            if cell_i <= cell_j: #NOTE: UnCommitted change: <= instead of <
                _,VnpeopleIJ, _, VdestinationsIJ = SubsampleByCellFluxDestination(VnpeopleI, VoriginsI, VdestinationsI,cell_j)
                _,_,VgridPopulationJ = SubSampleGridByCell(VgridIdx, VgridPopulation,cell_j)
#                       if debug:
#                           DebuggingGetd0Celli(count_j,NCycleControl,VnpeopleXj,VoriginsXj,VdestinationsXj,cell_j)
#                           count_j += 1
                if len(VnpeopleIJ)==1:
                    idx_i0 = VoriginsI[0]
                    idx_i1 = VdestinationsIJ[0]
                    OriginDestination[count][0] = idx_i0
                    OriginDestination[count][1] = idx_i1
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
    idx = np.where(Fluxes > 0)
    Fluxes = Fluxes[idx]
#    idx = np.where(Massi != 0)
    Massi = Massi[idx]
#    idx = np.where(Massj != 0)
    Massj = Massj[idx]
#    idx = np.where(Fluxes != 0)
    DistanceVector = DistanceVector[idx]
    OriginDestination = OriginDestination[idx]
    VespignaniVector = [Massi,Massj,DistanceVector]
    return VespignaniVector,Fluxes,OriginDestination

##-------------- VESPIGNANI FEATURES -----------------##
def ComputeVespignaniVectorFluxesOD(df_distance,grid,Tij,verbose):
    '''
        Input:  
            df_distance: distance matrix
            grid: grid
            Tij: (int32[:], int32[:], int32[:]) -> number of people, origin, destination
        Output:
            VespignaniVector: (Massi,Massj,Dij)                         -> ([massi,massi,...,],[massj,massj1,...],[dij,dij1,...])
            Fluxes: (int32[:])                                          ->  [fluxesij,fluxesij1,...]
            OriginDestination: ([[int32, int32]...,[int32, int32]...])  -> [[i,j],[i,j1],...]
    '''
    # INITIALIZING POPULATION, DISTANCE, FLUXES
    distance_matrix = DistanceDf2Matrix(df_distance) # 1.2 s with 3470 grids
    VgridIdx,VgridPopulation = Grid2Arrays(grid) # 0.0003 with 3470
    Vnpeople,Vorigins,Vdestinations = T2Arrays(Tij) # 0.07
    if verbose:
        print("Check Consistency Input Vectors: ")
        print("Distance Matrix: ",len(distance_matrix))
        print("Vector Indices Grid: ",len(VgridIdx))
        print("Vector Population Grid: ",len(VgridPopulation))
        print("Vector Number People: ",len(Vnpeople))
        print("Vector Origins: ",len(Vorigins))
        print("Vector Destinations: ",len(Vdestinations))
        fig,(ax00,ax01,ax10,ax11) = plt.subplots(2,2,figsize = (10,8))
        ax00.hist(Vnpeople,bins = 50)
        ax00.set_xlabel('Flux')
        ax00.set_ylabel('Count')
        ax01.plot(Vorigins,Vdestinations)
        ax01.set_xlabel('Origin')
        ax01.set_ylabel('Destination')
        ax10.plot(Vorigins,np.arange(len(df_distance)))
        ax10.set_xlabel('Distance Index')
        ax10.set_ylabel('Origin Index')
        ax10.set_title('Index Grid')
        ax11.hist(VgridPopulation,bins = 50)
        ax11.set_xlabel('Population')
        ax11.set_ylabel('Count')
        
    Vnpeople, Vorigins, Vdestinations = SubsampleFluxesByPop(Vnpeople, Vorigins, Vdestinations)
    #EstimateFluxesScaled,Fluxes,DistanceVector,ErrorEsteem,ErrorFluxes,ErrorDist,Massi,Massj = GetEstimationFluxesVector(Vnpeople, Vorigins, Vdestinations, VgridIdx, VgridPopulation, distance_matrix,d0)

    # [DistanceVector,Massi,Massj], Fluxes -> 1D vectors for all couples of OD.
    print('Start to compute vespignani Features')
    VespignaniVector,Fluxes,OriginDestination = PrepareVespignani(Vnpeople, Vorigins, Vdestinations, VgridIdx, VgridPopulation, distance_matrix)
    return VespignaniVector,Fluxes,OriginDestination



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
@numba.njit(['(int32[:], int32[:], int32[:],int32)'],parallel=True)
def FilterVespignaniVectorByDistance(Massi,Massj,DistanceVector,Fluxes,k):
    '''
        Input: 
            Massi: (int32[:]) ->  [massi,massi,...,]
            Massj: (int32[:]) ->  [massj,massj1,...]
            DistanceVector: (int32[:]) ->  [distanceij,distanceij1,...]
            Fluxes: (int32[:]) ->  [fluxesij,fluxesij1,...]
            k = Index of the bin used to cut the vectors
        Output:
            VespignaniVector0: [Massi,Massj,DistanceVector] With 
        NOTE:
            VespignaniVector: Contains just the entrances that are different from 0
        Description:
            This function cuts the vespignani vector by distance. That is the flux,mass,distance vectors.
        Goal:
            Extrcact The Fluxes,Masses and Distances for which the distance
            amonng the grids is in the bin.
            
    '''
    assert len(Massi) == len(Massj) == len(DistanceVector) == len(Fluxes),'The dimensions of the vectors are not consistent'
    assert 0 not in  Massi and 0 not in  Massj and 0 not in  Fluxes, 'You forgot to filter the Masses'
    # Partition Distance
    n,bins = np.histogram(DistanceVector,bins = 50)
    # Partition Fluxes
    length_vector =len(Massi)
    FilteredDistances0 = np.zeros(length_vector)
    FilteredMassi0 = np.zeros(length_vector)
    FilteredMassj0 = np.zeros(length_vector)
    FilteredFluxes0 = np.zeros(length_vector)
    FilteredDistancesEnd = np.zeros(length_vector)
    FilteredMassiEnd = np.zeros(length_vector)
    FilteredMassjEnd = np.zeros(length_vector)
    FilteredFluxesEnd = np.zeros(length_vector)

    for i in range(len(bins)-k -1):
        # Choose just entrances for which dij is contained in the bin
        IndicesWithDistanceInBin = np.where(((DistanceVector>bins[i]) & (DistanceVector < bins[i+1])))[0]
        # Put in those indices something different from 0
        for index in IndicesWithDistanceInBin:
            FilteredMassi0[index] = Massi[index]
            FilteredMassj0[index] = Massj[index] 
            FilteredDistances0[index] = DistanceVector[index]
            FilteredFluxes0[index] = Fluxes[index]
    # Choose just entrances for which dij that are bigger then the highest bin considered
    IndicesWithDistanceOutBin = np.where((DistanceVector>bins[len(bins)-k-1]))[0]
    for index in IndicesWithDistanceOutBin:
        FilteredMassiEnd[index] = Massi[index]
        FilteredMassjEnd[index] = Massj[index] 
        FilteredDistancesEnd[index] = DistanceVector[index]
        FilteredFluxesEnd[index] = Fluxes[index]
    # Remove 0 values
    ValidIndices = np.where(FilteredMassi0 > 0)[0]
    ComplementValidIndices = np.where(FilteredMassiEnd > 0)[0]

#    if verbose:
#        print('Number of Valid Indices: ',len(ValidIndices))
#        print('Number of Invalid Indices: ',len(FilteredMassi) - len(ValidIndices))
    return [FilteredMassi0[ValidIndices],FilteredMassj0[ValidIndices],FilteredDistances0[ValidIndices]],FilteredFluxes0[ValidIndices],[FilteredMassiEnd[ComplementValidIndices],FilteredMassjEnd[ComplementValidIndices],FilteredDistancesEnd[ComplementValidIndices]],FilteredFluxesEnd[ComplementValidIndices]
