import numpy as np
import sys
import matplotlib.pyplot as plt
import numba
import logging
import json
import os
from collections import defaultdict
from FittingProcedures import multilinear4variables
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
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
        @param Vnpeople: (np.array 1D) -> number of people
        @param Vorigins: (np.array 1D) -> origin
        @param Vdestinations: (np.array 1D) -> destination
        @return Vnpeople: (np.array 1D) -> number of people
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
    EstimateFluxesScaled = np.zeros(max_size,dtype = np.float64)
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
            if cell_i < cell_j: #NOTE: UnCommitted change: <= instead of <
                _,VnpeopleIJ, _, VdestinationsIJ = SubsampleByCellFluxDestination(VnpeopleI, VoriginsI, VdestinationsI,cell_j)
                _,_,VgridPopulationJ = SubSampleGridByCell(VgridIdx, VgridPopulation,cell_j)
#                       if debug:
#                           DebuggingGetd0Celli(count_j,NCycleControl,VnpeopleXj,VoriginsXj,VdestinationsXj,cell_j)
#                           count_j += 1
                if len(VnpeopleIJ)==1 and VnpeopleIJ[0] != 0 and VgridPopulationI[0] != 0 and VgridPopulationJ[0] != 0:
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
                    #print("Mass i {} Mass j {} Fluxes {}".format(VgridPopulationI[0],VgridPopulationJ[0],VnpeopleIJ[0]))
                    pass
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
def ComputeVespignaniVectorFluxesOD(df_distance,grid,Tij):
    '''
        @param df_distance: (pd.DataFrame) [i,j,distance]
        @param grid: (pd.DataFrame) [index,population]
        @param Tij: (pd.DataFrame) [number_people,origin,destination]
        @param verbose: (bool) -> Print the number of valid and invalid indices
        Input:  
            df_distance: distance matrix
            grid: grid
            Tij: (int32[:], int32[:], int32[:]) -> number of people, origin, destination
        Output:
            VespignaniVector: (Massi,Massj,Dij)                         -> ([massi,massi,...,],[massj,massj1,...],[dij,dij1,...])
            Fluxes: (int32[:])                                          ->  [fluxesij,fluxesij1,...]
            OriginDestination: ([[int32, int32]...,[int32, int32]...])  -> [[i,j],[i,j1],...]
    '''
    # pivot into distance matrix: [d00,d01,...]
    logger.info("df_dist -> [d00,d01,...] ...")
    distance_matrix = DistanceDf2Matrix(df_distance) # 1.2 s with 3470 grids
    # Transform grid
    logger.info("grid -> [mass0,mass1,...] ...")
    logger.info("grid -> [index0,index1,...,max(grid.index)] ...")
    VgridIdx,VgridPopulation = Grid2Arrays(grid) # 0.0003 with 3470
    logger.info("Tij -> NT_i = [number_people00,number_people01,...] ...")
    logger.info("Tij -> O_i = [origin0,origin0,...] ...")
    logger.info("Tij -> D_i = [destination0,destination1,...] ...")
    Vnpeople,Vorigins,Vdestinations = T2Arrays(Tij) # 0.07   
    logger.info("Subsampling (NT,O,I)_i where NT > 0 ...")     
    Vnpeople, Vorigins, Vdestinations = SubsampleFluxesByPop(Vnpeople, Vorigins, Vdestinations)
    #EstimateFluxesScaled,Fluxes,DistanceVector,ErrorEsteem,ErrorFluxes,ErrorDist,Massi,Massj = GetEstimationFluxesVector(Vnpeople, Vorigins, Vdestinations, VgridIdx, VgridPopulation, distance_matrix,d0)
    
    # [DistanceVector,Massi,Massj], Fluxes -> 1D vectors for all couples of OD.
    logger.info("Computing: [Massi,Massj,Dij],Fluxes,OriginDestination ...")
    VespignaniVector,Fluxes,OriginDestination = PrepareVespignani(Vnpeople, Vorigins, Vdestinations, VgridIdx, VgridPopulation, distance_matrix)
    VespignaniVector = np.array([np.array(VespignaniVector[0],dtype = np.int32),np.array(VespignaniVector[1],dtype = np.int32),np.array(VespignaniVector[2],dtype = np.float32)])
    Massi = np.array(VespignaniVector[0],dtype = np.int32)
    Massj = np.array(VespignaniVector[1],dtype = np.int32)
    DistanceVector = np.array(VespignaniVector[2],dtype = np.float32)
    Fluxes = np.array(Fluxes,dtype = np.int32)
    assert len(Massi) == len(Massj) == len(DistanceVector) == len(Fluxes),'ComputeVespignaniVectorFluxesOD: The dimensions of the vectors are not consistent'
    assert 0 not in  Massi and 0 not in  Massj and 0 not in  Fluxes, 'ComputeVespignaniVectorFluxesOD: You forgot to filter the Masses'
    return Massi,Massj,DistanceVector,Fluxes,OriginDestination



def PlotVespignaniFit(EstimateFluxesScaled,Fluxes,DistanceVector,Massi,Massj):
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    n,bins = np.histogram(DistanceVector,bins = 50)
    AvgFluxGivenR =[Fluxes[np.where(DistanceVector>bins[i]) and DistanceVector < bins[i+1]]/(Massi[np.where(DistanceVector>bins[i]) and DistanceVector < bins[i+1]]*Massj[np.where(DistanceVector>bins[i]) and DistanceVector < bins[i+1]]) for i in range(len(bins)-1)]
    ax.boxplot(bins[:-1],AvgFluxGivenR)
    ax.set_xlabel('R(km)')
    ax.set_ylabel('W/(mi*mj)')
    ax.legend(['Fluxes','Gravity'])
    plt.show()


##----------------------------- VESPIGNANI FITTING -----------------------------##

def VespignaniBlock(df_distance,grid,Tij,potentialdir):
    """
        @df_distance: (pd.DataFrame) [i,j,distance]
        @grid: (pd.DataFrame) [index,population]
        @Tij: (pd.DataFrame) [number_people,origin,destination]
        @potentialdir: (str) -> Directory where to save the potential
    """
    Massi,Massj,DistanceVector,Fluxes,OriginDestination = ComputeVespignaniVectorFluxesOD(df_distance,grid,Tij)
    # FIT
    VespignaniVector = np.array([Massi,Massj,DistanceVector])
    # SAVE FIT
    if not os.path.isfile(os.path.join(potentialdir,'FitVespignani.json')):
        logger.info("Fitting Gravitational Model ...")
        # NOTE: The Guess For the fitting Procedure is that the multiplicative factor is = 0, therefore the normalization is = 1, then the masses are linearly related to the fluxes, and the typical length is 100 km
        k,error = Fitting(VespignaniVector,np.array(Fluxes),label = 'vespignani',initial_guess = [0,1,1,0.001] ,bounds = (np.array([-50,0,0,-2]),np.array([50,2,2,0])) ,maxfev = 30000)
        with open(os.path.join(potentialdir,'FitVespignani.json'),'w') as f:
            json.dump({'logk':k[0],'alpha': k[1],'gamma': k[2],'1/d0':k[3]},f)
    else:
        logger.info("Loading Fitting Gravitational Model ...")
        with open(os.path.join(potentialdir,'FitVespignani.json'),'r') as f:
            d = json.load(f)
        k = [d['logk'],d['alpha'],d['gamma'],d['1/d0']]
    logger.info("Checking Fit ...")
    n,bins = np.histogram(DistanceVector,bins = 50)
    # PLOT '$W_{ij}/(m_i^{{\\alpha}} m_j^{{\\gamma}})$'
    EstimatedVectorFluxesVespignani = multilinear4variables(VespignaniVector,k[0],k[1],k[2],k[3])
    assert len(Fluxes) == len(EstimatedVectorFluxesVespignani),"Estimated Fluxes and Fluxes have not same shape"
    logger.info("EFluxes_ij/(M_i*Mj)...")
    WOverMM = [EstimatedVectorFluxesVespignani[np.where(((VespignaniVector[2]>bins[i]) & (VespignaniVector[2] < bins[i+1])))]/(VespignaniVector[0][np.where(((VespignaniVector[2]>bins[i]) & (VespignaniVector[2] < bins[i+1])))]**k[0]*VespignaniVector[1][np.where(((VespignaniVector[2]>bins[i]) & (VespignaniVector[2] < bins[i+1])))])**k[1] for i in range(len(bins)-1)]
    error = [np.std(WOverMM[i])/np.sqrt(len(WOverMM[i])) for i in range(len(WOverMM))]
    mean = [np.mean(WOverMM[i]) for i in range(len(WOverMM))]
    # PLOT '$W_{ij} (M)/W_{ij} (D)$'
    WM = [EstimatedVectorFluxesVespignani[np.where(((VespignaniVector[2]>bins[i]) & (VespignaniVector[2] < bins[i+1])))] for i in range(len(bins)-1)]
    WD = [Fluxes[np.where(VespignaniVector[0]>bins[i]) and VespignaniVector[0] < bins[i+1]] for i in range(len(bins)-1)]
    meanWM = np.array([np.mean(WM[i]) if np.mean(WM[i])>0 else 0 for i in range(len(WM))] )
    errorWM = np.array([np.std(WM[i])/np.sqrt(len(WM[i])) for i in range(len(WM))]) 
    meanWD = np.array([np.mean(WD[i]) for i in range(len(WD))]) 
    errorWD = np.array([np.std(WD[i])/np.sqrt(len(WD[i])) for i in range(len(WD))]) 


    WMoverWD = meanWM/meanWD
    error1 = errorWM/meanWD + meanWM/(meanWD)**2*errorWD 
    # PLOTTING
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    plt.errorbar(bins[:-1],mean,yerr = error, fmt='o', capsize=5, color='red')
    plt.yscale('log')
    plt.xlim(0,90)
    plt.xlabel('R(km)')
    plt.ylabel('$W_{ij}/(m_i^{{\\alpha}} m_j^{{\\gamma}})$')
    plt.savefig(os.path.join(potentialdir,'PlotFitVespignani.png'),dpi = 200)
    plt.show()

    fig,ax = plt.subplots(1,1,figsize = (10,10))
    plt.errorbar(bins[:-1],WMoverWD[:],yerr = error1[:], fmt='o', capsize=5, color='red')
    plt.yscale('log')
    plt.xlabel('R(km)')
    plt.xlim(0,90)
    plt.ylabel('$W_{ij} (M)/W_{ij} (D)$')
    plt.savefig(os.path.join(potentialdir,'PlotFitVespignaniDataModel.png'),dpi = 200)
    plt.show()

    return k[0],k[1],k[2],k[3]