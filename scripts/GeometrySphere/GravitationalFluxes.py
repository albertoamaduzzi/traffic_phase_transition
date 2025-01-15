import numpy as np
import sys
import matplotlib.pyplot as plt
import numba
import logging
import json
import os
from collections import defaultdict
from FittingProcedures import *
import polars as pl
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.append('~/berkeley/traffic_phase_transition/scripts')

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
def _ComputeVespignaniVectorFluxesOD(df_distance,grid,Tij):
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

def ComputeVespignaniVectorFluxesOD(df_distance,grid,Tij):
    '''
        @param df_distance: (pl.DataFrame) [i,j,distance]
        @param grid: (pd.DataFrame) [index,population]
        @param Tij: (pd.DataFrame) [number_people,origin,destination]
        @return: Tij_dist: pl.DataFrame:
        ['origin': index of the origin from grid index set.
         'destination': index of the destination from grid index set.
         'number_people': number of people exchanged between origin and destination.
         'dir_vector': unit direction vector between OD
         'distance': distance between OD
         'population_origin': population of the origin
         'population_destination': population of the destination]
        NOTE: Contains all the informations about the fluxes and the population and the distance
        and therefore about the fit.
             '''

    import polars as pl
    df_distance = pl.DataFrame(df_distance)
    Tij = pl.DataFrame(Tij)
    g_renamed = grid.rename(columns={"index":"index","population":"population_origin"})
    df_distance = df_distance.join(pl.DataFrame(g_renamed[["index","population_origin"]]), left_on="i", right_on = "index", how="inner")
    g_renamed = grid.rename(columns={"index":"index","population":"population_destination"})
    df_distance = df_distance.join(pl.DataFrame(g_renamed[["index","population_destination"]]), left_on="j", right_on = "index", how="inner")
    Tij_dist = Tij.join(df_distance, left_on=["origin","destination"],right_on = ["i","j"], how="inner")
    Tij_dist = Tij_dist.drop(["(i,j)D","(i,j)O"])
    return Tij_dist

##----------------------------- VESPIGNANI FITTING -----------------------------##
def GravityModel(Mi,Mj,Dij,k,alpha,gamma,d0minus1):
    """
        @param Mi: (int) -> Mass of the origin
        @param Mj: (int) -> Mass of the destination
        @param Dij: (float) -> Distance between origin and destination
        @param k: (float) -> Multiplicative factor
        @param alpha: (float) -> Exponent of the origin
        @param gamma: (float) -> Exponent of the destination
        @param d0minus1: (float) -> Exponential factor
        @return: (float) -> Gravity Fluxes
    """
    return k*Mi**alpha*Mj**gamma* np.exp(Dij*d0minus1)
def FluxesOverProductMasses(GravityFluxes,Mi,Mj,Alpha,Beta):
    return GravityFluxes/(Mi**Alpha*Mj**Beta)

def FilterGridWithPeopleAndFluxes(Tij_dist):
    """
        @param Tij_dist: (pl.DataFrame) [origin,destination,number_people,population_origin,population_destination,distance]
        NOTE: Filter the grid with the fluxes and the population
    """
    FilterPerGravityFit = (pl.col("number_people")>0,pl.col("population_origin")>0,pl.col("population_destination")>0)
    Tij_dist_fit_gravity = Tij_dist.filter(FilterPerGravityFit)
    print("Fraction Cells not Considered: ",len(Tij_dist_fit_gravity)/len(Tij_dist))
    return Tij_dist_fit_gravity
def AddGravityColumnTij(Tij_dist_fit_gravity,K,Alpha,Gamma,D0minus1):
    Tij_dist_fit_gravity = Tij_dist_fit_gravity.with_columns(
        pl.struct(["distance","population_origin","population_destination"])
        .map_batches(lambda x: GravityModel(
            x.struct.field("population_origin"),
            x.struct.field("population_destination"),
            x.struct.field('distance'),
            K,Alpha,Gamma,D0minus1)).alias("gravity_fluxes"))
    K1 = K*np.sum(Tij_dist_fit_gravity["number_people"].to_numpy())/np.sum(Tij_dist_fit_gravity["gravity_fluxes"].to_numpy())
    Tij_dist_fit_gravity = Tij_dist_fit_gravity.with_columns(
        pl.struct(["distance","population_origin","population_destination"])
        .map_batches(lambda x: GravityModel(
            x.struct.field("population_origin"),
            x.struct.field("population_destination"),
            x.struct.field('distance'),
            K1,Alpha,Gamma,D0minus1)).alias("gravity_fluxes"))
    return Tij_dist_fit_gravity,K1

def VespignaniBlock(df_distance,grid,Tij,potentialdir):
    """
        @df_distance: (pd.DataFrame) [i,j,distance]
        @grid: (pd.DataFrame) [index,population]
        @Tij: (pd.DataFrame) [number_people,origin,destination]
        NOTE: New Procedure to Fit.
    """
    
    Tij_dist = ComputeVespignaniVectorFluxesOD(df_distance,grid,Tij)
    Tij_dist_fit_gravity = FilterGridWithPeopleAndFluxes(Tij_dist)
    mimjdij = np.array([Tij_dist_fit_gravity['population_origin'].to_numpy(),Tij_dist_fit_gravity['population_destination'].to_numpy(),Tij_dist_fit_gravity['distance'].to_numpy()])
    Fluxes = Tij_dist_fit_gravity['number_people'].to_numpy()
    if not os.path.isfile(os.path.join(potentialdir,'FitVespignani.json')):
        logger.info("Fitting Gravitational Model ...")
        # NOTE: The Guess For the fitting Procedure is that the multiplicative factor is = 0, therefore the normalization is = 1, then the masses are linearly related to the fluxes, and the typical length is 100 km
        k,error = FittingGravity(mimjdij,np.log(np.array(Fluxes)),initial_guess = [EstimateLogk(Tij_dist_fit_gravity),0.01,0.01,-EstimateD0minus1(Tij_dist_fit_gravity)] ,bounds = (np.array([-50,0,0,-2]),np.array([50,2,2,0])) ,maxfev = 300000)
        K = np.exp(k[0][0])
        Alpha = k[0][1]
        Gamma = k[0][2]
        D0minus1 = k[0][3]
        Tij_dist_fit_gravity,K1 = AddGravityColumnTij(Tij_dist_fit_gravity,K,Alpha,Gamma,D0minus1)
        with open(os.path.join(potentialdir,'FitVespignani.json'),'w') as f:
            json.dump({'logk':np.log(K1),'alpha': k[0][1],'gamma': k[0][2],'1/d0':k[0][3]},f)
    else:
        logger.info("Loading Fitting Gravitational Model ...")
        with open(os.path.join(potentialdir,'FitVespignani.json'),'r') as f:
            d = json.load(f)
        k = [d['logk'],d['alpha'],d['gamma'],d['1/d0']]
        K = np.exp(k[0][0])
        Alpha = k[0][1]
        Gamma = k[0][2]
        D0minus1 = k[0][3]
    
    n,bins = np.histogram(Tij_dist["distance"].to_numpy(),bins = 50)
    # PLOT '$W_{ij}/(m_i^{{\\alpha}} m_j^{{\\gamma}})$'
    print("Maximum Fluxes: ",max(Tij_dist_fit_gravity["number_people"].to_numpy()))
    print("Maximum Gravity Fluxes: ",max(Tij_dist_fit_gravity["gravity_fluxes"].to_numpy()))
    print("Maximum Error In Fraction: ",max(np.sqrt((Tij_dist_fit_gravity["gravity_fluxes"].to_numpy() - Tij_dist_fit_gravity["number_people"].to_numpy())**2).mean()/Tij_dist_fit_gravity["number_people"].to_numpy()))
    print("Minimum Error in Fraction: ",min(np.sqrt((Tij_dist_fit_gravity["gravity_fluxes"].to_numpy() - Tij_dist_fit_gravity["number_people"].to_numpy())**2).mean()/Tij_dist_fit_gravity["number_people"].to_numpy()))
    print("Total Fluxes: ",sum(Tij_dist_fit_gravity["number_people"].to_numpy()))
    print("Total Gravity Fluxes: ",sum(Tij_dist_fit_gravity["gravity_fluxes"].to_numpy()))

    Distance = []
    WOverMM = []
    ErrorWOverMM = []
    WOverWD = []
    ErrorWOverWD = []
    for i in range(len(bins)-1):
        Tij_dist_bin_i = Tij_dist_fit_gravity.filter(pl.col('distance') > bins[i], 
                                pl.col('distance') < bins[i+1])
        Tij_dist_bin_i = Tij_dist_bin_i.with_columns(pl.struct(["gravity_fluxes","population_origin","population_destination"]).map_batches(lambda x: FluxesOverProductMasses(x.struct.field('gravity_fluxes'),x.struct.field("population_origin"),x.struct.field("population_destination"),k[0][1],k[0][2])).alias("W/(Mi^(a)Mj^(b))"))
        ErrorW0 = Tij_dist_bin_i.select((pl.col("W/(Mi^(a)Mj^(b))").std()).alias("error_W/(Mi^(a)Mj^(b))"))
        Tij_dist_bin_i = Tij_dist_bin_i.with_columns((pl.col("gravity_fluxes")/pl.col("number_people")).alias("We/Wd"))
        ErrorW1 = Tij_dist_bin_i.select((pl.col("We/Wd").std()).alias("error_We/Wd"))
        Distance.append(bins[i])
        WOverMM.append(np.mean(Tij_dist_bin_i['W/(Mi^(a)Mj^(b))'].to_numpy()))
        WOverWD.append(np.mean(Tij_dist_bin_i['We/Wd'].to_numpy()))
        if ErrorW0["error_W/(Mi^(a)Mj^(b))"][0] is None:
            ErrorWOverMM.append(0)
        else:
            ErrorWOverMM.append(ErrorW0["error_W/(Mi^(a)Mj^(b))"][0]*2)
        if ErrorW1["error_We/Wd"][0] is None:
            ErrorWOverWD.append(0)
        else:
            ErrorWOverWD.append(ErrorW1['error_We/Wd'][0]*2)

    MaxWOverWD = WOverWD[0]
    WOverWD = np.array(WOverWD)/MaxWOverWD
    ErrorWOverWD = np.array(ErrorWOverWD)/MaxWOverWD
    WOverMM = np.array(WOverMM)
    ErrorWOverMM = np.array(ErrorWOverMM)
    # PLOTTING
    fig,ax = plt.subplots(1,1,figsize = (10,10))

    ax.errorbar(bins[:-1],WOverMM,yerr = ErrorWOverMM, fmt='o', capsize=5, color='red')
    ax.plot(bins[:-1],WOverMM[0]*np.exp(bins[:-1]*D0minus1), color='black',linestyle='--')
    ax.text(10,0.1,r'$\frac{{1}}{{d_0}} = $' + str(round(D0minus1,3)),fontsize = 15)
    ax.set_yscale('log')
    ax.set_xlim(0,90)
    ax.set_xlabel('R(km)')
    ax.set_ylabel(r'$W_{ij}/(m_i^{\alpha} m_j^{\gamma})$')
    plt.savefig(os.path.join(potentialdir,'PlotFitVespignaniNew.png'),dpi = 200)
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    ax.errorbar(bins[:-1],WOverWD[:],yerr = ErrorWOverWD[:], fmt='o', capsize=5, color='red')
    ax.hlines(1,0,90,linestyle='--',color='black')
    ax.set_yscale('log')
    ax.set_xlabel('R(km)')
    ax.set_xlim(0,90)
    ax.set_ylabel('$W_{ij} (M)/W_{ij} (D)$')
    plt.savefig(os.path.join(potentialdir,'PlotFitVespignaniNewDataModel.png'),dpi = 200)
def _VespignaniBlock(df_distance,grid,Tij,potentialdir):
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