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
    distance_matrix = pivot_df.to_numpy()
    return distance_matrix

def Grid2Arrays(grid):
    gridIdx = grid['index'].to_numpy()
    gridPopulation = grid['population'].to_numpy()
    return gridIdx,gridPopulation

def T2Arrays(T):
    Vnpeople = T['number_people'].to_numpy()
    Vorigins = T['origin'].to_numpy()
    Vdestinations = T['destination'].to_numpy()
    return Vnpeople,Vorigins,Vdestinations

## SUBSAMPLING

#------------------------------------------------- FLUXES  -------------------------------------------------#
@numba.njit(nopython=True)
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


@numba.njit(nopython=True)
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

@numba.njit(nopython=True)
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

@numba.njit(nopython=True)
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
## POTENTIAL FITTING
@numba.njit(nopython=True)
def d0PotentialFitOptimized(Vnpeople, Vorigins, Vdestinations, VgridIdx, VgridPopulation, DistanceMatrix,debug = False):
#    if debug:
#        count = 0
#        NCycleControl = 1000
    d0s = []
    Vnpeople, Vorigins, Vdestinations = SubsampleFluxesByPop(Vnpeople, Vorigins, Vdestinations)
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
                        _,VgridIdxXj,VgridPopulationXj = SubSampleGridByCell(VgridIdx, VgridPopulation,cell_j)
 #                       if debug:
 #                           DebuggingGetd0Celli(count_j,NCycleControl,VnpeopleXj,VoriginsXj,VdestinationsXj,cell_j)
 #                           count_j += 1
                        if len(VnpeopleXi)==1 and len(VnpeopleXj)==1:
                            dxi = DistanceMatrix[VoriginsXi[0], VdestinationsXi[0]]
                            dxj = DistanceMatrix[VoriginsXj[0], VdestinationsXj[0]]
                            Txi = VnpeopleXi[0]
                            Txj = VnpeopleXj[0]
                            mi = VgridPopulationXi[0]
                            mj = VgridPopulationXj[0]
#                            if np.isnan(np.log(Txi * mi / (Txj * mj))):
#                                print('Invalid: ',' Tx{0}: {1} Tx{2} {3} mi: {4},mj: {5}'.format(cell_i,Txi,cell_j,Txj,mi,mj))
#                            else:
#                                print('Valid: ',' Tx{0}: {1} Tx{2} {3} mi: {4},mj: {5}, idxj {6} idxi {7}'.format(cell_i,Txi,cell_j,Txj,mi,mj,VgridIdxXj[0],VgridIdxXi[0]))
                            d0s.append(dxi - dxj / (-np.log(Txi * mi / (Txj * mj))))
                        else:
                            raise ValueError('More than 1 flux')
#                        count += 1
        median = np.median(np.asarray(d0s))
    return median,d0s

@numba.njit(parallel=True)
def d0PotentialFitOptimizedParallel(Vnpeople, Vorigins, Vdestinations, VgridIdx, VgridPopulation, DistanceMatrix,debug = False):
    num_origins = len(Vorigins)
    d0s = np.empty(num_origins, dtype=np.float64)
    d0_count = 0
    
    Vnpeople, Vorigins, Vdestinations = SubsampleFluxesByPop(Vnpeople, Vorigins, Vdestinations)
    
    for idx_x in numba.prange(num_origins):
        cell_x = Vorigins[idx_x]
        _, VnpeopleX, VoriginsX, VdestinationsX = SubsampleByCellFluxOrigin(Vnpeople, Vorigins, Vdestinations, cell_x)
        
        for cell_i in VdestinationsX:
            if cell_x < cell_i:
                _, VnpeopleXi, _, VdestinationsXi = SubsampleByCellFluxDestination(VnpeopleX, VoriginsX, VdestinationsX, cell_i)
                _, VgridIdxXi, VgridPopulationXi = SubSampleGridByCell(VgridIdx, VgridPopulation, cell_i)
                
                for cell_j in VdestinationsX:
                    if cell_i < cell_j:
                        _, VnpeopleXj, _, VdestinationsXj = SubsampleByCellFluxDestination(VnpeopleX, VoriginsX, VdestinationsX, cell_j)
                        _, VgridIdxXj, VgridPopulationXj = SubSampleGridByCell(VgridIdx, VgridPopulation, cell_j)
                        
                        if len(VnpeopleXi) == 1 and len(VnpeopleXj) == 1:
                            dxi = DistanceMatrix[VoriginsXi[0], VdestinationsXi[0]]
                            dxj = DistanceMatrix[VoriginsXj[0], VdestinationsXj[0]]
                            Txi = VnpeopleXi[0]
                            Txj = VnpeopleXj[0]
                            mi = VgridPopulationXi[0]
                            mj = VgridPopulationXj[0]
                            
                            d0s[d0_count] = dxi - dxj / (-np.log(Txi * mi / (Txj * mj)))
                            d0_count += 1
                        else:
                            raise ValueError('More than 1 flux')

    median = np.median(d0s[:d0_count])
    return median, d0s[:d0_count]


def d0PotentialFit(grid, df_distance, T, chunk_size=1000):
    '''
    Input:
        grid: output of GetGrid
        df_distance: output of distance_matrix
        T: output of OD2Grid
        chunk_size: Number of combinations to process in each chunk
    Output:
        d0: float -> The value of d0
    '''
    # Create all possible combinations of cell indices

    subsetflux = T[T['number_people'] > 0]
    cell_x = np.random.choice(subsetflux['origin'])
    d0s = []
    count = 0
    for cell_x,subsetorigin in subsetflux.groupby('origin'):
        print('cell_x: ',cell_x)
        subsetorigin = subsetorigin[subsetorigin['number_people'] > 0]
        for cell_i,subsetdestinationi in subsetorigin.groupby('destination'):
            subsetdestinationi = subsetdestinationi[subsetdestinationi['number_people'] > 0]
            if count == 0 or count % 1000 == 0:
                print('comibnation OD subsetdestinationi: ',len(subsetdestinationi))
            for cell_j,subsetdestinationj in subsetorigin.groupby('destination'):
                if count == 0 or count % 1000 == 0:
                    print('combinations OD subsetdestinationj: ',len(subsetdestinationj))
                if cell_i < cell_j:
                    subsetdestinationj = subsetdestinationj[subsetdestinationj['number_people'] > 0]
                    dxi = df_distance.loc[df_distance['i'].values == cell_x]
                    dxi = dxi[dxi['j'] == cell_i]['distance'].values[0]
                    dxj = df_distance.loc[df_distance['i'].values == cell_x]
                    dxj = dxj[dxj['j'] == cell_j]['distance'].values[0]      
                    Txi = subsetdestinationi.loc[subsetdestinationi['destination'].values == cell_i]['number_people'].values[0]
                    Txj = subsetdestinationj.loc[subsetdestinationj['destination'].values == cell_j]['number_people'].values[0]
                    mi = grid.loc[grid['index'] == cell_i, 'population'].values[0]
                    mj = grid.loc[grid['index'] == cell_j, 'population'].values[0]
                    if np.isnan(np.log(Txi * mi / (Txj * mj))):
                        print('Invalid: ',' Tx{}:'.format(cell_i),Txi,' Tx{}:'.format(cell_j),Txj,' mi: ',mi,' mj: ',mj)
                    else:
                        if count == 0:
                            print('Valid: ',' Tx{}:'.format(cell_i),Txi,' Tx{}:'.format(cell_j),Txj,' mi: ',mi,' mj: ',mj)
                    d0s.append(dxi - dxj / (-np.log(Txi * mi / (Txj * mj))))
            count += 1
    print('np.shape(d0s): ',np.shape(d0s))
    print('median(d0s): ',np.median(d0s))
    plt.hist(d0s)
    plt.show()

    return np.median(d0s)

def GetParametersPotential(grid,df_distance,Tij,save_dir):
    '''
        Function to get the parameters of the potential
        cell_i is the origin
        cell_j is the destination
    '''
    if not os.path.isfile(os.path.join(save_dir,'FitFluxesParameters.json')):
        d0 = d0PotentialFit(grid,df_distance,Tij)
        print('d0: ',d0)
        subsetflux = Tij[Tij['number_people'] > 0]
        EstimateFluxesScaled = []
        Fluxes = []
        for cell_i,subsetorigin in subsetflux.groupby('origin'):
            subsetorigin = subsetorigin[subsetorigin['number_people'] > 0]
            for cell_j,subsetdestination in subsetorigin.groupby('destination'):
                if cell_i < cell_j:
                    subsetdestination = subsetdestination[subsetdestination['number_people'] > 0]
                    dij = df_distance.loc[df_distance['i'].values == cell_i]
                    dij = dij[dij['j'] == cell_j]['distance'].values[0]
                    mi = grid.loc[grid['index'] == cell_i, 'population'].values[0]
                    mj = grid.loc[grid['index'] == cell_j, 'population'].values[0]
                    Tij = subsetdestination['number_people'].loc[subsetdestination['destination'].values == cell_j]['number_people'].values[0]
                    Testimatedij = np.exp(mi*mj*np.exp(dij/d0))
                    EstimateFluxesScaled.append(Testimatedij)
                    Fluxes.append(Tij)

        k = Fitting(np.array(EstimateFluxesScaled),np.array(Fluxes),label = 'linear',initial_guess = max(Tij['number_people'])/max(EstimateFluxesScaled) ,maxfev = 10000)
        json.dump({'d0':d0,'k':k},open(os.path.join(save_dir,'FitFluxesParameters.json'),'w'))
    else:
        d0,k = json.load(open(os.path.join(save_dir,'FitFluxesParameters.json'),'r'))
    return d0,k


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
