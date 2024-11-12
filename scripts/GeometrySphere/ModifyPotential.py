import numpy as np
import pandas as pd
import geopandas as gpd
import numba
import time
from Potential import *
from Polycentrism import *
from GeometrySphere import *
import logging
logger = logging.getLogger(__name__)

def GenerateIndexCenters(grid,num_peaks,verbose = False):
    '''
        NOTE: The center is the center of mass. Required: Pre-compute 'distance_from_center' 
        NOTE: Grid set at 1.5 km per side, (grid_size = 0.02), filter 300 people. (150 people per km^2)
        NOTE: On average each grid contains >1000 people for Boston (4M pop, 3500 grids)
        Input:
            grid: geodataframe with the grid.
            num_peaks: int -> number of centers
        Description:
            1) Filters the grid by population. (is_populated)
            2) Extract the center coordinates (from exponential whose only parameter is the average distance from center)
            3) From the set of grids that are at the distance extracted, extract uniformly and take the index.
            4) Store index in index_centers.
    '''
    coords_center,_ = ExtractCenterByPopulation(grid)
    if 'is_populated' not in grid.columns:
        grid['is_populated'] = grid['population']>300
    if 'coords' not in grid.columns:
        grid['coords'] = grid.apply(lambda x: ProjCoordsTangentSpace(x['centroidx'],x['centroidy'],coords_center[0],coords_center[1]),axis = 1)
        grid['distance_from_center'] = grid.apply(lambda x: polar_coordinates(np.array([x['centroidx'],x['centroidy']]),np.array(x['coords']))[0],axis = 1)
    index_centers = []
    scale = np.mean(grid.loc[grid['is_populated']]['distance_from_center'])
    # Choose Just From The Populated And With Roads Grids
    populated_grid = grid.loc[grid['is_populated']]
    populated_grid = populated_grid.loc[populated_grid['with_roads']]
    # PICK RANDOMLY THE CENTERS (with exponentially decreasing probability in distance from center)
    random_values = np.random.exponential(scale,num_peaks)
    _, bin_edges = np.histogram(populated_grid['distance_from_center'].to_numpy(), bins=30)
    logger.info("Gnerating Index Centers")
    if verbose:
        print("++++++++++++ Generate Index Centers ++++++++++++")
        print("Number of Populated Grids: ",len(populated_grid))
        print("Average distance from Center: ",scale)
#        print("Distance from Center Extracted: ")
#        for rv in random_values:
#            print(rv) 
    for rv in random_values:
        if rv > bin_edges[-1]:
            while(rv > bin_edges[-1]):
                rv = np.random.exponential(scale)
#                print('extracting rv: ',rv)
                bin_index = np.digitize(rv, bin_edges)
#                print('bin index: ',bin_index)     
            filtered_grid = populated_grid[(populated_grid['distance_from_center'] >= bin_edges[bin_index - 1])] 
            filtered_grid = filtered_grid[filtered_grid['distance_from_center'] < bin_edges[bin_index]]   
#            if verbose:
#                print('Distance from center extracted: ',rv,'Number of grids available: ',len(filtered_grid))
        else:
            bin_index = np.digitize(rv, bin_edges)
#            print('bin index: ',bin_index)  
            while(bin_index > bin_edges[-1]):
                bin_index = np.digitize(rv, bin_edges)
            filtered_grid = populated_grid[(populated_grid['distance_from_center'] >= bin_edges[bin_index - 1])] 
            filtered_grid = filtered_grid[filtered_grid['distance_from_center'] < bin_edges[bin_index]]        
#            print('Length filtered grid: ',len(filtered_grid), ' rv: ',rv)
        if filtered_grid.shape[0] == 0:
            print('Empty bin')
        else:
            selected_row = filtered_grid.sample()
            # Step 6: Get the index of the selected row
            selected_index = selected_row['index'].values[0]
            index_centers.append(selected_index)
#    if verbose:
#        print('Grid selected: ',index_centers)

    return index_centers

def SetCovariances(index_centers,cov = {"cvx":5,"cvy":5},Isotropic = True,Random = False,verbose = False):
    '''
        Input:
            index_centers: list of indices of the centers. (int: 0,...,Ngrids)
            cov: dictionary with the covariance in x and y.
            Isotropic: boolean to set the covariance isotropic.
            Random: boolean to set the covariance randomly.
        Output:
            covariances: list of covariances for each center. (they are constant in the case of non random.)
            That is, each center is equal to the other one in terms of covariance.
            '''
    covariances = []
    if verbose:
        print('+++++++++ Setting Covariances ++++++++')
    if Isotropic:
        if Random:
            for i in range(len(index_centers)):
                cv = np.random.uniform(2,15)
                rvs = [[cv,0],[0,cv]]
                covariances.append(rvs)
            if verbose:
                print('Isotropic and Random')
#                for i in range(len(index_centers)):
#                    print("Center ",i,":\nsigma_x: ",covariances[i][0][0],"\nsigma_y: ",covariances[i][1][1])
        else:
            assert 'cvx' in cov.keys()
            for i in range(len(index_centers)):
                cv = cov['cvx']
                rvs = [[cv,0],[0,cv]]
                covariances.append(rvs)
            if verbose:
                print('Isotropic and Not Random')
#                for i in range(len(index_centers)):
#                    print("Center ",i,":\nsigma_x: ",covariances[i][0][0],"\nsigma_y: ",covariances[i][1][1])
    else:
        if Random:
            for i in range(len(index_centers)):
                rvs = [[np.random.uniform(2,15),0],[0,np.random.uniform(2,15)]]
                covariances.append(rvs)
            if verbose:
                print('Not Isotropic and Random')
#                for i in range(len(index_centers)):
#                    print("Center ",i,":\nsigma_x: ",covariances[i][0][0],"\nsigma_y: ",covariances[i][1][1])
        else:
            assert 'cvx' in cov.keys() and 'cvy' in cov.keys()
            for i in range(len(index_centers)):
                cvx = cov['cvx']
                cvy = cov['cvy']
                rvs = [[cvx,0],[0,cvy]]
                covariances.append(rvs)
            if verbose:
                print('Not Isotropic and Not Random')
#                for i in range(len(index_centers)):
#                    print("Center ",i,":\nsigma_x: ",covariances[i][0][0],"\nsigma_y: ",covariances[i][1][1])


    return covariances

def ComputeNewPopulation(grid,index_centers,covariances,total_population,Distribution = 'exponential',verbose = False):
    '''
        Input:
            grid: geodataframe with the grid.
            index_centers: list of indices of the centers. (int: 0,...,Ngrids)
            covariances: list of covariances for each center. 
            total_population: total population of the city.
            Distribution: type of distribution to use. ('exponential','gaussian')
    '''
    total_population_center = total_population/len(index_centers)
    new_population = np.ones(len(grid))
    centers = grid.loc[index_centers][['centroidx','centroidy']].to_numpy()
    count_center = 0
#    if verbose:
#        print('++++++++++ POPULATION DISTRIBUTION +++++++++')
#        print("Distribution: ",Distribution)
#        print('Covariance -> x {0}, y {1}'.format(covariances[count_center][0][0],covariances[count_center][1][1]))
    for center in centers:
        for i,row in grid.iterrows():
            point = np.array([grid['centroidx'][i], grid['centroidy'][i]])
            if grid['is_populated'][i] and grid['with_roads'][i]:
                if Distribution == 'exponential':
                    new_population[i] += total_population_center*np.exp(-(np.linalg.norm(ProjCoordsTangentSpace(center[0],center[1],point[0],point[1]))/(10**3))/covariances[count_center][0][0])/2
                elif Distribution == 'gaussian':
                    new_population[i] += total_population_center*np.exp(-(np.linalg.norm(ProjCoordsTangentSpace(center[0],center[1],point[0],point[1]))/(10**3))**2/(covariances[count_center][0][0]**2 + covariances[count_center][1][1]**2))/2
            else:
                new_population[i] = 0
        count_center += 1
    return new_population    

def GenerateRandomPopulation(grid,num_peaks,total_population,args = {'center_settings': {"type":"exponential"},
                                                                      'covariance_settings':{"covariances":{"cvx":5,"cvy":5},
                                                                                             "Isotropic": True,
                                                                                             "Random": False
                                                                                             }
                                                                      },verbose = False):
    '''
        Input:
            grid: geodataframe with the grid.
            num_peaks: number of centers in the city 
            total_population: total population of the city
        Help:
            args:
                [center_settings][type]: exponential/gaussian.
        Description:

            
        Output:
            new_population: new population [np.array(dtype = np.int32)] -> to be used in the GravitationalModel
            index_centers: list of indices of the centers.
        NOTE: Consider Just the Grids that have either population and road network in them.
    '''
    required_keys = ['center_settings', 'covariance_settings']
    required_keys_center_settings = ['type']
    allowed_keys_center_type = ['exponential','gaussian']
    required_keys_covariance_settings = ['covariances']
    required_keys_covariances = ['cvx','cvy']
    # Check Required Keys
    assert all(key in args for key in required_keys), "Dictionary is missing required keys check the missing from: {}".format(required_keys)
    assert all(key in args['center_settings'] for key in required_keys_center_settings), "Dictionary is missing required keys check the missing from: {}".format(['Distribution'])
    assert all(key in args['covariance_settings'] for key in required_keys_covariance_settings), "Dictionary is missing required keys check the missing from: {}".format(required_keys_covariance_settings)
    assert all(key in args['covariance_settings']['covariances'] for key in required_keys_covariances), "Dictionary is missing required keys check the missing from: {}".format(required_keys_covariances)
    # Check Allowed Keys
    assert args['center_settings']['type'] in allowed_keys_center_type, "center_settings['type'] must be in: {}".format(allowed_keys_center_type)
    # Compute the indices of the new centers at random (Chosen From Places with Population and Roads)
    index_centers = GenerateIndexCenters(grid,num_peaks,verbose)
    covariances = SetCovariances(index_centers,args['covariance_settings']['covariances'],args["covariance_settings"]["Isotropic"],args["covariance_settings"]["Random"],verbose)
    new_population = ComputeNewPopulation(grid,index_centers,covariances,total_population,args["center_settings"]["type"],verbose)    
    Factor = np.sum(grid['population'])/np.sum(new_population)
    new_population = new_population*Factor
    return new_population,index_centers

##----------------------- FLUXES MODEL ---------------------------------##
@numba.njit(['(float32[:], float32[:], float32,float32,float32,float32)'],parallel=True)
def GravitationalModel(population,df_distance,k,alpha,beta,d0):
    '''
        Input:
            population: array of potential values [Pot_O,...,Pot_(Ngrids with non 0 potential)]
            df_distance: array of distances
            Parameters k, alpha, beta, d0 for gravitational model k m**(alpha) m**(beta) exp(-d/d0)
        Output:
            Modified_Fluxes: array of modified fluxes (1 dimensional np.float32 array) [i*Ngrids(row) + j(column)]

        '''
#    print('k: ',k,' alpha: ',alpha,' beta: ',beta,' d0: ',d0)
#    print('<k*p>: ',np.mean(k*population))
#    print('<p**alpha>: ',np.mean(population**alpha))
#    print('<p**beta>: ',np.mean(population**beta))
#    print('<exp(-d/d0)>: ',np.mean(np.exp(df_distance*d0)))
#    print('<k*population**(alpha)*population**(beta)>: ',np.mean(k*np.tensordot(population**(alpha),population**(beta),axes = 0)))
    kMiMjedij = np.zeros(len(population)*len(population),dtype=np.float64)
    count_close_centers = 0
    for i in range(len(population)):
        for j in range(len(population)):
            if int(population[i])!=0 and int(population[j])!=0 and np.exp(df_distance[i*len(population) + j]*d0)>10**(-4):
#                print('pop i: ',population[i],' pop j: ',population[j])
#                print('dij: ',df_distance[i*len(population) + j])
#                print('kMiMj: ',k*population[i]**(alpha)*population[j]**(beta))
#                print('exp(dij/d0): ',np.exp(df_distance[i*len(population) + j]*d0))
#                print('kMiMjedij: ',k*population[i]**(alpha)*population[j]**(beta)*np.exp(df_distance[i*len(population) + j]*d0))
                kMiMjedij[i*len(population) + j] += k*population[i]**(alpha)*population[j]**(beta)*np.exp(df_distance[i*len(population) + j]*d0)
                count_close_centers += 1
            else:
                kMiMjedij[i*len(population) + j] = 0
#    print('Number of close centers: ',count_close_centers)
#    kMiMjedij = [k*population[i]**(alpha)*population[j]**(beta)*np.exp(-df_distance[i*len(population) + j]/d0) for i in range(len(population)) for j in range(len(population))]
    return kMiMjedij

def GenerateModifiedFluxes(new_population,df_distance,k,alpha,beta,d0,total_flux,verbose = False):
    '''
        Generate the new fluxes according to the gravitational model and scale them down to the fluxes measured in the data.
    '''
    if isinstance(new_population,pd.Series):
        Modified_Fluxes =  GravitationalModel(new_population.to_numpy(dtype = np.float32),df_distance['distance'].to_numpy(dtype = np.float32),np.float32(k),np.float32(alpha),np.float32(beta),np.float32(d0))
    else:
        Modified_Fluxes =  GravitationalModel(new_population.astype(np.float32),df_distance['distance'].to_numpy(dtype = np.float32),np.float32(k),np.float32(alpha),np.float32(beta),np.float32(d0))
    Multiplicator = total_flux/Modified_Fluxes.sum()
    Modified_Fluxes = Modified_Fluxes*Multiplicator
    if verbose:
        print("Multiplicator: ",Multiplicator)
        gammas = [1,5,10,20,30,50,100]
        for gamma in gammas:
            print("Number of people in grid with flux > ",gamma,": ",(Modified_Fluxes>gamma).sum())
            print("Number of couples of grids with flux > ",gamma,": ",len(Modified_Fluxes[Modified_Fluxes>gamma]))
            print("Fraction of couples of grids with flux > ",gamma,": ",len(Modified_Fluxes[Modified_Fluxes>gamma])/len(Modified_Fluxes))
    return Modified_Fluxes

def ComputeNewVectorField(Tij,df_distance,verbose = False):
    t0 = time.time()
    New_Vector_Field = GetVectorField(Tij,df_distance)
    t1 = time.time()
    if verbose:
        print('Time to compute the vector field: ',t1 - t0)
    return New_Vector_Field

def ComputeNewPotential(New_Vector_Field,lattice,grid,verbose = False):
    t0 = time.time()
    lattice = GetPotentialLattice(lattice,New_Vector_Field)
    lattice = SmoothPotential(lattice)
    t1 = time.time()
    New_Potential_DataFrame = ConvertLattice2PotentialDataframe(lattice)
    New_Potential_DataFrame = CompletePotentialDataFrame(grid,New_Potential_DataFrame)
    if verbose:
        print('Time to compute Lattice: ',time.time() - t0)
        print('Time to compute Potential: ',time.time() - t1)
    return New_Potential_DataFrame
