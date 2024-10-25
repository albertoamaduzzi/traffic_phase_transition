import numpy as np
import json
import os
import socket
from Potential import *
from ModifyPotential import *
from Polycentrism import *
from PolycentrismPlot import *
from GenerateModifiedFluxesSimulation import *
from ODfromfma import *
import logging
logger = logging.getLogger(__name__)

# ----- UPLOAD GRAVITATIONAL FIT ------
def UploadGravitationalFit(TRAFFIC_DIR,name):
    with open(os.path.join(TRAFFIC_DIR,'data','carto',name,'potential','FitVespignani.json'),'r')as f:
        fitGLM = json.load(f)
    k = np.exp(fitGLM['logk'])
    alpha =fitGLM['alpha']
    beta = fitGLM['gamma']
    d0 = fitGLM['1/d0']    
    return k,alpha,beta,d0
###------------------------------- MODIFY FLUXES -------------------------------------------###
def LogModifyMorphologyCity(name,num_peaks,InfoCenters):
    logger.info('Modify Morphology City: {}'.format(name))
    logger.info("Number of Centers: {}".format(num_peaks))
    logger.info("Type: ",InfoCenters['center_settings']['type'])
    logger.info(f"Covariance: ({InfoCenters['covariance_settings']['covariances']['cvx']},{InfoCenters['covariance_settings']['covariances']['cvy']}")



def ModifyMorphologyCity(InfoConfigurationPolicentricity,grid,SFO_obj,Tij,df_distance,lattice,num_peaks,TRAFFIC_DIR,name,grid_size,InfoCenters = {'center_settings': {"type":"exponential"},'covariance_settings':{"covariances":{"cvx":5,"cvy":5}}},fraction_fluxes = 80,verbose = True):
    '''
        Returns The Fluxes in the Tij That needs to be put in the simulation as Configuration Files.
    '''
#    InfoConfigurationPolicentricity,grid,SFO_obj,Tij,df_distance,num_peaks = args
    k,alpha,beta,d0 = UploadGravitationalFit(TRAFFIC_DIR,name)
    #  Total Population and Fluxes
    total_population = np.sum(grid['population'])
    total_flux = np.sum(Tij['number_people'])
    LogModifyMorphologyCity(name,num_peaks,InfoCenters)
    # Generate random indices for centers
    logger.info('Plotting fluxes coming from raw data')
    PlotFluxes(grid,Tij,SFO_obj,os.path.join(TRAFFIC_DIR,'data','carto',name),fraction_fluxes)
    logger.info('PIPELINE MODIFICATION FLUXES starting...')
    # Store the DataFrames that will store the generated fluxes and population
    InfoConfigurationPolicentricity = InitConfigurationPolicentricity(num_peaks,InfoConfigurationPolicentricity,grid,Tij)
    # Generate Random Population
    new_population,index_centers = GenerateRandomPopulation(grid,num_peaks,total_population,InfoCenters,verbose)
    # From Population, using the gravitational model, generate the fluxes
    Modified_Fluxes = GenerateModifiedFluxes(new_population,df_distance,k,alpha,beta,d0,total_flux,verbose)
    InfoConfigurationPolicentricity[num_peaks]['Tij']['number_people'] = Modified_Fluxes
    InfoConfigurationPolicentricity[num_peaks]['grid']['population'] = new_population    
    # Compute new vector field and Potential with relative UCI
    if (np.zeros(len(Tij)) == (InfoConfigurationPolicentricity[num_peaks]['Tij']['number_people'].to_numpy() - Tij['number_people'].to_numpy())).all():
        raise ValueError('Fluxes not modified correctly')
    if (np.zeros(len(grid)) == (InfoConfigurationPolicentricity[num_peaks]['grid']['population'].to_numpy() - grid['population'].to_numpy())).all():
        raise ValueError('Population not modified correctly')
    New_Vector_Field = ComputeNewVectorField(InfoConfigurationPolicentricity[num_peaks]['Tij'],df_distance)
    New_Potential_Dataframe = ComputeNewPotential(New_Vector_Field,lattice,InfoConfigurationPolicentricity[num_peaks]['grid'])
    InfoConfigurationPolicentricity = StoreConfigurationsPolicentricity(new_population, Modified_Fluxes,New_Vector_Field,New_Potential_Dataframe,num_peaks,InfoConfigurationPolicentricity)
    PI,LC,UCI,result_indices,_,cumulative,Fstar = ComputeUCI(InfoConfigurationPolicentricity[num_peaks]['grid'],InfoConfigurationPolicentricity[num_peaks]['potential'],df_distance)
    InfoConfigurationPolicentricity[num_peaks]['PI'] = PI
    InfoConfigurationPolicentricity[num_peaks]['LC'] = LC
    InfoConfigurationPolicentricity[num_peaks]['UCI'] = UCI
    if verbose:
        print('After Population Generation and Gravity:')
        PrintInfoFluxPop(InfoConfigurationPolicentricity[num_peaks]['grid'],InfoConfigurationPolicentricity[num_peaks]['Tij'])
        if len(Tij) == len(InfoConfigurationPolicentricity[num_peaks]['Tij']):
            print("Total Absolute Difference Original/Generated Fluxes: ",np.sum(np.abs(InfoConfigurationPolicentricity[num_peaks]['Tij']['number_people'].to_numpy() - Tij['number_people'].to_numpy())))
            print("Average Absolute Difference Original/Generated Fluxes: ",np.mean(np.abs(InfoConfigurationPolicentricity[num_peaks]['Tij']['number_people'].to_numpy() - Tij['number_people'].to_numpy())))
            print("Number of Fluxes: ",len(InfoConfigurationPolicentricity[num_peaks]['Tij']))
            print("Number of Fluxes > 0: ",len(InfoConfigurationPolicentricity[num_peaks]['Tij'][InfoConfigurationPolicentricity[num_peaks]['Tij']['number_people']>0]))
            print("Fraction (>0) to Total Fluxes: ",len(InfoConfigurationPolicentricity[num_peaks]['Tij'][InfoConfigurationPolicentricity[num_peaks]['Tij']['number_people']>0])/len(InfoConfigurationPolicentricity[num_peaks]['Tij']))
        print('PI: ',PI)
        print('LC: ',LC)
        print('UCI: ',UCI)
        print('************PLOTTING************')
        print("Comparison Among Fluxes: ")
        count_peak0 = 0
        for num_peak in InfoConfigurationPolicentricity.keys():
            count_peak1 = 0
            for num_peak1 in InfoConfigurationPolicentricity.keys():
                if num_peak != num_peak1 and count_peak0 < count_peak1:
                    if 'Tij' in InfoConfigurationPolicentricity[num_peak].keys() and 'Tij' in InfoConfigurationPolicentricity[num_peak1].keys():
                        print("Comparison between {} and {}".format(num_peak,num_peak1))
                        print("Average Difference Population",np.mean(np.abs(InfoConfigurationPolicentricity[num_peak]['Tij']['number_people'].to_numpy() - InfoConfigurationPolicentricity[num_peak1]['Tij']['number_people'].to_numpy())))
#                        print("Average Difference Potential",np.mean(np.abs(np.array(InfoConfigurationPolicentricity[num_peak]['potential']) - np.array(InfoConfigurationPolicentricity[num_peak1]['potential']))))
                        print("Potential: ",np.array(InfoConfigurationPolicentricity[num_peak]['potential']))
                count_peak1 += 1
            count_peak0 += 1
        dir_grid = GetDirGrid(TRAFFIC_DIR,name,grid_size,num_peaks,InfoCenters['covariance_settings']['covariances']['cvx'],InfoCenters['center_settings']['type'],UCI)        
        PlotRoutineOD(InfoConfigurationPolicentricity[num_peaks]['grid'],
                      InfoConfigurationPolicentricity[num_peaks]['Tij'],
                      SFO_obj.gdf_polygons,
                      InfoConfigurationPolicentricity[num_peaks]['potential'],
                        InfoConfigurationPolicentricity[num_peaks]['vector_field'],
                      dir_grid,
                      fraction_fluxes,
                      UCI,
                      index_centers,
                      Tij,
                      cumulative,
                      Fstar,
                      result_indices)
    return InfoConfigurationPolicentricity,UCI



def ConcatenateODsInputSimulation(SFO_obj,grid_size,NameCity):
    FirstFile = True
    for file in os.listdir(os.path.join(SFO_obj.ODfma_dir)):
        if file.endswith('.fma'):
            start = int(file.split('.')[0].split('D')[1])
            end = start + 1
            ODfmaFile = os.path.join(SFO_obj.ODfma_dir,file)
            if start == 7:
                pass
            else:
                if FirstFile:
                    df1,_,ROutput = OD_from_fma(SFO_obj.polygon2OD,
                                        SFO_obj.osmid2index,
                                        SFO_obj.grid,
                                        grid_size,
                                        SFO_obj.OD2grid,
                                        NameCity,
                                        ODfmaFile,
                                        start,
                                        end,
                                        SFO_obj.save_dir_local,
                                        seconds_in_minute = 60,
                                        )
                    FirstFile = False
                else:
                    df2,_,ROutput = OD_from_fma(SFO_obj.polygon2OD,
                                        SFO_obj.osmid2index,
                                        SFO_obj.grid,
                                        grid_size,
                                        SFO_obj.OD2grid,
                                        NameCity,
                                        ODfmaFile,
                                        start,
                                        end,
                                        SFO_obj.save_dir_local,
                                        seconds_in_minute = 60,
                                        )
                    df1 = pd.concat([df1,df2],ignore_index = True)
    return df1

def GenerateParallelODs(num_peaks, cv, distribution,InfoConfigurationPolicentricity,grid,SFO_obj,Tij,df_distance,lattice,TRAFFIC_DIR,NameCity,grid_size,osmid2index,grid2OD,CityName2RminRmax):
    """
        Description: Generate the ODs for the city with the given parameters.
        NOTE: It is Very Heavy to run this function in parallel. Since Need to Load each time the Tij,grid,df_distance,lattice, that are around 2
          GB for Boston. 
        
    """
    InfoCenters = {'center_settings': {"type":distribution},
                   'covariance_settings':{"covariances":{"cvx":cv,"cvy":cv},
                                          "Isotropic": True,
                                          "Random": False}
                    }
    InfoConfigurationPolicentricity,UCI = ModifyMorphologyCity(InfoConfigurationPolicentricity,
                                                               grid,
                                                               SFO_obj,
                                                               Tij,
                                                               df_distance,
                                                               lattice,
                                                               num_peaks,
                                                               TRAFFIC_DIR,
                                                               NameCity,
                                                               grid_size,
                                                               InfoCenters,
                                                               fraction_fluxes = 200,
                                                               verbose = True)
    if socket.gethostname()=='artemis.ist.berkeley.edu':
        SaveOd = "/home/alberto/LPSim/LivingCity/berkeley_2018/new_full_network"
    else:
        SaveOd = f'/home/aamad/Desktop/phd/traffic_phase_transition/data/carto/{NameCity}/OD'
    df1 = ConcatenateODsInputSimulation(SFO_obj,grid_size,NameCity)
    df2 = OD_from_T_Modified(InfoConfigurationPolicentricity[num_peaks]['Tij'],
                    CityName2RminRmax,
                    NameCity,
                    osmid2index,
                    grid2OD,
                    1,
                    SaveOd,
                    7,
                    8,
                    round(UCI,3),
                    df1)

    del InfoConfigurationPolicentricity
