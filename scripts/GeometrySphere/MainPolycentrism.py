import numpy as np
import json
from Potential import *
from ModifyPotential import *
from Polycentrism import *
from PolycentrismPlot import *
###------------------------------- MODIFY FLUXES -------------------------------------------###
def ModifyMorphologyCity(InfoConfigurationPolicentricity,grid,SFO_obj,Tij,df_distance,lattice,num_peaks,TRAFFIC_DIR,name,grid_size,InfoCenters = {'center_settings': {"type":"exponential"},'covariance_settings':{"covariances":{"cvx":5,"cvy":5}}},fraction_fluxes = 80,verbose = True):
    '''
        Returns The Fluxes in the Tij That needs to be put in the simulation as Configuration Files.
    '''
#    InfoConfigurationPolicentricity,grid,SFO_obj,Tij,df_distance,num_peaks = args
    if verbose:
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print("Modify Morphology {}".format(num_peaks))
    with open('/home/aamad/Desktop/phd/berkeley/traffic_phase_transition/data/carto/BOS/potential/FitVespignani.json','r')as f:
        fitGLM = json.load(f)
    k = np.exp(fitGLM['logk'])
    alpha =fitGLM['alpha']
    beta = fitGLM['gamma']
    d0 = fitGLM['1/d0']    
    # Total Population and Fluxes
    total_population = np.sum(grid['population'])
    total_flux = np.sum(Tij['number_people'])
    # Generate random indices for centers
    if verbose:
        PrintInfoFluxPop(grid,Tij)
        print('Plotting fluxes coming from raw data')
        PlotFluxes(grid,Tij,SFO_obj,'/home/aamad/Desktop/phd/berkeley/traffic_phase_transition/data/carto/BOS',fraction_fluxes)
        print('PIPELINE MODIFICATION FLUXES starting...')
    InfoConfigurationPolicentricity = InitConfigurationPolicentricity(num_peaks,InfoConfigurationPolicentricity,grid,Tij)
    new_population,index_centers = GenerateRandomPopulation(grid,num_peaks,total_population,InfoCenters)
    Modified_Fluxes = GenerateModifiedFluxes(new_population,df_distance,k,alpha,beta,d0,total_flux)
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
        print("InfoconfigurationPolicentricity: ",id(InfoConfigurationPolicentricity))
        print("InfoconfigurationPolicentricity[num_peaks]: ",id(InfoConfigurationPolicentricity[num_peaks]))
        print("InfoConfigurationPolicentricity[num_peaks]['grid']: ",id(InfoConfigurationPolicentricity[num_peaks]['grid']))
        print("InfoConfigurationPolicentricity[num_peaks]['Tij']: ",id(InfoConfigurationPolicentricity[num_peaks]['Tij']))
        print("grid: ",id(grid))
        print("Tij: ",id(Tij))
        print("Modified_Fluxes: ",id(Modified_Fluxes))
        print("New_Vector_Field: ",id(New_Vector_Field))
        print("New_Potential_Dataframe: ",id(New_Potential_Dataframe))
        print("df_distance: ",id(df_distance))
        print("SFO_obj: ",id(SFO_obj))
        print("total_population: ",id(total_population))
        print("total_flux: ",id(total_flux))
        print("k: ",id(k))
        print("alpha: ",id(alpha))
        print("beta: ",id(beta))
        print("d0: ",id(d0))
        print('After Population Generation and Gravity:')
        PrintInfoFluxPop(InfoConfigurationPolicentricity[num_peaks]['grid'],InfoConfigurationPolicentricity[num_peaks]['Tij'])
        print('************PLOTTING************')
        dir_grid = GetDirGrid(TRAFFIC_DIR,name,grid_size,num_peaks,UCI)
        PlotFluxes(InfoConfigurationPolicentricity[num_peaks]['grid'],InfoConfigurationPolicentricity[num_peaks]['Tij'],SFO_obj,dir_grid,fraction_fluxes)
        PlotPositionCenters(grid,SFO_obj,index_centers,dir_grid)
        PlotNewPopulation(InfoConfigurationPolicentricity[num_peaks]['grid'],   SFO_obj,dir_grid)
        PlotOldNewFluxes(InfoConfigurationPolicentricity[num_peaks]['Tij'],Tij)
        PlotFluxes(InfoConfigurationPolicentricity[num_peaks]['grid'],InfoConfigurationPolicentricity[num_peaks]['Tij'],SFO_obj,dir_grid)
        PlotVFPotMass(InfoConfigurationPolicentricity[num_peaks]['grid'],SFO_obj,InfoConfigurationPolicentricity[num_peaks]['potential'],InfoConfigurationPolicentricity[num_peaks]['vector_field'],dir_grid,label_potential = 'population',label_fluxes = 'Ti')
        PotentialContour(InfoConfigurationPolicentricity[num_peaks]['grid'],InfoConfigurationPolicentricity[num_peaks]['potential'],SFO_obj,dir_grid)
        PotentialSurface(InfoConfigurationPolicentricity[num_peaks]['grid'],SFO_obj,InfoConfigurationPolicentricity[num_peaks]['potential'],dir_grid)
        PlotRotorDistribution(InfoConfigurationPolicentricity[num_peaks]['grid'],InfoConfigurationPolicentricity[num_peaks]['potential'],dir_grid)
        PlotLorenzCurve(cumulative,Fstar,result_indices,dir_grid,shift = 0.1,verbose = False)
    return InfoConfigurationPolicentricity