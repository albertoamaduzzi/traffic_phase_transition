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
sys.path.append('~/berkeley/traffic_phase_transition/scripts')
from FittingProcedures import Fitting
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