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
import logging
logger = logging.getLogger(__name__)

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
    '''
        @param Tij: Dataframe with the number of people from i to j
        @param df_distance: Dataframe with the distance matrix and the direction vector
        @return VectorField: Dataframe with the vector field in the square lattice
        @description: Compute the vector field in the square lattice from the Tij and the distance matrix.
        Columns of Tij: (i,j)O, (i,j)D, number_people

    '''
    assert 'dir_vector' in df_distance.columns, 'The column "dir_vector" is not in the DataFrame'
    assert 'number_people' in Tij.columns, 'The column "number_people" is not in the DataFrame'    
    if isinstance(df_distance.iloc[0]['dir_vector'],str):
        Tij['vector_flux'] = df_distance['dir_vector'].apply(lambda x: parse_dir_vector(x) ) * Tij['number_people']
    else:
        pass
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
    VectorField.to_csv(os.path.join(save_dir,'VectorField.csv'),index=False)

def GetSavedVectorFieldDF(save_dir):
    return pd.read_csv(os.path.join(save_dir,'VectorField.csv'))

#------------------------------------------ POTENTIAL ----------------------------------------------------------#
#####
#####                   NOTE: Usage ->  lattice = GetPotentialLattice(lattice,VectorField)
#####                                   PotentialDataframe = ConvertLattice2PotentialDataframe(lattice)
#####                                   SavePotentialDataframe(PotentialDataframe,dir_grid)

def GetPotentialLattice(lattice,VectorField):
    '''
        @param lattice: Graph with the square lattice
        @param VectorField: Dataframe with the vector field
        @return lattice: Graph 
        @description: lattice -> nodes features: V_in, V_out, index, rotor_z_in, rotor_z_out, HarmonicComponentOut, HarmonicComponentIn
                      lattice -> edges features: dx, dy, d/dx, d/dy
        'V_in': potential for the incoming fluxes
        'V_out': potential for the outgoing fluxes
        'rotor_z_in': Is the rotor at the point (i,j) for the ingoing flux. (Tj) sum over i. So I look at a source and I say that the field
                    is the ingoing flux. This is strange as it does not give any information about where to go to find the sink.
        'rotor_z_out': Is the rotor at the point (i,j) for the ingoing flux. (Ti) sum over j. So I look at a source and I say that the field
                    is the outgoing flux. In this way I am considering the analogue case to the google algorithm for
                    page rank as I am at a random point and the field points at the direction with smaller potential, the sink, that is 
                    the higher rank of importance.
        'HarmonicComponentIn': Harmonic Component for the ingoing flux
        'HarmonicComponentOut': Harmonic Component for the outgoing flux
                    
    '''
    assert 'Ti' in VectorField.columns, 'The column "Ti" is not in the DataFrame'
    assert 'Tj' in VectorField.columns, 'The column "Tj" is not in the DataFrame'
    assert 'index' in VectorField.columns, 'The column "index" is not in the DataFrame'
    assert '(i,j)' in VectorField.columns, 'The column "(i,j)" is not in the DataFrame'
    nx.set_node_attributes(lattice, 0, 'V_in')
    nx.set_node_attributes(lattice, 0, 'V_out')
    nx.set_node_attributes(lattice, 0, 'index')
    nx.set_node_attributes(lattice, 0, 'rotor_z_out')
    nx.set_node_attributes(lattice, 0, 'rotor_z_in')
#    max_i = max(ast.literal_eval(node_str)[0] for node_str in lattice.nodes)
#    max_j = max(ast.literal_eval(node_str)[1] for node_str in lattice.nodes)
    # Initialize Potential To 0 -> In this way the edges will be 0
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
        lattice = ComputeHarmonicComponents(lattice)
    return lattice

def ComputeHarmonicComponents(lattice):
    """
        @param lattice: Graph with the square lattice
        @return lattice: Graph with the square lattice
        @description: Compute the Harmonic Component for the ingoing and outgoing fluxes
    """
    nx.set_node_attributes(lattice, 0, 'HarmonicComponentIn')
    nx.set_node_attributes(lattice, 0, 'HarmonicComponentOut')
    for node_str in lattice.nodes:
        ij = ast.literal_eval(node_str)
        i = ij[0]
        j = ij[1]
        lattice.nodes[node_str]['HarmonicComponentIn'] = 0
        lattice.nodes[node_str]['HarmonicComponentOut'] = 0
        HarmonicComponentNodeOut = 0
        HarmonicComponentNodeIn = 0
        for Neighbor in lattice.neighbors(node_str):
            if math.isinf(lattice[node_str][Neighbor]['d/dx']):
                HarmonicComponentNodeOut += (lattice[node_str][Neighbor]['d/dy']**2)*(lattice.nodes[node_str]['V_out'] - lattice.nodes[Neighbor]['V_out'])
                HarmonicComponentNodeIn += (lattice[node_str][Neighbor]['d/dy']**2)*(lattice.nodes[node_str]['V_in'] - lattice.nodes[Neighbor]['V_in'])
            elif math.isinf(lattice[node_str][Neighbor]['d/dy']):
                HarmonicComponentNodeOut += (lattice[node_str][Neighbor]['d/dx']**2)*(lattice.nodes[node_str]['V_out'] - lattice.nodes[Neighbor]['V_out'])
                HarmonicComponentNodeIn += (lattice[node_str][Neighbor]['d/dx']**2)*(lattice.nodes[node_str]['V_in'] - lattice.nodes[Neighbor]['V_in'])            
        lattice.nodes[node_str]['HarmonicComponentOut'] = HarmonicComponentNodeOut
        lattice.nodes[node_str]['HarmonicComponentIn'] = HarmonicComponentNodeIn
    return lattice
        

def SmoothPotential(lattice):
    """
        @param lattice: Graph with the square lattice
        @return lattice: Graph with the square lattice
    """
    from math import isnan
    # Smooth V_in and V_out by taking the average over all the neighbors
    for node_str in lattice.nodes:
        neighbors = list(lattice.neighbors(node_str))
        
        # Initialize sums and counts
        sum_Vin = 0
        sum_Vout = 0
        count_Vin = 0
        count_Vout = 0
        
        # Calculate the sum and count of V_in and V_out for the neighbors, ignoring NaN values
        for neighbor in neighbors:
            Vin = lattice.nodes[neighbor]['V_in']
            Vout = lattice.nodes[neighbor]['V_out']
            
            if not isnan(Vin):
                sum_Vin += Vin
                count_Vin += 1
            
            if not isnan(Vout):
                sum_Vout += Vout
                count_Vout += 1
        
        # Calculate the average values, handling the case where count is zero
        avg_Vin = sum_Vin / count_Vin if count_Vin > 0 else float('nan')
        avg_Vout = sum_Vout / count_Vout if count_Vout > 0 else float('nan')
        
        # Assign the average values to the current node
        lattice.nodes[node_str]['V_in'] = avg_Vin
        lattice.nodes[node_str]['V_out'] = avg_Vout
    return lattice

def ConvertLattice2PotentialDataframe(lattice):
    '''
        @param lattice: Graph with the square lattice
        @return PotentialDataframe: Dataframe with V_in, V_out, centroid (x,y), index, node_id(i,j), HarmonicComponentIn, HarmonicComponentOut
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
    # Add Harmonic Components
    AddHarmonicComponents2PotentialDataframe(PotentialDataframe,lattice)
        # Format the 'node_id' column using ast.literal_eval
#        PotentialDataframe['node_id'] = PotentialDataframe['node_id'].apply(ast.literal_eval)
    return PotentialDataframe

def AddHarmonicComponents2PotentialDataframe(PotentialDataframe,lattice):
    '''
        @param PotentialDataframe: Dataframe with V_in, V_out, index, node_id(i,j)
        @param lattice: Graph with the potential
        @description: Add the Harmonic Components to the PotentialDataframe
        @return PotentialDataframe: Dataframe with V_in, V_out, index, node_id(i,j), HarmonicComponentIn, HarmonicComponentOut
    '''
    HarmonicComponentsIn = []
    HarmonicComponentsOut = []
    for node,data in lattice.nodes(data=True):
        # Extract the indices of the nodes
        ij = ast.literal_eval(node)    
        node_id = (ij[0],ij[1])
        # Compute the value of V_in for the edge
        HarmonicComponentsIn.append(lattice.nodes[node]['HarmonicComponentIn'])  
        # Compute the value of V_out for the edge
        HarmonicComponentsOut.append(lattice.nodes[node]['HarmonicComponentOut'])
    PotentialDataframe['HarmonicComponentIn'] = HarmonicComponentsIn
    PotentialDataframe['HarmonicComponentOut'] = HarmonicComponentsOut
    return PotentialDataframe

## ---- Aggregeted Functions Generation Potential from Fluxes ---- ##
def ComputeVectorFieldFromTijDistance(Tij,df_distance):
    """
        @param Tij: Dataframe with the number of people from i to j
        @param df_distance: Dataframe with the distance matrix and the direction vector
        @return VectorField: Dataframe with the vector field
    """
    logger.info("Computing Vector Field ...")
    VectorField = GetVectorField(Tij,df_distance)
    return VectorField

    

def GeneratePotentialFromFluxes(Tij,df_distance,lattice,grid,city,save_dir):
    """
        @param Tij: Dataframe with the number of people from i to j
        @param df_distance: Dataframe with the distance matrix and the direction vector
        @param lattice: Graph with the square lattice
        @param grid: Dataframe with the grid
        @param city: Name of the city
        @return PotentialDataframe: Dataframe with the potential
    """
    if not os.path.exists(os.path.join(save_dir,'VectorField.csv')):
        VectorField = ComputeVectorFieldFromTijDistance(Tij,df_distance)
        logger.info(f"Saving Vector Field {city} ...")
        SaveVectorField(VectorField,save_dir)
    else:
        logger.info(f"Loading Vector Field {city} ...")
        VectorField = GetSavedVectorFieldDF(save_dir)
    if not os.path.exists(os.path.join(save_dir,'PotentialDataFrame.csv')):
        logger.info(f"Getting Potential in Lattice from VF {city} ...")
        lattice = GetPotentialLattice(lattice,VectorField)
        logger.info(f"Smoothing Potential in Lattice {city} ...")
        lattice = SmoothPotential(lattice)
        logger.info(f"Converting Lattice to Potential Dataframe {city} ...")
        PotentialDataframe = ConvertLattice2PotentialDataframe(lattice)
        logger.info(f"Add Population to Potential Dataframe {city} ...")
        PotentialDataframe = CompletePotentialDataFrame(grid,PotentialDataframe)
        logger.info(f"Saving Potential Dataframe {city} ...")
        SavePotentialDataframe(PotentialDataframe,save_dir)
    else:
        logger.info(f"Loading Potential Dataframe {city} ...")
        PotentialDataframe = GetSavedPotentialDF(save_dir)
    return PotentialDataframe,lattice,VectorField


def CompletePotentialDataFrame(grid,PotentialDataframe):
    """
        @param grid: Dataframe with the grid
        @param PotentialDataframe: Dataframe with the potential
        @description: Add the population to the PotentialDataframe
    """
    if isinstance(grid,pd.DataFrame):
        PotentialDataframe['population'] = grid['population']    
    else:
        PotentialDataframe['population'] = grid
    return PotentialDataframe

def SavePotentialDataframe(PotentialDataFrame,save_dir):
    PotentialDataFrame.to_csv(os.path.join(save_dir,'PotentialDataFrame.csv'),index=False)    

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

