'''
    Description:
        I take the shape file and convert it to nodes.csv, edges.csv 
        Take OD and convert it to origin_destination.csv

'''

import geopandas as gpd
import numpy as np
import os
import pandas as pd
from global_functions import ifnotexistsmkdir

##------------------------------------- ENVIRONMENT -------------------------------------##
def set_environment():
    base_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
    data_dir = os.path.join(base_dir,'data')
    carto = os.path.join(data_dir,'carto')
    ifnotexistsmkdir(data_dir)
    ifnotexistsmkdir(carto)
    return data_dir,carto

##------------------------------------- READ FILE -------------------------------------##
def read_file_gpd(file_name):
    '''
        Input:
            file_name: string [name of the file to read]
        Output:
            df: dataframe [dataframe with the data of the file]
        Description:
            Read the file and return the dataframe
    '''
    df = gpd.read_file(file_name)
    return df

def read_file_fma(file_name):
    '''
        Input:
            file_name: string [name of the file to read]
        Output:
            df: dataframe [dataframe with the data of the file]
        Description:
            Read the file and return the dataframe
    '''
    df = pd.read_csv(file_name,sep=';',header=None)
    return df

##------------------------------------- CONVERT TO CSV -------------------------------------##

def nodes_file_from_gpd(df):
    '''
        Reads from shape and returns the nodes.csv
    '''
    vertices = []
    col_id = pd.DataFrame(vertices,name='osmid')
    col_x = pd.DataFrame(,name='x')
    col_y = pd.DataFrame(,name='y')
    ref = pd.DataFrame(,name='ref')
    highway = pd.DataFrame(,name='highway')
    df1 = pd.concat([col_id,col_x,col_y,ref,highway],join='inner',ignore_index=True)
    return df1
def edges_file_from_gpd(df):
    df_unique_id = pd.DataFrame(vertices,name='uniqueid')
    df_u = pd.DataFrame(,name='u')
    df_v = pd.DataFrame(,name='v')
    df_length = pd.DataFrame(,name='length')
    df_lanes = pd.DataFrame(,name='lanes')
    df_speed = pd.DataFrame(,name='speed_mph')
    df1 = pd.concat([df_unique_id,df_u,df_v,df_length,df_lanes,df_speed],join='inner',ignore_index=True)
    
    return df1

def OD_from_fma(df):
    df_sampn = pd.DataFrame(,name='SAMPN')
    df_perno = pd.DataFrame(,name='PERNO')
    df_origin = pd.DataFrame(,name='origin')
    df_destination = pd.DataFrame(,name='destination')
    df1 = pd.concat([df_sampn,df_perno,df_origin,df_destination],join='inner',ignore_index=True)

    return df1

##------------------------------------- SAVE FILES -------------------------------------##

def save_files(save_dir,df_nodes,df_edges,df_od):
    '''
        Input:
            carto: string [path to the folder where to save the files]
        Description:
            Save the files in the carto folder
    '''
    df_nodes.to_csv(os.path.join(save_dir,'nodes.csv'),sep=',',index=False)
    df_edges.to_csv(os.path.join(save_dir,'edges.csv'),sep=',',index=False)
    df_od.to_csv(os.path.join(save_dir,'OD.csv'),sep=',',index=False)

    pass


if __name__=='__main__':
    data_dir,carto = set_environment()
    for root,dirs,files in os.walk(data_dir):
        for file in files:
            if file.endswith('.shp'):
                file_name = os.path.join(data_dir,dirs,file)
                df = read_file_gpd(file_name)
                df_nodes,name_nodes = nodes_file_from_gpd(df)
                df_edges,name_edges = edges_file_from_gpd(df)
            elif file.endswith('.fma'):
                file_name = os.path.join(data_dir,dirs,file)
                df = read_file_fma(file_name)
                df_od,name_origin_destination = get_origin_destination_cities(df)
        save_dir = os.path.join(carto,dirs)
        save_files(save_dir,df_nodes,df_edges,df_od)
    