import geopandas as gpd
import numpy as np
import os
import pandas as pd
from global_functions import ifnotexistsmkdir
import osmnx as ox
from collections import defaultdict
from shapely.geometry import Point,Polygon
import matplotlib.pyplot as plt
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

##------------------------------------- CONVERT TO CSV -------------------------------------##

def nodes_file_from_gpd(carto):
    '''
        Reads from shape and returns the nodes.csv
        Index is the index that will be used in too and origin and destination.
    '''
    osmid2id = defaultdict()
    i = 0
    for node in carto.nodes():
        osmid2id[node] = i
        i += 1  
    id_ = [i for i in range(len(carto.nodes()))]
    vertices = [node for node in carto.nodes()]
    df1 = pd.DataFrame({
        'osmid':vertices,
        'x':[carto.nodes[node]['x'] for node in vertices],
        'y':[carto.nodes[node]['y'] for node in vertices],
        'ref':['' for node in vertices],
        'index':id_
        })

    return df1,osmid2id
##------------------------------------- EDGES -------------------------------------##
def edges_file_from_gpd(carto,osmid2id):
    all_edges = carto.edges(data=True)
#    osmid = [BO[edge[0]][edge[1]]['osmid'] for edge in all_edges]    
#    name = [BO[edge[0]][edge[1]]['name'] for edge in all_edges]
#    df_osmid_u = pd.DataFrame([edge[0] for edge in all_edges],name='osmid_u')
#    df_osmid_v = pd.DataFrame([edge[1] for edge in all_edges],name='osmid_v')    
    try:
        lanes = [edge[2]['lanes'] if type(edge[2]['lanes']) != list else '' for edge in all_edges]
    except KeyError:
        lanes = ['' for edge in all_edges]
    try:
        speed_mph = [edge[2]['maxspeed'] for edge in all_edges]
    except KeyError:
        speed_mph = ['' for edge in all_edges]
    try:
        highway = [edge[2]['highway'] for edge in all_edges]
    except KeyError:
        highway = ['' for edge in all_edges]
    df1 = pd.DataFrame({
        'unique_id':[i for i in range(len(all_edges))],
        'u':[osmid2id[edge[1]] for edge in all_edges],
        'v':[osmid2id[edge[0]] for edge in all_edges],
        'length':[edge[2]['length'] for edge in all_edges],
        'lanes':lanes,
        'speed_mph':speed_mph,
        'highway':highway
        })
    
    return df1

##------------------------------------- OD -------------------------------------##

def polygon2origin_destination(gdf,carto):
    '''
        Given a network taken from the cartography or ours:
            Build the set of origin and destinations from the polygons that are coming from the 
            geodataframe.
        
    '''
    polygon2origindest = defaultdict(list)
    possiblekeys = ['tractID','tractid','TRACTID','tract_id','tract_ID','TRACT_ID','TRACTCE']
    for node in carto.nodes():
        containing_polygon = gdf.geometry.apply(lambda x: x.contains(Point(carto.nodes[node]['x'],carto.nodes[node]['y'])))
#        print('containing_polygon: ',containing_polygon)
        idx_containing_polygon = gdf[containing_polygon].index
#        print('idx_containing_polygon: ',idx_containing_polygon)
        found_key = False
        for key in possiblekeys:
            if key in gdf.columns:
                found_key = True
                break
        if found_key:
            tract_id = gdf.loc[idx_containing_polygon][key]
        else:
            raise KeyError('No key found in: ',gdf.columns)
        if len(tract_id)==1: 
            tract_id = tract_id.tolist()[0]
#            print('tract_id: ',tract_id)
            polygon2origindest[tract_id].append(node)
        elif len(tract_id)>1:
            raise ValueError('more than one tract id: THIS IS STRANGE')
        else:
            pass

    return polygon2origindest

def histo_point2polygon(polygin2origindest):
    bins = np.arange(len(polygin2origindest.keys()))
    value = np.array([len(polygin2origindest[polygon]) for polygon in polygin2origindest.keys()])
    plt.bar(bins,value)
    plt.xlabel('number of polygons')
    plt.ylabel('number of points per polygon')
    plt.show()
def OD_from_fma(file,
                polygon2origindest,
                R = 1,
                offset = 6,
                seconds_in_minute = 60,
                start_hour = 7
                ):
    '''
        Each fma file contains the origin and destinations with the rate of people entering the graph.
        This function, takes advantage of the polygon2origindest dictionary to build the origin and destination
        selecting at random one of the nodes that are contained in the polygon.
    '''
    users_id = []
    time_ = []
    origins = []
    destinations = []
    with open(file,'r') as infile:
        count_line = 0
        print('OPENED FILE: ',file)
        for line in infile:
            count_line += 1
            if count_line > offset:
                tok = line.split(' ')
                origin = int(tok[0])
                destination = int(tok[1])
                number_people = int(float(tok[2].split('\n')[0]))
                bin_width = 1
                iterations = number_people/bin_width    
                if count_line%10000==0:
                    print('iterations: ',iterations,' number_people: ',number_people,' origin: ',origin,' destination: ',destination,' R: ',R)        
                    
                if number_people > 0:
                    if count_line%10000==0:
                        print('number_people: ',number_people)
                    for it in range(int(iterations)):
                        if count_line%10000==0:
                            print('NUMBER DEST per polygon {}: '.format(destination),len(polygon2origindest[destination]),' NUMBER ORIGIN per polygon {}: '.format(origin),len(polygon2origindest[origin]))
                            print('it: ',it)
                        if len(polygon2origindest[destination])>0 and len(polygon2origindest[origin])>0:
                            if count_line%10000==0:
                                print('NUMBER NODES ORIGIN: ',len(polygon2origindest[origin]))
                                print('NUMBER NODES DESTINATION: ',len(polygon2origindest[destination]))                            
                            for r in range(R):
                                users_id.append(count_line-offset)
                                time_.append(start_hour*seconds_in_minute + it*seconds_in_minute)
                                i = np.random.randint(0,len(polygon2origindest[origin]))
                                origins.append(polygon2origindest[origin][i])
                                i = np.random.randint(0,len(polygon2origindest[destination]))                        
                                destinations.append(polygon2origindest[destination][i])
    print('number elements OD: ',len(destinations))
    df1 = pd.DataFrame({
        'SAMPN':users_id,
        'PERNO':users_id,
        'origin':origins,
        'destination':destinations,
        'dep_time':time_,
        'origin_osmid':origins,
        'destination_osmid':destinations
        })
    return df1
##------------------------------------- SAVE FILES -------------------------------------##

def save_nodes_edges(save_dir,df_nodes,df_edges):
    '''
        Input:
            carto: string [path to the folder where to save the files]
        Description:
            Save the files in the carto folder
    '''
    df_nodes.to_csv(os.path.join(save_dir,'nodes.csv'),sep=',',index=False)
    df_edges.to_csv(os.path.join(save_dir,'edges.csv'),sep=',',index=False)
    pass

def save_od(save_dir,df_od,R = 1,start=7,end=8):
    '''
        Input:
            carto: string [path to the folder where to save the files]
        Description:
            Save the files in the carto folder
    '''
    df_od.to_csv(os.path.join(save_dir,'od_demand_{0}to{1}_R_{2}.csv'.format(start,end,R)),sep=',',index=False)

    pass

if __name__=='__main__':
## GLOBAL    
    dir2labels = {
        'BO':"Boston, Massachusetts, USA",
        'SFO':"San Francisco, California, USA",
        'LA':"Los Angeles, California, USA",
    }
    names_carto = {
        'BOS':"Boston",
        'SFO':"San_Francisco",
        'LAX':"Los_Angeles",
    }
    data_dir,carto_base = set_environment()
    for root,dirs,files in os.walk(data_dir,topdown=True):
        print('root: ',root)
        for dir_ in dirs:
            carto_computed = False
            if dir_ != 'carto':
                print('dir_: ',dir_)
                carto = ox.graph_from_place(dir2labels[dir_], network_type="drive")
                save_dir = os.path.join(carto_base,dir_)            
                ifnotexistsmkdir(save_dir)
                print('save_dir: ',save_dir)            
                for file in os.listdir(os.path.join(root,dir_)):
                    print('file: ',file)
                    file_name = os.path.join(data_dir,dir_,file)
                    if file.endswith('.fma'):
                        if carto_computed == False:
                            gdf = read_file_gpd(file_name.split('.')[0] + '.shp')
                        pol_ordest = polygon2origin_destination(gdf,carto)  
                        histo_point2polygon(pol_ordest)      
                        for R in range(1,11):
                            df_od =OD_from_fma(file_name,pol_ordest,R)
                            print('df_od: ',df_od)
                            save_od(save_dir,df_od,R)
                    elif carto_computed == False:
                        file_name = file_name.split('.')[0] + '.shp'
                        gdf = read_file_gpd(file_name)                        
                        df_nodes,osmid2id = nodes_file_from_gpd(carto)
                        df_edges = edges_file_from_gpd(carto,osmid2id)
                        save_nodes_edges(save_dir,df_nodes,df_edges)
                        carto_computed = True

'''
    Description:
        I take the shape file and convert it to nodes.csv, edges.csv 
        Take OD and convert it to origin_destination.csv

    Output:
        _routes.csv ->  each row is index of the person
        _people.csv -> time_arrival is time_departure+numb_steps


'''
