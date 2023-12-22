import geopandas as gpd
import numpy as np
import os
import pandas as pd
from global_functions import ifnotexistsmkdir
import osmnx as ox
from collections import defaultdict
from shapely.geometry import Point,Polygon
import matplotlib.pyplot as plt
import json
import networkx as nx
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

def polygon2origin_destination(gdf,carto,dir_,debug = False):
    '''
        Given a network taken from the cartography or ours:
            Build the set of origin and destinations from the polygons that are coming from the 
            geodataframe.
        
    '''
    polygon2origindest = defaultdict(list)
    possiblekeys = ['tractid','tract_id','tractID_ne','TRACTCE']
    for node in carto.nodes():
        containing_polygon = gdf.geometry.apply(lambda x: Point(carto.nodes[node]['x'],carto.nodes[node]['y']).within(x))
#        print('containing_polygon: ',containing_polygon)
        idx_containing_polygon = gdf[containing_polygon].index
#        print('idx_containing_polygon: ',idx_containing_polygon)
        if dir_ == 'SFO':
            key = 'TRACT'
        if dir_ == 'LAX':
            key = 'external_i'
        if dir_ == 'LIS':
            key = 'ID'
        if dir_ == 'RIO':
            key = 'Zona'
        tract_id = gdf.loc[idx_containing_polygon][key]
        if len(tract_id)==1: 
            try:
                tract_id = int(tract_id.tolist()[1])
            except IndexError:
                tract_id = int(tract_id.tolist()[0])
#            print('tract_id: ',tract_id)
            polygon2origindest[int(tract_id)].append(node)
            if debug:
                print('found polygon: ',idx_containing_polygon)
                print('tract_id: ',tract_id)
                print('type tract_id: ',type(tract_id))
                print('values dict: ',polygon2origindest[tract_id])
                print('dict: ',polygon2origindest)

        elif len(tract_id)>1:
            print('tract_id: ',tract_id)
            print('more than one tract id: THIS IS STRANGE')

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
    PRINTING_INTERVAL = 10000000
    users_id = []
    time_ = []
    origins = []
    destinations = []
    total_number_people_not_considered = 0
    total_number_people_considered = 0
    with open(file,'r') as infile:
        count_line = 0
        print('OPENED FILE: ',file)
        for line in infile:
            count_line += 1
            if count_line > offset:
                tok = line.split(' ')
                origin = tok[0]
                destination = tok[1]
                number_people = int(float(tok[2].split('\n')[0]))
                bin_width = 1
                iterations = number_people/bin_width    
                if count_line%PRINTING_INTERVAL==0:
                    print('iterations: ',iterations,' number_people: ',number_people,' origin: ',origin,' destination: ',destination,' R: ',R)        
                if number_people > 0:
                    if count_line%PRINTING_INTERVAL==0:
                        print('number_people: ',number_people)
                    try:
                        for it in range(int(iterations)):
                            if count_line%PRINTING_INTERVAL==0:
                                print('NUMBER DEST per polygon {}: '.format(destination),len(polygon2origindest[destination]),' NUMBER ORIGIN per polygon {}: '.format(origin),len(polygon2origindest[origin]))
                                print('it: ',it)                        
                            if len(polygon2origindest[destination])>0 and len(polygon2origindest[origin])>0:
                                if count_line%PRINTING_INTERVAL==0:
                                    print('NUMBER NODES ORIGIN: ',len(polygon2origindest[origin]))
                                    print('NUMBER NODES DESTINATION: ',len(polygon2origindest[destination]))                                                    
                                users_id.append(count_line-offset)
                                time_.append(start_hour*seconds_in_minute + it*seconds_in_minute)
                                i = np.random.randint(0,len(polygon2origindest[origin]))
                                origins.append(polygon2origindest[origin][i])
                                i = np.random.randint(0,len(polygon2origindest[destination]))                        
                                destinations.append(polygon2origindest[destination][i])
                                total_number_people_considered += 1
                    except KeyError:
                        total_number_people_not_considered += number_people
                        #print('Key not found at iteration: ',iterations,' number_people: ',number_people,' origin: ',origin,' destination: ',destination,' R: ',R)        
        print('number_people considered: ',total_number_people_considered)
        print('number_people not considered: ',total_number_people_not_considered)
        print('Loss bigger 5%',total_number_people_not_considered/total_number_people_considered>0.05)
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

def configOD(carto_dir,
             shape_file_dir,
            name,
            start,
            end,
            df1,
            R,
            p= 0.5,
            number_of_rings = 10,
            grid_size = 1,
            tif_file='usa_ppp_2020_UNadj_constrained.tif'):
    config = {'carto_dir':carto_dir, # Directory where the cartographic data is stored
        'shape_file_dir': shape_file_dir, # Directory where the shape files are stored
        'start': start, # Time where cumulation of trips starts
        'end': end, # If not specified, it is assumed that the OD is for 1 hour (time where cumlation of trips ends)
        "name":name,
        "number_users":len(df1),
        "R":R,
        "grid_size":grid_size, # Grid size in km
        'number_of_rings':number_of_rings,
        "p":p,
        "tif_file":tif_file
              }
    return config
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

def save_pol2orddest(save_dir,polygon2origindest):
    '''
        Input:
            carto: string [path to the folder where to save the files]
        Description:
            Save the files in the carto folder
    '''
    json_obj = json.dumps(polygon2origindest,indent=4)
    with open(os.path.join(save_dir,'polygon2origindest.json'),'w') as f:
        f.write(json_obj)

    pass
if __name__=='__main__':
## GLOBAL    
    dir2labels = {
        'BOS':"Boston, Massachusetts, USA",
        'SFO':"San Francisco, California, USA",
        'LAX':"Los Angeles, California, USA",
        'RIO':"Rio de Janeiro, Brazil",
        "LIS":"Lisbon, Portugal",
    }
    city2file_tif = {
        'BOS':"usa_ppp_2020_UNadj_constrained.tif",
        'SFO':"usa_ppp_2020_UNadj_constrained.tif",
        'LAX':"usa_ppp_2020_UNadj_constrained.tif",
        'RIO':"bra_ppp_2020_UNadj_constrained.tif",
        "LIS":"prt_ppp_2020_UNadj_constrained.tif",
    }
    R_city = {
        'BOS':np.linspace(120,200,80),
        'SFO':np.linspace(120,200,80),
        'LAX':np.linspace(120,200,80),
        'RIO':np.linspace(50,100,50),
        "LIS":np.linspace(20,70,50),
    }
    map_already_computed ={
        'BOS':True,
        'SFO':True,
        'LAX':True,
        'RIO':False,
        "LIS":False,
    }
    names_carto = {
        'BOS':"Boston",
        'SFO':"San_Francisco",
        'LAX':"Los_Angeles",
    }
    list_cities = ['BOS','SFO','LAX','RIO','LIS']
    populations = [4.5e6,7.5e6,13e6,12.6e6,2.8e6]
    config_dir = '/home/aamad/Desktop/phd/berkeley/traffic_phase_transition/config'
    root = '/home/aamad/Desktop/phd/berkeley/data/carto'
    base_dir_shape = '/home/aamad/Desktop/phd/berkeley/data'
#    infocity = {city: {'population':0,'Area':0,'Roads length':0,'Volume':0} for city in list_cities}
    pop_count = 0
    for dir_ in list_cities:
        save_dir = os.path.join(root,dir_)            
        ifnotexistsmkdir(save_dir)
        print(os.path.join(root,dir_,dir_ + '_new_tertiary_simplified.graphml'))
        G = ox.load_graphml(filepath=os.path.join(save_dir,dir_ + '_new_tertiary_simplified.graphml'))
        gdf = read_file_gpd(os.path.join(base_dir_shape,dir_,dir_ + '.shp'))
#        infocity[dir_]['population'] = populations[pop_count]
#        infocity[dir_]['Area'] = gdf.to_crs(4326).area.sum()
#        infocity[dir_]['Roads length'] = G.size(weight='length')
        ## FROM POLYGON TO ORIGIN DESTINATION -> OD FILE
        if os.path.isfile(os.path.join(save_dir,'polygon2origindest.json')):
            if map_already_computed[dir_]:
                print('already computed origin destination; pass.')
                pass
            else:
                with open(os.path.join(save_dir,'polygon2origindest.json'),'r') as infile:
                    pol_ordest = json.load(infile)
                for file in os.listdir(os.path.join(base_dir_shape,dir_)):
                    if file.endswith('.fma'):
                        start = int(file.split('.')[0].split('D')[1])
                        end = start + 1
                        R = 1
                        print('file.fma: ',file)
                        file_name = os.path.join(base_dir_shape,dir_,file)
    #                        histo_point2polygon(pol_ordest)      
    #                    for R in R_city[dir_]:
    #                        print('R: ',R)
                        df_od = OD_from_fma(file_name,pol_ordest)#,R
#                        infocity[dir_]['Volume']= len(df_od)
                        print('saving od:\n',save_dir,' start: ',start,' end: ',end)
                        save_od(save_dir,df_od,R,start,end)#,R
                        config = configOD(os.path.join(root,dir_),os.path.join(root,dir_),dir_,start,end,df_od,R,city2file_tif[dir_])
                        with open(os.path.join(config_dir,'{%sdir_}configOD_{%sstart}_{%send}_R_{%sR}.json'),'w') as f:
                            json.dump(config,f,indent=4)


        else:
            print('computing polygon2origindest')
            pol_ordest = polygon2origin_destination(gdf,G,dir_)  
            print('saving polygon2origindest')
            save_pol2orddest(save_dir,pol_ordest)
            for file in os.listdir(os.path.join(base_dir_shape,dir_)):
                print('file.fma: ',file)
                if file.endswith('.fma'):
                    start = int(file.split('.')[0][-1])
                    end = start + 1
                    R = 1
                    file_name = os.path.join(base_dir_shape,dir_,file)
#                        histo_point2polygon(pol_ordest)      
#                    for R in R_city[dir_]:
#                        print('R: ',R)
                    df_od = OD_from_fma(file_name,pol_ordest)#,R
#                    infocity[dir_]['Volume']= len(df_od)
                    print('saving od')
                    save_od(save_dir,df_od,R,start,end)
                    config = configOD(os.path.join(root,dir_),os.path.join(root,dir_),dir_,start,end,df_od,R,city2file_tif[dir_])
                    with open(os.path.join(config_dir,'{%sdir_}configOD_{%sstart}_{%send}_R_{%sR}_p_{%sround(p,3)}.json'),'w') as f:
                        json.dump(config,f,indent=4)
        pop_count += 1
#    with open(os.path.join(root,'infocity.json'),'w') as f:
#        json.dump(infocity,f,indent=4)
'''
    Description:
        I take the shape file and convert it to nodes.csv, edges.csv 
        Take OD and convert it to origin_destination.csv
    Input:
        G: cartography that is polished in mobility_planner.py
        gdf: polygon file utilized in mobility planner to reconstruct a polished cartography
    Output:
        _routes.csv ->  each row is index of the person
        _people.csv -> time_arrival is time_departure+numb_steps


'''
