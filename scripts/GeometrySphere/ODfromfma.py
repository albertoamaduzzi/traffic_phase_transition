from termcolor import cprint
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from Grid import *
PRINTING_INTERVAL = 10000000
offset = 6
CityName2RminRmax = {'SFO':[145,180], 'LAX':[100,200],'LIS':[60,80],'RIO':[75,100],'BOS':[150,200]}


def GetTotalMovingPopulation(OD_vector):
    return np.sum(OD_vector)
##-------------------------------------------------##
def MapFile2Vectors(ODfmaFile):
    '''
        Read the file and store the origin, destination and number of people in the vectors O_vector, D_vector and OD_vector
    '''
    O_vector = []
    D_vector = []
    OD_vector = []
    # THESE POINTS ARE NATURALLY ON THE POLYGONS        
    with open(ODfmaFile,'r') as infile:
        count_line = 0
        for line in infile:
            count_line += 1
            if count_line > offset:
                tok = line.split(' ')
                O_vector.append(int(tok[0]))
                D_vector.append(int(tok[1]))
                OD_vector.append(int(float(tok[2].split('\n')[0])))
    return O_vector,D_vector,OD_vector

def GetRightTypeOD(origin,destination,polygon2OD):
    if type(list(polygon2OD.keys())[0]) == str:
        origin = str(origin)
        destination = str(destination)
    elif type(list(polygon2OD.keys())[0]) == int:
        origin = int(origin)
        destination = int(destination)
    elif type(list(polygon2OD.keys())[0]) == float:
        origin = float(origin)
        destination = float(destination)
    return origin,destination                    

def GetODGrid(save_dir_local,grid_size):
    if os.path.isfile(os.path.join(save_dir_local,'grid',grid_size,'ODgrid.csv')):
        return pd.read_csv(os.path.join(save_dir_local,'grid',grid_size,'ODgrid.csv'))
    else:
        return None

def GetOD(save_dir_local,NameCity,start,end,R):
    if os.path.isfile(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(NameCity,start,end,int(R)))):
        return pd.read_csv(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(NameCity,start,end,int(R))))
    else:
        return None
##---------`----------------------------------------##
def SaveOD(df,df1,save_dir_local,NameCity,start,end,R,grid_size):
    '''
        Save the OD grid and the OD demand:
            TRAFFIC_DIR/data/carto/{NameCity}/grid/{grid_size}/ODgrid.csv
            TRAFFIC_DIR/data/carto/{NameCity}/OD/{NameCity}_oddemand_{start}_{end}_R_{R}.csv
    '''
    if not os.path.isfile(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),'ODgrid.csv')):
        df.to_csv(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),'ODgrid.csv'),sep=',',index=False)
    else:
        df.to_csv(os.path.join(save_dir_local,'grid',str(round(grid_size,3)),'ODgrid.csv'),sep=',',index=False)
    if not os.path.isfile(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(NameCity,start,end,str(int(R))))):
        df1.to_csv(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(NameCity,start,end,str(int(R)))),sep=',',index=False)



##-------------------------------------------------##
def CumulativeODPolygonGivenR(O_vector,D_vector,OD_vector,polygon2OD,osmid2index,OD2Grid,gridIdx2dest,start,end,multiplicative_factor,seconds_in_minute,R):
    total_number_people_considered = 0
    total_number_people_not_considered = 0
    count_line = 0
    users_id = []
    time_ = []
    origins = []
    destinations = []
    osmid_origin = []
    osmid_destination = []
    print('map Nodes Carto to grid:\n ')
    count = 0
    for key, value in OD2Grid.items():
        print(key, ':', value)
        count += 1
        if count == 3:
            break    
    print('number of couples of origin-destination: ',len(O_vector))
    for i in range(len(O_vector)):
        origin = O_vector[i]
        destination = D_vector[i]
        number_people = OD_vector[i]
        bin_width = 1                        
        if number_people > 0:
            iterations = multiplicative_factor*number_people/bin_width   
            time_increment = 1/iterations
            for it in range(int(iterations)):
                origin,destination = GetRightTypeOD(origin,destination,polygon2OD)
                try:
                    Originbigger0 = len(polygon2OD[origin])>0
                except KeyError:
                    total_number_people_not_considered += number_people
                    break
                try:
                    Destinationbigger0 = len(polygon2OD[destination])>0
                except KeyError:
                    total_number_people_not_considered += number_people
                    break
                if  Originbigger0 and Destinationbigger0:
                    users_id.append(count_line)
                    t = start*(seconds_in_minute**2) + it*time_increment*seconds_in_minute**2
                    time_.append(t) # TIME IN HOURS
                    i = np.random.randint(0,len(polygon2OD[origin]))
                    try:
                        origins.append(osmid2index[polygon2OD[origin][i]])
                    except KeyError:
                        total_number_people_not_considered += number_people
                        raise KeyError('KeyError Polygon 2 OD: origin {0} i {1}'.format(origin,i))
                    j = np.random.randint(0,len(polygon2OD[destination]))                        
                    try:
                        destinations.append(osmid2index[polygon2OD[destination][j]])
                    except KeyError:
                        total_number_people_not_considered += number_people
                        raise KeyError('KeyError Polygon 2 OD: destination {0} j {1}'.format(origin,i))
                    osmid_origin.append(polygon2OD[origin][i])
                    osmid_destination.append(polygon2OD[destination][j])
                    ## FILLING ORIGIN DESTINATION GRID ACCORDING TO THE ORIGIN DESTINATION NODES
                    try:
                        ogrid = OD2Grid[str(polygon2OD[origin][i])]
                    except KeyError:
                        total_number_people_not_considered += number_people
                        raise KeyError('KeyError OD 2 Grid: origin {0} i {1}'.format(origin,i))
                    try:
                        dgrid = OD2Grid[str(polygon2OD[destination][j])]
                    except KeyError:
                        total_number_people_not_considered += number_people
                        raise KeyError('KeyError OD 2 Grid: destination {0} j {1}'.format(destination,j))
                    gridIdx2dest[(ogrid,dgrid)] += 1
                    count_line += 1
                    if count_line%PRINTING_INTERVAL==0:
                        print('Origin: ',origin,' Osmid Origin: ',osmid2index[polygon2OD[str(origin)][i]],' Destination: ',destination,' Osmid Destination: ',osmid2index[polygon2OD[str(destination)][i]],' Number of people: ',number_people,' R: ',R)
                        print(len(destinations),len(origins))
                        cprint('iteration: ' + str(it) + ' number_people: ' + str(number_people) + ' origin: ' + str(origin) + ' #nodes('+ str(len(polygon2OD[origin])) + ') ' + ' destination: ' + str(destination) + ' #nodes('+ str(len(polygon2OD[destination])) + ') '+ ' R: ' + str(R),'green')        
                        print('time insertion: ',start*seconds_in_minute**2 + it*time_increment*seconds_in_minute**2,' time min: ',start*seconds_in_minute**2,' time max: ',end*seconds_in_minute**2,' time max iteration: ',start*seconds_in_minute**2 + (iterations)*time_increment*seconds_in_minute**2)
                    total_number_people_considered += 1
    print('total_number_people_considered: ',total_number_people_considered)
    print('total_number_people_not_considered: ',total_number_people_not_considered)
    print('ratio: ',total_number_people_considered/(total_number_people_considered+total_number_people_not_considered))
    return users_id,time_,origins,destinations,gridIdx2dest,osmid_origin,osmid_destination


##--------------------------ALL Rs-----------------------##
def OD_from_fma(polygon2OD,
                osmid2index,
                grid,
                grid_size,
                OD2Grid,
                NameCity,
                ODfmaFile,
                start,
                end,
                save_dir_local,
                number_of_rings,
                grid_sizes,
                resolutions,
                offset = 6,
                seconds_in_minute = 60,
                ):
    '''
        Each fma file contains the origin and destinations with the rate of people entering the graph.
        This function, takes advantage of the polygon2origindest dictionary to build the origin and destination
        selecting at random one of the nodes that are contained in the polygon.
    '''
    ROutput = []
    # NOTE: ADD HERE THE POSSIBILITY OF HAVING OD FROM POTENTIAL CONSIDERATIONS
    O_vector,D_vector,OD_vector = MapFile2Vectors(ODfmaFile)
    # START TAB
    R = GetTotalMovingPopulation(OD_vector)/3600 # R is the number of people that move in one second (that is the time interval for the evolution )
    Rmin = CityName2RminRmax[NameCity][0]
    Rmax = CityName2RminRmax[NameCity][1]
    spacing = (Rmax/R - Rmin/R)/20
    cprint('OD_from_fma {} '.format(NameCity) + ODfmaFile,'cyan')
    cprint('R: ' + str(R) + ' Rmin: ' + str(Rmin) + ' Rmax: ' + str(Rmax) + ' spacing: ' + str(spacing),'cyan')
    gridIdx2ij = {grid['index'][i]: (grid['i'].tolist()[i],grid['j'].tolist()[i]) for i in range(len(grid))}
    for multiplicative_factor in np.arange(Rmin/R,Rmax/R,spacing):
        R = GetTotalMovingPopulation(OD_vector)/3600 
        if os.path.isfile(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(NameCity,start,end,int(multiplicative_factor*R)))):
            cprint(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(NameCity,start,end,int(multiplicative_factor*R))),'cyan')
            ROutput.append(int(multiplicative_factor*R))
            df = pd.DataFrame({})
            df1 = pd.DataFrame({})
            continue
        else:
            gridIdx2dest = GridIdx2OD(grid)    
            cprint('COMPUTING {}'.format(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(NameCity,start,end,int(multiplicative_factor*R)))),'cyan')
            users_id,time_,origins,destinations,gridIdx2dest,osmid_origin,osmid_destination = CumulativeODPolygonGivenR(O_vector,D_vector,OD_vector,polygon2OD,osmid2index,OD2Grid,gridIdx2dest,start,end,multiplicative_factor,seconds_in_minute,R)
            df = ODGrid(gridIdx2dest,gridIdx2ij)
            print('df:\n',df.head())
            print('population moving: ',df['number_people'].sum())
            df1 = pd.DataFrame({
                'SAMPN':users_id,
                'PERNO':users_id,
                'origin_osmid':osmid_origin,
                'destination_osmid':osmid_destination,
                'dep_time':time_,
                'origin':origins,
                'destination':destinations,
                })
            print('df1:\n',df1.head())
            R = multiplicative_factor*R
            ROutput.append(int(R))
            SaveOD(df,df1,save_dir_local,NameCity,str(start),str(end),str(int(R)),round(grid_size,3))
    return df1,df,ROutput


# def OD_from_potential TODO: Inputs O_vector,D_vector,OD_vector from potential in the grid.

##-------------------------------------------------##
def AdjustDetailsBeforeConvertingFma2Csv(GeometricalInfo):
    GeometricalInfo.gdf_polygons['index'] = GeometricalInfo.gdf_polygons.index
    GeometricalInfo.gdf_polygons[['geometry','index']].to_file(os.path.join(GeometricalInfo.shape_file_dir_local,GeometricalInfo.city + 'new'+'.shp'))
    cprint('Setting the graph right','yellow')
    GeometricalInfo.nodes = pd.read_csv(os.path.join(GeometricalInfo.save_dir_local,'nodes.csv'))
    GeometricalInfo.osmid2index = dict(zip(GeometricalInfo.nodes['osmid'], GeometricalInfo.nodes['index']))
    cprint('osmid2index: ','yellow')
    with open(os.path.join(GeometricalInfo.save_dir_local,'osmid2idx.json'),'w') as f:
        json.dump(GeometricalInfo.osmid2index,f,indent=4)
    cprint('index2osmid: ','yellow')
    GeometricalInfo.index2osmid = dict(zip(GeometricalInfo.nodes['index'], GeometricalInfo.nodes['osmid']))
    with open(os.path.join(GeometricalInfo.save_dir_local,'idx2osmid.json'),'w') as f:
        json.dump(GeometricalInfo.index2osmid,f,indent=4)
    AdjustEdges(GeometricalInfo.save_dir_local,GeometricalInfo.osmid2index,GeometricalInfo.index2osmid)
    return True


def AdjustEdges(save_dir_local,osmid2index,index2osmid):
    '''
        If The edges file has got columns u,v that are osmid, replaces them with the index
        If The edges file has got columns u,v that are index, creates osmid_u and osmid_v
        If both the columns are already there does nothing
    '''
    cprint('Adjust edges file','green')
    try:
        edges = pd.read_csv(os.path.join(save_dir_local,'edges.csv'))
        edges['u'] = edges['u'].apply(lambda x: osmid2index[x])
        edges['v'] = edges['v'].apply(lambda x: osmid2index[x])
        edges.to_csv(os.path.join(save_dir_local,'edges.csv'),index=False)
    except KeyError:
        cprint('edges.csv ALREADY COMPUTED','green')
        try:
            edges['osmid_u'] = edges['u'].apply(lambda x: index2osmid[x])
            edges['osmid_v'] = edges['v'].apply(lambda x: index2osmid[x])
            edges.to_csv(os.path.join(save_dir_local,'edges.csv'),index=False)
        except KeyError:
            cprint('edges.csv HAS GOT ALREADY, [u,v,osmid_u,osmid_v]','green')
            pass
