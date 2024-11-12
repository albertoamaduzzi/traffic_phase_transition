from termcolor import cprint
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from Grid import *
import logging
logger = logging.getLogger(__name__)

PRINTING_INTERVAL = 10000000
NUMBER_SIMULATIONS = 20
offset = 6
CityName2RminRmax = {'SFO':[140,200], 'LAX':[100,220],'LIS':[60,100],'RIO':[75,120],'BOS':[150,220]}
#CityName2RminRmax = {'SFO':[145,180], 'LAX':[100,200],'LIS':[60,80],'RIO':[75,100],'BOS':[150,200]}


def GetTotalMovingPopulation(OD_vector):
    return np.sum(OD_vector)
##-------------------------------------------------##
def MapFile2Vectors(ODfmaFile):
    '''
        @param ODfmaFile: str -> Path to the fma file
        @return O_vector: list -> List of origins (index of the polygon)
        @return D_vector: list -> List of destinations (index of the polygon)
        @return OD_vector: list -> List of number of people moving from origin to destination
        @brief: Read the file and store the origin, destination and number of people in the vectors O_vector, D_vector and OD_vector
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

def GetRightTypeList(list_,i):
    if type(list_[0]) == str:
        return str(i)
    elif type(list_[0]) == int:
        return int(i)
    elif type(list_[0]) == float:
        return float(i)

def ObtainODMatrixGrid(save_dir_local,grid_size,grid):
    """
        Either Upload the OD grid or compute it
    """
    if os.path.isfile(os.path.join(save_dir_local,'grid',str(grid_size),'ODgrid.csv')):
        logger.info(f"Upload OD Grid from: {os.path.join(save_dir_local,'grid',str(grid_size),'ODgrid.csv')}...")
        return pd.read_csv(os.path.join(save_dir_local,'grid',str(grid_size),'ODgrid.csv'))
    else:
        logger.info(f"Computing OD Grid to save in: {os.path.join(save_dir_local,'grid',str(grid_size),'ODgrid.csv')}...")
        gridIdx2ij = {grid['index'][i]: (grid['i'].tolist()[i],grid['j'].tolist()[i]) for i in range(len(grid))}
        gridIdx2dest = GridIdx2OD(grid)
        return ODGrid(gridIdx2dest,gridIdx2ij)
        

def GetODGrid(save_dir_local,grid_size):
    if os.path.isfile(os.path.join(save_dir_local,'grid',str(grid_size),'ODgrid.csv')):
        return pd.read_csv(os.path.join(save_dir_local,'grid',str(grid_size),'ODgrid.csv'))
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
    if not os.path.isfile(os.path.join(save_dir_local,'grid',str(grid_size),'ODgrid.csv')):
        df.to_csv(os.path.join(save_dir_local,'grid',str(grid_size),'ODgrid.csv'),sep=',',index=False)
    else:
        df.to_csv(os.path.join(save_dir_local,'grid',str(grid_size),'ODgrid.csv'),sep=',',index=False)
    if not os.path.isfile(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(NameCity,start,end,str(int(R))))):
        df1.to_csv(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}.csv'.format(NameCity,start,end,str(int(R)))),sep=',',index=False)

def ScaleOD(OD_vector,R):
    '''
        @param OD_vector: list -> List of number of people moving from origin to destination
        @param R: int -> Rate of people moving in the simulation
        @return OD_vector: list -> People moving from origin to destination scaled
        @brief Scale the OD vector. In This way we obtain the WANTED rate of people in the simulation.
    '''
    logger.info("Scaling OD...")
    TotalNonModifiedFlux = np.sum(OD_vector)
    FluxWantedIn1HourForControlGroup = R*3600
    ScaleFactor = FluxWantedIn1HourForControlGroup/TotalNonModifiedFlux
    OD_vector = [int(ScaleFactor*OD_vector[i]) for i in range(len(OD_vector))]
    return OD_vector

##-------------------------------------------------##
def GenerateBeginDf(Hour2Files,
                       ODfma_dir,
                       StartControlGroup,
                        polygon2OD,
                        osmid2index,
                        OD2grid,
                        city,
                        gridIdx2dest):
    """
        @param Hour2Files: dict -> Dictionary that maps the hour to the file fma (NOTE: is ordered)
        @return dfBegin: pd.DataFrame -> DataFrame that contains the first 7 hours of the simulation
        @return dfEnd: pd.DataFrame -> DataFrame that contains the last 7 hours of the simulation
        @ Description: Returns the dataframe of the fist StartControlGroup hours of the simulation
        It will be used appending o it the different generated dataframes depending on R (insertion rate)
    """
    FileMidnight = True
    OffsetNPeople = 0
    for start,file in Hour2Files.items():
        logger.info(f"BeginDf {city} start time: {start}")
        ODfmaFile = os.path.join(ODfma_dir,file)
        # Generate the Origin Destination For non modified mobility
        # Concatenated for all different hours.
        O_vector,D_vector,OD_vector = MapFile2Vectors(ODfmaFile)
        logger.info(f"size OD: {len(OD_vector)}, N people: {np.sum(OD_vector)}")
        if start < StartControlGroup:
            # Total Number of People Moving in One Hour
            Nhour = np.sum(OD_vector)
            R = int(Nhour/3600)
            # Do Not Change the OD and Concatenate the input for Simulation                            
            df2 = GetODForSimulationFromFmaPolygonInput(O_vector,
                              D_vector,
                              OD_vector,
                              R,
                              OffsetNPeople,
                              polygon2OD,
                              osmid2index,
                              OD2grid,
                              gridIdx2dest,
                              start,
                              60)
            OffsetNPeople += Nhour
        else:
            break
        if FileMidnight:
            dfBegin = df2
            FileMidnight = False
        else:
            dfBegin = pd.concat([dfBegin, df2], ignore_index=True)
    return dfBegin




##-------------------------------------------------##
def GetODForSimulationFromFmaPolygonInput(O_vector,
                              D_vector,
                              OD_vector,
                              R,
                              OffsetNPeople,
                              polygon2OD,
                              osmid2index,
                              OD2Grid,
                              gridIdx2dest,
                              start,
                              seconds_in_minute
                              ):
    """
        @param O_vector: list -> List of origins (index of the polygon)
        @param D_vector: list -> List of destinations (index of the polygon)
        @param OD_vector: list -> List of number of people moving from origin to destination
        @param polygon2OD: dict -> Maps polygon ids (That are contained in OD <- ODfma_file) to the OD of Tij (in the grid): NOTE: i,j in I,J
        @param osmid2index: dict -> Maps osmid to Index
        @param OD2Grid: dict -> Maps PolygonIds to grid index
        @param gridIdx2dest: dict -> Maps grid index to number of people moving from origin to destination
        @param start: int -> Start time of control group
        @param end: int -> End time of control group
        NOTE:
            GEOMETRY:
                I,J: Set of x,y coordinates of the grid (int numbers)
                Index: Set of 1D indeces of the grid (int number)
                PolygonId: Set of 1D ids of the polygon (int number)
            GRAPH:
                Osmid: Set of 1D ids of the node (int number)
        Output:
            df1: pd.DataFrame -> DataFrame with the OD demand
            df: pd.DataFrame -> DataFrame with the OD grid
            ROutput: list -> List of Rs that have been considered
        Description: 
            Each fma file contains the origin and destinations with the rate of people entering the graph.
            This function, takes advantage of the polygon2origindest dictionary to build the origin and destination
            selecting at random one of the nodes that are contained in the polygon.


    """ 
    # Check the type, If computed is string otherwise int
    KeyPolygon2Od = list(polygon2OD.keys())[0]
    KeyOD2Grid = list(OD2Grid.keys())[0]
    Keyosmid2index = list(osmid2index.keys())[0]
    if type(polygon2OD[KeyPolygon2Od]) == str:
        Polygon2OdStr = True
    else:
        Polygon2OdStr = False
    if type(OD2Grid[KeyOD2Grid]) == str:
        OD2GridStr = True
    else:
        OD2GridStr = False
    if type(osmid2index[Keyosmid2index]) == str:
        osmid2indexStr = True
    else:
        osmid2indexStr = False
    # Rescale OD -> Ensures that the total number of people moving in the simulation is R*3600 otherwhise invariant
    OD_vector = ScaleOD(OD_vector,R)
    logger.info(f"Scaled OD: {np.sum(OD_vector)}")   
    total_number_people_considered = 0
    total_number_people_not_considered = 0
    count_line = OffsetNPeople
    users_id = []
    time_ = []
    origins = []
    destinations = []
    osmid_origin = []
    osmid_destination = []
    # For Each Couple of Origin and Destination
    logger.info(f"Offset: {OffsetNPeople}")
    for i in range(len(O_vector)):
        origin = O_vector[i]
        destination = D_vector[i]
        number_people = OD_vector[i]                   
        # If a flux exists
        if number_people > 0:
            time_increment = 1/number_people
            for person in range(int(number_people)):
                # Get Index of the Origin and Destination From Polygon Index Set
                origin,destination = GetRightTypeOD(origin,destination,polygon2OD)
                # Control there are nodes in the polygon
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
                    t = start*(seconds_in_minute**2) + person*time_increment*seconds_in_minute**2
                    time_.append(t) # TIME SECONDS
                    i = np.random.randint(0,len(polygon2OD[origin]))
                    i = GetRightTypeList(polygon2OD[origin],i)
                    try:
                        idx,_ = GetRightTypeOD(polygon2OD[origin][i],polygon2OD[origin][i],osmid2index)
                        origins.append(osmid2index[idx])
                    except KeyError:
                        total_number_people_not_considered += number_people
                        raise KeyError('KeyError Polygon 2 OD: origin {0} i {1}'.format(origin,i))
                    j = np.random.randint(0,len(polygon2OD[destination]))                        
                    j = GetRightTypeList(polygon2OD[origin],j)
                    try:
                        idx,_ = GetRightTypeOD(polygon2OD[destination][j],polygon2OD[destination][j],osmid2index)
                        destinations.append(osmid2index[idx])
                    except KeyError:
                        total_number_people_not_considered += number_people
                        raise KeyError('KeyError Polygon 2 OD: destination {0} j {1}'.format(origin,i))
                    osmid_origin.append(polygon2OD[origin][i])
                    osmid_destination.append(polygon2OD[destination][j])
                    ## FILLING ORIGIN DESTINATION GRID ACCORDING TO THE ORIGIN DESTINATION NODES
                    try:
                        idx,_ = GetRightTypeOD(polygon2OD[origin][i],polygon2OD[origin][i],OD2Grid)
                        ogrid = OD2Grid[idx]                        
                    except KeyError:
                        total_number_people_not_considered += number_people
                        raise KeyError('KeyError OD 2 Grid: origin {0} i {1}'.format(origin,i))
                    try:
                        idx = GetRightTypeOD(polygon2OD[destination][j],polygon2OD[destination][j],OD2Grid)
                        dgrid = OD2Grid[str(polygon2OD[destination][j])]
                    except KeyError:
                        total_number_people_not_considered += number_people
                        raise KeyError('KeyError OD 2 Grid: destination {0} j {1}'.format(destination,j))
                    gridIdx2dest[(ogrid,dgrid)] += 1
                    count_line += 1
                    if count_line%PRINTING_INTERVAL==0:
                        logger.debug(f"Iteration: {person}")
                        logger.debug(f'People Considered: {total_number_people_considered} (Not: {total_number_people_not_considered})')
                        logger.debug('Lost People: {}'.format(total_number_people_considered/(total_number_people_considered+total_number_people_not_considered)))
                    total_number_people_considered += 1
    logger.info(f'Tot People Considered: {total_number_people_considered} (Not: {total_number_people_not_considered})')
    logger.info('Tot Lost People: {}'.format(total_number_people_considered/(total_number_people_considered+total_number_people_not_considered)))
    df1 = pd.DataFrame({
        'SAMPN':users_id,
        'PERNO':users_id,
        'origin_osmid':osmid_origin,
        'destination_osmid':osmid_destination,
        'dep_time':time_,
        'origin':origins,
        'destination':destinations,
        })

    return df1


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
                Rmin,
                Rmax,
                seconds_in_minute = 60
                ):
    '''
        NOTE:
            GEOMETRY:
                I,J: Set of x,y coordinates of the grid (int numbers)
                Index: Set of 1D indeces of the grid (int number)
                PolygonId: Set of 1D ids of the polygon (int number)
            GRAPH:
                Osmid: Set of 1D ids of the node (int number)
        Input:
            polygon2OD: dict -> Maps polygon ids (That are contained in OD <- ODfma_file) to the OD of Tij (in the grid): NOTE: i,j in I,J
            osmid2index: dict -> Maps osmid to Index
            grid: Geopandas -> ["i": int, "j": int, "centroidx": float, "centroidy": float, "area":float, "index": int, "population":float, "with_roads":bool, "geometry":Polygon]
            grid_size: float -> Size of the grid (0.02 for Boston is 1.5 km^2)
            OD2Grid: dict -> Maps PolygonIds to grid index
            NameCity: str -> Name of the city
            ODfmaFile: str -> Path to the fma file
            start: int -> Start time of the simulation
            end: int -> End time of the simulation
            save_dir_local: str -> Path to the directory where the data is stored
            number_of_rings: int -> Number of rings to consider
            grid_sizes: list -> List of grid sizes to consider
            resolutions: list -> List of resolutions to consider
            offset: int -> Offset of the fma file
            seconds_in_minute: int -> Number of seconds in a minute
        Output:
            df1: pd.DataFrame -> DataFrame with the OD demand
            df: pd.DataFrame -> DataFrame with the OD grid
            ROutput: list -> List of Rs that have been considered
        Description: 
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
    spacing = (Rmax/R - Rmin/R)/NUMBER_SIMULATIONS
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
            df1 = GetODForSimulationFromFmaPolygonInput(O_vector,
                                                        D_vector,
                                                        OD_vector,
                                                        R,
                                                        polygon2OD,
                                                        osmid2index,
                                                        OD2Grid,
                                                        gridIdx2dest,
                                                        start,
                                                        seconds_in_minute
                                                        )
            df = ODGrid(gridIdx2dest,gridIdx2ij)
            print('df:\n',df.head())
            print('population moving: ',df['number_people'].sum())
            print('df1:\n',df1.head())
            R = multiplicative_factor*R
            ROutput.append(int(R))
            SaveOD(df,df1,save_dir_local,NameCity,str(start),str(end),str(int(R)),str(grid_size))
    return df1,df,ROutput




# def OD_from_potential TODO: Inputs O_vector,D_vector,OD_vector from potential in the grid.

##-------------------------------------------------##
def AdjustDetailsBeforeConvertingFma2Csv(GeometricalInfo):
    GeometricalInfo.gdf_polygons['index'] = GeometricalInfo.gdf_polygons.index
    GeometricalInfo.gdf_polygons[['geometry','index']].to_file(os.path.join(GeometricalInfo.shape_file_dir_local,GeometricalInfo.city + 'new'+'.shp'))
    GeometricalInfo.nodes = pd.read_csv(os.path.join(GeometricalInfo.save_dir_local,'nodes.csv'))
    GeometricalInfo.osmid2index = dict(zip(GeometricalInfo.nodes['osmid'], GeometricalInfo.nodes['index']))
    with open(os.path.join(GeometricalInfo.save_dir_local,'osmid2idx.json'),'w') as f:
        json.dump(GeometricalInfo.osmid2index,f,indent=4)
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
