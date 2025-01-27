from termcolor import cprint
import os
import numpy as np
import pandas as pd
from ODfromfma import CityName2RminRmax
import logging
logger = logging.getLogger(__name__)

def CheckAndConvert2AvailableTypesKeys(origin,AvailableTypesKeys,grid2OD):
    if type(origin) not in AvailableTypesKeys:
        for type_ in AvailableTypesKeys:
            origin = type_(origin)
            if origin in grid2OD.keys():
                return origin 
            else:
                print("No Conversion Possible {}".format(origin)) 
    else:
        return origin
def GetAvailableTypesKeys(grid2OD):
    AvailableTypesKeys = []
    for key in grid2OD.keys():
        if type(key) not in AvailableTypesKeys:
            AvailableTypesKeys.append(type(key))
    return AvailableTypesKeys


def GenerateInfoInputSimulationFromGridOD(O_vector,D_vector,OD_vector,osmid2index,grid2OD,start,NOffset,seconds_in_minute = 60):
    '''
        @brief: Generate the Input for the Simulation from the Grid2OD
        @param: O_vector: (np.array 1D) -> origin subset with repetition [0,...,Ngridx*Ngridy-1]
        @param: D_vector: (np.array 1D) -> destination subset with repetition [0,...,Ngridx*Ngridy-1]
        @param: OD_vector: (np.array 1D) -> number of people from Tij['number_people'].to_numpy()
        @param: osmid2index: (dict) -> osmid2index = {osmid:i}
        @param: grid2OD: (dict) -> grid2OD = {(i,j):OD}
        @param: start: (int) -> start time
        @param: seconds_in_minute: (int) -> seconds in a minute
        @return: df1: (pd.DataFrame) -> 
        df1 = pd.DataFrame({"SAMPN":users_id,"PERNO":users_id,"origin_osmid":osmid_origin,"destination_osmid":osmid_destination,"dep_time":time_,"origin":origins,"destination":destinations})
        NOTE: grid2OD,keys() in [0,...,Ngridx*Ngridy-1] <- Stored like Strings
        NOTE: grid2OD,values() in [osmid0,...,osmidN-1]
    '''
    assert len(O_vector) == len(D_vector) == len(OD_vector), 'Lengths of O_vector, D_vector, and OD_vector must be the same'
    assert osmid2index is not None, 'osmid2index must be provided'
    assert grid2OD is not None, 'grid2OD must be provided'
    assert start is not None, 'start must be provided'
    total_number_people_considered = 0
    total_number_people_not_considered = 0
    count_line = NOffset
    users_id = []
    time_ = []
    origins = []
    destinations = []
    osmid_origin = []
    osmid_destination = []
    NoneInOrigins = len([i for i in range(len(O_vector)) if O_vector[i] is None])
    NoneInDestinations = len([i for i in range(len(D_vector)) if D_vector[i] is None])
    NoneInOD = len([i for i in range(len(OD_vector)) if OD_vector[i] is None])
    logger.info(f'Number of Origin-Destination: {len(O_vector)}')
    logger.debug(f'NoneInOrigins: {NoneInOrigins} NoneInDestinations: %s NoneInOD: %s',NoneInDestinations,NoneInOD)
    for i in range(len(O_vector)):
        # Index Origin
        origin = O_vector[i]
        # Index Destination
        destination = D_vector[i]
        # Number of people Origin-Destination
        number_people = OD_vector[i]
        # Add to File Just if There are People for The Origin-Destination                        
        if number_people > 0:
            iterations = number_people   
            time_increment = 1/iterations
            # Adequate the key to the grid2OD dictionary
            AvailableTypesKeysGrid2OD = GetAvailableTypesKeys(grid2OD)
            AvailableTypesKeysOsimd2Index = GetAvailableTypesKeys(osmid2index)
            origin = CheckAndConvert2AvailableTypesKeys(origin,AvailableTypesKeysGrid2OD,grid2OD)
            destination = CheckAndConvert2AvailableTypesKeys(destination,AvailableTypesKeysGrid2OD,grid2OD)            
            # Load People 
            for it in range(int(iterations)):
                # Are there Origin and Destination in the Grid?
                if origin in grid2OD.keys():
                    Originbigger0 = len(grid2OD[origin])>0
                else:
                    total_number_people_not_considered += number_people
                    print(f'Couple {origin} not considered as one of the 2 not present in grid2OD')
                    break
                if destination in grid2OD.keys():
                    Destinationbigger0 = len(grid2OD[destination])>0
                else:
                    total_number_people_not_considered += number_people
                    print(f'Couple {origin}-{destination} not considered as one of the 2 not present in grid2OD')
                    break
                # Handle the Case there are Origins and Destinations (belonging to the road network) in the Grid
                if  Originbigger0 and Destinationbigger0:
                    # User Id is the count of times this row is applied
                    users_id.append(count_line)
                    # Time in seconds from start*3600 to end*3600
                    t = start*(seconds_in_minute**2) + it*time_increment*seconds_in_minute**2
                    time_.append(t) 
                    # Randomly Select Origin and Destination and Make sure to Map them into the Road Network
                    i = np.random.randint(0,len(grid2OD[origin]))
                    osmidorigin = CheckAndConvert2AvailableTypesKeys(grid2OD[origin][i],AvailableTypesKeysOsimd2Index,osmid2index)
                    if osmidorigin in osmid2index.keys():
                        origins.append(osmid2index[osmidorigin])
                    else:
                        print(f'{osmidorigin} not considered as one of the 2 not present in osmid2index')
                        total_number_people_not_considered += number_people
                    j = np.random.randint(0,len(grid2OD[destination]))                        
                    osmiddestination = CheckAndConvert2AvailableTypesKeys(grid2OD[destination][j],AvailableTypesKeysOsimd2Index,osmid2index)
                    if osmiddestination in osmid2index.keys():
                        destinations.append(osmid2index[osmiddestination])
                    else:                        
                        print(f'{osmidorigin}-{osmiddestination} not considered as one of the 2 not present in osmid2index')
                        total_number_people_not_considered += number_people
                    osmid_origin.append(grid2OD[origin][i])
                    osmid_destination.append(grid2OD[destination][j])
                    ## FILLING ORIGIN DESTINATION GRID ACCORDING TO THE ORIGIN DESTINATION NODES
                    count_line += 1
                    total_number_people_considered += 1
    df1 = pd.DataFrame({
        'SAMPN':users_id,
        'PERNO':users_id,
        'origin_osmid':osmid_origin,
        'destination_osmid':osmid_destination,
        'dep_time':time_,
        'origin':origins,
        'destination':destinations,
        })

    print('total_number_people_considered: ',total_number_people_considered)
    print('total_number_people_not_considered: ',total_number_people_not_considered)
    print('ratio: ',total_number_people_considered/(total_number_people_considered+total_number_people_not_considered))
    return df1

def ODVectorFromTij(Tij_modified,R):
    """
        Input:
            Tij_modified: (pd.DataFrame) -> Tij_modified = Tij[['origin','destination','number_people']]
        Output:
            O_vector: (np.array 1D) -> origin subset with repetition [0,...,Ngridx*Ngridy-1]
            D_vector: (np.array 1D) -> destination subset with repetition [0,...,Ngridx*Ngridy-1]
            OD_vector: (np.array 1D) -> number of people from Tij['number_people'].to_numpy()
    """
    logger.info("Convert Tij to OD_vector...")
    # NOTE: ADD HERE THE POSSIBILITY OF HAVING OD FROM POTENTIAL CONSIDERATIONS
    O_vector = np.array(Tij_modified['origin'],dtype=int)
    D_vector = np.array(Tij_modified['destination'],dtype=int)
    OD_vector = np.array(Tij_modified['number_people'],dtype=int)
    TotPeople = np.sum(OD_vector)
    WantedPeople = R*3600
    Multiply = WantedPeople/TotPeople
    OD_vector = np.array([int(OD_vector[i]*Multiply) for i in range(len(OD_vector))])
    return O_vector,D_vector,OD_vector

def GenerateDfFluxesFromTij(Tij_modified,osmid2index,grid2OD,start,NOffset,R):
    """
        Input:
            Tij_modified: (pd.DataFrame) -> Tij_modified = Tij[['origin','destination','number_people']]
            osmid2index: (dict) -> osmid2index = {osmid:i}
            grid2OD: (dict) -> grid2OD = {(i,j):OD}
            start: (int) -> start time
        Output:
            df1: (pd.DataFrame) -> df1 = pd.DataFrame({
                'SAMPN':users_id,
                'PERNO':users_id,
                'origin_osmid':osmid_origin,
                'destination_osmid':osmid_destination,
                'dep_time':time_,
                'origin':origins,
                'destination':destinations,
                })
    """
    O_vector,D_vector,OD_vector = ODVectorFromTij(Tij_modified,R)
    df1 = GenerateInfoInputSimulationFromGridOD(O_vector,D_vector,OD_vector,osmid2index,grid2OD,start,NOffset,60)
    return df1
