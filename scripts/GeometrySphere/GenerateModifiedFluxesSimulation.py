from termcolor import cprint
import os
import numpy as np
import pandas as pd
from ODfromfma import CityName2RminRmax

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

def CumulativeODFromGrid(O_vector,D_vector,OD_vector,osmid2index,grid2OD,start,multiplicative_factor,seconds_in_minute = 60):
    '''
        Input:
            O_vector: (np.array 1D) -> origin subset with repetition [0,...,Ngridx*Ngridy-1]
            D_vector: (np.array 1D) -> destination subset with repetition [0,...,Ngridx*Ngridy-1]
            OD_vector: (np.array 1D) -> number of people from Tij['number_people'].to_numpy() 
            osmid2index: (dict) -> osmid2index = {osmid:i} 
            grid2OD: (dict) -> grid2OD = {(i,j):OD}
            start: (int) -> start time

        NOTE: grid2OD,keys() in [0,...,Ngridx*Ngridy-1] <- Stored like Strings
        NOTE: grid2OD,values() in [osmid0,...,osmidN-1]
    '''
    assert len(O_vector) == len(D_vector) == len(OD_vector), 'Lengths of O_vector, D_vector, and OD_vector must be the same'
    assert osmid2index is not None, 'osmid2index must be provided'
    assert grid2OD is not None, 'grid2OD must be provided'
    assert start is not None, 'start must be provided'
    assert multiplicative_factor is not None, 'multiplicative_factor must be provided'    
    total_number_people_considered = 0
    total_number_people_not_considered = 0
    count_line = 0
    users_id = []
    time_ = []
    origins = []
    destinations = []
    osmid_origin = []
    osmid_destination = []
    print('number of couples of origin-destination: ',len(O_vector))
    NoneInOrigins = len([i for i in range(len(O_vector)) if O_vector[i] is None])
    NoneInDestinations = len([i for i in range(len(D_vector)) if D_vector[i] is None])
    NoneInOD = len([i for i in range(len(OD_vector)) if OD_vector[i] is None])
    print('NoneInOrigins: ',NoneInOrigins," NoneInDestinations: ",NoneInDestinations," NoneInOD: ",NoneInOD)
    for i in range(len(O_vector)):
        # Index Origin
        origin = O_vector[i]
        # Index Destination
        destination = D_vector[i]
        # Number of people Origin-Destination
        number_people = OD_vector[i]
        bin_width = 1
        # Add to File Just if There are People for The Origin-Destination                        
        if number_people > 0:
            iterations = multiplicative_factor*number_people/bin_width   
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
    print('total_number_people_considered: ',total_number_people_considered)
    print('total_number_people_not_considered: ',total_number_people_not_considered)
    print('ratio: ',total_number_people_considered/(total_number_people_considered+total_number_people_not_considered))
    return users_id,time_,origins,destinations,osmid_origin,osmid_destination

def OD_from_T_Modified(Tij_modified,
                       CityName2RminRmax,
                       NameCity,
                       osmid2index,
                       grid2OD,
                       p,
                       save_dir_local,
                       start = 7,
                       end = 8,
                       UCI = None
                       ):
    # Set the number of OD files to generate
    NumberOfConfigurationsR = 20
    ROutput = []
    # NOTE: ADD HERE THE POSSIBILITY OF HAVING OD FROM POTENTIAL CONSIDERATIONS
    O_vector = np.array(Tij_modified['origin'],dtype=int)
    D_vector = np.array(Tij_modified['destination'],dtype=int)
    OD_vector = np.array(Tij_modified['number_people'],dtype=int)
    # START TAB
    R = np.sum(OD_vector)/3600 # R is the number of people that move in one second (that is the time interval for the evolution )
    Rmin = CityName2RminRmax[NameCity][0]
    Rmax = CityName2RminRmax[NameCity][1]
    spacing = (Rmax/R - Rmin/R)/NumberOfConfigurationsR
    print("Generation Fluxes UCI: ",UCI)
    for multiplicative_factor in np.arange(Rmin/R,Rmax/R,spacing):
        R = np.sum(OD_vector)/3600 
#        if os.path.isfile(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(NameCity,start,end,int(multiplicative_factor*R),UCI))):
        if os.path.isfile(os.path.join(save_dir_local,'{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(NameCity,start,end,int(multiplicative_factor*R),UCI))):
#            cprint(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(NameCity,start,end,int(multiplicative_factor*R),UCI)),'cyan')
            cprint(os.path.join(save_dir_local,'{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(NameCity,start,end,int(multiplicative_factor*R),UCI)),'cyan')
            ROutput.append(int(multiplicative_factor*R))
            continue
        else:
#            cprint('COMPUTING {}'.format(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(NameCity,start,end,int(multiplicative_factor*R),UCI))),'cyan')
            cprint('COMPUTING {}'.format(os.path.join(save_dir_local,'{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(NameCity,start,end,int(multiplicative_factor*R),UCI))),'cyan')
            users_id,time_,origins,destinations,osmid_origin,osmid_destination = CumulativeODFromGrid(O_vector,D_vector,OD_vector,osmid2index,grid2OD,start,multiplicative_factor,60)
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
#            df1.to_csv(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(NameCity,start,end,int(R),UCI)))
            df1.to_csv(os.path.join(save_dir_local,'{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(NameCity,start,end,int(R),UCI)))
    
