from termcolor import cprint
import os
import numpy as np
import pandas as pd
from ODfromfma import CityName2RminRmax

def CumulativeODFromGrid(O_vector,D_vector,OD_vector,osmid2index,grid2OD,start,multiplicative_factor,,seconds_in_minute = 60):
    '''
        Input:
            O_vector: (np.array 1D) -> origin subset with repetition [0,...,Ngridx*Ngridy-1]
            D_vector: (np.array 1D) -> destination subset with repetition [0,...,Ngridx*Ngridy-1]
            OD_vector: (np.array 1D) -> number of people from Tij['number_people'].to_numpy() 
            osmid2index: (dict) -> osmid2index = {osmid:i} 
            grid2OD: (dict) -> grid2OD = {(i,j):OD}
            start: (int) -> start time
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
    for i in range(len(O_vector)):
        origin = O_vector[i]
        destination = D_vector[i]
        number_people = OD_vector[i]
        bin_width = 1                        
        if number_people > 0:
            iterations = multiplicative_factor*number_people/bin_width   
            time_increment = 1/iterations
            for it in range(int(iterations)):
                try:
                    Originbigger0 = len(grid2OD[origin])>0
                except KeyError:
                    total_number_people_not_considered += number_people
                    break
                try:
                    Destinationbigger0 = len(grid2OD[destination])>0
                except KeyError:
                    total_number_people_not_considered += number_people
                    break
                if  Originbigger0 and Destinationbigger0:
                    users_id.append(count_line)
                    t = start*(seconds_in_minute**2) + it*time_increment*seconds_in_minute**2
                    time_.append(t) # TIME IN HOURS
                    i = np.random.randint(0,len(grid2OD[origin]))
                    try:
                        origins.append(osmid2index[grid2OD[origin][i]])
                    except KeyError:
                        total_number_people_not_considered += number_people
                        raise KeyError('KeyError Polygon 2 OD: origin {0} i {1}'.format(origin,i))
                    j = np.random.randint(0,len(grid2OD[destination]))                        
                    try:
                        destinations.append(osmid2index[grid2OD[destination][j]])
                    except KeyError:
                        total_number_people_not_considered += number_people
                        raise KeyError('KeyError Polygon 2 OD: destination {0} j {1}'.format(origin,i))
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
    ROutput = []
    # NOTE: ADD HERE THE POSSIBILITY OF HAVING OD FROM POTENTIAL CONSIDERATIONS
    O_vector = Tij_modified['origin']
    D_vector = Tij_modified['destination']
    OD_vector = Tij_modified['number_people']
    # START TAB
    R = np.sum(OD_vector)/3600 # R is the number of people that move in one second (that is the time interval for the evolution )
    Rmin = CityName2RminRmax[NameCity][0]
    Rmax = CityName2RminRmax[NameCity][1]
    spacing = (Rmax/R - Rmin/R)/20
    for multiplicative_factor in np.arange(Rmin/R,Rmax/R,spacing):
        R = np.sum(OD_vector)/3600 
        if os.path.isfile(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(NameCity,start,end,int(multiplicative_factor*R),UCI))):
            cprint(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(NameCity,start,end,int(multiplicative_factor*R),UCI)),'cyan')
            ROutput.append(int(multiplicative_factor*R))
            continue
        else:
            cprint('COMPUTING {}'.format(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(NameCity,start,end,int(multiplicative_factor*R),UCI))),'cyan')
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
            df1.to_csv(os.path.join(save_dir_local,'OD','{0}_oddemand_{1}_{2}_R_{3}_UCI_{4}.csv'.format(NameCity,start,end,int(R),UCI)))
    
