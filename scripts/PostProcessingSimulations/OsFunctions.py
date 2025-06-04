import os
import pandas as pd
import logging
import shutil
from polars import read_parquet, read_csv, Utf8,col

logger = logging.getLogger(__name__)
# CONSTANT DIRECTORIES
LPSIM_DIR = os.environ["LPSim"]
TRAFFIC_DIR = os.environ["TRAFFIC_DIR"] 
LIVING_CITY_DIR = os.path.join(LPSIM_DIR,"LivingCity")
BERKELEY_DIR = os.path.join(LIVING_CITY_DIR,"berkeley_2018")
NEW_FULL_NETWORK_DIR = os.path.join(BERKELEY_DIR,"new_full_network")
CARTO_DIR = os.path.join(TRAFFIC_DIR,"data","carto")

def JoinDir(BaseDir,TreeDir):
    return os.path.join(BaseDir,TreeDir)
def IsFile(File):
    return os.path.isfile(File)

def MakeDirs(Dir):
    os.makedirs(Dir,exist_ok=True)
    return Dir

## FILE MANAGEMENT


def DeleteFile(dir_file):
    """
        @params dir_file: File to delete
        @description: Delete the file in input
    """
    if os.path.exists(dir_file):
        os.remove(dir_file)
    else:
        pass


def CopyCartoGraphy(SourceFile,CityName):
    """
        @params SourceFile: File to copy
        @params DestDir: Directory where to copy the file
        @description : Copy the cartography file in the directory of the simulation
    """
    # Input
    SourceDir = os.path.join(CARTO_DIR,CityName)
    SourceFileComplete = os.path.join(SourceDir,SourceFile)
    # Output

    if os.path.exists(SourceFileComplete):
        logger.info(f"Copying {SourceFileComplete} to {NEW_FULL_NETWORK_DIR}")
        shutil.copy(SourceFileComplete,NEW_FULL_NETWORK_DIR)
    else:
        raise Exception(f"CopyCartoGraphy: Cannot copy {SourceFileComplete} : It does not exist")


def RenameMoveOutput(source_file,
                     destination_file,
                     destination_file_parquet):
    """
        @params source_file: Copmlete path of the file to move -> /LPSim/source_file.csv
        @params output_file: Complete path of the output file -> /data_dir/output_file.csv
        @description: Move the output file in the correct
    """
    logger.info(f"Transfer: {source_file} -> {destination_file}")
    if os.path.exists(source_file):
        shutil.move(source_file, destination_file)
    else:
        logger.info(f"File {source_file} does not exist")
    case = "route" if "route" in destination_file_parquet else "default"
    Csv2Parquet(destination_file,destination_file_parquet,case)
#     shutil.move(os.path.join(LPSIM_DIR,f"R_{R}_UCI_{round(UCI,3)}_{output_file}"),saving_dir)  

def Csv2Parquet(file_csv,file_parquet,case):
    """
        @params file_csv: File to convert
        @params file_parquet: File to convert
        @description: Convert the csv file in parquet
    """
    if os.path.exists(file_csv):
        logger.info(f"Csv2Parquet: Converting {file_csv} -> {file_parquet}")
        if case == "route":
            df = pd.read_csv(file_csv,delimiter=':')
        else:
            # Read
            df = pd.read_csv(file_csv)
        # Convert
        df.to_parquet(file_parquet, engine='pyarrow', compression='snappy')
        # Delete Not Compressed File
        DeleteFile(file_csv)
    else:
        logger.info(f"Csv2Parquet: File {file_csv} does not exist")


    # ------ POSTPROCESSING -> read Routes and People info per simulation --------- #
def GetRouteInfo(iter_2_DfRoute,
                 RouteFiles,
                 city,
                 iter_2_is_available_locally_DfRoute,
                 iters):
    """
        @params iter_2_DfRoute: Iterator of DataFrame with the route
        @params RouteFiles: List of files to read
        @params city: City name
        @params iter_2_is_available_locally_DfRoute: Iterator of DataFrame with the route
        @params iters: Iterator of DataFrame with the route
        @description: Get the route information from the files
        @return: iter_2_DfRoute: Iterator of DataFrame with the route
        @return: iter_2_is_available_locally_DfRoute: Iterator of DataFrame with the route
        @return: ReadRouteInfoBool: Boolean to check if the route information was read
    """
    count_route_file = 0
    for RouteFile in RouteFiles:
        iter_ = iters[count_route_file]
        if IsFile(RouteFile):
            if ".parquet" in RouteFile:
                iter_2_DfRoute[iter_] = read_parquet(RouteFile)
            elif ".csv" in RouteFile:
                iter_2_DfRoute[iter_] = read_csv(RouteFile,separator = ':')
                if iter_2_DfRoute[iter_].schema["route"] == Utf8:
                    pass
            iter_2_is_available_locally_DfRoute[iter_] = True
            ReadRouteInfoBool = True
            logger.info(f"{city}: GetRouteInfo: {RouteFile} was read -> updated iter_2_DfRoute")
        else:
            logger.error(f"{city}: GetRouteInfo: {RouteFile} was not read -> updated iter_2_DfRoute")
            iter_2_DfRoute[iter_] = None
            iter_2_is_available_locally_DfRoute[iter_] = False
        count_route_file += 1
    return iter_2_DfRoute, iter_2_is_available_locally_DfRoute, ReadRouteInfoBool

def GetPeopleInfo(iter_2_DfPeople,
                  iter_2_is_available_locally_DfPeople,
                  PeopleFiles,
                  iters,
                  city,
                  t_start_control_group):
    """
        @params iter_2_DfPeople: Iterator of DataFrame with the people
        @params iter_2_is_available_locally_DfPeople: Iterator of DataFrame with the people
        @params PeopleFiles: List of files to read
        @params iters: Iterator of DataFrame with the people
        @params city: City name
        @description: Get the people information from the files
        @return: iter_2_DfPeople: Iterator of DataFrame with the people
        @return: iter_2_is_available_locally_DfPeople: Iterator of DataFrame with the people
    """
    count_people_file = 0
    for PeopleFile in PeopleFiles:
        iter_ = iters[count_people_file]
        if IsFile(PeopleFile):
            if ".parquet" in PeopleFile:
                iter_2_DfPeople[iter_] = read_parquet(PeopleFile)
            elif ".csv" in PeopleFile:
                iter_2_DfPeople[iter_] = read_csv(PeopleFile)
            logger.info(f"{city}: GetPeopleInfo: {PeopleFile} was read, updated iter_2_DfPeople")
            iter_2_DfPeople[iter_] = iter_2_DfPeople[iter_].filter(col("time_departure") >= t_start_control_group)
            iter_2_is_available_locally_DfPeople[iter_] = True
        else:
            logger.error(f"{city}: GetPeopleInfo: {PeopleFile} was not read, updated iter_2_DfPeople")
            iter_2_DfPeople[iter_] = None
            iter_2_is_available_locally_DfPeople[iter_] = False
        count_people_file += 1
    return iter_2_DfPeople, iter_2_is_available_locally_DfPeople
