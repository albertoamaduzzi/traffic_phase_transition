import os
import pandas as pd
import logging
import shutil

logger = logging.getLogger(__name__)
# CONSTANT DIRECTORIES
LPSIM_DIR = os.environ["LPSim"]
TRAFFIC_DIR = os.environ["TRAFFIC_DIR"] 
LIVING_CITY_DIR = os.path.join(LPSIM_DIR,"LivingCity")
BERKELEY_DIR = os.path.join(LIVING_CITY_DIR,"berkeley_2018")
NEW_FULL_NETWORK_DIR = os.path.join(BERKELEY_DIR,"new_full_network")


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
    SourceDir = os.path.join(BERKELEY_DIR,CityName)
    SourceFileComplete = os.path.join(SourceDir,SourceFile)
    # Output

    if os.path.exists(SourceFileComplete):
        logger.info(f"Copying {SourceFileComplete} to {NEW_FULL_NETWORK_DIR}")
        shutil.copy(SourceFile,NEW_FULL_NETWORK_DIR)
    else:
        raise Exception(f"CopyCartoGraphy: Cannot copy {SourceFileComplete} : It does not exist")


def RenameMoveOutput(output_file,CityName,R,UCI,verbose = False):
    """
        @params output_file: File to move
        @params CityName: Name of the city
        @params R: Number of people per unit time
        @params UCI: Urban Centrality Index
        @description: Move the output file in the correct
    """
    saving_dir = os.path.join(LIVING_CITY_DIR,'berkeley_2018',CityName,'Output')
#    name_parquet = f"R_{0}_UCI_{1}_{2}".format(R,round(UCI,3),output_file.split('.csv')[0] + '.parquet')
    name_parquet = f"R_{R}_UCI_{round(UCI, 3)}_{output_file.replace('.csv', '.parquet')}"
    # Complete Path
    source_file = os.path.join(LPSIM_DIR, output_file)
    destination_file = os.path.join(saving_dir, f"R_{R}_UCI_{round(UCI,3)}_{output_file}")
    destination_file_parquet = os.path.join(saving_dir, name_parquet)
    logger.info(f"Transfer: {source_file} -> {destination_file}")
    if not os.path.exists(saving_dir):
        # Make Sure saving_dir exists
        os.mkdir(saving_dir)
    if os.path.exists(source_file):
        os.rename(source_file, destination_file)
    else:
        logger.info(f"File {source_file} does not exist")
    case = "route" if "route" in name_parquet else "default"
    Csv2Parquet(destination_file,destination_file_parquet,case)
#     shutil.move(os.path.join(LPSIM_DIR,f"R_{R}_UCI_{round(UCI,3)}_{output_file}"),saving_dir)  

def Csv2Parquet(file_csv,file_parquet,case):
    """
        @params file_csv: File to convert
        @params file_parquet: File to convert
        @description: Convert the csv file in parquet
    """
    logger.info(f"Converting {file_csv} -> {file_parquet}")
    if case == "route":
        df = pd.read_csv(file_csv,delimiter=':')
    else:
        # Read
        df = pd.read_csv(file_csv)
    # Convert
    df.to_parquet(file_parquet, engine='pyarrow', compression='snappy')
    # Delete Not Compressed File
    DeleteFile(file_csv)