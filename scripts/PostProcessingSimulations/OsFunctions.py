import os
import pandas as pd
import logging
logger = logging.getLogger(__name__)
LPSIM_DIR = '/home/alberto/LPSim' 
HOME_DIR = '/home/alberto/LPSim/LivingCity'

def JoinDir(BaseDir,TreeDir):
    return os.path.join(BaseDir,TreeDir)
def IsFile(File):
    return os.path.isfile(File)

def MakeDirs(Dir):
    os.makedirs(Dir,exist_ok=True)
    return Dir


def DeleteFile(dir_file):
    """
        @params dir_file: File to delete
        @description: Delete the file in input
    """
    if os.path.exists(dir_file):
        os.remove(dir_file)
    else:
        pass

def RenameMoveOutput(output_file,CityName,R,UCI,verbose = False):
    """
        @params output_file: File to move
        @params CityName: Name of the city
        @params R: Number of people per unit time
        @params UCI: Urban Centrality Index
        @description: Move the output file in the correct
    """
    saving_dir = os.path.join(HOME_DIR,'berkeley_2018',CityName,'Output')
    source_file = os.path.join(LPSIM_DIR, output_file)
    destination_file = os.path.join(saving_dir, f"R_{R}_UCI_{round(UCI,3)}_{output_file}")
    logger.info(f"Transfer: {output_file} -> {destination_file}")
    if not os.path.exists(saving_dir):
        # Make Sure saving_dir exists
        os.mkdir(saving_dir)
    if os.path.exists(source_file):
        os.rename(source_file, destination_file)
    else:
        logger.info(f"File {source_file} does not exist")
    Csv2Parquet(destination_file,destination_file.replace('.csv','.parquet'))
#     shutil.move(os.path.join(LPSIM_DIR,f"R_{R}_UCI_{round(UCI,3)}_{output_file}"),saving_dir)  

def Csv2Parquet(file_csv,file_parquet):
    """
        @params file_csv: File to convert
        @params file_parquet: File to convert
        @description: Convert the csv file in parquet
    """
    # Read
    df = pd.read_csv(file_csv)
    # Convert
    df.to_parquet(file_parquet, engine='pyarrow')
    # Delete Not Compressed File
    DeleteFile(file_csv)