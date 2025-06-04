from TrajectoryAnalysis import *
from OutputStats import *
from GenerateConfiguraiton import *
import warnings
import multiprocessing
warnings.filterwarnings("ignore")
BaseConfig = os.path.join(os.environ["TRAFFIC_DIR"],"config")
list_cities = os.listdir(os.path.join(os.environ["TRAFFIC_DIR"],'data','carto'))
list_cities = ["BOS"]#["BOS","SFO","RIO"]
ListPeopleFile = []
ListRoutesFile = []
complete_path_info_simulations_df = os.path.join(TRAFFIC_DIR,"info_simulations.csv")

def MainMultiprocessing(City):
    """
        @City: str
        Description:
            - For each city, load the configuration file
            - For each UCI, load the output file
            - For each R, load the output file
            - Compute the average unload curve
            - Compute the average volume traffic
            - Compute the average speed
    """

#        City2Config,Rs,UCIs = InitConfigPolycentrismAnalysis(City)  
    Df_Info_Simulations = pl.read_csv(complete_path_info_simulations_df)
    Rs = Df_Info_Simulations.filter(pl.col("city") == City)["R"].unique().to_numpy()
    UCIs = Df_Info_Simulations.filter(pl.col("city") == City)["UCI"].unique().to_numpy()
#    UCIs = [UCI for UCI in UCIs if UCI > 0.9]
    print("Rs\n",Rs)
    print("UCIs\n",UCIs)
    Df_City2Config = Df_Info_Simulations.filter(pl.col("city") == City)    
    # 10) NOTE: Analyze the Output
    PCTA = Polycentrism2TrafficAnalyzer(Df_City2Config,Rs,UCIs)  
    PCTA.CompleteAnalysis()

if False:
    with multiprocessing.Pool(3) as p:
        p.map(MainMultiprocessing, list_cities)
City = list_cities[0]
for City in list_cities:
    print("City:", City)
    MainMultiprocessing(City)