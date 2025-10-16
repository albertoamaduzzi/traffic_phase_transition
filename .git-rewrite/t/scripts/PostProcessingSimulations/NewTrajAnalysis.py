"""
    This script contains the analysis of the simulations launched with the LivingCity program.
    The configuration file is for each city and contains the following information:
        - name: name of the city
        - delta_t: time step (in seconds)
        - dir_output_simulation_analysis: directory where the simulation output is stored for the analysis
        - graphml_file: file containing the graph of the city
        - UCI: {
                R: {
                    "route_file": "/home/alberto/LPSim/LivingCity/berkeley_2018/BOS/Output/R_152_UCI_0.666_0_route7to24.csv",
                    "start_time": "7",
                    "end_time": "24",
                    "people_file": "/home/alberto/LPSim/LivingCity/berkeley_2018/BOS/Output/R_152_UCI_0.666_0_people7to24.csv"
                },

"""
import pandas as pd
import os
import matplotlib.pyplot as plt 
import numpy as np
import polars as pl
from collections import defaultdict
from GeoJsonFunctions import *
from OutputStats import *
from PlotsPercolation import *
import json
from Beta import *
import sys
sys.path.append(os.path.join(os.environ["TRAFFIC_DIR"],"scripts","GeometrySphere"))
from ODfromfma import UCI2CityName2RminRmax
BASE_PATH = os.path.join(os.environ["LPSim"],'LivingCity/berkeley_2018/BOS/Output/')
from dictionaries_percolation_analysis import *



def ChooseKeyFromUCI2CityName2RminRmax(UCI,UCI2CityName2RminRmax):
    """
        @UCI: float -> UCI
        @CityName: str -> City Name
        @UCI2CityName2RminRmax: dict -> {UCI:{CityName:{Rmin:Rmax}}}
        @description: Choose the key from the dictionary
    """
    UCI_keys = list(UCI2CityName2RminRmax.keys())
    last_key = UCI_keys[-1]
    first_key = UCI_keys[0]
    for i in range(len(UCI_keys)-1):
        UCI_key = UCI_keys[i]
        if UCI_key != last_key and UCI_key != first_key:
            if UCI >= UCI_key and UCI < UCI_keys[i+1]:
                return UCI_key
        elif UCI_key == first_key:
            if UCI < UCI_key:
                return UCI_key
        else:
            if UCI >= UCI_key:
                return UCI_key
    
def Cut_Rs_from_ODfromfma(Rs,UCI,CityName,UCI2CityName2RminRmax):
    """
        @Rs: list -> List of Rs
        @UCI: float -> UCI
        @CityName: str -> City Name
        @UCI2CityName2RminRmax: dict -> {UCI:{CityName:{Rmin:Rmax}}}
        @description: Cut the Rs that are not present in the ODfromfma
    """
    UCI_key = ChooseKeyFromUCI2CityName2RminRmax(UCI,UCI2CityName2RminRmax)
    Rmin = UCI2CityName2RminRmax[UCI_key][CityName][0]
    Rmax = UCI2CityName2RminRmax[UCI_key][CityName][1]
    Rs = [R for R in Rs if R >= Rmin and R <= Rmax]
    return Rs

class Polycentrism2TrafficAnalyzer:
    def __init__(self,Df_City2Config,Rs,UCIs):
        assert isinstance(Df_City2Config, pl.DataFrame), f"Df_City2Config must be a polars DataFrame, instead {type(Df_City2Config)}"
        assert isinstance(Rs,list) or isinstance(Rs,tuple) or isinstance(Rs,np.ndarray), f"Rs must be a list, instead {type(Rs)}"
        assert isinstance(UCIs,list) or isinstance(UCIs,tuple) or isinstance(UCIs,np.ndarray), f"UCIs must be a list, instead {type(UCIs)}"
        assert len(Rs) > 0, f"Rs must be a list of length > 0, instead {len(Rs)}"
        assert len(UCIs) > 0, f"UCIs must be a list of length > 0, instead {len(UCIs)}"
        assert "name" in Df_City2Config.columns, f"Df_City2Config must contain a column 'name', instead {Df_City2Config.columns}"
        assert "delta_t" in Df_City2Config.columns, f"Df_City2Config must contain a column 'delta_t', instead {Df_City2Config.columns}"
        assert "dir_output_simulation_analysis" in Df_City2Config.columns, f"Df_City2Config must contain a column 'dir_output_simulation_analysis', instead {Df_City2Config.columns}"
        assert "complete_path_graphml" in Df_City2Config.columns, f"Df_City2Config must contain a column 'complete_path_graphml', instead {Df_City2Config.columns}"
        self.Df_config = Df_City2Config
        # Unique variables
        self.Name = Df_City2Config['name'].unique().to_numpy()[0]
        self.delta_t = Df_City2Config['delta_t'].unique().to_numpy()[0]
        self.BaseDir = os.path.dirname(Df_City2Config['dir_output_simulation_analysis'].unique().to_numpy()[0])
        self.GraphmlFile = Df_City2Config['complete_path_graphml'].unique().to_numpy()[0]
        self.R2UCI2OutputStats = defaultdict()
        # Phase Transition
        self.UCI2R2Time2PeopleInNet = defaultdict()
        self.UCI2R2ErrorFit = defaultdict()
        self.UCI2R2Tau = defaultdict()
        self.UCI2NR = defaultdict()
        # FLAGS
        self.GetGeoJsonBool = False
        self.GetGeoJson()
        self.Nodes = pl.read_csv(os.path.join(BERKELEY_DIR,self.Name,'nodes.csv'),dtypes={'ref':pl.Utf8})
        self.Edges = pl.read_csv(os.path.join(BERKELEY_DIR,self.Name,'edges.csv'))
        self.Rs = Rs
        self.UCIs = UCIs
        # Aggregated Variables
        # One per (UCI,R)
        self.Taus = {round(UCI,3):[] for UCI in UCIs}
        self.Gammas = {round(UCI,3):[] for UCI in UCIs}
        self.UsedRs = {round(UCI,3):[] for UCI in UCIs}
        # One per UCI
        self.UCI2CriticalParams = {round(UCI,3):defaultdict() for UCI in UCIs}
        self.Rcs = {round(UCI,3):[] for UCI in UCIs}
        self.Alphas = {round(UCI,3):[] for UCI in UCIs}
        # Range Fit
        self.RangeFit = None
        # Time For Plots and Fit Beta
        self.t_vect = None
        # 
        self.UCI2PhaseTransition = {round(UCI,3):{"Beta":None,"best_A_fit_Beta":None,"time_best_fit":None,"int_time_best_fit":None} for UCI in UCIs}
        self.City2Color = {"BOS":"red","LIS":"blue","LAX":"green","SFO":"orange","RIO":"purple"}
        self.City2Marker = {"BOS":"o","LIS":"s","LAX":"^","SFO":"D","RIO":"P"}
        # Look at how the error of the fit of n_R(t) changes in R
        self.list_err_exp_nt_R = []
        self.list_err_pl_nt_R = []
        # Scalar informations contained for each R
        self.list_column_scalar_info_for_each_R = ["R","gamma","tau","error_pl_nRt","error_exp_nRt","is_Rc"]                                             # NOTE: useful to plot things related to the R
        self.dict_scalar_info_for_each_R = {key:[] for key in self.list_column_scalar_info_for_each_R} # NOTE: useful to plot things related to the R

    def CompleteAnalysis(self):
        """
            Description:
                - For each UCI:
                    - For each R:
                        - DfUnload: pl.DataFrame -> Columns: 
                            - FractionPeople: np.array (Fraction of people in the network)
                            - Time_hours: np.array (Time in hours from t_start_hour)
                            - Time_hours_shifted: np.array (Time in hours after the unloading period starts) 
                            - IsPeople: np.array (True if people are in network, used as mask for fit)
                            - IsUnload: np.array (True if people are unloading, used to make the y,and y fit coincide at the plotting stage)
                            - NumberPeople: np.array (Number of people, not fraction)
                            - NumberPeopleFit: np.array (Number of people fitted)
                            - Time: np.array (Time in seconds)
                            - y_fit_exp: np.array (Fitted exponential)
                            - y_fit_pl: np.array (Fitted powerlaw)
                            - log_y: np.array (log of the fraction of people)
                            - log_x: np.array (log of the time in hours)
                        - InfoTraffic: dict ->
        """

        # Treat each UCI independently
        logger.info(f"Working on: {self.BaseDir}")
        self.Rs = [int(R) for R in self.Rs]
        self.UCI2R2InfoTraffic = {round(UCI,3):{R: defaultdict() for R in self.Rs} for UCI in self.UCIs}                        # NOTE: Information about the traffic 
        self.UCI2AlphaBeta = {round(UCI,3):{"alpha":[],"beta":[]} for UCI in self.UCIs}                                         # NOTE: information about the transition. (figure 5 paper marta)
        # NOTE: Info About 
        if os.path.exists(os.path.join(os.environ["TRAFFIC_DIR"],"data","info_DP_simulations.json")):
            with open(os.path.join(os.environ["TRAFFIC_DIR"],"data","info_DP_simulations.json"), 'r') as f:
                self.City2UCI2ImportantFeaturesForComparisonCities = json.load(f)
        else:
            self.City2UCI2ImportantFeaturesForComparisonCities = init_City2UCI2ImportantFeaturesForComparisonCities([self.Name],self.UCIs)
        for UCI in self.UCIs:  
            # Initialize the final comparison dictionary  
            UCI_plot_dir = os.path.join(self.BaseDir,str(UCI),"Plots")
            print(f"Before filter UCI: {UCI}, len(Df_config): {len(self.Df_config)}")
            Df_config_UCI = self.Df_config.filter(pl.col("UCI") == UCI)
            FoundRc = False
            # NOTE: Put a cut on R according to some considerations or not
            Rs = self.Rs
#            Rs = Cut_Rs_from_ODfromfma(self.Rs,UCI,self.Name,UCI2CityName2RminRmax)
            R2NtNtFit = {R:defaultdict() for R in sorted(Rs)}
            fig,ax = plt.subplots(1,1,figsize = (10,10))
            # Compute The Gamma Beta Tau for each UCI for which I did not compute it already
#            if not os.path.exists(os.path.join(self.BaseDir,f'{self.Name}_{round(UCI,3)}_GammaTau.parquet')):
            for R in sorted(Rs):
#                print(Df_config_UCI["R"].to_numpy())
#                print(f"UCIs in DfConfig: {Df_config_UCI['UCI'].unique().to_numpy()}")
                Df_config_UCI_R = Df_config_UCI.filter(pl.col("R") == int(R),
                                                  )
#                print(f"R: {R}, UCI: {UCI}, len(Df_config_UCI_R): {len(Df_config_UCI_R)}")
                if len(Df_config_UCI_R) != 0:
                    # Make The Analysis For The sinle Couple Of UCI and R
                    OS = OutputStats(R,
                                    UCI,
                                    Df_config_UCI_R,
                                    self.GeoJsonEdges,
                                    self.Nodes,
                                    self.Edges)
                    logger.info(f"{self.Name}: Completed Single Analysis: UCI: {UCI} and R: {R}")
                    OS.CompleteAnalysis()
                    # NOTE: To look for the best Beta
                    self.DfUnload = OS.DfUnload
                    self.InfoTraffic = OS.InfoTraffic
                    ax.plot(self.DfUnload["Time_hours_shifted"],self.DfUnload["FractionPeople"],label = f"R: {R}")
                    ax.set_xlabel("Time (h)")
                    ax.set_ylabel("Fraction of People")
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.legend()
                    # Collect all the needed informations required for DP analysis
                    self.list_err_exp_nt_R.append(OS.err_exp_nt_R)
                    self.list_err_pl_nt_R.append(OS.err_pl_nt_R)
                    #
                    self.UCI2R2InfoTraffic[round(UCI,3)][R] = self.InfoTraffic
                    # Evidence for Traffic Changing R
                    self.Taus[round(UCI,3)].append(float(self.InfoTraffic["Tau"]))
                    self.Gammas[round(UCI,3)].append(float(self.InfoTraffic["Gamma"]))
                    self.UsedRs[round(UCI,3)].append(R)
                    # NOTE: Each of these is a list, take the smaller to 
                    if self.InfoTraffic["IsJam"]:
                        self.Rcs[round(UCI,3)].append(R)
                        self.Alphas[round(UCI,3)].append(self.InfoTraffic["alpha_pl"])
                        if not FoundRc:
                            FoundRc = True
                            self.UCI2CriticalParams[round(UCI,3)] = compute_UCI2CriticalParams(R,self.InfoTraffic)
                            logger.info(f"{self.Name}: Found Rc: {R} for UCI: {UCI}, alpha: {self.InfoTraffic['alpha_pl']}, tauc: {self.InfoTraffic['Tau']}, gammac: {self.InfoTraffic['Gamma']}")
                            with open(os.path.join(UCI_plot_dir,f"CriticalParams_UCI_{round(UCI,3)}.json"),'w') as f:
                                json.dump(self.UCI2CriticalParams,f,indent = 4)
                    # OS.Time2nt contains both n,n_fit, for different times.
                    # We need to find the t that best gives a fit to beta.
                    R2NtNtFit[R]["n"] = self.DfUnload["FractionPeople"].to_numpy()[:-1]
    #                        R2NtNtFit[R]["t"] = OS.InfoTraffic["Time_hours_shifted"].to_numpy()
                    if self.t_vect is None:
                        self.t_vect = self.DfUnload["Time_hours_shifted"].to_numpy()[:-1]
                    if OS.InfoTraffic["IsJam"]:
                        y_fit = self.DfUnload["y_fit_pl"].to_numpy()
                    else:
                        y_fit = self.DfUnload["y_fit_exp"].to_numpy()
                    R2NtNtFit[R]["n_fit"] = y_fit[:-1]
                    self.dict_scalar_info_for_each_R["R"].append(R)
                    self.dict_scalar_info_for_each_R["gamma"].append(float(self.InfoTraffic["Gamma"]))
                    self.dict_scalar_info_for_each_R["tau"].append(float(self.InfoTraffic["Tau"]))
                    self.dict_scalar_info_for_each_R["error_pl_nRt"].append(float(OS.err_pl_nt_R))
                    self.dict_scalar_info_for_each_R["error_exp_nRt"].append(float(OS.err_exp_nt_R))
                    self.dict_scalar_info_for_each_R["is_Rc"].append(True if FoundRc else False)
                    del OS
                # Do not make computations for Rs that are not present 
                # NOTE: There are different Rs for different UCIs and are not present in all the UCIs
                else:
                    logger.info(f"R: {R} not present in UCI: {UCI}")
                    pass
            plt.savefig(os.path.join(self.BaseDir,f"SmoothedFractionPeople_UCI_{round(UCI,3)}.png"))
            plt.close(fig)
            # NOTE: Initialize the Aggregated analysis of a given UCI
            logger.info(f"{self.Name}: Start Aggregated Analysis: UCI: {UCI}")
            self.R2NtNtFit = R2NtNtFit
#                R2NtNtFit_converted = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in R2NtNtFit.items()}
            R2NtNtFit_converted = convert_ndarray(R2NtNtFit)
            with open(os.path.join(UCI_plot_dir,f"R2NtNtFit_UCI_{round(UCI,3)}.json"),'w') as f:
                json.dump(R2NtNtFit_converted,f,indent = 4)
            
            plot_error_fit_expo_powerlaw(self.UsedRs[round(UCI,3)],self.list_err_exp_nt_R,self.list_err_pl_nt_R,UCI_plot_dir)
            # Compute Beta, such that the fit for an increasing powerlaw is top (starting from 6:30 h) -> long lasting curve is this
            self.RangeFit = range(39,len(self.t_vect))
            if FoundRc:
                IndexRc = [i for i, R in enumerate(self.UsedRs[round(UCI,3)]) if R == self.UCI2CriticalParams[round(UCI,3)]["Rc"]][0]
                PlotGammaTau(self.Taus[round(UCI,3)],self.Gammas[round(UCI,3)],IndexRc,self.Name,UCI,UCI_plot_dir)
                # NOTE: we are considering the curve up until the last hour, since it goes to 0
                # NOTE: It must not depend on the moment in which the transition occurs
                Time2FitBeta = ComputeTime2FitBeta(self.RangeFit,self.t_vect,self.R2NtNtFit,self.UCI2CriticalParams,Rs,UCI,self.Name)
                with open(os.path.join(UCI_plot_dir,f"Time2FitBeta_UCI_{round(UCI,3)}.json"),'w') as f:
                    json.dump(convert_ndarray(Time2FitBeta),f,indent = 4)
                # NOTE: independent on the choice of 
                try:
                    BestBeta,BestA,BestT,int_t_star,is_found_beta = ChooseBestBeta(Time2FitBeta,self.t_vect)
                    # NOTE: Plot the best beta
                    PlotBestBeta(Time2FitBeta,BestT,int_t_star,self.Name,self.t_vect,UCI,UCI_plot_dir)
                    # NOTE: Plot error of the fit (evolving in time)
                    PlotTime2FitBetaError(Time2FitBeta,self.t_vect,int_t_star,self.Name,UCI,UCI_plot_dir)
                    self.UCI2PhaseTransition[round(UCI,3)] = {"Beta":BestBeta,"best_A_fit_Beta":BestA,"time_best_fit":BestT,"int_time_best_fit":int_t_star}     
                    try:
                        PlotFig5A5B(self.UsedRs[round(UCI,3)],                                          # NOTE: Rs
                                    self.UCI2CriticalParams[round(UCI,3)]["Rc"],                        # NOTE: Rc
                                    self.R2NtNtFit,                                                     # NOTE: N(R), by choosing the int_t_star position of the vector associated to R
                                    int_t_star,                                                         # NOTE: t*, time at which the best transition occurs.
                                    BestBeta,                                                           # NOTE: Best Beta, is the coefficient for the increase of the N(R) at time t*    
                                    self.Name,
                                    UCI,
                                    UCI_plot_dir
                                )       
                    except Exception as e:
                        print(e)
                except Exception as e:
                    print(e)
                # NOTE: Plot figure 4
                Rc = self.UCI2CriticalParams[round(UCI,3)]["Rc"]
                alpha = self.UCI2CriticalParams[round(UCI,3)]["Alpha"]
                R2Epsilon = {R:(R**2 - Rc**2)/Rc**2 for R in sorted(Rs)}
                PlotFigure4(R2NtNtFit,
                            R2Epsilon,
                            alpha,
                            self.t_vect,
                            UCI,
                            Rc,
                            self.Name,
                            BestT,
                            20,
                            100,
                            200,
                            UCI_plot_dir)
                # Plot R/Rc vs Tau/Tauc
                PlotRRcTauTauc(self.Taus[round(UCI,3)],
                            self.UCI2CriticalParams[round(UCI,3)]["Tauc"],
                            self.UsedRs[round(UCI,3)],
                            self.UCI2CriticalParams[round(UCI,3)]["Rc"],
                            alpha,
                            self.Name,
                            UCI,
                            UCI_plot_dir
                            )
                uci2beta_path = os.path.join(UCI_plot_dir,"uci2beta.json")
                if os.path.exists(uci2beta_path):
                    with open(uci2beta_path, 'r') as f:
                        existing_data = json.load(f)
                else:
                    existing_data = self.UCI2PhaseTransition
                # Update the existing data with the new results
                existing_data.update({round(UCI,3):self.UCI2PhaseTransition[round(UCI,3)]})

                # Save the updated data back to the JSON file
                with open(uci2beta_path, 'w') as f:
                    json.dump(existing_data, f, indent=4)
                
                self.City2UCI2ImportantFeaturesForComparisonCities[self.Name][str(round(UCI,3))]["Rc"] = self.UCI2CriticalParams[round(UCI,3)]["Rc"]
                self.City2UCI2ImportantFeaturesForComparisonCities[self.Name][str(round(UCI,3))]["Tauc"] = self.UCI2CriticalParams[round(UCI,3)]["Tauc"]
                self.City2UCI2ImportantFeaturesForComparisonCities[self.Name][str(round(UCI,3))]["index_Rc"] = IndexRc
                self.City2UCI2ImportantFeaturesForComparisonCities[self.Name][str(round(UCI,3))]["Taus"] = self.Taus[round(UCI,3)]
                self.City2UCI2ImportantFeaturesForComparisonCities[self.Name][str(round(UCI,3))]["Gammas"] = self.Gammas[round(UCI,3)]
                self.City2UCI2ImportantFeaturesForComparisonCities[self.Name][str(round(UCI,3))]["Rs"] = self.UsedRs[round(UCI,3)]
                self.City2UCI2ImportantFeaturesForComparisonCities[self.Name][str(round(UCI,3))]["alpha"] = self.UCI2CriticalParams[round(UCI,3)]["Alpha"]
                self.City2UCI2ImportantFeaturesForComparisonCities[self.Name][str(round(UCI,3))]["beta"] = self.UCI2PhaseTransition[round(UCI,3)]["Beta"]
                self.City2UCI2ImportantFeaturesForComparisonCities[self.Name][str(round(UCI,3))]["n(R)"] = [self.R2NtNtFit[R]["n"] for R in self.UsedRs[round(UCI,3)]]
        with open(os.path.join(os.environ["TRAFFIC_DIR"],"data","info_DP_simulations.json"), 'w') as f:
            json.dump(convert_ndarray(self.City2UCI2ImportantFeaturesForComparisonCities), f, indent=4)
                # Extract The informations for the analysis to be Global.
                # In this way I do not loose Informations but the process is very power consuming
#                self.R2UCI2OutputStats[UCI][R] =  OS

        pl.DataFrame(self.dict_scalar_info_for_each_R).write_parquet(os.path.join(UCI_plot_dir,f"ScalarInfo_UCI_{round(UCI,3)}.parquet"))
        pass


    def CollectUnloadAndFitAllRGivenUCI(self,UCI):
        """
            @UCI: float
            This Function is used to collect time and n(t),fit_nt and plot the unload curve for all Rs.
            The DfUnloads: pd.DataFrame -> Columns: nt_{R},n_fit_{R},time_{R} for all Rs
        """
        pass
    def GetGeoJson(self):
        """
            Description:
                GeoJsonEdges: gpd.DataFrame -> u,v, uv, geometry, highway, lanes, maxspeed, capacity
        """
        self.GeoJsonNodes,self.GeoJsonEdges,self.GetGeoJsonBool = GetGeopandas(self.GraphmlFile)
        self.GeoJsonEdges = CleanGeojson(self.GeoJsonEdges)




if __name__ == "__main__":
    import warnings
    import multiprocessing
    from OutputStats import *
    from GenerateConfiguraiton import *

    warnings.filterwarnings("ignore")
    BaseConfig = os.path.join(os.environ["TRAFFIC_DIR"],"config")
    list_cities = os.listdir(os.path.join(os.environ["TRAFFIC_DIR"],'data','carto'))
    list_cities = ["SFO"]#["BOS","SFO","RIO"]
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
        print("Rs\n",Rs)
        print("UCIs\n",UCIs)
        Df_City2Config = Df_Info_Simulations.filter(pl.col("city") == City)
        # 10) NOTE: Analyze the Output
        PCTA = Polycentrism2TrafficAnalyzer(Df_City2Config,Rs,UCIs)  
        PCTA.CompleteAnalysis()

    if True:
        with multiprocessing.Pool(3) as p:
            p.map(MainMultiprocessing, list_cities)
