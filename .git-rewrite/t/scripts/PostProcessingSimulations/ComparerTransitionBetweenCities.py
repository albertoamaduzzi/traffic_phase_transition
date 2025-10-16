"""
    Description:
        This code is responsible for comparing different cities. 
        In particular it is responsible for comparing the transitions to gridlocks.
"""
import os
import json
class ComparerTransitionBetweenCities:
    """
    self.dict_info contains all the informations that are needed in order to plot the comparison between cities.
    """
    def __init__(self, cities, path_to_data):
        self.cities = cities
        self.path_to_data = path_to_data
        with open(os.path.join(os.environ["TRAFFIC_DIR"],"data","info_DP_simulations.json"), 'r') as f:
            self.dict_info = json.load(f)
