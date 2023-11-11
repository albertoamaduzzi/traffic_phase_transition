import json
import os
from collectoins import defaultdict
def ifnotexistsmkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path
# Local directory
local_dir = os.getcwd()

# Set input output
output_dir = os.path.join(local_dir,'output')
input_dir = os.path.join(local_dir,'input')
ifnotexistsmkdir(output_dir)
ifnotexistsmkdir(input_dir)

# Setting the configuration file
cities_of_interest = ['San Francisco','Boston','Lisbon','Porto','Rio']
dir_output_cities = []
dir_input_cities = []
for city in cities_of_interest:
    dir_output_cities.append(ifnotexistsmkdir(os.path.join(output_dir,city))) 
    dir_input_cities.append(ifnotexistsmkdir(os.path.join(input_dir,city)))

config = defaultdict()
for c in range(len(cities_of_interest)):
    config[cities_of_interest]["input_dir"]
    config["output_dir"] 