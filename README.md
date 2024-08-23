# Phase transition for traffic
This description is highly incomplete and working progress...
Is a library that is build to study the phase transition to traffic whose order parameter is the fraction of people that remain trapped in the network after some limiting time, and
whose control parameter is the number of people that enter in the network per unit time (time of update for the simulator used).
In particular, tries to show the dependence of the phase transition on different regimes of polycentricity and mopmcentricity.
# SetUp
Set the base directory where the project is to be downloaded:
```bash
export TRAFFIC_DIR=/path/to/ProjectBaseDir
```


# Preprocessing
## Population/Origin-Destination/Grid
`python3 ./scripts/GeometrySphere/ComputeGeometryMain.py`   
The script reads the shape file of the region of interest with the relative Origin-Destinations file (.fma) (see `Input` Section).
It accordingly creates a grid (the script is set to 1.5km^2 for Boston but it changes from place to place according to the size of the region considered.
NOTE: having it too fine grained is very expensive computationally).
It assigns the population to the grid, and creates an origin destination according to the one read from the file. 
Since the file origin destination is taken from polygons of the shape file, the population is redistributed uniformly on the grid that has more resolution by counting.
### OUTPUT
`/TRAFFIC_DIR/grid/size_grid/ODgrid.csv`
`/TRAFFIC_DIR/grid/size_grid/direction_distance_matrix.csv`    
`/TRAFFIC_DIR/grid/size_grid/centroid_lattice.graphml`    


## Computation Vector Field - Potential

## Fit


## Generation New Configuration
# PostProcessing
To postprocess the results create a new environment so that the packages needed do not clash, in the old environment probably built in random order.    
`conda config --set channel_priority strict`    
`conda install geoplot -c conda-forge`
`conda install osmnx`
`conda install conda-forge::imageio`

Hierarchy of Calls Post Processing:   
MainAnalysis: for City in ListCities -> TrajectoryAnalysis: for UCI in UCIs: for R in R -> OutputStats: Compute things that are conditional on these variables.


# Input
`{NameCity}.shp`: Shape File Containing Polygons
`OD_{NameCity}_{StartTime}_{EndTime}.fma`: This file contains information about origin and destinations.    
*Format*: `IndexOrigin` (space) `IndexDestination` (space) `NumberPeople` (new line)   
*NOTE*:  `IndexOrigin` and `IndexDestination` are those of the polygons `{NameCity}.shp`