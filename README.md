# Description
This project consists in a pipeline for studying traffic phase transition to the absorbing state of gridlock.
In particular, we analyze the number of people that remain trapped in the road network after long times.   
The project defines a pipeline that evolves into three different phases.
1. `geometry_creation`: we define the geometrical objects necessary to compute what is needed.
2. `morphology_handling`:     
    2a. we define and compute urban centrality index `UCI`    
    2b. compute the gravitational model for fluxes in terms of population.        
3. `morphology_generation`:     
    3a. We define a model to generate different morpholgies (the difference is thought as difference in `UCI`)    
    3b. Produce new fluxes accordingly exploiting infomration about previous step.    
4. `simulation`:    
    4a. Generate the origin and destination in the road network    
    4b. Launch the simulations    
5. `analysis_DP`:    
    We handle the results of the simulation and we focus on the analysis of directed percolation universality class.

# SetUp
Set the base directory where the project is to be downloaded:
```
bash export TRAFFIC_DIR=/path/to/ProjectBaseDir
```


# Preprocessing
In this project we are considering the realization of synthetic traffic conditions for different cities: Boston, San Francisco,Rio De Janeiro, Los Angeles, Lisbon.
The alorithm, starting from informations about population and fluxes in different hours of the day has the goal to produce diff
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
Either:
`ModifyPotential.ipynb` (first cell)   
Or:   

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
# Directed Percolation
Objects of Interest:
`GeoJsonEdges`
# Input
`{NameCity}.shp`: Shape File Containing Polygons
`OD_{NameCity}_{StartTime}_{EndTime}.fma`: This file contains information about origin and destinations.    
*Format*: `IndexOrigin` (space) `IndexDestination` (space) `NumberPeople` (new line)   
*NOTE*:  `IndexOrigin` and `IndexDestination` are those of the polygons `{NameCity}.shp`


# Description:
The Project can be divided into 2 parts:
1) Preprocessing
2a) Launch Simulation
2b) Analysis Simulation
The Preprocessing part is all contained in `/$TRAFFIC_DIR/scripts/GeometrySphere/` directory.
The task computed here are the following:
## Preprocessing
1) Extract information about population from .tiff files extracted from Meta project for demography. They come in `hexagon` format, whose `hexagon_resolution` must be specified
2) Extract information about fluxes in .fma files obtained from previous work. They come in `polygon` format. No parameter needed.
3) Adjust the data to `grid` as it is the only format that allows the framework developed for UCI used (Urban Centrality Index).
4) 
## Geometry
`grid`: GeoJson 
Columns:  
- "i": int, Bidimensional index for grid in the x-axis
- "j": int, Bidimensional index for grid in the y-axis
- "centroidx": float, "centroidy": float, Centroid coorindates
- "area": float, Area of the grid
- "index": int, Unidimensional index for the grid once flattened: Useful for OD generation
- "population": float, Number of people living in the grid obtained as a intersection with the hexagons obtained from Facebook
- "with_roads": bool, Value that tells wether the map has roads in that grid
- "geometry":Polygon   

## Useful Commands
#### Check memory occupied on disk from home
du -sh /home/* 2>/dev/null