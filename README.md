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


# PostProcessing
To postprocess the results create a new environment so that the packages needed do not clash, in the old environment probably built in random order.    
`conda config --set channel_priority strict`    
`conda install geoplot -c conda-forge`
`conda install osmnx`
`conda install conda-forge::imageio`

Hierarchy of Calls Post Processing:   
MainAnalysis: for City in ListCities -> TrajectoryAnalysis: for UCI in UCIs: for R in R -> OutputStats: Compute things that are conditional on these variables.