In this directory I define all the geometrical object that I need in the study of traffic.
In particular Grids, Rings and Hexagons with different granularity.
This is thought in such a way that a renormalization approach can be applied naturally.
Compute-
        Grid.py
        Hexagon.py
        Ring.py
Are used to compute in parallel for each city and each resolution. 
NOTE: This is due to the fact that the partitions and mapping with origin and destinations are slow, and are a bottleneck of the whole pipeline.
NOTE that we need these tassellation for the analysis.
In particular for the description of the Potential.


