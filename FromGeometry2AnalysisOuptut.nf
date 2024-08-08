#!/usr/bin/env nextflow
// NOTE: The project is thought to be based on TRAFFIC_DIR (environment variable to be set beforehand)
params.TrafficDir = System.getenv('TRAFFIC_DIR')
// NOTE: For Each City 
params.BaseShapeFile = "${params.TrafficDir}/data/carto"
params.BaseTiffFile = "${params.TrafficDir}/tiff_files"

process ComputeGeometry{
    input:
    path tiff_file
    path shape_file
    each CityName
    
    output:
    path ...
    path ...
    """
    python3 ./scripts/GeometrySphere/ComputeGeometryMain.py
    """
}


workflow{
    // NOTE: For Each City 
    shape_file = "${params.BaseShapeFile}/shape_file.shp"
    tiff_file = "${params.BaseTiffFile}/tiff_file.tiff"
    // Repeat the same process for each city
    CityName = ["BOS","SFO","LAX","LIS","RIO"]   
    ComputeGeometry(tiff_file, shape_file, CityName)
}