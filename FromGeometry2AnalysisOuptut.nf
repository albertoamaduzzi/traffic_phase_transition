#!/usr/bin/env nextflow
// NOTE: The project is thought to be based on TRAFFIC_DIR (environment variable to be set beforehand)
params.TrafficDir = System.getenv('TRAFFIC_DIR')
// NOTE: For Each City 
params.BaseShapeFile = "${params.TrafficDir}/data/carto"
params.BaseTiffFile = "${params.TrafficDir}/tiff_files"
params.CityName = ["BOS"]//["BOS","SFO","LAX","LIS","RIO"]

// 

// Writing the Configuration File [For each City]

process OpenJsonConfig{
    input:
    each CityName
    """
    touch > ./config/${CityName}/Config_${CityName}.json
    nano 
    """
}

process AppendConfigFile{
    input:
    path ConfigFile
    each InputParameter
    """
    
    """

}

// Carto
process CartoGraphyFromShape{
    input:
    path shape_file
    each CityName

    output:
    path ...
    path ...
    """
    python3 ./scripts/Preprocessing/CreateCartoSimulator.py
    """
}

// Geometry For Model

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


process ComputeGrid{
    input:
    path tiff_file
    path shape_file
    each CityName

    output:
    path ...
    path ...
    """
    python3 ./scripts/GeometrySphere/ComputeGridMain.py
    """
}

// Launch

process LaunchSimulation{
    input:
    path tiff_file
    path shape_file
    each CityName

    output:
    path ...
    path ...
    """
    python3 ./scripts/PostProcessing/multiple_launches.py
    """
}

// Analysis

process Analysis{
    input:
    path tiff_file
    path shape_file
    each CityName

    output:
    path ...
    path ...
    """
    python3 ./scripts/PostProcessing/MainAnalysis.py
    """
}
workflow{
    // NOTE: For Each City 
    shape_file = "${params.BaseShapeFile}/shape_file.shp"
    tiff_file = "${params.BaseTiffFile}/tiff_file.tiff"
    // Repeat the same process for each city
    CityName = ["BOS","SFO","LAX","LIS","RIO"]   
    def hostname = getHostname()
    if (hostname == "artemis.ist.berkeley.edu")
    {
        ComputeGeometry(tiff_file, shape_file, CityName)
    }
    else
    {
        
    }
}















// AWS 
aws {
    accessKey = System.getenv('AWS_ACCESS_KEY_ALBERTO')
    secretKey = System.getenv('AWS_SECRET_ALBERTO')
    client {maxConnections = 20
            connectionTimeout = 10000
            uploadStorageClass = "INTELLIGENT_TIERING"
            region = "us-west-2"
            }
}








// Useful Functions
def getHostname() {
    def hostname = "hostname".execute().text.trim()
    return hostname
}