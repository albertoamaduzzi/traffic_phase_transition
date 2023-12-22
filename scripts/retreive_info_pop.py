import requests
import rioxarray
'''
    This script contains functions to pull population data from the WorldPop:
        Given:
            - data type (population, births, pregnancies, urban change)
            - country (optional)
            - year (optional)
    https://github.com/Geo4Dev/Population-Weighted-Wealth/blob/main/Population-Weighted-Wealth.ipynb
'''

def gather_worldpop_data(data_type, country_iso=None, year=2015):
    """
    Build the url to pull WorldPop data from the API

    Inputs:
        data_type (string): Data type options are 'pop' (population),
            'births', 'pregnancies', and 'urban_change'.capitalize
        country_iso (string): The 3-letter country code, if desired. Default
            will be global. 
        year (int): the 4-digit year of interest for data. Default will be
            2015.

    Return (str, rioxarray DataArray): returns the name of the .tif file
        downloaded onto your computer containing the data and the DataArray
        containing the population counts read in using rioxarray.
    """

    # Build the API url according to user selection
    url_base = "https://www.worldpop.org/rest/data"
    url = url_base + '/'  + data_type + '/wpgp'
    if country_iso:
        url = url + '?iso3=' + country_iso

    # Request the desired data; filter by year 
    json_resp = requests.post(url).json()
    json_resp = json_resp['data']['popyear' == year]
    # Obtain exact .geotiff file name for the desired data
    geotiff_file = json_resp['files'][0]
    print('Obtaining file',geotiff_file)

    geotiff_data = requests.get(geotiff_file)
    
    file_name = 'worldpop_' + country_iso + '_' + str(year) + '.tif'
    print('Writing to',file_name)
    with open(file_name,'wb') as f:
        f.write(geotiff_data.content)

    # Read in the WorldPop data as a GeoTIFF
    worldpop_raster = rioxarray.open_rasterio(file_name)

    return file_name, worldpop_raster

