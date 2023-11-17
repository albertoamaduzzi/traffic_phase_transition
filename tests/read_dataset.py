import pytorch
import networkx as nx
import geopandas as gpd
import momepy
from shapely import Polygon,Point,LineString
import h3
from collections import defaultdict
class GraphLoader:
    '''
        The idea here is to create a class that handles the different possible ways to download the dataset.
        Essentially in output I would like some graph read from cartography.
        I want the graph containing informations about the points of interests.
        Above all the economic features and density features. If these data are available, in principle I have 
    '''
    def __init__(self,city,typegraph,config_dict):
        self.gdf = gpd.read_file(config_dict[city]['geojson'])
        self.city = city
        self.base_input_dir = config_dict[city]['input_dir']
        self.base_output_dir =  config_dict[city]['output_dir']
        self.variable_initialization = defaultdict()
        if typegraph == 'networkx':
            if config_dict['duality_net'] == True:
                self.G =momepy.gdf_to_nx(self.gdf,approach = 'dual')
            else:
                self.G =momepy.gdf_to_nx(self.gdf,approach = 'primal')
        elif typegraph == 'pytorch':

    


    def create_hexagonal_cover(self,polygon):
        '''
        Input: shapely.Polygon
        Description: Produces the hexagonal covering of the polygon
        The 13 resolution in polyfill corresponds to 40 m^2 area, similar to HDX data from meta
        '''
        # Obtain hexagonal ids by specifying geographic extent and hexagon resolution
        hexagon_ids = h3.polyfill(polygon.__geo_interface__, 13, geo_json_conformant = True)
        # Custom function to be applied to each hexagonal id (from previous step), converting them to valid shapely polygons
        polygonise = lambda hex_id: Polygon(
                                        h3.h3_to_geo_boundary(
                                            hex_id, geo_json=True)
                                            )
        # Map custom function to polygon ids with EPSG:4326 coordinate reference system
        hexagonal_polygons = gpd.GeoSeries(list(map(polygonise, hexagon_ids)), 
                                            index=hexagon_ids, 
                                            crs="EPSG:4326" 
                                            )    
        self.hex_gdf = gpd.GeoDataFrame(data=None, crs="EPSG:4326", geometry = hexagonal_polygons)
        # Reset index to move hexagonal polygon id list to its own column
        self.hex_gdf = self.hex_gdf.reset_index()
        # Rename column names to allow spatial overlay operation later
        self.hex_gdf.columns = ['hexid', 'geometry']
        if not os.path.is_file(os.path.join(self.base_output_dir,self.city+"_hexagonal_geometry.geojson")):
            self.hex_gdf.to_file(os.path.join(self.base_output_dir,self.city+"_hexagonal_geometry.geojson",driver = "GeoJson"))
        self.variable_initialization['hex_gdf'] = True