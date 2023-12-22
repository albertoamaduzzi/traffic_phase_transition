import json
import argparse
import numpy as np
import os
import geopandas as gpd
import osmnx as ox
import shapely as shp
import pandas as pd
from collections import defaultdict
import h3
import shapely.geometry as sg
import rasterio
from rasterio.plot import show
from rasterio.windows import Window
from rasterio.mask import mask




argparser = argparse.ArgumentParser()
'''
    This is done for each city [BOS,SFO,LIS,RIO,LAX], for each time, these are the input parameters.
    config = {
    config = {'carto_dir':carto_dir, # Directory where the cartographic data is stored
        'shape_file_dir': shape_file_dir, # Directory where the shape files are stored
        'start': start, # Time where cumulation of trips starts
        'end': end, # If not specified, it is assumed that the OD is for 1 hour (time where cumlation of trips ends)
        "name":name,
        "number_users":len(df1),
              }
        
'''


def lognormal(mean, std):
    '''
        Distribution of distances for each trip (the OD pair must be generated in this way)
        For policentric cities the variance is bigger. [ maybe we can invent some way of defining the variance according to the policentricity]
    '''
    return np.random.lognormal(mean, std)

def Weibull(shape, scale):
    '''
        Distribution of distances of trapped cars after 1 hour
    '''
    return np.random.weibull(shape, scale)

    

class OD:
    '''
        The origin destination file is OD{start}_{}end_Rk.csv -> It counts how many people go from any origin to any destination leaving at time start and arriving at time end
    '''
    def __init__(self,config):
        if 'carto_dir' in config:
            self.carto_dir = config['carto_dir']
        else:
            self.carto_dir = '/home/aamad/Desktop/phd/traffic_phase_transition/data/carto/'
        if 'shape_file_dir' in config:
            self.shape_file_dir = config['shape_file_dir']
        else:
            self.shape_file_dir = self.carto_dir
        if 'tif_file' in config:
            self.tif_file = config['tif_file']
        else:
            self.tif_file = '/home/aamad/Desktop/phd/berkeley/data/usa_ppp_2020_UNadj_constrained.tif'
        if 'file_OD' in config:
            self.file_OD = config['file_OD']
        else:
            self.file_OD = '/home/aamad/Desktop/phd/berkeley/data/carto/BOS/od_demand_7to8_R_1.csv'
        if 'name' in config:
            self.name = config['name']
        else:
            self.name = 'BOS'
        if 'start' in config:
            self.start = config['start']
        if 'end' in config:
            self.end = config['end']
        else:
            self.end = self.start + 1
        if 'number_trips' in config:
            self.number_trips = config['number_trips']
        else:
            self.number_trips = 10000
        if 'R' in config:
            self.R = config['R']
        else:   
            self.R = 1
        if 'p' in config:
            self.p = config['p']
        else:
            self.p = 0.5
        if 'grid_size' in config:
            self.grid_size = config['grid_size']    
        else:
            self.grid_size = 1
        if 'number_of_rings' in config:
            self.number_of_rings = config['number_of_rings']
        else:
            self.number_of_rings = 10
        ## GRAPH -> Nx
        self.graph = ox.load_graphml(filepath=os.path.join(self.carto_dir,self.name + '_new_tertiary_simplified.graphml'))
        self.crs = self.graph.graph['crs']
        ## POLYGONS -> geopandas
        self.gdf_polygons = gpd.read_file(os.path.join(self.shape_file_dir,self.name + '.shp'))
        self.Area = self.gdf_polygons.unary_union.area
        self.radius = np.sqrt(self.Area/np.pi)
        ## TILING
#        self.get_squared_grid()
#        self.get_rings()
#        self.get_hexagon_tiling()

        ## OD -> df format 
        self.df_od = pd.read_csv(config['file_OD'],index_col=0)
        ## POLYGON2OD
        if os.path.exists(os.path.join(self.carto_dir,self.name,'polygon2od.json')):
            with open(os.path.join(self.carto_dir,self.name,'polygon2od.json'),'r') as f:
                self.polygon2od = json.load(f)
        else:
            raise ValueError('polygon2od.json does not exist')
        self.realization2UCI = defaultdict(list)

##------------------------------------------------ TILING ------------------------------------------------##

    def get_squared_grid(self):
        '''
            centroid: Point -> centroid of the city
            bounding_box: tuple -> (minx,miny,maxx,maxy)
            grid: GeoDataFrame -> grid of points of size grid_size
            idx2grid: dict -> {idx:grid_point}
        '''
        self.centroid = self.gdf_polygons.geometry.unary_union.centroid
        self.bounding_box = self.gdf_polygons.geometry.unary_union.bounds
        bbox = shp.geometry.box(*self.bounding_box)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=self.crs)
        x = np.arange(self.bounding_box[0], self.bounding_box[2], self.grid_size)
        y = np.arange(self.bounding_box[1], self.bounding_box[3], self.grid_size)
        grid_points = gpd.GeoDataFrame(geometry=[shp.geometry.box(x, y) for x in x for y in y], crs=self.crs)
        # Clip the grid to the bounding box
        self.grid = gpd.overlay(grid_points, bbox_gdf, how='intersection')
        self.idx2grid = {idx:grid for idx,grid in enumerate(self.grid)}

    def get_rings(self):
        '''
            Compute the rings of the city and the intersection with polygons
            rings: dict -> {idx:ring}
        '''
        self.rings = defaultdict(list)
        gdf_original_crs = gpd.GeoDataFrame(geometry=[self.centroid], crs=self.crs)
        self.radius = max([abs(self.bounding_box[0] -self.bounding_box[2])/2,abs(self.bounding_box[1] - self.bounding_box[3])/2]) 
        self.radiuses = np.linspace(0,self.radius,self.number_of_rings)
        for i,r in enumerate(self.radiuses):
            if i == 0:
                intersection_ = gdf_original_crs.buffer(r)
                self.rings[i] = intersection_
            else:
                intersection_ = gdf_original_crs.buffer(r).intersection(gdf_original_crs.buffer(self.radiuses[i-1]))
                complement = gdf_original_crs.buffer(r).difference(intersection_)
                self.rings[i] = complement

    def get_hexagon_tiling(self,resolution=8):
        '''
            This function is used to get the hexagon tiling of the area, it can be used just for US right now as we have just
            tif population file just for that
        '''
        ## READ TIF FILE
        self.resolution = resolution
        if not os.path.exists(os.path.join(self.carto_dir,'{}_hexagonal_tiling.geojson'.format(self.name))):
            with rasterio.open(self.tif_file) as dataset:
                clipped_data, clipped_transform = mask(dataset, self.gdf_polygons.geometry, crop=True)
            ## CHANGE NULL ENTRANCIES (-99999) for US (may change for other Countries [written in United Nation page of Download])
            clipped_data = np.array(clipped_data)
            condition = clipped_data<0
            clipped_data[condition] = 0
            # Define hexagon resolution
            bay_area_geometry = self.gdf_polygons.unary_union
            # Convert MultiPolygon to a single Polygon
            bay_area_polygon = bay_area_geometry.convex_hull

            # Convert Polygon to a GeoJSON-like dictionary
            bay_area_geojson = sg.mapping(bay_area_polygon)
            # Get hexagons within the bay area
            hexagons = h3.polyfill(bay_area_geojson, resolution, geo_json_conformant=True)

            # Convert hexagons to Shapely geometries
            hexagon_geometries = [sg.Polygon(h3.h3_to_geo_boundary(h, geo_json=True)) for h in hexagons]


            # Aggregate population data for each hexagon
            population_data_hexagons = [aggregate_population(hexagon, clipped_transform, clipped_data) for hexagon in hexagon_geometries]
            centroid_hexagons = [h.centroid for h in hexagon_geometries]
            # Create GeoDataFrame
            self.gdf_hexagons = gpd.GeoDataFrame(geometry=hexagon_geometries, data=[population_data_hexagons,centroid_hexagons], columns=['population','centroid'])
            self.gdf_hexagons.reset_index(inplace=True)
            self.gdf_hexagons.to_file(os.path.join(self.carto_dir,'{}_hexagonal_tiling.geojson'.format(self.name)), driver="GeoJSON")
        else:
            self.gdf_hexagons = gpd.read_file(os.path.join(self.carto_dir,'{0}_hexagonal_tiling_resolution_{1}.geojson'.format(self.name,resolution)))


## ------------------------------------------------ MAP TILE TO POLYGONS ------------------------------------------------ ##

    def get_intersection_polygon2grid(self):
        if self.grid is None:
            raise ValueError('grid is None')
        elif self.gdf_polygons is None:
            raise ValueError('gdf_polygons is None')
        else:
            for k in self.idx2grid.keys():
                self.gdf_polygons['intersection_{}'.format(k)] = self.gdf_polygons.geometry.apply(lambda x: x.intersection(self.grid))
            

    def get_intersection_polygon2rings(self):
        if self.rings is None:
            raise ValueError('rings is None')
        elif self.gdf_polygons is None:
            raise ValueError('gdf_polygons is None')
        else:
            for r in self.radiuses:
                self.gdf_polygons['intersection_{}'.format(r)] = self.gdf_polygons.geometry.apply(lambda x: x.intersection(x.buffer(r)))
    
    def get_intersection_polygon2hexagon(self):
        self.polygon2hexagon = defaultdict(list)
        if self.gdf_hexagons is None:
            raise ValueError('gdf_hexagons is None')
        elif self.gdf_polygons is None:
            raise ValueError('gdf_polygons is None')
        else:
            for idx,hexagon in enumerate(self.gdf_hexagons.geometry):
                self.gdf_polygons['intersection_{}'.format(idx)] = self.gdf_polygons.geometry.apply(lambda x: x.intersection(hexagon))


##------------------------------------------------ MATRIX REPRESENTATION ------------------------------------------------##   
    


    def OD2matrix(self):
        '''
            Create: 
                1) 02mapidxorigin {0:origin1,1:origin2,...}
                2) 02mapidxdestination {0:destination1,1:destination2,...}
                3) mapidx2origin {origin1:0,origin2:1,...}
                4) mapidx2destination {destination1:0,destination2:1,...}
                origin_i,destination_i are the nominative of self.gdf_polygons
        '''
        origins = np.unique(self.df_od['origin'])
        destinations = np.unique(self.df_od['destination'])
        self.mapOD = self.df_od.groupby('origin')['destination'].unique().apply(lambda x: x[0]).to_dict()

        self.zero2mapidxorigin = {i:idx for idx,i in enumerate(origins)}
        self.zero2mapidxdestination = {i:idx for idx,i in enumerate(destinations)}
        self.mapidx2origin = {idx:i for idx,i in enumerate(origins)}
        self.mapidx2destination = {idx:i for idx,i in enumerate(destinations)}
        self.dimensionality_matrix = (len(origins),len(destinations))
        self.matrix_OD = np.zeros(self.dimensionality_matrix)
        self.trips_OD = self.df_od.groupby(['origin']).count()['destination'].to_numpy(dtype= int)
        for idx in range(len(origins)):
            self.matrix_OD[self.zero2mapidxorigin[origins[idx]],self.zero2mapidxdestination[self.mapOD[origins[idx]]]] = self.trips_OD[idx]
        self.matrix_out = np.sum(self.matrix_OD,axis=1)
        self.matrix_in = np.sum(self.matrix_OD,axis=0)

    def OD2Laplacian(self):
        '''
            Compute the Laplacian of the matrix
        '''
        self.Laplacian = np.diag(self.matrix_out) - self.matrix_OD
        return self.Laplacian
    
    def eigvals_eigvects_Laplacian(self):
        '''
            Compute the eigenvalues and eigenvectors of the Laplacian
        '''
        self.eigvals,self.eigvects = np.linalg.eig(self.Laplacian)
        return self.eigvals,self.eigvects
    
    def fiedler2beta(self):
        '''
            Compute the beta from the fiedler eigenvalue
        '''
        ev = np.sort(self.eigvals,ascending=False)
        i = 0
        while(ev[i]==0):
            i+=1
        self.beta = 1/ev[i]
        return self.beta

    def shuffle_matrix(self):
        '''
            This shuffling gives a random matrix with the same number of trips. So we are in the microcanonical ensemble
            of matrices that have fixed number_of_trips -> where the number of trips is the sum over all couples of origin and destinations
        '''
        entries = 0
        while(entries< self.dimensionality_matrix):
            i = np.random.randint(0,self.dimensionality_matrix[0])
            j = np.random.randint(0,self.dimensionality_matrix[1])
            l = np.random.randint(0,self.dimensionality_matrix[0])
            k = np.random.randint(0,self.dimensionality_matrix[1])
            entries += 1
            r = np.random.rand()
            if r<self.p:
                if(l!=i and k!=j):
                    self.matrix_OD[l,k] = self.matrix_OD[l,k] + self.matrix_OD[i,j]
                    if self.matrix_OD[i,j] != 0:
                        self.matrix_OD[i,j] = 0
                    else:
                        pass
            else:
                pass

    def shuffle_matrix_weighted():
            entries = 0
            while(entries< self.dimensionality_matrix):
                i = np.random.randint(0,self.dimensionality_matrix[0])
                j = np.random.randint(0,self.dimensionality_matrix[1])
                l = np.random.randint(0,self.dimensionality_matrix[0])
                k = np.random.randint(0,self.dimensionality_matrix[1])
                entries += 1
                r = np.random.rand()
                if r<np.exp(-self.beta*(self.matrix_OD[i,j])): ## If I have a lot of people I do not want to move them
                    if(l!=i and k!=j):
                        self.matrix_OD[l,k] = self.matrix_OD[l,k] + self.matrix_OD[i,j]
                        if self.matrix_OD[i,j] != 0:
                            self.matrix_OD[i,j] = 0
                        else:
                            pass
                else:
                    pass

##------------------------------------------------ MOBILITY VECTOR FIELD ------------------------------------------------##
    def mobility_vector_field(self):
        for i in range(len(self.gdf_hexagons.geometry)):
            for j in range(len(self.gdf_polygons.geometry)):
                if i!=j:
                    u = np.diff(self.gdf_polygons['centroid'][j].x -self.gdf_polygons['centroid'][i].x)
                    v = np.diff(self.gdf_polygons['centroid'][j].y -self.gdf_polygons['centroid'][i].y)
                    
                    (['centroid'][j]) 
                                

##------------------------------------------------ QUANTITIES OF SOCIOLOGICAL INTEREST ------------------------------------------------##
    def compute_UCI(self):
        '''
            Compute the UCI of the matrix
        '''
        if self.gdf_hexagons is None:
            self.get_hexagon_tiling()
        else:
            pass
        self.gdf_hexagons['s'] = (self.gdf_hexagons['population']/self.gdf_hexagons['population'].sum())
        self.LC = 0.5*(np.sum(self.gdf_hexagons['s']) - 1/len(self.gdf_hexagons))
        
        self.N = len(self.grid)
        self.Nrings = len(self.rings)


        return UCI
    
    def compute_Gini(self):
        '''
            Compute the Gini index of the matrix
        '''
        return Gini
    def multivariatePoisson(self,mean):
        '''
            Generate a multivariate Poisson distribution
        '''

        return np.random.multivariate_normal(mean,mean)
    

 ##------------------------------------------------ GLOBAL FUNCTION ABOUT TILING ------------------------------------------------##   

def aggregate_population(hexagon, hexagon_transform, clipped_data):
    # Convert hexagon to pixel coordinates
    hexagon_coords = np.array(hexagon.exterior.xy).T
    hexagon_pixel_coords = np.array(rasterio.transform.rowcol(hexagon_transform, hexagon_coords[:, 0], hexagon_coords[:, 1]))
    # Ensure coordinates are within valid bounds
    valid_coords = (
        (0 <= np.array(hexagon_pixel_coords[0])) & (np.array(hexagon_pixel_coords[0]) < clipped_data.shape[1]) &
        (0 <= np.array(hexagon_pixel_coords[1])) & (np.array(hexagon_pixel_coords[1]) < clipped_data.shape[2])
    )

    # Filter valid coordinates
    valid_pixel_coords = hexagon_pixel_coords[:, valid_coords]

    # Extract values from clipped data
    values = clipped_data[0, valid_pixel_coords[0], valid_pixel_coords[1]]

    # Aggregate population data (example: sum)
    population_sum = np.sum(values)

    return population_sum


if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    config_file = argparser.add_argument('--config_file','-c',type=str,default='/home/aamad/Desktop/phd/traffic_phase_transition/data/config/BOSconfigOD_7_8_R_1.json')
    with open(config_file,'r') as f:
        config = json.load(f)
    od = OD(config)
    od.OD2matrix()

    for r in range(1000):
        od.shuffle_matrix()
        UCI = od.compute_UCI()
        od.realization2UCI[r].append(UCI)
        print('-------------------------')
