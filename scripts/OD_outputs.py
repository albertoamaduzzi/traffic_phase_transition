import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import os

def plot_OD(self, start, end, R, p, grid_size, number_of_rings, number_trips):
    '''
        Plot the origin destination matrix
    '''
    OD = self.get_OD(start, end, R, p, grid_size, number_of_rings, number_trips)
    OD = OD / OD.sum()
    fig, ax = plt.subplots()
    self.gdf_polygons.plot(ax=ax, color='white', edgecolor='black')
    self.gdf_polygons.boundary.plot(ax=ax, color='black')
    OD.plot(ax=ax, legend=True, cmap='OrRd')
    plt.savefig(os.path.join(self.carto_dir,'OD_{}_{}_{}_{}_{}_{}.png'.format(start, end, R, p, grid_size, number_of_rings)))
    plt.show()

def plot_OD_ring(self, centroid, idx2ring,crs):
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    gdf_target_crs = gpd.GeoDataFrame(geometry=[centroid], crs=crs)
    for i, ring in enumerate(idx2ring):
        ax = gdf_target_crs.plot(marker='o', color='red', markersize=50)
        ring.plot(ax=ax, alpha=0.5)



def plot_density_population(hexagon_geometries,
                            population_data_hexagons,
                            polygon_gdf,
                            name,
                            save_dir,
                            resolution):
    '''
        Input:
            hexagon_geometries (list): list of hexagon geometries list(shapely.geometry.Polygon)
            population_data_hexagons (list): list of population data for each hexagon (int)
        Plot the population density of the hexagons for each different resolution.
    '''
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    gdf_hexagons = gpd.GeoDataFrame(geometry=hexagon_geometries, data=population_data_hexagons, columns=['population'])
    gdf_hexagons.reset_index(inplace=True)
    polygon_gdf.boundary.plot(ax = ax,alpha = 0.2,color = 'black')
    gdf_hexagons.loc[:,('population','geometry')].plot(ax=ax, column='population',legend=True,alpha = 0.6,cmap = 'inferno')
    ax.axis('off')
    ax.set_title('Boston Population')
    plt.savefig(os.path.join(save_dir,'cover_{}_{}.png'.format(name,resolution)))
    plt.show()


def plot_vector_field(G_centroid,fluxes_matrix,gdf_polygons):
    '''
        Plots the vector field between the centroids of the polygons
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    point_coords = np.array([np.array(G_centroid.nodes[node]['pos']) for node in G_centroid.nodes() if 'pos' in G_centroid.nodes[node].keys()])
    points_3d = point_coords[:, np.newaxis, :]
    #dx = point_coords.T[0] - point_coords[:,0]
    #dy = point_coords.T[1] - point_coords[:,1]
    vectors = point_coords - points_3d 
    norm_vect =  np.linalg.norm(vectors, axis=2)[:, :, np.newaxis]
    nv = np.array([np.array([norm_vect[i][j][0] if norm_vect[i][j][0] != 0 else 1 for j in range(len(norm_vect[i]))]) for i in range(len(norm_vect))])
    normalized_vectors = [[vectors[i][j] / nv[i][j] for j in range(len(vectors[i]))] for i in range(len(vectors))]
    not_yet_flux_vectors = np.array([np.array([normalized_vectors[i][j]*fluxes_matrix[i][j] for j in range(len(normalized_vectors[0]))]) for i in range(len(normalized_vectors))])
    flux_vectors = np.sum(not_yet_flux_vectors,axis=1)
    print(flux_vectors)
    X,Y = np.meshgrid(point_coords[:,0],point_coords[:,1]) 
    gdf_polygons.plot(ax=ax, color='white', edgecolor='black')
    ax.quiver(point_coords[:,0], point_coords[:,1], flux_vectors[:,0] ,flux_vectors[:,1], angles='xy', scale_units='xy',scale = 1000, color='blue')
    #ax.set_xlable('longitude')
    #ax.set_ylable('latitude')
    ax.set_title('Vector Field between Centroids polygons')
    plt.show()
