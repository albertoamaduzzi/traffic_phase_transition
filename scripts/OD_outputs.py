import matplotlib.pyplot as plt
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
