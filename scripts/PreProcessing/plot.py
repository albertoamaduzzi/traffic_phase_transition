import matplotlib.pyplot as plt
import os
import numpy as np
from shapely.geometry import Polygon
from mpl_toolkits.mplot3d import Axes3D

##------------------- GEOMETRY TO GRAPH -------------------##
def HistoPoint2Geometry(Geom2OD,GeomId,save_dir_local,resolution = None):
    '''
        Geom2OD: geopandas -> geometry contains (polygon,hexagon,ring,grid) Id {GeomID: [list of osmid]}
        GeomId: string -> name of the geometry
    '''
    LOCAL_DIR = os.path.join(save_dir_local,GeomId)
    if os.path.join(LOCAL_DIR,'histo_{0}.png'.format(GeomId)):
        print('{} ALREADY PLOTTED'.format(os.path.join(LOCAL_DIR,'histo_{0}.png'.format(GeomId))))
    else:
        bins = np.arange(len(Geom2OD.keys()))
        value = np.array([len(Geom2OD[polygon]) for polygon in Geom2OD.keys()])
        plt.bar(bins,value)
        plt.xlabel('number of {0}'.format(GeomId))
        plt.ylabel('number of points per {0}'.format(GeomId))
        if GeomId == 'polygon':
            plt.savefig(os.path.join(LOCAL_DIR,'NodeCount.png'),dpi=200)
        elif GeomId == 'hexagon':
            plt.savefig(os.path.join(LOCAL_DIR,resolution,'NodeCount.png'),dpi=200)
        elif GeomId == 'ring':
            plt.savefig(os.path.join(LOCAL_DIR,resolution,'NodeCount.png'),dpi=200)
        elif GeomId == 'grid':
            plt.savefig(os.path.join(LOCAL_DIR,resolution,'NodeCount.png'),dpi=200)

##------------------------------------------ PLOT ------------------------------------------##
def plot_grid_tiling(grid,gdf_polygons,save_dir_local,grid_size):
    # PLOT AREA
    if isinstance(grid_size,float):
        grid_size = str(round(grid_size,3))
    else:
        grid_size = str(grid_size)
    PLOT_DIR = os.path.join(save_dir_local,'grid',str(grid_size))
    _,ax = plt.subplots(1,1,figsize=(12,12))
    grid.plot(ax=ax,column='area',cmap='viridis', edgecolor='black')#color='white',
    gdf_polygons.plot(ax=ax, facecolor = 'none', edgecolor='black')
    cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Area $km^2$')
    ax.set_aspect('equal')
    plt.savefig(os.path.join(PLOT_DIR,'area.png'),dpi=200)
    # PLOT POPULATION
    _,ax = plt.subplots(1,1,figsize=(12,12))
    grid.plot(ax=ax,column='population',cmap='viridis', edgecolor='black')#color='white',
    gdf_polygons.plot(ax=ax, facecolor = 'none', edgecolor='black')
    cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Population')
    ax.set_aspect('equal')
    plt.savefig(os.path.join(PLOT_DIR,'population.png'),dpi=200)
    # PLOT DENSITY
    fig,ax = plt.subplots(1,1,figsize=(12,12))
    grid['density_population'] = grid['population']/grid['area']
    grid.plot(ax=ax,column='density_population',cmap='viridis', edgecolor='black')#color='white',
    gdf_polygons.plot(ax=ax, facecolor = 'none', edgecolor='black')
    cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Population/Area($km^2$)')
    ax.set_aspect('equal')
    plt.savefig(os.path.join(PLOT_DIR,'density.png'),dpi=200)
    # AREA DISTRIBUTION
    fig,ax = plt.subplots(1,1,figsize=(12,12))
    ax.hist(grid['area'])
    ax.set_xlabel('Area $km^2$')
    ax.set_ylabel('Number of squares')
    plt.savefig(os.path.join(PLOT_DIR,'histo_area.png'),dpi=200)

def plot_hexagon_tiling(gdf_hexagons,gdf_polygons,save_dir_local,resolution):
    PLOT_DIR = os.path.join(save_dir_local,'hexagon',resolution)
    if not os.path.isfile(os.path.join(PLOT_DIR,'area.png')):
        fig,ax = plt.subplots(1,1,figsize=(12,12))
        gdf_hexagons.plot(ax=ax,column='area',cmap='viridis', edgecolor='black')#color='white',
        gdf_polygons.plot(ax=ax, facecolor = 'none', edgecolor='black')
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Area $km^2$')
        ax.set_aspect('equal')
        plt.savefig(os.path.join(PLOT_DIR,'area.png'),dpi=200)
    else:
        pass
#            cprint('{} ALREADY PLOTTED'.format(os.path.join(PLOT_DIR,'area')),'green')
    if not os.path.isfile(os.path.join(PLOT_DIR,'population.png')):
        fig,ax = plt.subplots(1,1,figsize=(12,12))
        gdf_hexagons.plot(ax=ax,column='population',cmap='viridis', edgecolor='black')#color='white',
        gdf_polygons.plot(ax=ax, facecolor = 'none', edgecolor='black')
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Population')
        ax.set_aspect('equal')
        plt.savefig(os.path.join(PLOT_DIR,'population.png'),dpi=200)
    else:
        pass
#            cprint('{} ALREADY COMPUTED'.format(os.path.join(PLOT_DIR,'population')),'green')
    if not os.path.isfile(os.path.join(PLOT_DIR,'density.png')):
        fig,ax = plt.subplots(1,1,figsize=(12,12))
        gdf_hexagons.plot(ax=ax,column='density_population',cmap='viridis', edgecolor='black')#color='white',
        gdf_polygons.plot(ax=ax, facecolor = 'none', edgecolor='black')
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Population/Area($km^2$)')
        ax.set_aspect('equal')
        plt.savefig(os.path.join(PLOT_DIR,'density.png'),dpi=200)
    else:
        pass
#            cprint('{} ALREADY COMPUTED'.format(os.path.join(PLOT_DIR,'density')),'green')
    if not os.path.isfile(os.path.join(PLOT_DIR,'_histo_area.png')):
        fig,ax = plt.subplots(1,1,figsize=(12,12))
        ax.hist(gdf_hexagons['area'])
        ax.set_xlabel('Area $km^2$')
        ax.set_ylabel('Number of hexagons')
        plt.savefig(os.path.join(PLOT_DIR,'histo_area.png'),dpi=200)
    else:
        pass
#            cprint('{} ALREADY COMPUTED'.format(os.path.join(PLOT_DIR,'histo_area')),'green')

def plot_ring_tiling(gdf_polygons,save_dir_local,radiuses,radius):
    PLOT_DIR = os.path.join(save_dir_local,'ring',radius)
    if not os.path.isfile(os.path.join(PLOT_DIR,'area.png')):
        fig,ax = plt.subplots(1,1,figsize=(20,15))
        gdf_polygons.plot(ax = ax)
        centroid = gdf_polygons.geometry.unary_union.centroid
        for r in radiuses:
            circle = plt.Circle(np.array([centroid.x,centroid.y]), r, color='red',fill=False ,alpha = 0.5)
            ax.add_artist(circle)
            ax.set_in_layout(True)
            ax.grid(True)
            ax.get_shared_x_axes()
            ax.get_shared_y_axes()
            ax.set_aspect('equal')
            plt.savefig(os.path.join(PLOT_DIR,'ring.png'),dpi=200)
    else:
        pass
        #cprint('{} ALREADY PLOTTED'.format(os.path.join(PLOT_DIR,'area')),'blue')

def plot_departure_times(df,save_dir_local,start,end,R):
    time_dep = df['dep_time'].to_numpy()/3600
    plt.hist(time_dep , bins = 24)
    plt.xlabel('Departure time')
    plt.ylabel('Number of trips')
    plt.savefig(os.path.join(save_dir_local,'histo_departure_time_{0}to{1}_R_{2}.png'.format(start,end,R)),dpi=200)

##------------------------------------------ GEOMETRY ------------------------------------------##
def plot_in_polygon(point,polygon):
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    ax.plot(point[0], point[1], 'ro')  # 'ro' means red color and circular marker
    # Create a Polygon patch
    polygon_patch = Polygon(np.array([polygon[0],polygon[1],polygon[2],polygon[3]]), closed=True, edgecolor='b', facecolor='none')
    # Add the Polygon patch to the plot
    ax.add_patch(polygon_patch)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Point and Polygon')
    plt.savefig('point_in_polygon.png',dpi=200)

##------------------------------------------ POTENTIAL ------------------------------------------##

def PlotPotentialLevelCurve(x,y,z,save_dir_local):
    '''
        x,y: np.array 1D [containing x,y coordinates of the grid nodes]
        z: np.array 2D [containing the potential values] (should be the column of grid.geojson)
    '''
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    ax.contourf(x, y, z, 20, cmap='viridis')
    ax.set_aspect('equal')
    plt.savefig(os.path.join(save_dir_local,'level_curve.png'),dpi=200)

def PlotPotential3D(X,Y,Z,save_dir_local):
    '''
        X,Y: np.array 2D [containing x,y coordinates of the grid nodes]
        Z: np.array 2D [containing the potential values] (should be the column of grid.geojson)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Potential')
    ax.set_title('3D Plot of Potential')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(os.path.join(save_dir_local,'Potential.png'),dpi=200)
    plt.show()