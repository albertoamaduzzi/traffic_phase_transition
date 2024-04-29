import numpy as np
import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt

##------------------------------------- PLOTS -----------------------------------------##

def PrintInfoFluxPop(grid,Tij):
    print('******************')
    print('Number of grids with people: ',grid.loc[grid['population']>50].shape[0])
    print('Number of couples of grids with flux: ',Tij.loc[Tij['number_people']>0].shape[0])
    print('Total Population: ',np.sum(grid['population']))
    print('Total Flux: ',np.sum(Tij['number_people']))
    print('Fraction of grids populated: ',grid.loc[grid['population']>50].shape[0]/grid.shape[0])
    print('Fraction of couples of grids with fluxes: ',Tij.loc[Tij['number_people']>0].shape[0]/Tij.shape[0])
    print('******************')

def PlotOldNewFluxes(Tij_new,Tij,verbose = False):
    n,bins = np.histogram(Tij_new['number_people'].loc[Tij_new['number_people']>0],bins = 100)
    n1,bins1 = np.histogram(Tij['number_people'].loc[Tij['number_people']>0],bins = 100)
    fig,(ax0,ax1) = plt.subplots(ncols=2,figsize=(10, 10))
    ax0.scatter(bins[:-1],n)
    ax0.scatter(bins1[:-1],n1)
    ax0.set_yscale('log')
    ax0.legend(['Fitted','Original'])
    ax0.set_title('Distribution fluxes fitted')
    ax1.hist(Tij_new['number_people'],bins = 20)
    ax1.set_title('DIstribution fluxes fitted')
    if verbose:
        plt.show()


def PlotPositionCenters(grid,SFO_obj,index_centers,dir_grid,verbose = False):
    fig,ax = plt.subplots(figsize=(10, 10))
    SFO_obj.gdf_polygons.plot(ax=ax, color='white', edgecolor='black',alpha = 0.2)
    grid.plot(ax=ax, edgecolor='black', facecolor='none',alpha = 0.2)
    for i in index_centers:
        ax.scatter(grid['geometry'].apply(lambda geom: geom.centroid.x)[i],grid['geometry'].apply(lambda geom: geom.centroid.y)[i],marker = 'x',color = 'r')
    plt.savefig(os.path.join(dir_grid,'Position_Centers.png'),dpi = 200)
    if verbose:
        plt.show()
def PlotNewPopulation(grid,SFO_obj,dir_grid,verbose = False):
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    SFO_obj.gdf_polygons.plot(ax=ax, color='white', edgecolor='black',alpha = 0.2)
    grid.plot(column = 'population', cmap='Greys', facecolor = 'none',alpha = 0.2)
    contour_filled = ax.tricontourf(grid['geometry'].apply(lambda geom: geom.centroid.x), 
                                    grid['geometry'].apply(lambda geom: geom.centroid.y), 
                                    grid['population'], cmap='viridis', alpha=0.5)

    cbar = plt.colorbar(contour_filled)
    cbar.set_label('Population')
    ax.set_title('Mass Distribution')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.savefig(os.path.join(dir_grid,'Population_Distribution.png'),dpi = 200)
    if verbose:
        print('+++++ Plot Population Distribution +++++')
        print('Minimum population: ',min(grid['population'])," Maximum population: ",max(grid['population']))

        plt.show()

def PlotFluxes(grid,Tij,SFO_obj,dir_grid,top_fluxes = 50,verbose=False):
    fig,ax = plt.subplots(1,1, figsize = (10,10))
    SFO_obj.gdf_polygons.plot(ax=ax, color='white', edgecolor='black',alpha = 0.2)
    highest_row = Tij.nlargest(top_fluxes, 'number_people')
    # Extract indices 'i' and 'j' from the highest row
    highest_i = highest_row['origin'].to_numpy()
    highest_j = highest_row['destination'].to_numpy()
    fluxes = highest_row['number_people'].to_numpy()/max(highest_row['number_people'].to_numpy())
    norm = plt.Normalize(fluxes.min(), fluxes.max())
    if verbose:
        print("+++++ Plot Fluxes +++++")
        print('Type origin: ',type(highest_i))
        print('Type destination: ',type(highest_j))
        print('Type fluxes: ',type(fluxes))

    for grid_index in range(len(highest_i)):
        if verbose:
            print('Grid index: ',grid_index)
            print('i: ',highest_i[grid_index],' j: ',highest_j[grid_index])
            print('x of i: ',grid.loc[grid['index']==highest_i[grid_index]]['centroidx'].values[0])
            print('y of i: ',grid.loc[grid['index']==highest_i[grid_index]]['centroidy'].values[0])
            print('x of j: ',grid.loc[grid['index']==highest_j[grid_index]]['centroidx'].values[0])
            print('y of j: ',grid.loc[grid['index']==highest_j[grid_index]]['centroidy'].values[0])
        pointi = [grid.loc[grid['index']==highest_i[grid_index]]['centroidx'].values[0],grid.loc[grid['index']==highest_i[grid_index]]['centroidy'].values[0]]
        pointj = [grid.loc[grid['index']==highest_j[grid_index]]['centroidx'].values[0],grid.loc[grid['index']==highest_j[grid_index]]['centroidy'].values[0]]
        if pointi == pointj:
            ax.scatter(pointi[0],pointi[1],marker = 'o',color = 'r')
        ax.plot([pointi[0], pointj[0]], 
             [pointi[1], pointj[1]], 
             color=plt.cm.viridis(norm(fluxes[grid_index])), alpha=0.5, linewidth=4)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])    
    plt.colorbar(sm, label='Flux',ax=plt.gca())         
    plt.savefig(os.path.join(dir_grid,'Fluxes.png'),dpi = 200)
    if verbose:        
        gammas = [1,5,10,20,30,50,100]
        for gamma in gammas:
            print("Number of people in grid with flux > ",gamma,": ",(Tij['number_people'].to_numpy()>gamma).sum())
            print("Number of couples of grids with flux > ",gamma,": ",len(Tij['number_people'].to_numpy()[Tij['number_people'].to_numpy()>gamma]))
            print("Fraction of couples of grids with flux > ",gamma,": ",len(Tij['number_people'].to_numpy()[Tij['number_people'].to_numpy()>gamma])/len(Tij['number_people'].to_numpy()))
        plt.show()
        

def PotentialContour(grid,PotentialDataframe,SFO_obj,dir_grid,verbose = False):
    # Assuming you have a GeoDataFrame named 'grid' with a 'geometry' column containing polygons and a 'potential' column
    if 'potential' in grid.columns:
        pass
    else:
        grid['potential'] = PotentialDataframe['V_out']    
    grid['potential'] = PotentialDataframe['V_out']
    # Create a contour plot
    fig, ax = plt.subplots(figsize=(20, 20))
    SFO_obj.gdf_polygons.plot(ax=ax, color='white', edgecolor='black',alpha = 0.2)
    grid.plot(ax=ax, edgecolor='black', facecolor='none',alpha = 0.2)
    contour_filled = ax.tricontourf(grid['geometry'].apply(lambda geom: geom.centroid.x), 
                                    grid['geometry'].apply(lambda geom: geom.centroid.y), 
                                    grid['potential'], cmap='inferno', alpha=0.5)

    # Create contour lines
    contour_lines = ax.tricontour(grid['geometry'].apply(lambda geom: geom.centroid.x), 
                                grid['geometry'].apply(lambda geom: geom.centroid.y), 
                                grid['potential'], colors='black')

    #contour = ax.tricontour(grid['geometry'].apply(lambda geom: geom.centroid.x), 
    #                         grid['geometry'].apply(lambda geom: geom.centroid.y), 
    #                         grid['potential'], alpha=1, cmap='inferno')
    cbar = plt.colorbar(contour_filled)
    cbar.set_label('Potential')
    ax.set_title('Curve Level of Potential')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.savefig(os.path.join(dir_grid,'CountorPlot.png'),dpi = 200)
    if verbose:
        plt.show()

def PotentialSurface(grid,SFO_obj,PotentialDataframe,dir_grid,verbose = False):
    # Assuming you have a GeoDataFrame named 'grid' with a 'geometry' column containing polygons and a 'potential' column
    '''
        Draws the Potential surface considering just the potential inside and discarding the one outside.
    '''
    if 'potential' in grid.columns:
        pass
    else:
        grid['potential'] = PotentialDataframe['V_out']
    x = np.linspace(min(grid.centroidx), max(grid.centroidx), len(np.unique(grid['j'])))
    y = np.linspace(min(grid.centroidy), max(grid.centroidy), len(np.unique(grid['i'])))
    X, Y = np.meshgrid(x, y)
    Z = grid['potential'].values.reshape((len(y),len(x)))# Check the lengths of x, y, and the potential values array
    # Set Potential Null From Outside
    PositionOutside = grid['position'].values.reshape((len(y),len(x)))
    MaskOutside = [[True if PositionOutside[j][i] == 'inside' else False for i in range(len(x))] for j in range(len(y))]
    Z[MaskOutside] = 0
    # Create a contour plot
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    #polygon = SFO_obj.gdf_polygons.to_crs(grid.crs)  # Ensure polygon has the same CRS as the grid
    #polygon_patch = polygon.boundary.plot(ax=ax, color='black', alpha=0.5)
    #polygon_patch.set_zorder(10)  # Ensure polygon is plotted above the surface

    cbar = fig.colorbar(surf, ax=ax)
    cbar.set_label('Potential Height')
    ax.set_title('3D Surface Plot of Potential')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Potential')
    plt.savefig(os.path.join(dir_grid,'Potential3D.png'),dpi = 200)
    if verbose:
        plt.show()

def PlotRotorDistribution(grid,PotentialDataframe,dir_grid,verbose = False):
    if 'rotor' in grid.columns:
        pass
    else:
        grid['rotor'] = PotentialDataframe['rotor_z_out']
    fig, ax = plt.subplots(figsize=(20, 20))
    twin = ax.twinx()
    ax.hist(grid['rotor'],bins = 50, color = 'blue',label = 'Rotor')
    ax.set_title('Rotor Distribution')
    ax.set_xlabel('Rotor')
    ax.set_ylabel('Count')
    plt.savefig(os.path.join(dir_grid,'RotorDistr.png'),dpi = 200)
    if verbose:
        plt.show()

def PlotLorenzCurve(cumulative,Fstar,result_indices,dir_grid,shift = 0.1,verbose = False):
    
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    x = np.arange(len(cumulative))/len(cumulative)
    idxFstar = Fstar #int(Fstar*len(cumulative))
    line1, = ax.plot(x,cumulative,c='black',label='Potential')
    # Plot the straight line to F*
    line2, = ax.plot([x[idxFstar], 1], [0, cumulative[-1]], color='red',label = 'Potential angle')
    ax.plot(x[idxFstar],cumulative[idxFstar],'ro',label='Potential F*')
    if result_indices is not None:
        ax.text(x[Fstar] + shift, 0, f'I* = {x[Fstar]:.2f}', ha='right', va='bottom', color='black')
        ax.text(x[Fstar] + 2*shift , 0.1, f'Centers', ha='right', va='bottom', color='green')
        ax.text(x[Fstar] - 1.5*shift , 0.1, f'No Centers', ha='right', va='bottom', color='yellow')        
        ax.axhline(y=0.05, xmin=0 , xmax=(x[Fstar]), color='yellow', linestyle='--')
        ax.axhline(y=0.05, xmin=x[Fstar], xmax=1, color='green', linestyle='--')
        if verbose:
            print(Fstar)
    ax.set_ylim(0)
    ax.set_title('Lorenz Curve Potential')
    ax.set_xlabel('Index sorted grid')
    ax.set_ylabel('Cumulative Potential')
    plt.savefig(os.path.join(dir_grid,'LorenzCurve.png'),dpi = 200)
    if verbose:
        plt.show()
        print('index Fstar: ',Fstar)
        print('cumulative: ',cumulative)
        print('x: ',x)
        print('x[idxFstar]: ',x[idxFstar])
    return line1,line2


##-------- ##