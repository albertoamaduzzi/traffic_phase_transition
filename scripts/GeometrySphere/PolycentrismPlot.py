import numpy as np
import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
logger = logging.getLogger(__name__)

##------------------------------------- PLOTS -----------------------------------------##

def PrintInfoFluxPop(grid,Tij):
    logger.debug('******************')
    logger.debug('Number of grids with people: ',grid.loc[grid['population']>50].shape[0])
    logger.debug('Number of couples of grids with flux: ',Tij.loc[Tij['number_people']>0].shape[0])
    logger.debug('Total Population: ',np.sum(grid['population']))
    logger.debug('Total Flux: ',np.sum(Tij['number_people']))
    logger.debug('Fraction of grids populated: ',grid.loc[grid['population']>50].shape[0]/grid.shape[0])
    logger.debug('Fraction of couples of grids with fluxes: ',Tij.loc[Tij['number_people']>0].shape[0]/Tij.shape[0])
    logger.debug('******************')

def PlotOldNewFluxes(Tij_new,Tij):
    logger.info('Plotting Old and New Fluxes')
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
#    if verbose:
#        plt.show()


def PlotPositionCenters(grid,gdf_polygons,index_centers,dir_grid,UCI):
    logger.info('Plotting Position Centers')
    fig,ax = plt.subplots(figsize=(10, 10))
    gdf_polygons.plot(ax=ax, color='white', edgecolor='black',alpha = 0.2)
    grid.plot(ax=ax, edgecolor='black', facecolor='none',alpha = 0.2)
    for i in index_centers:
        ax.scatter(grid['geometry'].apply(lambda geom: geom.centroid.x)[i],grid['geometry'].apply(lambda geom: geom.centroid.y)[i],marker = 'x',color = 'r')
    plt.savefig(os.path.join(dir_grid,f'Position_Centers_{round(UCI,3)}.png'),dpi = 200)
#    if verbose:
#        plt.show()
def PlotNewPopulation(grid,gdf_polygons,dir_grid,UCI):
    logger.info('Plotting New Population')
    fig,ax = plt.subplots(1,1,figsize = (8,6))
    gdf_polygons.plot(ax=ax, color='white', edgecolor='black',alpha = 0.2)
    grid.plot(column = 'population', cmap='Greys', facecolor = 'none',alpha = 0.2)
    contour_filled = ax.tricontourf(grid['geometry'].apply(lambda geom: geom.centroid.x), 
                                    grid['geometry'].apply(lambda geom: geom.centroid.y), 
                                    grid['population'], cmap='viridis', alpha=0.5)

    cbar = plt.colorbar(contour_filled)
    cbar.set_label('Population')
    ax.set_title('Mass Distribution')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.savefig(os.path.join(dir_grid,f'Population_Distribution_{round(UCI,3)}.png'),dpi = 200)
#        plt.show()

def PlotInsideOutside(gdf_polygons,grid,dir_grid):
    cmap = mcolors.ListedColormap(['red', 'green'])  # Red for False, Green for True
    norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)
    # Plot the data
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    gdf_polygons.plot(ax=ax, color='white', edgecolor='black', alpha=0.2)
    grid.plot(column='position', ax=ax, alpha=0.5, cmap=cmap, norm=norm, legend=True)
    # Customize the plot
    ax.set_title('position')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.savefig(os.path.join(dir_grid,'InsideOutside.png'),dpi = 200)

def PlotRoads(gdf_polygons,grid,dir_grid):
    cmap = mcolors.ListedColormap(['red', 'green'])  # Red for False, Green for True
    norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)
    # Plot the data
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    gdf_polygons.plot(ax=ax, color='white', edgecolor='black', alpha=0.2)
    grid.plot(column='with_roads', ax=ax, alpha=0.5, cmap=cmap, norm=norm, legend=True)

    # Customize the plot
    ax.set_title('With Roads')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.savefig(os.path.join(dir_grid,'WithRoads.png'),dpi = 200)

def PlotEdges(gdf_polygons,grid,dir_grid):
    cmap = mcolors.ListedColormap(['red', 'green'])  # Red for False, Green for True
    norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)
    # Plot the data
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    gdf_polygons.plot(ax=ax, color='white', edgecolor='black', alpha=0.2)
    grid.plot(column='relation_to_line', ax=ax, alpha=0.5, cmap=cmap, norm=norm, legend=True)

    # Customize the plot
    ax.set_title('Edges')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.savefig(os.path.join(dir_grid,'Edges.png'),dpi = 200)


def PlotFluxes(grid,Tij,gdf_polygons,dir_grid,UCI,top_fluxes = 50):
    logger.info('Plotting Fluxes')
    fig,ax = plt.subplots(1,1, figsize = (8,6))
    gdf_polygons.plot(ax=ax, color='white', edgecolor='black',alpha = 0.2)
    highest_row = Tij.nlargest(top_fluxes, 'number_people')
    # Extract indices 'i' and 'j' from the highest row
    highest_i = highest_row['origin'].to_numpy()
    highest_j = highest_row['destination'].to_numpy()
    fluxes = highest_row['number_people'].to_numpy()/max(highest_row['number_people'].to_numpy())
    norm = plt.Normalize(fluxes.min(), fluxes.max())

    for grid_index in range(len(highest_i)):
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
    plt.savefig(os.path.join(dir_grid,f'Fluxes_{round(UCI,3)}.png'),dpi = 200)
    plt.close()
    gammas = [1,5,10,20,30,50,100]
    for gamma in gammas:
        print("Number of people in grid with flux > ",gamma,": ",(Tij['number_people'].to_numpy()>gamma).sum())
        print("Number of couples of grids with flux > ",gamma,": ",len(Tij['number_people'].to_numpy()[Tij['number_people'].to_numpy()>gamma]))
        print("Fraction of couples of grids with flux > ",gamma,": ",len(Tij['number_people'].to_numpy()[Tij['number_people'].to_numpy()>gamma])/len(Tij['number_people'].to_numpy()))
        
        

def PotentialContour(grid,PotentialDataframe,gdf_polygons,dir_grid,UCI):
    # Assuming you have a GeoDataFrame named 'grid' with a 'geometry' column containing polygons and a 'potential' column
    if 'potential' in grid.columns:
        pass
    else:
        grid['potential'] = PotentialDataframe['V_out']    
    grid['potential'] = PotentialDataframe['V_out']
    # Create a contour plot
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_polygons.plot(ax=ax, color='white', edgecolor='black',alpha = 0.2)
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
    plt.savefig(os.path.join(dir_grid,f'CountorPlot_{round(UCI,3)}.png'),dpi = 200)
#    if verbose:
        #plt.show()

def PotentialSurface(grid,PotentialDataframe,dir_grid,UCI):
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
    plt.savefig(os.path.join(dir_grid,f'Potential3D_{round(UCI,3)}.png'),dpi = 200)
#    if verbose:
        #plt.show()

def PlotRotorDistribution(grid,PotentialDataframe,dir_grid,UCI):
    if 'rotor' in grid.columns:
        pass
    else:
        grid['rotor'] = PotentialDataframe['rotor_z_out']
    fig, ax = plt.subplots(figsize=(8, 6))
    twin = ax.twinx()
    ax.hist(grid['rotor'],bins = 50, color = 'blue',label = 'Rotor')
    ax.set_title('Rotor Distribution')
    ax.set_xlabel('Rotor')
    ax.set_ylabel('Count')
    plt.savefig(os.path.join(dir_grid,f'RotorDistr_{round(UCI,3)}.png'),dpi = 200)
#    if verbose:
#        plt.show()

def PlotHarmonicComponentDistribution(grid,PotentialDataframe,dir_grid,UCI):
    if 'harmonic' in grid.columns:
        pass
    else:
        grid['harmonic'] = PotentialDataframe['HarmonicComponentOut']
    fig, ax = plt.subplots(figsize=(8, 6))
    twin = ax.twinx()
    ax.hist(grid['harmonic'],bins = 50, color = 'blue',label = 'Harmonic')
    ax.set_title('Harmonic Distribution')
    ax.set_xlabel('Harmonic')
    ax.set_ylabel('Count')
    plt.savefig(os.path.join(dir_grid,f'HarmonicDistr_{round(UCI,3)}.png'),dpi = 200)

def PlotLorenzCurve(cumulative,Fstar,result_indices,dir_grid,UCI,shift = 0.1,verbose = False):
    """
        @param cumulative: Cumulative Potential
        @param Fstar: Index of the Lorenz Curve
        @param result_indices: Indices of the Lorenz Curve
        @param dir_grid: Directory to save the plots
        @param UCI: UCI of the simulation
        @param shift: Shift of the text in the plot
        @param verbose: Print information
        @description: Plot the Lorenz Curve of the Potential
    """
    fig,ax = plt.subplots(1,1,figsize = (8,6))
    x = np.arange(len(cumulative))/len(cumulative)
    cumulative = np.array(cumulative)/np.sum(cumulative)
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
    plt.savefig(os.path.join(dir_grid,f'LorenzCurve_{round(UCI,3)}.png'),dpi = 200)
    logger.info(f"F* = {Fstar}")
    return line1,line2

def PlotLorenzCurveMassPot(cumulative,Fstar,result_indices,cumulativeM,FstarM,result_indicesM,dir_grid,UCI,UCIM,shift = 0.1,verbose = False):
    """
        @param cumulative: Cumulative Potential
        @param Fstar: Index of the Lorenz Curve
        @param result_indices: Indices of the Lorenz Curve
        @param dir_grid: Directory to save the plots
        @param UCI: UCI of the simulation
        @param shift: Shift of the text in the plot
        @param verbose: Print information
        @description: Plot the Lorenz Curve of the Potential
    """
    fig,ax = plt.subplots(1,1,figsize = (8,6))
    x = np.arange(len(cumulative))/len(cumulative)
    xm = np.arange(len(cumulativeM))/len(cumulativeM)
    cumulative = np.array(cumulative)/np.sum(cumulative)
    cumulativeM = np.array(cumulativeM)/np.sum(cumulativeM)
    idxFstar = Fstar #int(Fstar*len(cumulative))
    idxFstarM = FstarM #int(Fstar*len(cumulative))
    line1, = ax.plot(x,cumulative,c='black',label='Potential')
    line1m, = ax.plot(xm,cumulativeM,c='blue',label='Mass')
    # Plot the straight line to F*
    line2, = ax.plot([x[idxFstar], 1], [0, cumulative[-1]], color='red',label = 'Potential angle')
    line2m, = ax.plot([xm[idxFstarM], 1], [0, cumulativeM[-1]], color='blue',label = 'Mass angle')
    ax.plot(x[idxFstar],cumulative[idxFstar],'ro',label='Potential F*')
    ax.plot(xm[idxFstarM],cumulativeM[idxFstarM],'bo',label='Mass F*')
    if result_indices is not None and result_indicesM is not None:
        # POT
        ax.text(x[Fstar] + shift, 0, f'I* = {x[Fstar]:.2f}', ha='right', va='bottom', color='black')
        ax.text(x[Fstar] + 2*shift , 0.1, f'Centers', ha='right', va='bottom', color='green')
        ax.text(x[Fstar] - 1.5*shift , 0.1, f'No Centers', ha='right', va='bottom', color='yellow')        
        # MASS
        ax.text(xm[FstarM] + shift, 0.01, f'I* = {xm[FstarM]:.2f}', ha='right', va='bottom', color='black')
        ax.text(xm[FstarM] + 2*shift , 0.11, f'Centers', ha='right', va='bottom', color='green')
        ax.text(xm[FstarM] - 1.5*shift , 0.11, f'No Centers', ha='right', va='bottom', color='yellow')
        # POT
        ax.axhline(y=0.05, xmin=0 , xmax=(x[Fstar]), color='yellow', linestyle='--')
        ax.axhline(y=0.05, xmin=x[Fstar], xmax=1, color='green', linestyle='--')
        # MASS
        ax.axhline(y=0.04, xmin=0 , xmax=(xm[FstarM]), color='yellow', linestyle='--')
        ax.axhline(y=0.04, xmin=xm[FstarM], xmax=1, color='green', linestyle='--')

    ax.set_ylim(0)
    ax.set_title('Lorenz Curve Potential/Mass')
    ax.set_xlabel('Index sorted grid')
    ax.set_ylabel('Cumulative Potential/Mass')
    plt.savefig(os.path.join(dir_grid,f'LorenzCurve_{round(UCI,3)}_{round(UCIM,3)}.png'),dpi = 200)
    logger.info(f"F* = {Fstar}, F*M = {FstarM}, UCIM: {round(UCIM,3)}, UCI: {round(UCI,3)}")
    return line1,line2


##-------- ##

def PlotDistributionDistance(Tij,distance_df):
    fig,ax = plt.subplots(1,1,figsize = (8,6))
    merged_df = pd.merge(Tij, distance_df, on=['origin', 'destination'])
    TijDifferent0 = merged_df["number_people"].loc[merged_df["number_people"]>0] 
    # Repeat the distance value 'number_people' times for each row
    TijDifferent0['repeated_distance'] = TijDifferent0.apply(lambda row: [row['distance']] * row['number_people'], axis=1)
    # Transform the Series of lists into a single Series
    distance_vector = TijDifferent0['repeated_distance'].explode().reset_index(drop=True)    
    ax.hist(distance_vector['reapeted_distance'],bins = 50, color = 'blue',label = 'Distance')
    ax.set_title('Distance Distribution')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Count')
#    plt.show()

def PlotVFPotMass(grid,gdf_polygons,PotentialDataframe,VectorField,dir_grid,UCI,label_potential = 'V_out',label_fluxes = 'Ti',plot_mass = True,verbose = False):
    '''
        NOTE:
            label_potential:    V_in, V_out
            label_fluxes:       Tj  , Ti
        USAGE:
            PlotVFPotMass(grid,SFO_obj,PotentialDataframe,VectorField,label_potential = 'V_out',label_fluxes = 'Ti')
            PlotVFPotMass(grid,SFO_obj,PotentialDataframe,VectorField,label_potential = 'population',label_fluxes = 'Ti')

    '''
    labelf2title = {'Tj': 'Incoming','Ti':'Outgoing'}
    label2save = {'V_in':'Potential','V_out':'Potential','population':'Mass'}
    fig, ax = plt.subplots(figsize=(15, 15))
    centroid_coords = np.array([grid['centroidx'].to_numpy(),grid['centroidy'].to_numpy()])
    centroid_coords = centroid_coords.T
    #grav_vector_field = gravitational_field(fluxes_matrix,normalized_vectors,nv)
    gdf_polygons.plot(ax=ax, color='white', edgecolor='black')
    if label_potential in PotentialDataframe.columns: 
        grid[label_potential] = PotentialDataframe[label_potential]
    elif label_potential in grid.columns:
        pass
    else:
        raise KeyError(label_potential,' Neither in grid nor potential columns: ',grid.columns,PotentialDataframe.columns)
    if plot_mass:
        grid_plot = grid.plot(ax=ax, column = label_potential, cmap = 'viridis',edgecolor='black', alpha=0.3)
        grid_cbar = plt.colorbar(grid_plot.get_children()[1], ax=ax)
        grid_cbar.set_label('{}'.format(label_potential), rotation=270, labelpad=15)
    else:
        pass
    if type(VectorField[label_fluxes].iloc[0]) == str:
        VF = np.array([np.fromstring(vector_str.strip('[]'), sep=' ') for vector_str in VectorField[label_fluxes]])
    else:
        VF = np.stack(VectorField[label_fluxes].to_numpy(dtype = np.ndarray))
    VF_norm = np.linalg.norm(VF, axis=1)
    VF_normalized = np.stack(np.array([VF[i] / VF_norm[i] if VF_norm[i] !=0 else [0, 0] for i in range(len(VF_norm))]))
    mask = [True if VF_norm[i]!=0 else False for i in range(len(VF_norm))]
    quiver_plot = ax.quiver(centroid_coords[mask,0], centroid_coords[mask,1], VF_normalized[mask,0], VF_normalized[mask,1],
            VF_norm[mask], cmap='inferno_r', angles='xy', scale_units='xy', scale=60, width=0.005,headwidth=1, headlength=3)
#    quiver_plot = ax.quiver(centroid_coords[:,0], centroid_coords[:,1], VF_normalized[:,0], VF_normalized[:,1],
#            VF_norm, cmap='inferno', angles='xy', scale_units='xy', scale=50, width=0.005,headwidth=1, headlength=2)
    quiver_cbar = plt.colorbar(quiver_plot, ax=ax)
    quiver_cbar.set_label('Normalized Vector Magnitude', rotation=270, labelpad=15)
    ax.set_title('Vector Field {} Fluxes'.format(labelf2title[label_fluxes]))
    plt.savefig(os.path.join(dir_grid,f'{labelf2title[label_fluxes]}_Flux_{label2save[label_potential]}_{round(UCI,3)}.png'),dpi = 200)

def PlotRoutineOD(grid,Tij,gdf_polygons,PotentialDataFrame,VectorField,dir_grid,fraction_fluxes,UCI,index_centers,Tij1,cumulative,Fstar,result_indices):
    """
        @param grid: DataFrame with the population of each grid 
        @param Tij: DataFrame with the fluxes between each grid
        @param SFO_obj: Object with the information of the city
        @param PotentialDataFrame: DataFrame with the potential of each grid
        @param VectorField: DataFrame with the vector field of each grid
        @param dir_grid: Directory to save the plots
        @param fraction_fluxes: Fraction of fluxes to plot
        @param verbose: Print information
    """
    PlotFluxes(grid,Tij,gdf_polygons,dir_grid,UCI,fraction_fluxes)
    if index_centers is not None:
        PlotPositionCenters(grid,gdf_polygons,index_centers,dir_grid,UCI)
    PlotNewPopulation(grid, gdf_polygons,dir_grid,UCI)
    # Comparison New Fluxes and From File
    if Tij1 is not None:
        PlotOldNewFluxes(Tij,Tij1)
    PlotVFPotMass(grid,gdf_polygons,PotentialDataFrame,VectorField,dir_grid,UCI,'population','Ti')
    PotentialContour(grid,PotentialDataFrame,gdf_polygons,dir_grid,UCI)
#    PotentialSurface(grid,PotentialDataFrame,dir_grid,UCI)
    PlotRotorDistribution(grid,PotentialDataFrame,dir_grid,UCI)
    PlotLorenzCurve(cumulative,Fstar,result_indices,dir_grid, UCI,0.1)
    PlotHarmonicComponentDistribution(grid,PotentialDataFrame,dir_grid,UCI)


def PlotDepTime(Df, CityName, PlotDir):
    from collections import Counter
    import datetime
    time2c = Counter(Df["dep_time"])
    times = [datetime.datetime.fromtimestamp(int(time)).strftime("%Y-%m-%d %H:%M:%S") for time in time2c.keys()]
    counts = list(time2c.values())

    fig, ax = plt.subplots()
    ax.scatter(times, counts, label='Dep Time')

    # Set x-axis labels, rotate them, and choose one every 20
    ax.set_xticks(np.arange(0, len(times)))
    ax.set_xticklabels(times[::30], rotation=90)

    ax.set_xlabel('Time')
    ax.set_ylabel('Number of People')
    plt.tight_layout()
    plt.savefig(os.path.join(PlotDir, f'{CityName}_DepTime.png'))
    plt.close()