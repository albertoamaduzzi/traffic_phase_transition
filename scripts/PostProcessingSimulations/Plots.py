import matplotlib.pyplot as plt
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import imageio

def PlotPeopleInNetwork(Interval2NumberPeopleInNet,interval15,FileName):
    '''
        Input:
            df_people: DataFrame containing the people information (stay time in the network, time of departure, travel time, etc.)
            save_dir: Directory where the plot will be saved
            name: Name of the plot
            hour_in_day: Number of hours in a day
            minutes_in_hour: Number of minutes in an hour
            seconds_in_minute: Number of seconds in a minute
            interval_in_minutes: Number of minutes in each interval
    '''
    fig,ax = plt.subplots(1,1,figsize = (15,15))
    Interval2Plot = [t for t in interval15 if Interval2NumberPeopleInNet[t] > 0]
    NPeople2Count = [Interval2NumberPeopleInNet[t] for t in interval15 if Interval2NumberPeopleInNet[t] > 0]
    ax.plot(Interval2Plot,NPeople2Count)
#    ax.set_xticks(second2hour(interval15))
#    ax.set_xticklabels([str(t) for t in second2hour(interval15)])
    ax.set_xlabel('time')
    ax.set_ylabel('Number people in graph')
    plt.savefig(FileName,dpi = 200)
    plt.show()

def PlotTrafficInGeopandasNet(TrafficGdf,TrafficLevel,ColorBarExplanation,PlotFile,Title,dpi = 300,IsLognorm = False):
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    gs = GridSpec(1, 2, width_ratios=[20, 1])
    ax = fig.add_subplot(gs[0,0],projection=gcrs.AlbersEqualArea())
    if IsLognorm:
        gplt.sankey(
            TrafficGdf,
            scale= TrafficLevel,
            limits=(0.1, 10),
            hue= TrafficLevel,
            cmap = 'inferno',
            norm = LogNorm(),
            ax=ax  # Use the created axes
        )
    else:
        gplt.sankey(
            TrafficGdf,
            scale= TrafficLevel,
            limits=(0.1, 10),
            hue= TrafficLevel,
            cmap = 'inferno',
            ax=ax  # Use the created axes
        )
    cax = fig.add_subplot(gs[0,1])
    # Create a ScalarMappable object for the colorbar
    sm = plt.cm.ScalarMappable(cmap='inferno', norm=LogNorm(vmin=TrafficGdf[TrafficLevel].min(), vmax=TrafficGdf[TrafficLevel].max()))
    # Empty array for the data range
    sm.set_array([])
    # Add the colorbar to the figure
    cbar = fig.colorbar(sm, cax=cax)
    # Set the colorbar label
    cbar.set_label(ColorBarExplanation)
    ax.set_title(Title)
    plt.savefig(PlotFile,dpi = dpi)

    plt.show()
    return PlotFile

def AnimateNetworkTraffic(PlotFiles,AnimationFile,TrafficGdf,TrafficLevel,ColorBarExplanation,Title,dpi = 300,IsLognorm = False):
    images = []
    for PlotFile in PlotFiles:
        PlotTrafficInGeopandasNet(TrafficGdf,TrafficLevel,ColorBarExplanation,PlotFile,Title,dpi,IsLognorm)
        images.append(imageio.imread(PlotFile))
    imageio.mimsave(AnimationFile, images, duration = 0.5)
    return 'movie.gif'

