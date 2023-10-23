import matplotlib.pyplot as plt
import networkx as nx
from contextily import add_basemap


def xy_plot(x,y,xlabel,ylabel,title,set_logx=False,set_logy=False):
    '''
        Input:
            x = x values
            y = y values
            xlabel = x label
            ylabel = y label
            title = title of the plot
            set_logx = True/False
            set_logy = True/False
    '''
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    ax.plot(x,y)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if set_logx:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')
    if set_logy:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    ax.grid()
    plt.show()



def plot_adjacent_nx_maps(G_gdf,G_nx,number_axes):
    '''
        Input:
            G_gdf = geopandas graph
            G_nx = networkx graph
            number_axes = number of adjacent axes
    '''    
    f, ax = plt.subplots(1, number_axes, figsize=(12, 6), sharex=True, sharey=True)
    G_gdf.plot(color="k", ax=ax[0])
    for i, facet in enumerate(ax):
        facet.set_title(("Streets", "Graph")[i])
        facet.axis("off")
        add_basemap(facet)
    nx.draw(G_nx, {n: [n[0], n[1]] for n in list(G_nx.nodes)}, ax=ax[1], node_size=50)
