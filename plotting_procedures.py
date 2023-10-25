import matplotlib.pyplot as plt
import numpy as np
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

def edges_over_nodes(nodes,edges,path_to_save=None):
    '''
        Figure 1 (a)
        Modeling Urban Street Patterns Marc Barthélemy and Alessandro Flammini
        Expected E(N) = aplha*N  (alpha in [1,2] [tree-like,regular lattice])
        Input:
            nodes: list(np.array) int = number of nodes (for each different reazlization of a graph)
            edges: list(np.array) int = number of edges (for each different reazlization of a graph)
    '''
    from scipy.stats import linregress 
    from os.path import join
    _,ax = plt.subplots(1,1,figsize=(12,6))
    N = np.linspace(0,max(nodes))
    ax.scatter(np.array(nodes,dtype = int),np.array(edges,dtype = int),xlabel='Number of nodes',ylabel='Number of edges',title='Edges over nodes')
    ax.plot(N,linregress(nodes,edges).slope*N+linregress(nodes,edges).intercept,'b--')
    ax.plot(N,N,'r--')
    ax.plot(N,2*N,'g--')
    ax.legend(['fit','tree','lattice'])
    ax.savefig(join(path_to_save,'edges_over_nodes.png'),dpi = 150)
    plt.show()

def nodes_total_length(nodes,total_length,path_to_save=None):
    '''
        Figure 1 (b)
            Modeling Urban Street Patterns Marc Barthélemy and Alessandro Flammini
        Input:
            nodes: list int = number of nodes (for each different reazlization of a graph)
            total_length: list double = total length streets (for each different reazlization of a graph)
        Description:
            Expected total_length[i] = nodes[i]^(\frac{1}{2}))   
    '''
    from os.path import join
    _,ax = plt.subplots(1,1,figsize=(12,6))
    N = np.linspace(0,max(nodes))
    ax.scatter(np.array(nodes,dtype = int),np.array(total_length,dtype=float),xlabel='Number of nodes',ylabel='Number of edges',title='Edges over nodes')
    ax.plot(N,np.sqrt(N),'b--')
    ax.savefig(join(path_to_save,'edges_over_nodes.png'),dpi=150)
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
