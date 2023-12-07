import matplotlib

matplotlib.use('Agg')#.use('TkAgg')#.use('GTK3Agg')
#import gi
#gi.require_version('Gtk', '3.0')
#from gi.repository import Gtk
#Gtk.init()
#Gtk.Window()
from termcolor import cprint
import matplotlib.pyplot as plt
import matplotlib.lines as lines
#from gi.repository import Gtk, Gdk, GdkPixbuf, GObject, GLib
import numpy as np
import os
from global_functions import ifnotexistsmkdir
#os.environ['DISPLAY'] = 'localhost:0.0'
# FROM PROJECT
'''
    Color Legend:
        Blue: general node
        Yellow: End point
        Brown: Old Attracting node
        Grey: New Attracting node
        In graph: Cyan
'''

def default_axis(ax,title,xmin = -10,xmax = 10,ymin = -10,ymax = 10):
    ax.grid()
    ax.set_aspect('equal')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_title(title)
    return ax

def plot_old_attractors(planar_graph,ax,debug = False):
    ids = [planar_graph.graph.vp['id'][v] for v in planar_graph.old_attracting_vertices]
    coords = [np.array([planar_graph.graph.vp['x'][v],planar_graph.graph.vp['y'][v]]) for v in planar_graph.old_attracting_vertices]
    r = planar_graph.rate_growth
    ax = default_axis(ax,'old attracting vertices')
    if debug:
        cprint('old attracting vertices','light_green','on_white')
        for id_ in ids:
            cprint('old vertex: ' + str(id_),'light_green','on_white')

    if len(coords)!=0:
        if len(coords)==1:
            ax.scatter(coords[0],coords[1],color = 'brown')
        else:
            ax.scatter(coords[:][0],coords[:][1],color = 'brown')
        for av in range(len(coords)):
            ax.text(coords[av][0],coords[av][1], f'({ids[av]})', verticalalignment='bottom', horizontalalignment='center', fontsize=10)
            circle1 = plt.Circle(coords[av], r, color='black', linestyle = '--',fill=False ,alpha = 0.1)
            ax.add_artist(circle1)
    else:
        pass    
    ids = [planar_graph.graph.vp['id'][v] for v in planar_graph.end_points]
    coords = [np.array([planar_graph.graph.vp['x'][v],planar_graph.graph.vp['y'][v]]) for v in planar_graph.end_points]
    if len(coords)!=0:
        if len(coords)==1:
            ax.scatter(coords[0][0],coords[0][1],color = 'yellow')
        else:
            ax.scatter(coords[:][0],coords[:][1],color = 'yellow')
        for av in range(len(coords)):
            ax.text(coords[av][0],coords[av][1], f'({ids[av]})', verticalalignment='bottom', horizontalalignment='center', fontsize=10)
            circle1 = plt.Circle(coords[av], r, color='black', linestyle = '--',fill=False ,alpha = 0.1)
            ax.add_artist(circle1)
        if debug:
            for id_ in ids:
                cprint('end point: ' + str(id_),'light_green','on_white')
    else:
        pass   
    ax.legend(['old','end points']) 
    return ax  

       
def plot_new_attractors(planar_graph,ax,debug=False):
    r = planar_graph.rate_growth
    ids = [planar_graph.graph.vp['id'][v] for v in planar_graph.newly_added_attracting_vertices]
    coords = [np.array([planar_graph.graph.vp['x'][v],planar_graph.graph.vp['y'][v]]) for v in planar_graph.newly_added_attracting_vertices]
    ax = default_axis(ax,'new attracting vertices')    
    if debug:
        cprint('new attracting vertices','light_green','on_white')
        for id_ in ids:
            cprint('new vertex: ' + str(id_),'light_green','on_white')

    if len(coords)!=0:        
        if len(coords)==1:
            ax.scatter(coords[0][0],coords[0][1],color = 'grey')
        else:
            ax.scatter(coords[:][0],coords[:][1],color = 'grey')
            for av in range(len(coords)):
                ax.text(coords[av][0],coords[av][1], f'({ids[av]})', verticalalignment='bottom', horizontalalignment='center', fontsize=10)
    else:
        pass
    coords = [np.array([planar_graph.graph.vp['x'][v],planar_graph.graph.vp['y'][v]]) for v in planar_graph.graph.vertices() if v in planar_graph.is_in_graph_vertices]
    if len(coords)!=0:
        if len(coords)==1:
            ax.scatter(coords[0][0],coords[0][1],color = 'cyan')
        else:
            ax.scatter(coords[:][0],coords[:][1],color = 'cyan')
        for av in range(len(coords)):
            circle1 = plt.Circle(coords[av], r, color='black', linestyle = '--',fill=True ,alpha = 0.1)
            ax.add_artist(circle1)
        if debug:
            for id_ in planar_graph.graph.vertices():
                cprint('vertex: ' + str(planar_graph.graph.vp['id'][id_]),'light_green','on_white')

    else:
        pass
    ax.legend(['new','in graph'])
    return ax

def plot_active_vertices(planar_graph,ax):
    default_axis(ax,'active vertices')
    ids = [planar_graph.graph.vp['id'][v] for v in planar_graph.active_vertices]
    coords = [np.array([planar_graph.graph.vp['x'][v],planar_graph.graph.vp['y'][v]]) for v in planar_graph.active_vertices]
    if len(coords)!=0:
        ax.scatter(coords[:][0],coords[:][1],color = 'white')
        for av in range(len(coords)):
            ax.text(coords[av][0],coords[av][1], f'({ids[av]})', verticalalignment='bottom', horizontalalignment='center', fontsize=10)
    else:
        pass
    return ax


def plot_graph(planar_graph,ax):  
    ax = default_axis(ax,'evolving graph')        
    for v in planar_graph.important_vertices:
#        ax.scatter(planar_graph.graph.vp['x'][v],planar_graph.graph.vp['y'][v],color = 'red')   
        for r in planar_graph.graph.vp['roads'][v]:
            for edge in r.list_edges:
                ax.add_artist(lines.Line2D([planar_graph.graph.vp['x'][edge[0]],planar_graph.graph.vp['x'][edge[1]]],[planar_graph.graph.vp['y'][edge[0]],planar_graph.graph.vp['y'][edge[1]]]))

    return ax
def plot_rn(planar_graph,vi,ax,debug = False):
    '''
        Plots: 
            1) growing node
            2) Circle of attraction
            3) Relative neighbors
            4) All the other nodes
    '''
    default_axis(ax,'relative neighbors')
    coordinates_all = np.array([planar_graph.graph.vp['x'].a ,planar_graph.graph.vp['y'].a]).T
    coordinates_vi = np.array([planar_graph.graph.vp['x'][vi],planar_graph.graph.vp['y'][vi]])
    ax.scatter(coordinates_all[:,0],coordinates_all[:,1],color = 'blue')
    ax.scatter(planar_graph.graph.vp['x'][vi],planar_graph.graph.vp['y'][vi],color = 'red')
    if debug:
        cprint('relative neighbors','light_green','on_white')
        cprint('vertex: ' + str(planar_graph.graph.vp['id'][vi]),'light_green','on_white')
        for vj in planar_graph.graph.vp['relative_neighbors'][vi]:
            cprint('relative neighbor: ' + str(planar_graph.graph.vp['id'][planar_graph.graph.vertex(vj)]),'light_green','on_white')
    for vj in planar_graph.graph.vp['relative_neighbors'][vi]:
        vertex_vj = planar_graph.graph.vertex(vj)
        if vertex_vj in planar_graph.newly_added_attracting_vertices:
            ax.scatter(planar_graph.graph.vp['x'][vertex_vj],planar_graph.graph.vp['y'][vertex_vj],color = 'grey')
        elif vertex_vj in planar_graph.old_attracting_vertices:
            ax.scatter(planar_graph.graph.vp['x'][vertex_vj],planar_graph.graph.vp['y'][vertex_vj],color = 'brown')
        coordinates_vj = np.array([planar_graph.graph.vp['x'][vertex_vj],planar_graph.graph.vp['y'][vertex_vj]])
        r = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][vi]][vj]
        circle1 = plt.Circle(coordinates_vi, r, color='red', linestyle = '--',fill=True ,alpha = 0.1)
        circle2 = plt.Circle(coordinates_vj, r, color='green', linestyle = '--',fill=True ,alpha = 0.1)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        ax.scatter(planar_graph.graph.vp['x'][planar_graph.graph.vertex(vj)],planar_graph.graph.vp['y'][planar_graph.graph.vertex(vj)],color = 'green')
#        ax.plot([planar_graph.graph.vp['x'][vi],planar_graph.graph.vp['x'][planar_graph.graph.vertex(vj)]],[planar_graph.graph.vp['y'][vi],planar_graph.graph.vp['y'][planar_graph.graph.vertex(vj)]],linestyle = '--',color = 'green')
    ax.legend(['all','growing','old attracting','relative neighbors','circle of attraction'])
    return ax
def plot_old_delauney(planar_graph,ax):
    '''
        Delauney graph of the old attracting vertices
    '''
    default_axis(ax,'delauny triangle old important nodes')
    if len(planar_graph.old_attracting_vertices)!=0:
        x = [planar_graph.graph.vp['x'][v] for v in planar_graph.graph.vertices() if v in planar_graph.end_points or v in planar_graph.old_attracting_vertices]
        y = [planar_graph.graph.vp['y'][v] for v in planar_graph.graph.vertices() if v in planar_graph.end_points or v in planar_graph.old_attracting_vertices]
        ax.triplot(np.array([x,y]).T[:,0],np.array([x,y]).T[:,1])    
    else:
        pass
    return ax

def plot_new_delauney(planar_graph,ax): 
    '''
        Delauney graph of the new attracting vertices
    '''
    default_axis(ax,'delauny triangle new important nodes')
    if len(planar_graph.newly_added_attracting_vertices)!=0:
        x = [planar_graph.graph.vp['x'][v] for v in planar_graph.graph.vertices()]
        y = [planar_graph.graph.vp['y'][v] for v in planar_graph.graph.vertices()]
        ax.triplot(np.array([x,y]).T[:,0],np.array([x,y]).T[:,1])    
    else:
        pass
    return ax


def plot_relative_neighbors(planar_graph,vi,new_added_vertex,available_vertices,debug = False):
    '''
        Plots:
            1) Vertices that can attract just the end points of the graph (upper-left)
            2) Vertices that can attract any in graph point (upper-right)
            3) Relative neighbors of the attracted vertex (middle-left)
            4) Added vertices (middle-right)
            5) active vertices (lower-left)
            6) graph (lower-right)
            7) Number of roads (lower-left)
            8) Total lenght streets (lower-right)
    '''
    if debug:
        if len(planar_graph.time)!=0:
            time = planar_graph.time[-1] + 1
        else:
            time = 0
        cprint('ITERATION: '+str(time),'light_green','on_white')
        cprint('Debug plots','light_green','on_white')
    fig,ax = plt.subplots(3,2,figsize = (20,20),sharex = True,sharey = True)
    if debug:
        cprint('plot old attractors','light_green','on_white')
    ax[0][0] = plot_old_attractors(planar_graph,ax[0][0])
    if debug:
        cprint('plot new attractors','light_green','on_white')
    ax[0][1] = plot_new_attractors(planar_graph,ax[0][1])
    if debug:
        cprint('plot relative neighbors','light_green','on_white')
    ax[1][0] = plot_rn(planar_graph,vi,ax[1][0])
    if debug:    
        cprint('plot new delauney','light_green','on_white')
    ax[1][1] = plot_new_delauney(planar_graph,ax[1][1])
    if debug:    
        cprint('plot old delauney','light_green','on_white')
    ax[2][0] = plot_old_delauney(planar_graph,ax[2][0])
#    ax[2][0] = plot_active_vertices(planar_graph,ax[2][0])
    if debug:    
        cprint('plot graph','light_green','on_white')
    ax[2][1] = plot_graph(planar_graph,ax[2][1]) 
#    plot_number_roads_time(planar_graph,ax[2][0])
#    plot_total_length_roads_time(planar_graph,ax[2][1])
    ## All attracting vertices
    if debug:
        for v in available_vertices:
            cprint('available vertices inside function: ' + str(planar_graph.graph.vp['id'][v]),'light_green','on_white')
    ifnotexistsmkdir(os.path.join(planar_graph.base_dir,'relative_neighbor'))
#    print(planar_graph.time)
    if len(planar_graph.time)!=0:
        time = planar_graph.time[-1] + 1
    else:
        time = 0
    id_added = planar_graph.graph.vp['id'][new_added_vertex]
    plt.savefig(os.path.join(planar_graph.base_dir,'relative_neighbor','iter_{0}_growing_{1}_added_{2}'.format(time,planar_graph.graph.vp['id'][vi],id_added)))
    plt.close(fig)
'''
def plot_growing_roads(planar_graph):
    fig,ax = plt.subplots(1,1,figsize = (20,20))
    colors = ['red','blue','green','yellow','orange','violet','black','brown','pink','grey','cyan','magenta']
    for starting_vertex in planar_graph.important_vertices:
        for r in planar_graph.graph.vp['roads'][starting_vertex]: 
            for edge in r.list_edges:
                if r.id < len(colors):
                    ax.plot([planar_graph.graph.vp['x'][edge.source()],planar_graph.graph.vp['x'][edge.target()]],[planar_graph.graph.vp['y'][edge.source()],planar_graph.graph.vp['y'][edge.target()]],linestyle = '-',color = colors[r.id])
                else:
                    i = r.id%len(colors) 
                    ax.plot([planar_graph.graph.vp['x'][edge.source()],planar_graph.graph.vp['x'][edge.target()]],[planar_graph.graph.vp['y'][edge.source()],planar_graph.graph.vp['y'][edge.target()]],linestyle = '-',color = colors[i])
    plt.show()

def plot_evolving_graph(planar_graph):
    fig,ax = plt.subplots(1,1,figsize = (20,20))
    for edge in planar_graph.graph.edges():
        if planar_graph.graph.vp['important_node'][edge.source()] == True:
            color_source = 'red'
            ax.scatter(planar_graph.graph.vp['x'][edge.source()],planar_graph.graph.vp['y'][edge.source()],color = color_source)
            ax.plot([planar_graph.graph.vp['x'][edge.source()],planar_graph.graph.vp['x'][edge.target()]],[planar_graph.graph.vp['y'][edge.source()],planar_graph.graph.vp['y'][edge.target()]],linestyle = '-',color = 'black')
    fig.show()
'''
def plot_number_roads_time(planar_graph,ax):
    if len(planar_graph.count_roads)!=0:
        ax.scatter(planar_graph.time,planar_graph.count_roads)
        ax.plot(planar_graph.time,np.array(planar_graph.time) + planar_graph.count_roads[0])
        ax.plot(planar_graph.time,2*np.array(planar_graph.time) + planar_graph.count_roads[0])
        ax.xlabel('time')
        ax.ylabel('number of roads')
        ax.legend(['graph','tree','lattice'])
        ax.set_title('number roads in nodes') 

def plot_total_length_roads_time(planar_graph,ax):
    if len(planar_graph.time)!=0:
        ax.scatter(planar_graph.time,planar_graph.length_total_roads)
        ax.plot(planar_graph.time,np.sqrt(np.array(planar_graph.time))*planar_graph.length_total_roads[0])
        ax.set_xlabel('time')
        ax.set_ylabel('total length (m)')
        ax.legend(['graph','square root'])
        ax.set_title('total length network in time') 

