import matplotlib
matplotlib.use('Agg')#.use('TkAgg')#.use('GTK3Agg')
#import gi
#gi.require_version('Gtk', '3.0')
#from gi.repository import Gtk
#Gtk.init()
#Gtk.Window()

import matplotlib.pyplot as plt
import matplotlib.lines as lines
#from gi.repository import Gtk, Gdk, GdkPixbuf, GObject, GLib
import numpy as np
import os
from global_functions import ifnotexistsmkdir
#os.environ['DISPLAY'] = 'localhost:0.0'
# FROM PROJECT

def plot_old_attractors(planar_graph,ax):
    ids = [planar_graph.graph.vp['id'][v] for v in planar_graph.old_attracting_vertices]
    coords = [np.array([planar_graph.graph.vp['x'][v],planar_graph.graph.vp['y'][v]]) for v in planar_graph.old_attracting_vertices]
    if len(coords)!=0:
        ax.scatter(coords[:][0],coords[:][1],color = 'white')
        for av in range(len(coords)):
            ax.text(coords[av][0],coords[av][1], f'({ids[av]})', verticalalignment='bottom', horizontalalignment='center', fontsize=10)
        ax.set_title('old attracting vertices')
    else:
        pass       
    return ax         
def plot_new_attractors(planar_graph,ax):
    ids = [planar_graph.graph.vp['id'][v] for v in planar_graph.newly_added_attracting_vertices]
    coords = [np.array([planar_graph.graph.vp['x'][v],planar_graph.graph.vp['y'][v]]) for v in planar_graph.newly_added_attracting_vertices]
    if len(coords)!=0:
        ax.scatter(coords[:][0],coords[:][1],color = 'white')
        for av in range(len(coords)):
            ax.text(coords[av][0],coords[av][1], f'({ids[av]})', verticalalignment='bottom', horizontalalignment='center', fontsize=10)
        ax.set_title('newly attracting vertices')
    else:
        ax.set_title('newly attracting vertices')
        pass
    return ax

def plot_active_vertices(planar_graph,ax):
    ids = [planar_graph.graph.vp['id'][v] for v in planar_graph.active_vertices]
    coords = [np.array([planar_graph.graph.vp['x'][v],planar_graph.graph.vp['y'][v]]) for v in planar_graph.active_vertices]
    if len(coords)!=0:
        ax.scatter(coords[:][0],coords[:][1],color = 'white')
        for av in range(len(coords)):
            ax.text(coords[av][0],coords[av][1], f'({ids[av]})', verticalalignment='bottom', horizontalalignment='center', fontsize=10)
        ax.set_title('active vertices')
    else:
        ax.set_title('active vertices')
        pass
    return ax
def plot_graph(planar_graph,ax):  
    for v in planar_graph.important_vertices:
        ax.scatter(planar_graph.graph.vp['x'][v],planar_graph.graph.vp['y'][v],color = 'red')   
        for r in planar_graph.graph.vp['roads'][v]:
            for edge in r.list_edges:
                ax.add_artist(lines.Line2D([planar_graph.graph.vp['x'][edge[0]],planar_graph.graph.vp['x'][edge[1]]],[planar_graph.graph.vp['y'][edge[0]],planar_graph.graph.vp['y'][edge[1]]]))
    return ax
def plot_rn(planar_graph,vi,ax):
    '''
        Plots: 
            1) growing node
            2) Circle of attraction
            3) Relative neighbors
            4) All the other nodes
    '''
    coordinates_all = np.array([planar_graph.graph.vp['x'].a ,planar_graph.graph.vp['y'].a]).T
    coordinates_vi = np.array([planar_graph.graph.vp['x'][vi],planar_graph.graph.vp['y'][vi]])
    ax.scatter(coordinates_all[:,0],coordinates_all[:,1])
    ax.scatter(planar_graph.graph.vp['x'][vi],planar_graph.graph.vp['y'][vi],color = 'red')
    for vj in planar_graph.graph.vp['relative_neighbors'][vi]:
        vertex_vj = planar_graph.graph.vertex(vj)
        coordinates_vj = np.array([planar_graph.graph.vp['x'][vertex_vj],planar_graph.graph.vp['y'][vertex_vj]])
        r = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][vi]][vj]
        circle1 = plt.Circle(coordinates_vi, r, color='red', linestyle = '--',fill=True ,alpha = 0.1)
        circle2 = plt.Circle(coordinates_vj, r, color='green', linestyle = '--',fill=True ,alpha = 0.1)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        ax.scatter(planar_graph.graph.vp['x'][planar_graph.graph.vertex(vj)],planar_graph.graph.vp['y'][planar_graph.graph.vertex(vj)],color = 'green')
#        ax.plot([planar_graph.graph.vp['x'][vi],planar_graph.graph.vp['x'][planar_graph.graph.vertex(vj)]],[planar_graph.graph.vp['y'][vi],planar_graph.graph.vp['y'][planar_graph.graph.vertex(vj)]],linestyle = '--',color = 'green')
    rel_neighbor = [planar_graph.graph.vertex(vj) in planar_graph.graph.vp['relative_neighbors'][vi]]
    other_vertices = np.array([planar_graph.graph.vp['pos'][v] for v in planar_graph.is_in_graph_vertices if v !=vi and v not in rel_neighbor])
    if len(other_vertices)!=0:
        ax.scatter(other_vertices[:,0],other_vertices[:,1])
    else:
        pass
    ax.legend(['red = growing','green = relative neighbors'])
    ax.set_title('relative neighbors')
    return ax
def plot_old_delauney(planar_graph,ax):
    '''
        Delauney graph of the old attracting vertices
    '''
    if len(planar_graph.old_attracting_vertices)!=0:
        x = [planar_graph.graph.vp['x'][v] for v in planar_graph.graph.vertices() if v in planar_graph.end_points or v in planar_graph.old_attracting_vertices]
        y = [planar_graph.graph.vp['y'][v] for v in planar_graph.graph.vertices() if v in planar_graph.end_points or v in planar_graph.old_attracting_vertices]
        ax.triplot(np.array([x,y]).T[:,0],np.array([x,y]).T[:,1])    
        ax.set_title('delauny triangle old important nodes')
        ax.legend(['old added '])
    else:
        ax.set_title('delauny triangle old important nodes')
    return ax

def plot_new_delauney(planar_graph,ax): 
    '''
        Delauney graph of the new attracting vertices
    '''
    if len(planar_graph.newly_added_attracting_vertices)!=0:
        x = [planar_graph.graph.vp['x'][v] for v in planar_graph.graph.vertices()]
        y = [planar_graph.graph.vp['y'][v] for v in planar_graph.graph.vertices()]
        ax.triplot(np.array([x,y]).T[:,0],np.array([x,y]).T[:,1])    
        ax.set_title('delauny triangle new important nodes')
        ax.legend(['newly added '])
    else:
        ax.set_title('delauny triangle new important nodes')
    return ax


def plot_relative_neighbors(planar_graph,vi,new_added_vertex,available_vertices):
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
    fig,ax = plt.subplots(3,2,figsize = (20,20),sharex = True,sharey = True)
    print('plot old attractors')
    ax[0][0] = plot_old_attractors(planar_graph,ax[0][0])
    print('plot new attractors')
    ax[0][1] = plot_new_attractors(planar_graph,ax[0][1])
    print('plot relative neighbors')
    ax[1][0] = plot_rn(planar_graph,vi,ax[1][0])
    print('plot new delauney')
    ax[1][1] = plot_new_delauney(planar_graph,ax[1][1])
    print('plot old delauney')
    ax[2][0] = plot_old_delauney(planar_graph,ax[2][0])
#    ax[2][0] = plot_active_vertices(planar_graph,ax[2][0])
    print('plot graph')
    ax[2][1] = plot_graph(planar_graph,ax[2][1]) 
#    plot_number_roads_time(planar_graph,ax[2][0])
#    plot_total_length_roads_time(planar_graph,ax[2][1])
    ## All attracting vertices
    print('available vertices inside function: ',available_vertices)
    ifnotexistsmkdir(os.path.join(planar_graph.base_dir,'relative_neighbor'))
#    print(planar_graph.time)
    if len(planar_graph.time)!=0:
        time = planar_graph.time[-1] + 1
    else:
        time = 0
    id_added = planar_graph.graph.vp['id'][new_added_vertex]
    plt.savefig(os.path.join(planar_graph.base_dir,'relative_neighbor','iter_{0}_growing_{1}_added_{2}'.format(time,planar_graph.graph.vp['id'][vi],id_added)))

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

## ANIMATION (DEPRECATED)
'''
def animate_growth(planar_graph):        
    # Setting general parameters
    black = [0, 0, 0, 1]           # Black color (attracting nodes)
    red = [1, 0, 0, 1]             # Red color (important nodes)
    green = [0, 1, 0, 1]           # Green color (growing nodes)
    blue = [0, 0, 1, 1]            # Blue color 
    pos = planar_graph.graph.vp['pos']
    # Generate path save 
    if not os.path.exists(os.path.join(root,'animation')):
        os.mkdir(os.path.join(root,'animation'))
    planar_graph.max_count = 500
    # Generate the graph window
    if not planar_graph.offscreen:
        win = gt.GraphWindow(planar_graph.graph, pos, geometry=(500, 400),
                        edge_color=[0.6, 0.6, 0.6, 1],
                        vertex_fill_color=planar_graph.graph.vp['important_node'],
                        vertex_halo=planar_graph.graph.vp['is_active'],
                        vertex_halo_color=[0.8, 0, 0, 0.6])
    else:
        win = Gtk.OffscreenWindow()
        win.set_default_size(500, 400)
        win.graph = gt.GraphWidget(planar_graph.graph, pos,
                            edge_color=[0.6, 0.6, 0.6, 1],
                            vertex_fill_color=planar_graph.graph.vp['state'],
                            vertex_halo=planar_graph.graph.vp['is_active'],
                            vertex_halo_color=[0.8, 0, 0, 0.6])
        win.add(win.graph)
    # Bind the function above as an 'idle' callback.
    cid = GLib.idle_add(planar_graph.update_state())
    # We will give the user the ability to stop the program by closing the window.
    win.connect("delete_event", Gtk.main_quit)
    # Actually show the window, and start the main loop.
    win.show_all()
    Gtk.main()

def update_state(planar_graph,win):
    # Filter out the recovered vertices
    planar_graph.graph.set_vertex_filter(planar_graph.graph.vp['is_active'], inverted=True)
    # The following will force the re-drawing of the graph, and issue a

    # re-drawing of the GTK window.
    win.graph.regenerate_surface()
    win.graph.queue_draw()
    # if doing an offscreen animation, dump frame to disk

    if offscreen:
        global count
        pixbuf = win.get_pixbuf()
        pixbuf.savev(r'./frames/sirs%06d.png' % count, 'png', [], [])
        if count > max_count:
            sys.exit(0)
        count += 1
    # We need to return True so that the main loop will call this function more
    # than once.
    return True
'''