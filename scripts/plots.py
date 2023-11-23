import matplotlib.pyplot as plt
#from gi.repository import Gtk, Gdk, GdkPixbuf, GObject, GLib
import numpy as np
import os
# FROM PROJECT

def plot_relative_neighbors(planar_graph,vi,attracting_vertex,new_added_vertex,available_vertices):
    fig,ax = plt.subplots(1,2,figsize = (20,20))
    ## All attracting vertices
    print('available vertices inside function: ',available_vertices)
    attracting_vertices = np.array([np.array([planar_graph.graph.vp['x'][v],planar_graph.graph.vp['y'][v]]) for v in planar_graph.graph.vertices() if planar_graph.graph.vp['is_active'][v] == True])
    attracting_vertices_indices = np.array([planar_graph.graph.vp['id'][v] for v in planar_graph.graph.vertices() if planar_graph.graph.vp['is_active'][v] == True])
    ## Attracting vertex whose growing relative neighbors are updated
    coords_attracting_vertex = np.array([planar_graph.graph.vp['x'][attracting_vertex],planar_graph.graph.vp['y'][attracting_vertex]])
    ## Growing node v
    coordinates_vi = np.array([planar_graph.graph.vp['x'][vi],planar_graph.graph.vp['y'][vi]])
    coordinates_new_added_vertex = np.array([planar_graph.graph.vp['x'][new_added_vertex],planar_graph.graph.vp['y'][new_added_vertex]])  
    ## vector toward attracting vertices
    coords_available_vertices = np.array([np.array([planar_graph.graph.vp['x'][planar_graph.graph.vertex(vj)],planar_graph.graph.vp['y'][planar_graph.graph.vertex(vj)]]) for vj in available_vertices])
    toward_attr_vertices = coords_available_vertices - coordinates_vi
    ua_plus_ub = np.sum(toward_attr_vertices,axis = 0)
    utoward_attr_vertices = ua_plus_ub/np.sqrt(np.sum(ua_plus_ub**2))
    print(np.shape(utoward_attr_vertices))
    vector_edge = planar_graph.graph.vp['pos'][new_added_vertex].a - planar_graph.graph.vp['pos'][vi].a   
    uvector_edge = vector_edge/np.sqrt(np.sum(vector_edge**2))
    ## plot (attracting vertices, attracting vertex, growing node)
    ax[0].scatter(planar_graph.graph.vp['x'].a,planar_graph.graph.vp['y'].a,color = 'black')
    ax[0].scatter(attracting_vertices[:,0],attracting_vertices[:,1],color = 'blue')
    for av in range(len(attracting_vertices_indices)):
        ax[0].text(attracting_vertices[av,0],attracting_vertices[av,1], f'({attracting_vertices_indices[av]})', verticalalignment='bottom', horizontalalignment='center', fontsize=10)                
    ax[0].scatter(coords_attracting_vertex[0],coords_attracting_vertex[1],color = 'yellow')
    ax[0].scatter(coordinates_vi[0],coordinates_vi[1],color = 'red')
    ## Plot relative neighbors
    ax[1].scatter(planar_graph.graph.vp['x'].a,planar_graph.graph.vp['y'].a,color = 'black')
    ax[1].scatter(attracting_vertices[:,0],attracting_vertices[:,1],color = 'white')
    for av in range(len(attracting_vertices_indices)):
        ax[1].text(attracting_vertices[av,0],attracting_vertices[av,1], f'({attracting_vertices_indices[av]})', verticalalignment='bottom', horizontalalignment='center', fontsize=10)                
    ax[1].scatter(coords_attracting_vertex[0],coords_attracting_vertex[1],color = 'yellow')
    ax[1].scatter(coordinates_vi[0],coordinates_vi[1],color = 'red')        
    ax[1].scatter(coordinates_new_added_vertex[0],coordinates_new_added_vertex[1],color = 'orange')
    ax[1].plot([coordinates_vi[0],coordinates_new_added_vertex[0]],[coordinates_vi[1],coordinates_new_added_vertex[1]],linestyle = '-',linewidth = 1.5,color = 'black')
    ax[1].plot(vector_edge[0],vector_edge[1],linestyle = '-',linewidth = 1.5,color = 'violet')
    ax[1].grid()
    print('growth line: ',uvector_edge)
    print('new added: ',utoward_attr_vertices)
    print('theta: ',np.arccos(np.dot(uvector_edge,utoward_attr_vertices)/np.sqrt(np.sum(vector_edge**2)))/np.pi*180)
    print('growth line is orthogonal to line (vi,new_node): ',np.dot(uvector_edge,utoward_attr_vertices))
    for vj in planar_graph.graph.vp['relative_neighbors'][vi]:
        coordinates_vj = np.array([planar_graph.graph.vp['x'][vj],planar_graph.graph.vp['y'][vj]])
        r = planar_graph.distance_matrix_[planar_graph.graph.vp['id'][vi]][vj]
        circle1 = plt.Circle(coordinates_vi, r, color='red', linestyle = '--',fill=True ,alpha = 0.1)
        circle2 = plt.Circle(coordinates_vj, r, color='green', linestyle = '--',fill=True ,alpha = 0.1)
        ax[1].add_artist(circle1)
        ax[1].add_artist(circle2)
#            intersection = plt.Circle((coordinates_vi + coordinates_vj) / 2, np.sqrt(r ** 2 - (1/(2*r)) ** 2), color='green',alpha = 0.2)
        # Add the intersection to the axis
#            ax.add_artist(intersection)            
        ax[1].scatter(planar_graph.graph.vp['x'][planar_graph.graph.vertex(vj)],planar_graph.graph.vp['y'][planar_graph.graph.vertex(vj)],color = 'green')
        ax[1].plot([planar_graph.graph.vp['x'][vi],planar_graph.graph.vp['x'][planar_graph.graph.vertex(vj)]],[planar_graph.graph.vp['y'][vi],planar_graph.graph.vp['y'][planar_graph.graph.vertex(vj)]],linestyle = '--',color = 'green')
    ax[0].legend(['any vertex','attracting vertices','responsable attracting vertex','growing node'])
    ax[1].legend(['any vertex','attracting vertices','responsable attracting vertex','growing node','new added vertex','connection new node','growth line','relative neighbors','circle growing','circle relative neighbor','line relative neighbor'])
    plt.title('attracting {0}, growing {1} '.format(planar_graph.graph.vp['id'][attracting_vertex],planar_graph.graph.vp['id'][vi]))
    plt.show()

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

def plot_number_roads_time(planar_graph):
    plt.scatter(planar_graph.time,planar_graph.count_roads)
    plt.plot(planar_graph.time,np.array(planar_graph.time) + planar_graph.count_roads[0])
    plt.plot(planar_graph.time,2*np.array(planar_graph.time) + planar_graph.count_roads[0])
    plt.xlabel('time')
    plt.ylabel('number of roads')
    plt.legend(['graph','tree','lattice'])
    plt.plot() 

def plot_total_length_roads_time(planar_graph):
    plt.scatter(planar_graph.time,planar_graph.length_total_roads)
    plt.plot(planar_graph.time,np.sqrt(np.array(planar_graph.time))*planar_graph.length_total_roads[0])
    plt.xlabel('time')
    plt.ylabel('total length (m)')
    plt.legend(['graph','square root'])
    plt.plot() 

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