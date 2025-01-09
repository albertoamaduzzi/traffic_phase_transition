import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import numba 
import polars as pl
from DateTime_ import *
import logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# UNIVERSALITY EXPONENTS DIRECTED PERCOLATION
# Equal time correlation function for directed percolation : Beljakov Hinrichsen
BETA1 = 0.276487 # 1D percolation (survived fraction n^(beta)(R))
NUORTH1 = 1.096854 # 1D percolation (correlation length in time n^(nuorth)(R))
NUPAR1 = 1.733847 # 1D percolation (correlation length in space n^(nupar)(R))
Z1 = NUPAR1/NUORTH1
CHI1 = BETA1/NUORTH1
DELTA1 = BETA1/NUPAR1
BETA2 = 0.58343 # 2D percolation (survived fraction n^(beta)(R))
NUORTH2 = 0.7334 # 2D percolation (correlation length in time n^(nuorth)(R))
NUPAR2 = 1.2951 # 2D percolation (correlation length in space n^(nupar)(R))
Z2 = NUPAR2/NUORTH2
CHI2 = BETA2/NUORTH2
DELTA2 = BETA2/NUPAR2

#NOTE: The measures we are going too look at are n-points functions \Phi(r_1,r_2) = C|r_1 - r_2|^(-2CHI)  

## PERCOLATION WITH SPEED 
def ComputePercolationColumns(EdgesWithFluxesAndSpeed,IntTimeArray,q0,q1,q2):
    """
        @param EdgesWithFluxesAndSpeed: DataFrame with the edges of the network and the fluxes and speeds of the edges.
            [uniqueid','u','v','length','speed_mph','lanes','capacity','flux_t','speed_kmh_t']
        @param t: Time to compute the percolation
        @param q: Quantile to compute the percolation
        @return EdgesWithFluxesAndSpeed: DataFrame with the percolation column
    """
    for t in range(len(IntTimeArray)):
        if f"speed_kmh_{t}" in EdgesWithFluxesAndSpeed.columns:
            EdgesWithFluxesAndSpeed = EdgesWithFluxesAndSpeed.with_columns((pl.col("speed_mph")*1.6).alias(f"maxspeed_kmh")) 
            EdgesWithFluxesAndSpeed = EdgesWithFluxesAndSpeed.with_columns((pl.col(f"speed_kmh_{t}")/pl.col("maxspeed_kmh") < q0).alias(f"severe_traffic_{t}"))
            EdgesWithFluxesAndSpeed = EdgesWithFluxesAndSpeed.with_columns((pl.col(f"speed_kmh_{t}")/pl.col("maxspeed_kmh") < q1).alias(f"moderate_traffic_{t}"))
            EdgesWithFluxesAndSpeed = EdgesWithFluxesAndSpeed.with_columns((pl.col(f"speed_kmh_{t}")/pl.col("maxspeed_kmh") < q2).alias(f"free_flow_{t}"))
    return EdgesWithFluxesAndSpeed

def ConvertGeoJsonEdges2NetworkX(EdgesWithFluxesAndSpeed, IntTimeArray):
    """
    Convert GeoJsonEdges to a NetworkX graph.

    :param EdgesWithFluxesAndSpeed: DataFrame with the edges of the network and the fluxes and speeds of the edges.
        [uniqueid', 'u', 'v', 'length', 'speed_mph', 'lanes', 'capacity', 'flux_t', 'speed_kmh_t']
    :param IntTimeArray: Array of time intervals
    :return: G: NetworkX Graph with the edges of the network and the fluxes and speeds of the edges.
    """
    G = nx.Graph()
    
    for t in range(len(IntTimeArray)):
        # Ensure the column f"percolation_trafficked_{t}" exists
        if f"severe_traffic_{t}" in EdgesWithFluxesAndSpeed.columns:
            # Iterate over the rows of the DataFrame
            for row in EdgesWithFluxesAndSpeed.iter_rows(named=True):
                
                u = row['u']
                v = row['v']
                uniqueid = row['uniqueid']
                severe_traffic= row[f"severe_traffic_{t}"]
                moderate_traffic= row[f"moderate_traffic_{t}"]
                free_flow= row[f"free_flow_{t}"]
                # Add nodes
                G.add_node(u)
                G.add_node(v)    
                # Add edge with attributes
                G.add_edge(u, v, uniqueid=uniqueid, **{f"severe_traffic_{t}": severe_traffic, f"moderate_traffic_{t}": moderate_traffic, f"free_flow_{t}": free_flow})

    return G

def ComputeConnectedComponentsGonT(G, IntTimeArray):
    """
    Compute the number of connected components of the graph G at each time interval.

    :param G: NetworkX Graph with the edges of the network and the fluxes and speeds of the edges.
    :param IntTimeArray: Array of time intervals
    :return: connected_components: List of the number of connected components at each time interval.
    """
    Components = []    
    for t in range(len(IntTimeArray)):
        # Filter edges based on the feature f"q_{t}"
        edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d.get(f"severe_traffic_{t}", False)]
        subgraph = G.edge_subgraph(edges_to_keep).copy()
        # Compute the number of connected components
        Components_t = list(nx.connected_components(subgraph))
        # Sort components by size in descending order
        Components_t = sorted(Components_t, key=len, reverse=True)

#        Components_t = nx.number_connected_components(subgraph)
        if len(Components_t) == 1:
            Components.append([list(Components_t[0]),0])
        elif len(Components_t) > 1:
            Components.append([list(Components_t[0]),list(Components_t[1])])
        else:
            Components.append([0,0])
    return Components

def PlotNumberLinksTwoBiggestConnectedComponents(connected_components, t_hours,UCI,R,PlotDir):
    """
    Plot the number of links in the two biggest connected components at each time interval.
    """
    logger.info(f"Plotting the number of links in the two biggest connected components at each time interval.")
    SaveDir  = os.path.join(PlotDir,f"ConnectedComponents_{UCI}_{R}")
    os.makedirs(SaveDir,exist_ok=True)
    num_links_component1 = []
    num_links_component2 = []
    legend = ["component 1", "component 2"]
    if not os.path.isfile(os.path.join(SaveDir, f'number_links_two_biggest_connected_components.png')):
        for t in range(len(t_hours)):
            if isinstance(connected_components[t], int):
                num_links_component1.append(connected_components[t])
                num_links_component2.append(0)

            elif isinstance(connected_components[t], list):
                if len(connected_components[t]) == 1:
                    if len(connected_components[t][0]) == 1:
                        num_links_component1.append(connected_components[t][0])
                    else:
                        num_links_component1.append(len(connected_components[t][0]))
                    num_links_component2.append(0)
                elif len(connected_components[t]) == 2:
                    if isinstance(connected_components[t][0],int):
                        num_links_component1.append(connected_components[t][0])
                    else:
                        num_links_component1.append(len(connected_components[t][0]))
                    if isinstance(connected_components[t][1], int):
                        num_links_component2.append(connected_components[t][1])
                    elif isinstance(connected_components[t][1], list):
                        if len(connected_components[t][1]) == 1:
                            num_links_component2.append(connected_components[t][1])
                        else:
                            num_links_component2.append(len(connected_components[t][1]))
            if t <10:
                str_t = f'00{t}'
            elif t <100:
                str_t = f'0{t}'
            else:
                str_t = f'{t}'

        _,ax = plt.subplots(1,1,figsize=(10,10)) 
        if len(t_hours[:-1]) == len(num_links_component1[:-1]) and len(t_hours[:-1]) == len(num_links_component2[:-1]):
            ax.plot(t_hours[:-1], num_links_component1[:-1], marker='o')
            ax.plot(t_hours[:-1], num_links_component2[:-1], marker='o')
            ax.set_xticks(t_hours[::8])
            ax.set_xticklabels(t_hours[::8], rotation=90)
            ax.set_xlabel('Time')
            ax.set_ylabel('Number of Links')
            ax.set_title('Number of Links in the Two Biggest Connected Components')
            ax.legend(legend)
            plt.savefig(os.path.join(SaveDir, f'number_links_two_biggest_connected_components.png'))
            plt.close()


def SinglePlotPercolation(GeoJsonEdges,EdgesWithFluxesAndSpeed,t,list_hours,UCI,R,q0,q1,q2,PlotDir):
    """
        @param GeoJsonEdges: GeoJson with the edges of the network 
            [key,lanes,length,geometry,uv,maxspeed_int,maxspeed_kmh,capacity]
        @param EdgesWithFluxesAndSpeed: DataFrame with the edges of the network and the fluxes and speeds of the edges.
            [uniqueid','u','v','length','speed_mph','lanes','capacity','flux_t','speed_kmh_t']
        @param t: Time to plot the traffic
        @param q0: 0.25-> Fraction of most traffic
        @param q1: 0.5-> Fraction of moderate traffic
        @param q2: 0.90-> Fraction Free Flow
        @param SaveDir: Directory to save the plot
    """
    import contextily as ctx
    if t <10:
        str_t = f'00{t}'
    elif t <100:
        str_t = f'0{t}'
    else:
        str_t = f'{t}'    
    PercolationDir = os.path.join(PlotDir,f"PercolationPlots_{round(UCI,3)}_{R}")
    os.makedirs(PercolationDir,exist_ok = True)
    SpeedDir = os.path.join(PlotDir,f"SpeedPlots_{UCI}_{R}")
    os.makedirs(SpeedDir,exist_ok = True)
    if not os.path.exists(os.path.join(SpeedDir,f'Speed_{str_t}.png')):
        if f"speed_kmh_{t}" in EdgesWithFluxesAndSpeed.columns:
            EdgesWithFluxesAndSpeed = EdgesWithFluxesAndSpeed.with_columns((pl.col("speed_mph")*1.6).alias(f"maxspeed_kmh")) 
            EdgesWithFluxesAndSpeed = EdgesWithFluxesAndSpeed.with_columns((pl.col(f"speed_kmh_{t}")/pl.col("maxspeed_kmh") < q0).alias(f"severe_traffic"))
            EdgesWithFluxesAndSpeed = EdgesWithFluxesAndSpeed.with_columns((pl.col(f"speed_kmh_{t}")/pl.col("maxspeed_kmh") < q1).alias(f"moderate_traffic"))
            EdgesWithFluxesAndSpeed = EdgesWithFluxesAndSpeed.with_columns((pl.col(f"speed_kmh_{t}")/pl.col("maxspeed_kmh") < q2).alias(f"free_flow"))
    #        EdgesWithFluxesAndSpeed = EdgesWithFluxesAndSpeed.with_columns(pl.col("percolation_trafficked") < q)
            EdgesWithFluxesAndSpeed_pd = EdgesWithFluxesAndSpeed.to_pandas()
            # Join GeoJsonEdges with EdgesWithFluxesAndSpeed
            if "level_0" not in GeoJsonEdges.columns:
                GeoJsonEdges.reset_index(inplace=True)
            EdgesWithFluxesAndSpeed_pd2Join = EdgesWithFluxesAndSpeed_pd[['uniqueid', f'speed_kmh_{t}',"severe_traffic","moderate_traffic","free_flow"]]
            GeoJsonEdges['uniqueid'] = GeoJsonEdges['uniqueid'].astype(int)
            EdgesWithFluxesAndSpeed_pd2Join['uniqueid'] = EdgesWithFluxesAndSpeed_pd['uniqueid'].astype(int)
            if "level_0" not in EdgesWithFluxesAndSpeed_pd.columns:
                EdgesWithFluxesAndSpeed_pd = EdgesWithFluxesAndSpeed_pd.reset_index(inplace=True)
            GeoJsonEdges = GeoJsonEdges.merge(EdgesWithFluxesAndSpeed_pd2Join, on='uniqueid', how='inner')
            GeoJsonEdges['severe_traffic'] = GeoJsonEdges['severe_traffic'].fillna(False)
            GeoJsonEdges['moderate_traffic'] = GeoJsonEdges['moderate_traffic'].fillna(False)
            GeoJsonEdges['free_flow'] = GeoJsonEdges['free_flow'].fillna(False)
            GeoJsonEdges = GeoJsonEdges.to_crs(epsg=3857)
            GeoJsonEdges = GeoJsonEdges[GeoJsonEdges.is_valid]
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            GeoJsonEdges.plot(column=f'speed_kmh_{t}', cmap='inferno', linewidth=1, ax=ax, legend=True)
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    #        GeoJsonEdges.plot(column=f'speed_kmh_{t}', cmap='viridis', linewidth=2, ax=ax, legend=True)
            ax.set_title(f'Speed at: {list_hours[t]}')
            plt.savefig(os.path.join(SpeedDir,f'Speed_{str_t}.png'), dpi=200)
            plt.close()
            if not os.path.exists(os.path.join(PercolationDir,f'percolation_{str_t}.png')):
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                GeoJsonEdges[GeoJsonEdges['free_flow']].plot(ax=ax, color='green', alpha=1, linewidth=1, label='True')
                GeoJsonEdges[GeoJsonEdges['moderate_traffic']].plot(ax=ax, color='yellow', alpha=1, linewidth=1, label='True')
                GeoJsonEdges[GeoJsonEdges['severe_traffic']].plot(ax=ax, color='red', alpha=1, linewidth=1, label='True')
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        #        GeoJsonEdges.plot(column='percolation_trafficked', ax=ax, legend=True)
                ax.set_title(f'Percolation: {list_hours[t]}')
                ax.legend(['Free Flow: q > 0.9', 'Moderate Traffic : 0.5 < q < 0.9', 'Severe Traffic : q < 0.25'])
                plt.savefig(os.path.join(PercolationDir,f'percolation_{str_t}.png'), dpi=200)
                plt.close()
    return GeoJsonEdges

def AnalysisPercolationSpeed(IntTimeArray,GeoJsonEdges,EdgesWithFluxesAndSpeed,UCI,R,PlotDir):
    """
        @param IntTimeArray: Array of time intervals 
    """
    MinutesTimeArray = np.array([int(t/60) for t in IntTimeArray])
    t_datetime = ConvertVectorMinutesToDatetime(MinutesTimeArray,0,"2024-08-15")
    t_hours = CastVectorDateTime2Hours(t_datetime)
    q0 = 0.25
    q1 = 0.5
    q2 = 0.9
    for t in range(len(MinutesTimeArray)):
        SinglePlotPercolation(GeoJsonEdges,EdgesWithFluxesAndSpeed,t,t_hours,UCI,R,q0,q1,q2,PlotDir)
    #CreateVideoFromImages(os.path.join(PlotDir,"PercolationPlots"),f"percolation")
    #DeleteImages(os.path.join(PlotDir,"PercolationPlots"))
    #CreateVideoFromImages(os.path.join(PlotDir,"SpeedPlots"),"Speed")
    #DeleteImages(os.path.join(PlotDir,"SpeedPlots"))
    EdgesWithFluxesAndSpeed = ComputePercolationColumns(EdgesWithFluxesAndSpeed,IntTimeArray,q0,q1,q2)
    G = ConvertGeoJsonEdges2NetworkX(EdgesWithFluxesAndSpeed, IntTimeArray) 
    Components = ComputeConnectedComponentsGonT(G, IntTimeArray)
    PlotNumberLinksTwoBiggestConnectedComponents(Components, t_hours,UCI,R,PlotDir)       


# DIRECTED PERCOLATION


