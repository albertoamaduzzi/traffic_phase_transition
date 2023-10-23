import city
import simulation
import json 
import graph
import parameters
'''

'''

if __name__=="__main__":
    total_iterations = 1000
    list_dynamics = ['NaSch','Cuda_Helbing']
    total_volume = 
    with open(config_file,'r') as f:
        config = json.load(f)
    fake_model = city(config['name'])
    for topology in config['graph_topologies']:
        # Create random graph
        lattice_structure = graph('graph-tool',topology)
        lattice_structure._connect_nodes()
        lattice_structure.create_origin_destination('monocentric')
        lattice_structure._compute_centrality_index()
        # Create simulation
        for model in list_dynamics:
            sim = simulation(model,lattice_structure)

    