import sys
from geometric_features import Grid,road
from barthelemy_vertex import barthelemy_graph,initial_graph
    '''
    Example config:
    {
        "path2graph": /workspace/traffic_phase_transition/data/graphs/whatever_specification/,
        "name_graph": "barthelemy_graph_parameters_specification.gt",
        "side_city": 10000,
        "number_grids": 10,
    "facilities": {
            "idx": 0,
            "capacity": 100,
            "type": "hospital"
        }
    }
    
    '''


class facility:
    '''
    1) To investigate the optimal density(average distance) = d^(2/3)
        Optimal design of spatial distribution networks: Michael T. Gastner, M. E. J. Newman
    2) As suggests:
        Deconstructing laws of accessibility and facility distribution in cities: Yanyan Xu, Luis E. Olmos, Sofiane Abbar, Marta C. González
    3) The optimality is observed just when:
        "the number of facilities (sinks) is small compared to the total number of blocks (sources) in the city "
    This recalls me the utility function in:

    U = \mu <G> + <d>
    That has got a maximum when there are a lot of sources and few sinks. (rho = d^0....) and couples of networks are used in similar ways.
    '''
    def __init__(self,config, capacity = sys.maxsize()):
        self.idx = config["idx"]
        self.capacity = capacity
        self.type = config["facilities"][self.idx]["type"]
        self.density: float = 0
        self.city_blocks = Grid(config["side_city"],config["side_city"]/config["number_grids"])


    

    
