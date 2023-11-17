import os
import sys
import json

class TrafficModel(mesa.Model):
    def __init__(self, N, width, velocity, p_slowdown, insertion_rate):
        self.num_agents = N
        self.width = width
        self.max_velocity = velocity
        self.p_slowdown = p_slowdown
        self.graph = lattice(directed=False) # create an undirected graph
        create_graph(graph_initializaion_string)
        self.graph = Graph(directed=False)
        self.graph.add_vertex(N) # add N vertices to the graph
        self.grid = space.NetworkGrid(self.graph) # create a grid with the graph
        self.schedule = time.RandomActivation(self) # create a random activation scheduler
        self.dc = datacollection.DataCollector(
            {"Speed": lambda a: a.velocity, "Position": lambda a: a.position}
        ) # create a data collector to collect data during the simulation
        self.insertion_rate = insertion_rate
        self.iteration = 0

    def step(self):
        # Insert agents
        if self.iteration % 100 == 0 and self.num_agents < 10000:
            num_to_insert = int(self.insertion_rate * self.width)
            for i in range(num_to_insert):
                agent = MeAgent(self.num_agents, self, self.max_velocity, i)
                self.grid.place_agent(agent, i)
                self.schedule.add(agent)
                self.num_agents += 1

        self.schedule.step()
        self.dc.collect(self)
        self.iteration += 1



def ifnotexistsmkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path
class configuration:
    def __init__(self):
        self.working_dir = sys.path[0] 



    def dump_configuration(self):
        dumping_dir = ifnotexistsmkdir(os.path.join(self.working_dir,'configuration'))
        with open(dumping_dir,'w') as f:
            json.dumps(f,indent = 4)