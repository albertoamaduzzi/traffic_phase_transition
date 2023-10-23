import mesa
import random
import numpy as np
import mesa.time as time
import mesa.space as space
import mesa.datacollection as datacollection
from graph_tool.all import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class MeAgent(mesa.Agent):
    '''
        Note: Person is a subclass of mesa.Agent
            1) requires unique_id and model (for default initialization)
            2) has got grid as object (in repast this is a ghost possibly)
    '''
    def __init__(self, unique_id, model, velocity, position):
        super().__init__(unique_id, model)
        self.velocity = velocity
        self.position = position

    def step(self):
        # Acceleration
        if self.velocity < self.model.max_velocity:
            self.velocity += 1

        # Deceleration
        distance = self.model.max_velocity + 1
        for neighbor in self.model.grid.get_neighbors(self.position, include_center=False):
            if neighbor.velocity < distance:
                distance = neighbor.velocity
        if distance < self.velocity:
            self.velocity = distance

        # Randomization
        if random.random() < self.model.p_slowdown:
            self.velocity -= 1

        # Movement
        self.position = (self.position + self.velocity) % self.model.width
        self.model.grid.move_agent(self, self.position)


class TrafficModel(mesa.Model):
    def __init__(self, N, width, velocity, p_slowdown, insertion_rate):
        self.num_agents = 0
        self.width = width
        self.max_velocity = velocity
        self.p_slowdown = p_slowdown
        self.graph = Graph(directed=False) # create an undirected graph
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

    def draw_graph(self, edge_weights):
        # Create a graph layout
        pos = sfdp_layout(self.graph)

        # Draw the graph with edge weights
        graph_draw(self.graph, pos=pos, vertex_size=5, edge_pen_width=edge_weights)

    def animate(self):
        # Create a list of edge weights
        edge_weights = np.zeros(self.graph.num_edges())

        # Create a graph layout
        pos = sfdp_layout(self.graph)

        # Create a figure and axis for the animation
        fig, ax = plt.subplots()

        # Define the animation function
        def update(frame):
            # Update the edge weights
            for i, e in enumerate(self.graph.edges()):
                edge_weights[i] = frame[i]

            # Draw the updated graph
            self.draw_graph(edge_weights)

            # Return the updated graph
            return ax

        # Create the animation
        anim = FuncAnimation(fig, update, frames=self.dc.get_model_vars_dataframe()['Speed'], interval=50)

        # Show the animation
        plt.show()



def main():
    model = TrafficModel(N=100, width=100, velocity=5, p_slowdown=0.2, insertion_rate=0.1)
    for i in range(100):
        model.step()
    model.animate()


if __name__ == "__main__":
    main()
