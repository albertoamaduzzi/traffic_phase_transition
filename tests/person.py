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






def main():
    model = TrafficModel(N=100, width=100, velocity=5, p_slowdown=0.2, insertion_rate=0.1)
    for i in range(100):
        model.step()
    model.animate()


if __name__ == "__main__":
    main()
