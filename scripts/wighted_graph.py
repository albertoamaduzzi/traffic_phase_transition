from planar_graph import initial_graph,planar_graph
import graph-tool as gt
from scipy.sparse.csgraph import laplacian
from scipy.linalg import expm
import numpy as np


##g = initial_graph()




def cofactor(matrix,i,j):
    return (-1)**(i+j)*matrix[i,j]





class analyzer_structure:
    def __init__(self,file_graph):
        self.g = gt.read_graph(file_graph)

    def three_cycles(self):
        self.adjacency_matrix = gt.spectral.adjacency_matrix(self.g)
        self.cube = self.adjacency_matrix**3

    def laplacian(self):
        self.laplacian = laplacian(self.adjacency_matrix)

    def gibbs_density_matrix(self,time):
        exp = expm(-self.laplacian*time)
        self.density_matrix = exp/np.trace(exp)