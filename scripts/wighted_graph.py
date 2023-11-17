from barthelemy_vertex import initial_graph,barthelemy_graph
import graph-tool as gt

##g = initial_graph()

def cofactor(matrix,i,j):
    return (-1)**(i+j)*matrix[i,j]





class analyzer_structure:
    def __init__(self,file_graph):
        self.g = gt.read_graph(file_graph)

    def three_cycles(self):
        self.adjacency_matrix = gt.spectral.adjacency_matrix(self.g)
        self.cube = self.adjacency_matrix**3
