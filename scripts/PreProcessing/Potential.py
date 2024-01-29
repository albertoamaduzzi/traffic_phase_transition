import networkx as nx
@jit
def Gradient(LatticeGrid):
    assert type(LatticeGrid) == nx.Graph():
    

@jit
def Potential(LatticeGrid):
    '''
        Stores the values of the potential in the nodes of the lattice
    '''

@jit
def Rotate(LatticeGrid):
    for node in LatticeGrid.nodes():
