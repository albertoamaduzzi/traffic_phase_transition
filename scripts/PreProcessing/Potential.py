import networkx as nx
import numpy as np

def EncodeIdx2Grid(LatticeGrid):


    return Idx2Grid



@jit
def GradientFromPotential(LatticeGrid):
    '''
        Create the vector field from the potential
    '''
    assert type(LatticeGrid) == nx.Graph():
    

@jit
def Potential(LatticeGrid):
    '''
        Stores the values of the potential in the nodes of the lattice.
    '''
    
@jit
def Rotate(LatticeGrid):
    for node in LatticeGrid.nodes():


def Lorenz(DensityVector):
    '''
        Lorenz curve is the comulative probability distribution function of the density (In our case potential).
        Output:
            CDF: vector(float) -> Increases
            SortedByDensity2PrimitiveIndex: dict -> {0: idx DensityVector that is smallest,....,n: Idx DensityVector that is biggest}
    '''
    Dv = np.argsort(DensityVector)
    SortedByDensity2PrimitiveIndex = {idx: Dv[idx] for idx in range(len(Dv))}
    Z = np.sum(Dv)
    CDF = np.cumsum(DensityVector)/Z
    return CDF,SortedByDensity2PrimitiveIndex

    

