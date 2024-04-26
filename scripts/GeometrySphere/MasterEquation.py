import numpy as np

def GetMaxIndicesGridFromTij(Tij):
    ij = Tij['(i,j)O'].apply(lambda x: [int((x.replace('(','').replace(')','').split(','))[0]),int((x.replace('(','').replace(')','').split(','))[1])]).T
    return max(ij[0]),max(ij[1])
def GetMaxIndicesGridFromGrid(grid):
    return max(grid['i']),max(grid['j'])


def ConvertTij2Matrix(Tij,grid):
    imax,jmax = GetMaxIndicesGridFromGrid(grid)
    MatrixTij = np.array(Tij['number_people'],dtype = np.float32).reshape((imax+1,jmax+1))
    return MatrixTij