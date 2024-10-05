import numpy as np
import logging
logger = logging.getLogger(__name__)

def GetMaxIndicesGridFromTij(Tij):
    ij = Tij['(i,j)O'].apply(lambda x: [int((x.replace('(','').replace(')','').split(','))[0]),int((x.replace('(','').replace(')','').split(','))[1])]).T
    return max(ij[0]),max(ij[1])
def GetMaxIndicesGridFromGrid(grid):
    return max(grid['i']),max(grid['j'])



def ConvertTij2Matrix(Tij,grid):
    imax,jmax = GetMaxIndicesGridFromGrid(grid)
    MatrixTij = np.array(Tij['number_people'],dtype = np.float32).reshape((imax+1,jmax+1))
    return MatrixTij

def EstimateGershgoring(MatrixTij):
    '''
        Gershgoring's method to estimate the area of the eigenvalue of a matrix (This case the transition of the master equation)
        For each:
            Re {eigval[i]} in MatrixTij[i,i] +  sum {j!=i} abs(MatrixTij[i,j]) 
    '''
    RealBoundEigenvalues = [np.abs(MatrixTij[i,i]) + np.sum([np.abs(MatrixTij[i,j]) for j in range(i+1,MatrixTij.shape[0])]) for i in range(MatrixTij.shape[0])] 
    return RealBoundEigenvalues