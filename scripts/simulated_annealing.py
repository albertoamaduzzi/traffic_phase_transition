import numpy as np 
class simulated_annealing_planar_graph:
    def __init__(self,points,distance_matrix,Tmax = 1,Tmin = 0.0001,alpha = 0.9,numIterations = 100):
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.alpha = alpha 
        self.numIterations = numIterations 
        self.points = points
        self.distance_matrix = distance_matrix
        self.current_state = []
        self.current_cost = 0

    def propose_perturbation(self):
        i = np.random.randint(len(self.points))        
        j = np.random.randint(len(self.points))
        if not [i,j] in self.current_state:
            
class Solution:
    '''
        Reproducing newman: 
            Input: fixed number of generated points
        Variables:
            current_state: list np.shape(number_connected_points,2) -> the state is the set of edges whose length is summed up to get the cost function

    '''

    def __init__(self, CVRMSE, configuration,points):
        self.CVRMSE = CVRMSE
        self.config = configuration

    
 
 
 
def genRandSol(points):
    '''
    
    '''
    initial_state = []
    for point_idx in range(len(points)):
        initial_state.append([point_idx,point_idx+1])
    return initial_state
 
 
def neighbor(currentSol):
    return currentSol
 
 
def cost(distance_matrix,current_state):
    cost = 0
    for edge in current_state:
        cost += distance_matrix[edge[0],edge[1]]
    return cost
 
# Mapping from [0, M*N] --> [0,M]x[0,N]
 
 
def indexToPoints(index):
    points = [index % M, index//M]
    return points
 
 
M = 5
N = 5
sourceArray = [['X' for i in range(N)] for j in range(M)]
min = Solution(float('inf'), None)
currentSol = genRandSol()
 
while(T > Tmin):
    for i in range(numIterations):
        # Reassigns global minimum accordingly
        if currentSol.CVRMSE < min.CVRMSE:
            min = currentSol
        newSol = neighbor(currentSol)
        ap = math.exp((currentSol.CVRMSE - newSol.CVRMSE)/T)
        if ap > random.uniform(0, 1):
            currentSol = newSol
    T *= alpha  # Decreases T, cooling phase
 
# Returns minimum value based on optimization
print(min.CVRMSE, "\n\n")
 
for i in range(M):
    for j in range(N):
        sourceArray[i][j] = "X"
 
# Displays
for obj in min.config:
    coord = indexToPoints(obj)
    sourceArray[coord[0]][coord[1]] = "-"
 
# Displays optimal location
for i in range(M):
    row = ""
    for j in range(N):
        row += sourceArray[i][j] + " "
    print(row)