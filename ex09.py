import numpy as np
from scipy.optimize import linprog
from basic_utils import nn2na

# parameters

NN = np.array([
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0]
])

# order of the arcs should be the same as nn2na returns
arcs = np.array( ['S2', 'S3', '24', '2T', '35', '4T', '5T'] )
Distances = np.array( [2, 1, 2, 5, 2, 1, 2] ) 
TravelTime = np.array( [[3, 1, 3, 1, 3, 3, 5]] ) 

B = [1, 0, 0, 0, 0, -1]

# solution

Aeq = nn2na(NN)

Beq = np.array(B)

bounds = tuple( [ (0, None) for a in range (0, Aeq.shape[1]) ] )

result = linprog(Distances, A_eq = Aeq, b_eq = Beq, A_ub=TravelTime, b_ub=9, bounds=bounds, method="simplex")

indexes = np.where(np.array(result.x) > 0.9)
arcs_result = [arcs[i] for i in indexes]

# results: Path to follow, total cost
print(*arcs_result)
print(result.fun)

# When we change b_ub to be less or equal than 8, Total distance becomes 5.4, wich is not a factible solution