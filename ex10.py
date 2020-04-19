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

step = 0.005
initValue = 0
lambdas = np.array(0)
while initValue < 1:
    initValue = initValue + step
    lambdas = np.append(lambdas, initValue)


# Solution

Aeq = nn2na(NN)
Beq = np.array(B)
bounds = tuple( [ (0, None) for a in range (0, Aeq.shape[1]) ] )

results = np.array(3)

for lambdaValue in np.nditer(lambdas):
    cCirconflexe = np.transpose(Distances) + (lambdaValue * TravelTime)
    print(cCirconflexe)
    result = linprog(cCirconflexe, A_eq = Aeq, b_eq = Beq, bounds=bounds, method="simplex")
    indexes = np.where(np.array(result.x) > 0.9)
    arcs_result = [arcs[i] for i in indexes]

    results = np.append(results, [lambdaValue, arcs_result, result.fun - (lambdaValue * 8)])


print(results)
