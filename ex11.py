import numpy as np
from scipy.optimize import linprog
from basic_utils import nn2na
import matplotlib.pyplot as plt

# parameters

NN = np.array([
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0]
])

arcs = np.array( ['S2', 'S3', '24', '2T', '35', '4T', '5T'] )
Distances = np.array( [2, 1, 2, 5, 2, 1, 2] ) 
TravelTime = np.array( [3, 1, 3, 1, 3, 3, 5] ) 
tTransposed = np.transpose(TravelTime)

B = [1, 0, 0, 0, 0, -1]


# Solution

Aeq = nn2na(NN)
Beq = np.array(B)
bounds = tuple( [ (0, None) for a in range (0, Aeq.shape[1]) ] )


currentLambda = 0
previousLambda = 1
tolerance = 0.001
iteration = 1

while abs(previousLambda - currentLambda) > tolerance:
    cCirconflexe = np.transpose(Distances) + (currentLambda * TravelTime)
    result = linprog(cCirconflexe, A_eq = Aeq, b_eq = Beq, bounds=bounds, method="simplex")
    gradient = np.dot(tTransposed, result.x) - 8
    step = 1 / iteration
    previousLambda = currentLambda
    currentLambda = currentLambda + (step * gradient)
    iteration = iteration + 1

print(currentLambda) 