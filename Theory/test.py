import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from numpy.matlib import repmat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.linalg import inv
import utilities

N = 100000

difference = np.arange(1, 1000, 10)
spectralRadius = np.zeros(len(difference))
numStates = np.zeros(len(difference))
for i in range(len(difference)):
    diff = difference[i]
    degrees = np.random.randint(1000 - diff, 1000 + diff, size=N)
    P = utilities.generateConfigurationModelP(degrees)
    k = len(P)
    numStates[i] = k
    l = np.max(np.abs(np.linalg.eigvals(P)))
    spectralRadius[i] = l

plt.figure()
plt.semilogy(numStates, spectralRadius)
plt.plot(numStates, np.divide(spectralRadius, numStates))
plt.show()
