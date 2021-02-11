import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from numpy.matlib import repmat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.linalg import inv
import utilities

N = 1000
nStates = 10
tStates = np.linspace(0, 20, nStates)
T = np.zeros((nStates, nStates))
S = np.diag(-np.ones(nStates), k=0) + np.diag(np.ones(nStates-1), k=-1)
p = np.arange(0.0, 1, 0.01)
T[0, :] += N*utilities.betaVL(tStates, 0.2, 1.0)
print(T)
print(inv(S))
print(-np.matmul(T,inv(S)))
l = np.linalg.eigvals(-np.matmul(T,inv(S)))
print(np.max(np.abs(l)))


plt.figure()
plt.plot(p, spectralRadius)
plt.show()
