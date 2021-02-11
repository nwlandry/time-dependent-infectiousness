import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv
import utilities

nInfectiousStates = 100
tStates = np.linspace(0, 20, nInfectiousStates)
dt = tStates[1] - tStates[0]
threshold = 0.0
pCrit =  utilities.calculateCriticalMax(nInfectiousStates, tStates, threshold, dt)
print(pCrit)
print(np.sum(utilities.betaVL(tStates, threshold, pCrit))*dt)
print(Stop)

T = np.zeros((nInfectiousStates, nInfectiousStates))
S = np.diag(-np.ones(nInfectiousStates), k=0)/dt + np.diag(np.ones(nInfectiousStates-1), k=-1)/dt
print(S)
p = np.linspace(0.0, 20*pCrit, 100)
spectralRadius = list()
for pMax in p:
    T[0, :] = N*utilities.betaVL(tStates, threshold, pMax)
    l = np.linalg.eigvals(-np.matmul(T,inv(S)))
    spectralRadius.append(np.max(np.abs(l)))

plt.figure()
plt.plot(p, spectralRadius, 'k-')
plt.plot([pCrit, pCrit],[0, 2], 'r--')
plt.ylabel("Spectral Radius")
plt.xlabel(r"$p_{max}$")
plt.show()



N = 1000
nStates = 100
tStates = np.linspace(0, 20, nStates)
T = np.zeros((nStates, nStates))
S = np.diag(-np.ones(nStates), k=0) + np.diag(np.ones(nStates-1), k=-1)
invS = inv(S)
p = np.arange(0.0, 0.1, 0.001)
spectralRadius = list()
for pMax in p:
    T[0, :] += betaVL(tStates, 0.0, pMax)
    l = np.linalg.eigvals(-np.matmul(T,invS))
    spectralRadius.append(np.max(np.abs(l)))

plt.figure()
plt.plot(p, spectralRadius)
plt.plot([min(p), max(p)], [1, 1], 'r--')
plt.show()
