import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from numpy.matlib import repmat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.linalg import inv
import utilities

nInfectiousStates = 100
lengthOfInfectiousness = 21
nStates = nInfectiousStates + 2
threshold = 0.2
maxRate = 1
timeToMaxRate = 4

tStates = np.linspace(0.0, lengthOfInfectiousness, nInfectiousStates)
dt = tStates[1] - tStates[0]

b = utilities.betaVL(tStates, threshold, maxRate, timeToMaxRate)

plt.figure()
plt.subplot(121)
plt.title(r"$\beta(t)$ and infectious threshold")
plt.plot(tStates, utilities.betaVL(tStates, 0, maxRate, timeToMaxRate), linewidth=2)
plt.plot([np.min(tStates), np.max(tStates)], [threshold, threshold], 'k--', linewidth=2)
plt.xlabel("Time (days)", fontsize=14)
plt.ylabel("Probability of transmission", fontsize=14)
plt.subplot(122)
plt.title(r"$\beta(t)I_{\beta(t)\geq threshold}$")
plt.plot(tStates, utilities.betaVL(tStates, threshold, maxRate, timeToMaxRate), linewidth=2)
plt.show()

plt.figure()
plt.subplot(121)
plt.title(r"$\beta(t)$ and infectious threshold")
plt.plot(tStates, utilities.betaVL(tStates, 0, maxRate, timeToMaxRate), linewidth=2)
plt.plot([np.min(tStates), np.max(tStates)], [threshold, threshold], 'k--', linewidth=2)
plt.xlabel("Time (days)", fontsize=14)
plt.ylabel("Probability of transmission", fontsize=14)
plt.subplot(122)
plt.title(r"$\beta(t)I_{\beta(t)\geq threshold}$")
plt.plot(tStates, utilities.betaVL(tStates, threshold, maxRate, timeToMaxRate), linewidth=2, zorder=1)
plt.scatter(tStates[::7], utilities.betaVL(tStates[::7], threshold, maxRate, timeToMaxRate), s=50, color="black", zorder=2)
plt.show()
