import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from numpy.matlib import repmat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.linalg import inv
from utilities import *

tmax = 75

nInfectiousStates = 15
lengthOfInfectiousness = 14
nStates = nInfectiousStates + 2
threshold = 0.2
maxRate = 1
timeToMaxRate = 4

tauStates = np.linspace(0.0, lengthOfInfectiousness, nInfectiousStates)
dtau = tauStates[1] - tauStates[0]

b = betaVL(tauStates, threshold, maxRate, timeToMaxRate)
bConst = betaConstant(tauStates, np.mean(b))

N = 1000
initialStates = np.zeros(nStates)
initialStates[1] = 1/N
initialStates[0] = 1 - np.sum(initialStates[1:])/N

sol1 = solve_ivp(viralLoadModelFullyMixed, (0, tmax), initialStates, t_eval=np.arange(0, tmax, 0.1), args=(b, dtau))
t1 = sol1.t
y1 = sol1.y.T

sol2 = solve_ivp(viralLoadModelFullyMixed, (0, tmax), initialStates, t_eval=np.arange(0, tmax, 0.1), args=(bConst, dtau))
t2 = sol2.t
y2 = sol2.y.T

plt.figure()
plt.plot([0, tmax], [0, 0], 'k--')
plt.plot(t1, y1[:, 0], label="S")
for i in np.arange(1, nStates-1, 4, dtype=int):
    plt.plot(t1, y1[:, i], label="Infected for "+ str(i-1) + " days")
plt.plot(t1, y1[:, -1], label="R")
plt.ylabel("Fraction of the population", fontsize=14)
plt.xlabel("Time (days)", fontsize=14)
plt.legend()
plt.show()

plt.figure()
plt.subplot(311)
plt.title(r"$\beta(\tau)\propto\tau e^{-\tau/4}$", fontsize=14)
plt.imshow(np.flipud(y1[:,1:-1].T), cmap=cm.coolwarm, aspect="auto", interpolation="spline16", extent=(0, tmax, 0, nStates), vmin=0, vmax=0.15)
plt.colorbar()

plt.subplot(312)
plt.title(r"$\beta(\tau)=c$", fontsize=14)
plt.imshow(np.flipud(y2[:,1:-1].T), cmap=cm.coolwarm, aspect="auto", interpolation="spline16", extent=(0, tmax, 0, nStates), vmin=0, vmax=0.15)
plt.ylabel("Days infected", fontsize=14)
plt.colorbar()

plt.subplot(313)
plt.title("Difference", fontsize=14)
plt.imshow(np.flipud(y1[:,1:-1].T - y2[:,1:-1].T), aspect="auto", interpolation="spline16", extent=(0, tmax, 0, nStates), vmin=-0.075, vmax=0.075)
plt.colorbar()
plt.xlabel("Time (days)", fontsize=14)
plt.tight_layout()
plt.savefig("heatmap.png", dpi=600)
plt.show()
