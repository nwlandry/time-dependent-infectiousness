import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from numpy.matlib import repmat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.linalg import inv
from utilities import *

nInfectiousStates = 15
lengthOfInfectiousness = 14
nStates = nInfectiousStates + 2
threshold = 0.0
maxRate = 100
timeToMaxRate = 4

tStates = np.linspace(0.0, lengthOfInfectiousness, nInfectiousStates)
dt = tStates[1] - tStates[0]

b = betaVL(tStates, threshold, maxRate, timeToMaxRate)
bConst = betaConstant(tStates, np.mean(b))
beta = np.mean(b)
gamma = 1/lengthOfInfectiousness
tmax = 50
initialFractionInfected = 0.01

degrees = np.random.randint(1, 11, size=100)
degrees = generatePowerLawDegreeSequence(100, 3, 100, 3)
P = generateConfigurationModelP(degrees)
k = len(P)

initialStatesVL = np.zeros(nStates*k)
initialInfected = np.random.rand(k)
initialStatesVL[k:2*k] = initialFractionInfected*initialInfected/np.sum(initialInfected)
initialSusceptible = np.random.rand(k)
initialStatesVL[:k] = (1 - initialFractionInfected)*initialSusceptible/np.sum(initialSusceptible)

initialStatesSIR = np.zeros(3*k)
initialStatesSIR[k:2*k] = initialFractionInfected*initialInfected/np.sum(initialInfected)
initialStatesSIR[:k] = (1 - initialFractionInfected)*initialSusceptible/np.sum(initialSusceptible)

sol1 = solve_ivp(viralLoadModelDegreeBased, (0, tmax), initialStatesVL, t_eval=np.arange(0, tmax, 0.1), args=(P, b, dt))
t1 = sol1.t
y1 = sol1.y.T

sol2 = solve_ivp(viralLoadModelDegreeBased, (0, tmax), initialStatesVL, t_eval=np.arange(0, tmax, 0.1), args=(P, bConst, dt))
t2 = sol2.t
y2 = sol2.y.T
#
sol3 = solve_ivp(SIRModelDegreeBased, (0, tmax), initialStatesSIR, t_eval=np.arange(0, tmax, 0.1), args=(P, beta, gamma))
t3 = sol3.t
y3 = sol3.y.T

plt.figure()
plt.plot(t1, np.sum(y1[:,k:-k], axis=1), label=r"$\beta(t)\propto\frac{e}{4} t e^{-t/4}$, (VL Model)")
plt.plot(t2, np.sum(y2[:,k:-k], axis=1), label=r"$\beta(t)=c$, (VL Model)")
plt.plot(t3, np.sum(y3[:,k:2*k], axis=1), label="SIR Model")
plt.xlabel("time (days)", fontsize=14)
plt.ylabel("Fraction infected (summed over all stages)", fontsize=14)
plt.ylim([0, 1])
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(t1, y1[:,k:-k].dot(repmat(b, 1, k).T), label=r"$\beta(t)\propto\frac{e}{4} t e^{-t/4}$, (VL Model)")
plt.plot(t2, y2[:,k:-k].dot(repmat(bConst, 1, k).T), label=r"$\beta(t)=c$, (VL Model)")
plt.plot(t3, beta*np.sum(y3[:,k:2*k], axis=1), label="SIR Model")
plt.xlabel("time (days)", fontsize=14)
plt.ylabel("Infection rate (summed over all stages)", fontsize=14)
plt.ylim([0, 7])
plt.legend()
plt.tight_layout()
plt.show()
