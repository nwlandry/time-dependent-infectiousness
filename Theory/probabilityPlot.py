import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from numpy.matlib import repmat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.linalg import inv
from utilities import *

nInfectiousStates = 100
lengthOfInfectiousness = 21
nStates = nInfectiousStates + 2
threshold = 0.2
maxRate = 1
timeToMaxRate = 4
n = 10000
b0 = 3

dtOutput = 5

tauStates = np.linspace(0.0, lengthOfInfectiousness, nInfectiousStates)
dtau = tauStates[1] - tauStates[0]

bFunction = betaVL(tauStates, threshold, maxRate, timeToMaxRate)


bScaled = bFunction/(np.sum(bFunction)*dtau)
b = b0*bScaled
bConst = betaConstant(tauStates, np.mean(b))
beta = np.mean(b)
gamma = 1/lengthOfInfectiousness

tmax = 100
initialFractionInfected = 0.01

### Fully mixed
initialStatesVL = np.zeros(nStates)
initialStatesVL[1] = initialFractionInfected
initialStatesVL[0] = 1 - initialFractionInfected

initialStatesSIR = [1 - initialFractionInfected, initialFractionInfected, 0]

sol1 = solve_ivp(viralLoadModelFullyMixed, (0, tmax), initialStatesVL, t_eval=np.arange(0, tmax, dtOutput), args=(b, dtau))
t1 = sol1.t
y1 = sol1.y.T

sol2 = solve_ivp(viralLoadModelFullyMixed, (0, tmax), initialStatesVL, t_eval=np.arange(0, tmax, dtOutput), args=(bConst, dtau))
t2 = sol2.t
y2 = sol2.y.T

sol3 = solve_ivp(SIRModelFullyMixed, (0, tmax), initialStatesSIR, t_eval=np.arange(0, tmax, dtOutput), args=(beta, gamma))
t3 = sol3.t
y3 = sol3.y.T

reds = cm.get_cmap("Reds")
blues = cm.get_cmap("Blues")
greens = cm.get_cmap("Greens")

plt.figure(figsize=(8,6))
plt.plot(t1, y1[:,1:-1].dot(b)/n*dtau, 'ks-', linewidth=2)
plt.plot(t2, y2[:,1:-1].dot(bConst)/n*dtau, 'ko-', linewidth=2)
plt.plot(t3, beta*y3[:,-2]/n*dtau, 'k^-', linewidth=2)

# Uniform degree distribution
degrees = np.random.randint(10, 30, size=n)
P = generateConfigurationModelP(degrees)
k = len(P)

spectralRadius = np.max(np.abs(np.linalg.eigvals(P)))
bScaled = bFunction/(spectralRadius*np.sum(bFunction)*dtau)
b = k*b0*bScaled
bConst = betaConstant(tauStates, np.mean(b))
beta = np.mean(b)
gamma = 1/lengthOfInfectiousness

initialStatesVL = np.zeros(nStates*k)
initialInfected = np.random.rand(k)
initialStatesVL[k:2*k] = initialFractionInfected*initialInfected/np.sum(initialInfected)
initialSusceptible = np.random.rand(k)
initialStatesVL[:k] = (1 - initialFractionInfected)*initialSusceptible/np.sum(initialSusceptible)

initialStatesSIR = np.zeros(3*k)
initialStatesSIR[k:2*k] = initialFractionInfected*initialInfected/np.sum(initialInfected)
initialStatesSIR[:k] = (1 - initialFractionInfected)*initialSusceptible/np.sum(initialSusceptible)

sol1 = solve_ivp(viralLoadModelDegreeBased, (0, tmax), initialStatesVL, t_eval=np.arange(0, tmax, dtOutput), args=(P, b, dtau))
t1 = sol1.t
y1 = sol1.y.T

sol2 = solve_ivp(viralLoadModelDegreeBased, (0, tmax), initialStatesVL, t_eval=np.arange(0, tmax, dtOutput), args=(P, bConst, dtau))
t2 = sol2.t
y2 = sol2.y.T
#
sol3 = solve_ivp(SIRModelDegreeBased, (0, tmax), initialStatesSIR, t_eval=np.arange(0, tmax, dtOutput), args=(P, beta, gamma))
t3 = sol3.t
y3 = sol3.y.T

plt.plot(t1, y1[:,k:-k].dot(repmat(b, 1, k).T)/n*dtau, 'ks--', linewidth=2)
plt.plot(t2, y2[:,k:-k].dot(repmat(bConst, 1, k).T)/n*dtau, 'ko--', linewidth=2)
plt.plot(t3, beta*np.sum(y3[:,k:2*k], axis=1)/n*dtau, 'k^--', linewidth=2)

### Power Law degree distribution
degrees = generatePowerLawDegreeSequence(n, 10, 1000, 3)
P = generateConfigurationModelP(degrees)
k = len(P)

spectralRadius = np.max(np.abs(np.linalg.eigvals(P)))
bScaled = bFunction/(spectralRadius*np.sum(bFunction)*dtau)
b = k*b0*bScaled
bConst = betaConstant(tauStates, np.mean(b))
beta = np.mean(b)
gamma = 1/lengthOfInfectiousness

initialStatesVL = np.zeros(nStates*k)
initialInfected = np.random.rand(k)
initialStatesVL[k:2*k] = initialFractionInfected*initialInfected/np.sum(initialInfected)
initialSusceptible = np.random.rand(k)
initialStatesVL[:k] = (1 - initialFractionInfected)*initialSusceptible/np.sum(initialSusceptible)

initialStatesSIR = np.zeros(3*k)
initialStatesSIR[k:2*k] = initialFractionInfected*initialInfected/np.sum(initialInfected)
initialStatesSIR[:k] = (1 - initialFractionInfected)*initialSusceptible/np.sum(initialSusceptible)

sol1 = solve_ivp(viralLoadModelDegreeBased, (0, tmax), initialStatesVL, t_eval=np.arange(0, tmax, dtOutput), args=(P, b, dtau))
t1 = sol1.t
y1 = sol1.y.T

sol2 = solve_ivp(viralLoadModelDegreeBased, (0, tmax), initialStatesVL, t_eval=np.arange(0, tmax, dtOutput), args=(P, bConst, dtau))
t2 = sol2.t
y2 = sol2.y.T
#
sol3 = solve_ivp(SIRModelDegreeBased, (0, tmax), initialStatesSIR, t_eval=np.arange(0, tmax, dtOutput), args=(P, beta, gamma))
t3 = sol3.t
y3 = sol3.y.T

plt.plot(t1, y1[:,k:-k].dot(repmat(b, 1, k).T)/n*dtau, 'ks-.', linewidth=2)
plt.plot(t2, y2[:,k:-k].dot(repmat(bConst, 1, k).T)/n*dtau, 'ko-.', linewidth=2)
plt.plot(t3, beta*np.sum(y3[:,k:2*k], axis=1)/n*dtau, 'k^-.', linewidth=2)
plt.xlabel("time (days)", fontsize=16)
plt.ylabel("Average infection probability", fontsize=16)

plt.tight_layout()
plt.savefig("Figures/probability_curves.png", dpi=600)
plt.savefig("Figures/probability_curves.pdf")
plt.show()
