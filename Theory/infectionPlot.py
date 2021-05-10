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

tStates = np.linspace(0.0, lengthOfInfectiousness, nInfectiousStates)
dt = tStates[1] - tStates[0]

b = betaVL(tStates, threshold, maxRate, timeToMaxRate)


bScaled = b/(np.sum(b)*dt)
b = b0*bScaled
bConst = betaConstant(tStates, np.mean(b))
beta = np.mean(b)
gamma = 1/lengthOfInfectiousness

tmax = 100
initialFractionInfected = 0.01

### Fully mixed
initialStatesVL = np.zeros(nStates)
initialStatesVL[1] = initialFractionInfected
initialStatesVL[0] = 1 - initialFractionInfected

initialStatesSIR = [1 - initialFractionInfected, initialFractionInfected, 0]

sol1 = solve_ivp(viralLoadModelFullyMixed, (0, tmax), initialStatesVL, t_eval=np.arange(0, tmax, 0.1), args=(b, dt))
t1 = sol1.t
y1 = sol1.y.T

sol2 = solve_ivp(viralLoadModelFullyMixed, (0, tmax), initialStatesVL, t_eval=np.arange(0, tmax, 0.1), args=(bConst, dt))
t2 = sol2.t
y2 = sol2.y.T

sol3 = solve_ivp(SIRModelFullyMixed, (0, tmax), initialStatesSIR, t_eval=np.arange(0, tmax, 0.1), args=(beta, gamma))
t3 = sol3.t
y3 = sol3.y.T

reds = cm.get_cmap("Reds")
blues = cm.get_cmap("Blues")
greens = cm.get_cmap("Greens")

plt.figure(figsize=(8,6))
plt.plot(t1, np.sum(y1[:,1:-1], axis=1), linewidth=2, color=reds(0.4))
plt.plot(t2, np.sum(y2[:,1:-1], axis=1), color=greens(0.4))
plt.plot(t3, y3[:,-2], linewidth=2, color=blues(0.4))
plt.ylim([0, 1])

# Uniform degree distribution
degrees = np.random.randint(10, 30, size=n)
P = generateConfigurationModelP(degrees)
k = len(P)

spectralRadius = np.max(np.abs(np.linalg.eigvals(P)))
bScaled = b/(spectralRadius*np.sum(b)*dt)
b = k*b0*bScaled
bConst = betaConstant(tStates, np.mean(b))
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

plt.plot(t1, np.sum(y1[:,k:-k], axis=1), label=r"$\beta(t)\propto t e^{-t/4}$, (VL Model)", linewidth=2, color=reds(0.6))
plt.plot(t2, np.sum(y2[:,k:-k], axis=1), label=r"$\beta(t)=c$, (VL Model)", linewidth=2, color=greens(0.6))
plt.plot(t3, np.sum(y3[:,k:2*k], axis=1), label="SIR Model", linewidth=2, color=blues(0.6))
plt.ylabel("Total fraction infected", fontsize=16)
plt.ylim([0, 1])

### Power Law degree distribution
degrees = generatePowerLawDegreeSequence(n, 10, 100, 3)
P = generateConfigurationModelP(degrees)
k = len(P)

spectralRadius = np.max(np.abs(np.linalg.eigvals(P)))
bScaled = b/(spectralRadius*np.sum(b)*dt)
b = k*b0*bScaled
bConst = betaConstant(tStates, np.mean(b))
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

plt.plot(t1, np.sum(y1[:,k:-k], axis=1), linewidth=2, color=reds(0.8))
plt.plot(t2, np.sum(y2[:,k:-k], axis=1), linewidth=2, color=greens(0.8))
plt.plot(t3, np.sum(y3[:,k:2*k], axis=1), linewidth=2, color=blues(0.8))
plt.xlabel("time (days)", fontsize=16)
plt.ylim([0, 1])
plt.text(1, 0.8, "Light: Fully-mixed\nMedium: Uniform degree distribution\nDark: Power-law degree distribution", fontsize=12)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("infection_curves.png", dpi=600)
plt.savefig("infection_curves.pdf")
plt.show()
