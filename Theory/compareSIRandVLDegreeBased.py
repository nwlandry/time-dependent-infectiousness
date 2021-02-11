import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from numpy.matlib import repmat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.linalg import inv
from utilities import *



nInfectiousStates = 20
tStates = np.linspace(0, 20, nInfectiousStates)
dt = tStates[1] - tStates[0]
nStates = len(tStates) + 2

b = betaVL(tStates, 0.0, 0.2)
bConst = betaConstant(tStates, np.mean(b))
beta = np.sum(b)/dt
gamma = 1/dt

tmax = 100

degrees = np.arange(1, 6, 1)
k = len(degrees)
meanDegree = 2
p = 0.1*np.ones(5)

P = generateConfigurationModelP(degrees, meanDegree, p)

initialFractionInfected = 0.01
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
#plt.plot(t1, np.sum(y1[:,-k:], axis=1), label=r"$\beta(t)\propto\frac{e}{4} t e^{-t/4}$, (VL Model)")
plt.plot(t2, np.sum(y2[:,k:-k], axis=1), label=r"$\beta(t)=c$, (VL Model)")
plt.plot(t3, np.sum(y3[:,k:2*k], axis=1), label="SIR Model")
plt.xlabel("time (days)")
plt.ylabel("Fraction infected (summed over all stages)")
plt.legend()
plt.show()
