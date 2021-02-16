import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from numpy.matlib import repmat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.linalg import inv
from utilities import *



nInfectiousStates = 8
lengthOfInfectiousness = 7
nStates = nInfectiousStates + 2
threshold = 0.2
maxRate = 1
timeToMaxRate = 4

tStates = np.linspace(0.0, lengthOfInfectiousness, nInfectiousStates)
dt = tStates[1] - tStates[0]

b = betaVL(tStates, threshold, maxRate, timeToMaxRate)
bConst = betaConstant(tStates, np.mean(b))
beta = np.sum(b)
gamma = 1/dt

tmax = 40
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

plt.figure()
plt.plot(t1, np.sum(y1[:,1:-1], axis=1), label=r"$\beta(t)\propto\frac{e}{4} t e^{-t/4}$, (VL Model)")
plt.plot(t2, np.sum(y2[:,1:-1], axis=1), label=r"$\beta(t)=c$, (VL Model)")
plt.plot(t3, y3[:,-2], label="SIR Model")
plt.xlabel("time (days)")
plt.ylabel("Fraction infected (summed over all stages)")
plt.legend()
plt.show()
