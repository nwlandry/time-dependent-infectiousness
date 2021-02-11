import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from numpy.matlib import repmat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.linalg import inv
from utilities import *



nInfectiousStates = 10
tStates = np.linspace(0, 20, nInfectiousStates)
dt = tStates[1] - tStates[0]
nStates = len(tStates)+2

b = betaVL(tStates, 0.0, 0.1)
bConst = betaConstant(tStates, np.mean(b))
beta = np.sum(b)*dt
gamma = 1/dt

N = 1000
tmax = 200

numInfected = 1
initialStatesVL = np.zeros(nStates)
initialStatesVL[1] = numInfected/N
initialStatesVL[0] = 1 - numInfected/N

initialStatesSIR = [(N-numInfected)/N, numInfected/N, 0]

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
