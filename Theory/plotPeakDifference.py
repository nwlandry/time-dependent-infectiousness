import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from numpy.matlib import repmat
from utilities import *

nInfectiousStates = [5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000]
lengthOfInfectiousness = 21
threshold = 0.2
maxRate = 1
timeToMaxRate = 4
n = 10000
b0 = 3
tmax = 100
initialFractionInfected = 0.01

timeDiffConstant = list()
timeDiffVL = list()
magDiffConstant = list()
magDiffVL = list()

for stages in nInfectiousStates:
    nStates = stages + 2

    if stages > 1:
        tauStates = np.linspace(0.0, lengthOfInfectiousness, stages)
        dtau = tauStates[1] - tauStates[0]
    else:
        tauStates = np.array([lengthOfInfectiousness])
        dtau = lengthOfInfectiousness
    bFunction = betaVL(tauStates, threshold, maxRate, timeToMaxRate)

    bScaled = bFunction/(np.sum(bFunction)*dtau)
    b = b0*bScaled
    bConst = betaConstant(tauStates, np.mean(b))
    beta = np.mean(b)
    gamma = 1/lengthOfInfectiousness

    ### Fully mixed
    initialStatesVL = np.zeros(nStates)
    initialStatesVL[1] = initialFractionInfected
    initialStatesVL[0] = 1 - initialFractionInfected

    initialStatesSIR = [1 - initialFractionInfected, initialFractionInfected, 0]

    sol1 = solve_ivp(viralLoadModelFullyMixed, (0, tmax), initialStatesVL, t_eval=np.arange(0, tmax, 0.01), args=(b, dtau))
    t1 = sol1.t
    y1 = sol1.y.T

    sol2 = solve_ivp(viralLoadModelFullyMixed, (0, tmax), initialStatesVL, t_eval=np.arange(0, tmax, 0.01), args=(bConst, dtau))
    t2 = sol2.t
    y2 = sol2.y.T

    sol3 = solve_ivp(SIRModelFullyMixed, (0, tmax), initialStatesSIR, t_eval=np.arange(0, tmax, 0.01), args=(beta, gamma))
    t3 = sol3.t
    y3 = sol3.y.T

    timeDiffVL.append(t1[np.argmax(np.sum(y1[:, 1:-1], axis=1))] - t3[np.argmax(y3[:,1])])
    timeDiffConstant.append(t2[np.argmax(np.sum(y2[:, 1:-1], axis=1))] - t3[np.argmax(y3[:,1])])
    magDiffVL.append(np.max(np.sum(y1[:, 1:-1], axis=1)) - np.max(y3[:,1]))
    magDiffConstant.append(np.max(np.sum(y2[:, 1:-1], axis=1)) - np.max(y3[:,1]))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.semilogx(nInfectiousStates, timeDiffVL, 'ks-', linewidth=2, label="VL model, " + r"$\beta(t)\propto t e^{-t/4}$")
plt.plot(nInfectiousStates, timeDiffConstant, 'ko-', linewidth=2, label="VL model, " + r"$\beta(t)=c$")
plt.xlabel("Number of infectious states", fontsize=14)
plt.ylabel(r"$\argmax_t\left(\sum_j I_j(t)\right)-\argmax_t(I_{SIR}(t))$", fontsize=14)
plt.subplot(1, 2, 2)
plt.semilogx(nInfectiousStates, magDiffVL, 'ks-', linewidth=2, label="VL model, " + r"$\beta(t)\propto t e^{-t/4}$")
plt.plot(nInfectiousStates, magDiffConstant, 'ko-', linewidth=2, label="VL model, " + r"$\beta(t)=c$")
plt.xlabel("Number of infectious states", fontsize=14)
plt.ylabel(r"$\max\left(\sum_j I_j(t)\right)-\max(I_{SIR}(t))$", fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("Figures/peak_difference.png", dpi=600)
plt.savefig("Figures/peak_difference.pdf")
plt.show()
