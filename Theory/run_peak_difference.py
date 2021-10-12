import numpy as np
from scipy.integrate import solve_ivp
from utilities import *
import shelve

nInfectiousStates = [5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000]
tauR = 21
threshold = 0.0
maxRate = 1
timeToMaxRate = 4
n = 10000
R0 = 3
tmax = 100
initialFractionInfected = 0.01

time_SIR = list()
time_VL_const = list()
time_VL_gamma = list()
magnitude_SIR = list()
magnitude_VL_const = list()
magnitude_VL_gamma = list()

for stages in nInfectiousStates:
    nStates = stages + 2

    if stages > 1:
        tau = np.linspace(0.0, tauR, stages)
        dtau = tau[1] - tau[0]
    else:
        tau = np.array([tauR])
        dtau = tauR
    bFunction = betaVL(tau, threshold, maxRate, timeToMaxRate)

    bScaled = bFunction/(np.sum(bFunction)*dtau)
    beta_gamma = R0*bScaled
    beta_const = betaConstant(tau, np.mean(beta_gamma))
    beta = np.sum(beta_gamma)*dtau/tauR
    gamma = 1/tauR

    ### Fully mixed
    initialStatesVL = np.zeros(nStates)
    initialStatesVL[1] = initialFractionInfected
    initialStatesVL[0] = 1 - initialFractionInfected

    initialStatesSIR = [1 - initialFractionInfected, initialFractionInfected, 0]

    sol = solve_ivp(SIRModelFullyMixed, (0, tmax), initialStatesSIR, t_eval=np.arange(0, tmax, 0.01), args=(beta, gamma))
    t = sol.t
    y = sol.y.T

    time_SIR.append(t[np.argmax(np.sum(y[:, 1:-1], axis=1))])
    magnitude_SIR.append(np.max(np.sum(y[:, 1:-1], axis=1)))

    sol = solve_ivp(viralLoadModelFullyMixed, (0, tmax), initialStatesVL, t_eval=np.arange(0, tmax, 0.01), args=(beta_const, dtau))
    t = sol.t
    y = sol.y.T

    time_VL_const.append(t[np.argmax(np.sum(y[:, 1:-1], axis=1))])
    magnitude_VL_const.append(np.max(np.sum(y[:, 1:-1], axis=1)))


    sol = solve_ivp(viralLoadModelFullyMixed, (0, tmax), initialStatesVL, t_eval=np.arange(0, tmax, 0.01), args=(beta_gamma, dtau))
    t = sol.t
    y = sol.y.T

    time_VL_gamma.append(t[np.argmax(np.sum(y[:, 1:-1], axis=1))])
    magnitude_VL_gamma.append(np.max(np.sum(y[:, 1:-1], axis=1)))



with shelve.open("Theory/peak_difference") as data:
    data["num-states"] = nInfectiousStates
    data["time-SIR"] = time_SIR
    data["mag-SIR"] = magnitude_SIR
    data["time-VL-const"] = time_VL_const
    data["mag-VL-const"] = magnitude_VL_const
    data["time-VL-gamma"] = time_VL_gamma
    data["mag-VL-gamma"] = magnitude_VL_gamma