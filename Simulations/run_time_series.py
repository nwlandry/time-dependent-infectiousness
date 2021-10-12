import networkx as nx
import numpy as np
from simulations import *
import math
from scipy.integrate import quad
from activity_model import activity_model
import shelve

n = 10000
tmin = 0
tmax = 300
dt = 1
rho = 0.01
tauR = 21
time_to_max = 4
R0 = 3

minDegree = 10
maxDegree = 1000
exponent = 3
k = generatePowerLawDegreeSequence(n, minDegree, maxDegree, exponent)
static_network = nx.configuration_model(k)

activities = invCDFPowerLaw(np.random.rand(n), 0.01, 1, exponent)
m = 10

temporal_network = activity_model(activities, m, tmin, tmax, dt)

beta_unscaled = lambda tau : math.e/time_to_max*tau*np.exp(-tau/time_to_max)
exposure = quad(beta_unscaled, 0, tauR)[0]
lambda_config_model = np.mean(np.power(k, 2)) / np.mean(k)
lambda_activity_model = m*np.mean(activities) + m*math.sqrt(np.mean(np.power(activities, 2)))

# static
beta_gamma = lambda tau : R0*math.e/time_to_max*tau*np.exp(-tau/time_to_max)/(exposure*lambda_config_model)
mean_beta_gamma = quad(beta_gamma, 0, tauR)[0]/tauR
beta_const = lambda tau : mean_beta_gamma
gamma = 1/tauR
beta = mean_beta_gamma

initial_infecteds = np.random.randint(0, n, size=int(rho*n))
t_SIR, S_SIR, I_SIR, R_SIR = SIR_model_static_network(static_network, gamma, beta, dt=dt, initial_infecteds=initial_infecteds, tmin=0, tmax=tmax)
t_VL_const, S_VL_const, I_VL_const, R_VL_const = VL_model_static_network(static_network, [beta_const]*n, [tauR]*n, dt=dt, initial_infecteds=initial_infecteds, tmin=0, tmax=tmax)
t_VL_gamma, S_VL_gamma, I_VL_gamma, R_VL_gamma = VL_model_static_network(static_network, [beta_gamma]*n, [tauR]*n, dt=dt, initial_infecteds=initial_infecteds, tmin=0, tmax=tmax)

with shelve.open("Simulations/numerical_sims") as data:
    data["t-SIR-static"] = t_SIR
    data["S-SIR-static"] = S_SIR/n
    data["I-SIR-static"] = I_SIR/n
    data["R-SIR-static"] = R_SIR/n

    data["t-VL-const-static"] = t_VL_const
    data["S-VL-const-static"] = S_VL_const/n
    data["I-VL-const-static"] = I_VL_const/n
    data["R-VL-const-static"] = R_VL_const/n

    data["t-VL-gamma-static"] = t_VL_gamma
    data["S-VL-gamma-static"] = S_VL_gamma/n
    data["I-VL-gamma-static"] = I_VL_gamma/n
    data["R-VL-gamma-static"] = R_VL_gamma/n


# temporal
beta_gamma = lambda tau : R0*math.e/time_to_max*tau*np.exp(-tau/time_to_max)/(exposure*lambda_activity_model)
mean_beta_gamma = quad(beta_gamma, 0, tauR)[0]/tauR
beta_const = lambda tau : mean_beta_gamma
gamma = 1/tauR
beta = 2*m*np.mean(activities)*mean_beta_gamma

initial_infecteds = np.random.randint(0, n, size=int(rho*n))

t_SIR, S_SIR, I_SIR, R_SIR = SIR_model_temporal_network(temporal_network, n, gamma, beta, dt=dt, initial_infecteds=initial_infecteds, tmin=0, tmax=tmax)
t_VL_const, S_VL_const, I_VL_const, R_VL_const = VL_model_temporal_network(temporal_network, n, [beta_const]*n, [tauR]*n, dt=dt, initial_infecteds=initial_infecteds, tmin=0, tmax=tmax)
t_VL_gamma, S_VL_gamma, I_VL_gamma, R_VL_gamma = VL_model_temporal_network(temporal_network, n, [beta_gamma]*n, [tauR]*n, dt=dt, initial_infecteds=initial_infecteds, tmin=0, tmax=tmax)

with shelve.open("Simulations/numerical_sims") as data:
    data["t-SIR-temporal"] = t_SIR
    data["S-SIR-temporal"] = S_SIR/n
    data["I-SIR-temporal"] = I_SIR/n
    data["R-SIR-temporal"] = R_SIR/n

    data["t-VL-const-temporal"] = t_VL_const
    data["S-VL-const-temporal"] = S_VL_const/n
    data["I-VL-const-temporal"] = I_VL_const/n
    data["R-VL-const-temporal"] = R_VL_const/n

    data["t-VL-gamma-temporal"] = t_VL_gamma
    data["S-VL-gamma-temporal"] = S_VL_gamma/n
    data["I-VL-gamma-temporal"] = I_VL_gamma/n
    data["R-VL-gamma-temporal"] = R_VL_gamma/n