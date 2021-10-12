import networkx as nx
import random
import numpy as np
from simulations import *
import math
from scipy.integrate import quad
from activity_model import activity_model
import shelve

n = 10000
tmin = 0
tmax = 125
dt = 1

minDegree = 10
maxDegree = 1000
exponent = 3
k = generatePowerLawDegreeSequence(n, minDegree, maxDegree, exponent)
static_network = nx.configuration_model(k)


activities = invCDFPowerLaw(np.random.rand(n), 0.01, 1, exponent)
m = 10
temporal_network = activity_model(activities, m, tmin, tmax, dt)

rho = 0.01
tauR = 21
interval = 10
tauR = np.random.uniform(tauR-interval/2, tauR+interval/2, n)
time_to_max = 4
interval = 4
time_to_max  = np.random.uniform(time_to_max-interval/2, time_to_max+interval/2, n)
R0 = 3

exposure = 0
for t, upper_bound in zip(time_to_max, tauR):
    beta_unscaled = lambda tau : math.e/t*tau*np.exp(-tau/t)
    exposure += quad(beta_unscaled, 0, upper_bound)[0]/n

lambda_config_model = np.mean(np.power(k, 2)) / np.mean(k)
lambda_activity_model = m*np.mean(activities) + m*math.sqrt(np.mean(np.power(activities, 2)))


# static
beta_gamma = [lambda tau : R0*math.e/t*tau*np.exp(-tau/t)/(exposure*lambda_config_model) for t in time_to_max]
initial_infecteds = np.random.randint(0, n, size=int(rho*n))
_, _, _, _, infection_distribution_static = VL_model_static_network(static_network, beta_gamma, tauR, dt=dt, initial_infecteds=initial_infecteds, tmin=0, tmax=tmax, return_infection_distribution=True)


# temporal
beta_gamma = [lambda tau : R0*math.e/t*tau*np.exp(-tau/t)/(exposure*lambda_activity_model) for t in time_to_max]

initial_infecteds = np.random.randint(0, n, size=int(rho*n))
_, _, _, _, infection_distribution_temporal = VL_model_temporal_network(temporal_network, n, beta_gamma, tauR, dt=dt, initial_infecteds=initial_infecteds, tmin=0, tmax=tmax, return_infection_distribution=True)

with shelve.open("Simulations/heatmap") as data:
    data["infection-distribution-static"] = infection_distribution_static
    data["infection-distribution-temporal"] = infection_distribution_temporal
    data["tmax"] = tmax
    data["tauR"] =  max(tauR)
    data["dt"] = dt