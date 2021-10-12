import networkx as nx
import random
import numpy as np
from numpy.random.mtrand import randint
from simulations import *
import math
from scipy.integrate import quad
from activity_model import activity_model
import shelve
import multiprocessing as mp
import os
from functools import partial


def get_static_extent(static_network, time_to_max, tauR, exposure, R0, lambda_config_model, dt, rho, num_sims):
    n - static_network.number_of_nodes()
    beta_gamma = lambda tau : R0*math.e/time_to_max*tau*np.exp(-tau/time_to_max)/(exposure*lambda_config_model)
    mean_beta_gamma = quad(beta_gamma, 0, tauR)[0]/tauR
    beta_const = lambda tau : mean_beta_gamma
    gamma = 1/tauR
    beta = mean_beta_gamma

    extent_SIR = 0
    extent_VL_const = 0
    extent_VL_gamma = 0
    for i in range(num_sims):
        initial_infecteds = np.random.choice(n, size=int(rho*n), replace=False)
        _, _, _, R_SIR = SIR_model_static_network(static_network, gamma, beta, dt=dt, initial_infecteds=initial_infecteds, tmin=0)
        extent_SIR += max(R_SIR)/(n*num_sims)
        _, _, _, R_VL_const = VL_model_static_network(static_network, [beta_const]*n, [tauR]*n, dt=dt, initial_infecteds=initial_infecteds, tmin=0)
        extent_VL_const += max(R_VL_const)/(n*num_sims)
        _, _, _, R_VL_gamma = VL_model_static_network(static_network, [beta_gamma]*n, [tauR]*n, dt=dt, initial_infecteds=initial_infecteds, tmin=0)
        extent_VL_gamma += max(R_VL_gamma)/(n*num_sims)
    print("R0 = " + str(R0) + " completed", flush=True)
    return extent_SIR, extent_VL_const, extent_VL_gamma
    
def get_temporal_extent(temporal_network, n, m, time_to_max, tauR, exposure, R0, lambda_activity_model, dt, rho, num_sims):
    beta_gamma = lambda tau : R0*math.e/time_to_max*tau*np.exp(-tau/time_to_max)/(exposure*lambda_activity_model)
    mean_beta_gamma = quad(beta_gamma, 0, tauR)[0]/tauR
    beta_const = lambda tau : mean_beta_gamma
    gamma = 1/tauR
    beta = R0*2*m*np.mean(activities)*gamma/lambda_activity_model # from Perra et al.

    extent_SIR = 0
    extent_VL_const = 0
    extent_VL_gamma = 0
    for i in range(num_sims):
        initial_infecteds = np.random.choice(n, size=int(rho*n), replace=False)
        _, _, _, R_SIR = SIR_model_temporal_network(temporal_network, n, gamma, beta, dt=dt, initial_infecteds=initial_infecteds, tmin=0)
        extent_SIR += max(R_SIR)/(n*num_sims)
        _, _, _, R_VL_const = VL_model_temporal_network(temporal_network, n, [beta_const]*n, [tauR]*n, dt=dt, initial_infecteds=initial_infecteds, tmin=0)
        extent_VL_const += max(R_VL_const)/(n*num_sims)
        _, _, _, R_VL_gamma = VL_model_temporal_network(temporal_network, n, [beta_gamma]*n, [tauR]*n, dt=dt, initial_infecteds=initial_infecteds, tmin=0)
        extent_VL_gamma += max(R_VL_gamma)/(n*num_sims)
    print("R0 = " + str(R0) + " completed", flush=True)
    return extent_SIR, extent_VL_const, extent_VL_gamma
    

num_processes = len(os.sched_getaffinity(0))

n = 10000


minDegree = 10
maxDegree = 1000
exponent = 3
k = generatePowerLawDegreeSequence(n, minDegree, maxDegree, exponent)
static_network = nx.configuration_model(k)


m = 10
tmin = 0
tmax = 500
dt = 1
activities = invCDFPowerLaw(np.random.rand(n), 0.01, 1, exponent)
temporal_network = activity_model(activities, m, tmin, tmax, dt)

rho = 0.01
tauR = 21
time_to_max = 4
R0_list = np.linspace(0, 2, 50)
num_sims = 100

beta_unscaled = lambda tau : math.e/time_to_max*tau*np.exp(-tau/time_to_max)
exposure = quad(beta_unscaled, 0, tauR)[0]
lambda_config_model = np.mean(np.power(k, 2)) / np.mean(k)
lambda_activity_model = m*np.mean(activities) + m*math.sqrt(np.mean(np.power(activities, 2)))


# Static
extent_SIR = np.zeros(len(R0_list))
extent_VL_const = np.zeros(len(R0_list))
extent_VL_gamma = np.zeros(len(R0_list))

arg_list = list()
for R0 in R0_list:
    arg_list.append((static_network, time_to_max, tauR, exposure, R0, lambda_config_model, dt, rho, num_sims))

with mp.Pool(processes=num_processes) as pool:
        extents = pool.starmap(get_static_extent, arg_list)

for index in range(len(extents)):
    data = extents[index]
    extent_SIR[index] = data[0]
    extent_VL_const[index] = data[1]
    extent_VL_gamma[index] = data[2]

with shelve.open("numerical_threshold") as data:
    data["R0"] = R0_list
    data["extent-SIR-static"] = extent_SIR
    data["extent-VL-const-static"] = extent_VL_const
    data["extent-VL-gamma-static"] = extent_VL_gamma



# Temporal
extent_SIR = np.zeros(len(R0_list))
extent_VL_const = np.zeros(len(R0_list))
extent_VL_gamma = np.zeros(len(R0_list))

arg_list = list()

for R0 in R0_list:
    arg_list.append((temporal_network, n, m, time_to_max, tauR, exposure, R0, lambda_activity_model, dt, rho, num_sims))

with mp.Pool(processes=num_processes) as pool:
        extents = pool.starmap(get_temporal_extent, arg_list)

for index in range(len(extents)):
    data = extents[index]
    extent_SIR[index] = data[0]
    extent_VL_const[index] = data[1]
    extent_VL_gamma[index] = data[2]

with shelve.open("numerical_threshold") as data:
    data["extent-SIR-temporal"] = extent_SIR
    data["extent-VL-const-temporal"] = extent_VL_const
    data["extent-VL-gamma-temporal"] = extent_VL_gamma
