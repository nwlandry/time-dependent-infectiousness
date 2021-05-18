# run da functions etc
import os
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import webweb
from Network import *
from utils import * # TODO this is where the network builing functions will live
os.system('clear')
c=4
r=1
m = 2
exponent = 2.8
nu = 100
epsilon = 1e-3
def Gamma(x):
    return math.gamma(x)

temp_inf = None
static_inf = None
times = None
nodes1, edges = readInSFHH()
if times is None:
    times = list(edges.keys())

for contagionModel in ['SIR', 'SEIR', 'VL']:
    for time_type in ['static']:
        if time_type == 'static':
            nodes, edges = readInCO90()

            if static_inf is None:
                static_inf = np.random.choice(list(nodes))

            exp_name = 'out/CO90_'+ contagionModel
            print(exp_name)
            edges_dict = {}
            for t in times:
                edges_dict[t] = edges
            avgDeg, degrees = getAvgDegree( nodes, edges_dict )

            network = Network(nodes, edges_dict, contagionType = contagionModel)
            network.run_temporal_contagion(0, 0, Gamma, exp_name = exp_name, initial_infected = static_inf)
            pickle.dump( network.edge_list, open( exp_name + "edge_list.p", "wb" ) )
            pickle.dump( network.node_list, open( exp_name + "node_list.p", "wb" ) )
            #uniform
            edges = Price_model(c, r, np.max(list(nodes)), usePA = False)
            exp_name = 'out/uniform_'+ contagionModel
            print(exp_name)
            edges_dict = {}
            for t in times:
                edges_dict[t] = edges
            network = Network(nodes, edges_dict, contagionType = contagionModel)
            network.run_temporal_contagion(0, 0, Gamma, exp_name = exp_name, initial_infected = static_inf)
            pickle.dump( network.edge_list, open( exp_name + "edge_list.p", "wb" ) )
            pickle.dump( network.node_list, open( exp_name + "node_list.p", "wb" ) )

            # power law
            edges = Price_model(c, r, np.max(list(nodes)), usePA = True)
            exp_name = 'out/powlaw_'+ contagionModel
            print(exp_name)
            edges_dict = {}
            for t in times:
                edges_dict[t] = edges
            network = Network(nodes, edges_dict, contagionType = contagionModel)
            network.run_temporal_contagion(0, 0, Gamma, exp_name = exp_name, initial_infected = static_inf)
            pickle.dump( network.edge_list, open( exp_name + "edge_list.p", "wb" ) )
            pickle.dump( network.node_list, open( exp_name + "node_list.p", "wb" ) )

            #configuration
            G = nx.configuration_model(degrees)
            nodes = G.nodes()
            edges = G.edges()
            exp_name = 'out/config_'+ contagionModel
            print(exp_name)
            edges_dict = {}
            for t in times:
                edges_dict[t] = edges
            network = Network(nodes, edges_dict, contagionType = contagionModel)
            network0.run_temporal_contagion(0, 0, Gamma, exp_name = exp_name, initial_infected = static_inf)
            pickle.dump( network0.edge_list, open( exp_name + "edge_list.p", "wb" ) )
            pickle.dump( network0.node_list, open( exp_name + "node_list.p", "wb" ) )

        else:
            nodes1, edges = readInSFHH()
            if times is None:
                times = list(edges.keys())

            if temp_inf is None:
                temp_inf = np.random.choice(list(nodes1))

            exp_name = 'out/SFHH_'+ contagionModel
            print(exp_name)
            print(nodes1)
            network = Network(nodes1, edges, contagionType = contagionModel)
            network.run_temporal_contagion(0, 0, Gamma, exp_name = exp_name, initial_infected = temp_inf)
            pickle.dump( network.edge_list, open( exp_name + "edge_list.p", "wb" ) )
            pickle.dump( network.node_list, open( exp_name + "node_list.p", "wb" ) )

            #activity model
            activities = generate_activities(nodes1, exponent, nu, epsilon)
            nodes, edges = construct_activity_driven_model(nodes1, m, activities, times)

            exp_name = 'out/activity_'+ contagionModel
            print(exp_name)
            network = Network(nodes1, edges, contagionType = contagionModel)
            network.run_temporal_contagion(0, 0, Gamma, exp_name = exp_name, initial_infected = temp_inf)
            pickle.dump( network.edge_list, open( exp_name + "edge_list.p", "wb" ) )
            pickle.dump( network.node_list, open( exp_name + "node_list.p", "wb" ) )



# -------------------------------------------------------------------------------------------------
# OLD FOR REFERENCE
# -------------------------------------------------------------------------------------------------

# n = 100
# p = 0.15
# m = 2
# exponent = 2.8
# nu = 100
# epsilon = 1e-3
# tmax = 1000
# net_type = 'static'
# #
# #---------------------------------------------------------------------------------
# #OOP Version
#
# for i in range(10):
#     nodesA = None
#     edgesA = None
#     if net_type == 'static':
#         #generate a random graph
#         G = nx.gnp_random_graph(n, p, directed=False)
#         nodesA = G.nodes()
#         edgesA = {0: list(G.edges())}
#
#     else:
#         activities = generate_activities(n, exponent, nu, epsilon)
#         nodesA, edgesA = construct_activity_driven_model(n, m, activities, tmin=0, tmax=tmax, dt=1)
#
#     initial_infected = np.random.choice(list(nodesA))
#
#
#     print("Starting SIR model")
#     exp_name_2 = net_type + '_SIR/run_' + str(i) + '/'
#     #network2 = Network(nodesA, temporalA, contagionType = 'SIR')
#     print(initial_infected)
#     network2 = Network(nodesA, edgesA, contagionType = 'SIR')
#
#     network2.run_temporal_contagion(0, 0, tmax=tmax, exp_name = exp_name_2, time_steps = timesteps, initial_infected = initial_infected)
#     #plot_stats(network.edge_list, network.node_list, tmax = 100, time_steps = 'day', exp_name = exp_name)
#
#     print('DONE.')
#     print("output/" + exp_name_2 + "edge_list.p")
#     # pickle things to plot later
#     pickle.dump( network2.edge_list, open( "output/" + exp_name_2 + "edge_list.p", "wb" ) )
#     pickle.dump( network2.node_list, open( "output/" + exp_name_2 + "node_list.p", "wb" ) )
#
#     exp_name = net_type + '_VL/run_' + str(i) + '/'
#
#     #temporal_to_static_network(temporalA)
#     #network = Network(nodesA, temporalA, contagionType = 'VL')
#     network = Network(nodesA, edgesA, contagionType = 'VL')
#
#     network.run_temporal_contagion(0, 0, tmax=tmax, exp_name = exp_name, time_steps = timesteps, initial_infected = initial_infected)
#     #plot_stats(network.edge_list, network.node_list, tmax = 100, time_steps = 'day', exp_name = exp_name)
#
#     print('DONE.')
#     print("output/" + exp_name + "edge_list.p")
#     # pickle things to plot later
#     pickle.dump( network.edge_list, open( "output/" + exp_name + "edge_list.p", "wb" ) )
#     pickle.dump( network.node_list, open( "output/" + exp_name + "node_list.p", "wb" ) )
