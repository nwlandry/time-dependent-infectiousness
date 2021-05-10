# run da functions etc
import os
import temporal_network
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import webweb
from Network import *
from utils import * # TODO this is where the network builing functions will live
os.system('clear')

timesteps = 'hour' # minute, day

#viral_load(timesteps)

# filename = "Data/School/thiers_2011.csv"
# filename = "Data/School/thiers_2012.csv"
# #filename = "Data/Workplace/tij_InVS.dat"
# filename = "Data/School/primaryschool.csv"
# filename = "Data/School/High-School_data_2013.csv" # 20 secs. -> convert to minutes
# delimiter = " "
#
#nodesA, temporalA = import_temporal_networks(filename, delimiter)
n = 100
p = 0.15
m = 2
exponent = 2.8
nu = 100
epsilon = 1e-3
tmax = 1000
net_type = 'static'
#
#---------------------------------------------------------------------------------
#OOP Version

for i in range(10):
    nodesA = None
    edgesA = None
    if net_type == 'static':
        #generate a random graph
        G = nx.gnp_random_graph(n, p, directed=False)
        nodesA = G.nodes()
        edgesA = {0: list(G.edges())}

    else:
        activities = generate_activities(n, exponent, nu, epsilon)
        nodesA, edgesA = construct_activity_driven_model(n, m, activities, tmin=0, tmax=tmax, dt=1)

    initial_infected = np.random.choice(list(nodesA))


    print("Starting SIR model")
    exp_name_2 = net_type + '_SIR/run_' + str(i) + '/'
    #network2 = Network(nodesA, temporalA, contagionType = 'SIR')
    print(initial_infected)
    network2 = Network(nodesA, edgesA, contagionType = 'SIR')

    network2.run_temporal_contagion(0, 0, tmax=tmax, exp_name = exp_name_2, time_steps = timesteps, initial_infected = initial_infected)
    #plot_stats(network.edge_list, network.node_list, tmax = 100, time_steps = 'day', exp_name = exp_name)

    print('DONE.')
    print("output/" + exp_name_2 + "edge_list.p")
    # pickle things to plot later
    pickle.dump( network2.edge_list, open( "output/" + exp_name_2 + "edge_list.p", "wb" ) )
    pickle.dump( network2.node_list, open( "output/" + exp_name_2 + "node_list.p", "wb" ) )

    exp_name = net_type + '_VL/run_' + str(i) + '/'

    #temporal_to_static_network(temporalA)
    #network = Network(nodesA, temporalA, contagionType = 'VL')
    network = Network(nodesA, edgesA, contagionType = 'VL')

    network.run_temporal_contagion(0, 0, tmax=tmax, exp_name = exp_name, time_steps = timesteps, initial_infected = initial_infected)
    #plot_stats(network.edge_list, network.node_list, tmax = 100, time_steps = 'day', exp_name = exp_name)

    print('DONE.')
    print("output/" + exp_name + "edge_list.p")
    # pickle things to plot later
    pickle.dump( network.edge_list, open( "output/" + exp_name + "edge_list.p", "wb" ) )
    pickle.dump( network.node_list, open( "output/" + exp_name + "node_list.p", "wb" ) )

#
'''web = webweb.Web(title="test")

i = 0
for time, A in temporalA.items():
    i += 1
    if i == 100:
        i = 0
        web.networks.__dict__[str(time)] = webweb.webweb.Network(adjacency=A)

web.display.sizeBy = 'strength'
web.display.showLegend = True
web.display.colorPalette = 'Dark2'
web.display.colorBy = 'degree'
web.show()'''
