import random
import networkx as nx

def activity_model(activities, m, tmin=0, tmax=100, dt=1):
    # at each time step turn on a node w.p. a_i *delta_t
    n = len(activities)
    t = tmin
    temporal_network_list = list()
    while t < tmax:
        edgelist = list()
        for index in range(n):
            if random.random() <= activities[index]*dt:
                indices = random.sample(range(n), m)
                edgelist.extend([(index, j) for j in indices])
                edgelist.extend([(j, index) for j in indices])
        temporal_network_list.append(nx.Graph(edgelist))
        t += dt
    return temporal_network_list