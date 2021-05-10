import numpy as np
import networkx as nx
import random
import csv
import math
import os
from utils import *

class Network:
    def __init__(self, nodes, edge_list, node_attrs = None, k = 2.5, SIR_prob = 0.3, contagionType = 'VL'):
        '''
            num_nodes : number of nodes in the network if not passing in a file
        '''
        #NEED TO UPDATE SOME OF THESE TO ACCOUNT FOR THE OTHER WAYS TO LOAD IN OR GENERATE NETWORKS...
        self.contagionModel = contagionType # could be SIR or VL

        default_node_attrs = node_attrs if node_attrs is not None else {'status': 'S', 'infect_time': -1, 'viral_loads': [], 'remove_time': -1}
        # this dictionary will be assigned to each node at initialization unless we pass in a list.
        self.node_list = dict({n:dict(default_node_attrs) for n in nodes})
        self.netType = 'temporal' if len(edge_list) > 1 else 'static'
        self.edge_list = edge_list # need to generate for each timestep based off of edge_probs[p]
        self.setPos(k) #for visualizations
        self.SIR_prob = SIR_prob

    def run_temporal_contagion(self, gamma, beta, tmin=0, tmax=100, dt=1, time_steps = 'day', initial_infected=None, initial_recovered=None, useEdgeDuration = False, exp_name = 'testing_drawFunc_nx/', toPrintNodeUpdates = False):
        # TO DO: NODES AREN'T RECOVERING.... MAYBE DUE TO TMAX?
        print('Running a ' + self.netType+ ' network with the ' + self.contagionModel + ' contagion model.')
        N = len(self.node_list)

        times = list(np.arange(tmin, tmax + 1))
        if len(self.edge_list) > 1: #case where we are doing dynamics on a dynamic network
            times = list(self.edge_list.keys())

        if initial_infected is None:
            #index_1 = random.randrange(N)
            #n = list(self.node_list.keys())[index_1]
            n = random.choice(list(self.node_list.keys()))
            #print(n)
            self.node_list[n]['status'] = 'I'
            self.node_list[n]['infect_time'] = times[0]
            self.node_list[n]['viral_loads'] = viral_load(time_steps) #might need to change this based on literature....
            self.node_list[n]['remove_time'] = len(self.node_list[n]['viral_loads'])
        else:
            print('add initial infected to I and update statuses in node_list')
            self.node_list[initial_infected]['status'] = 'I'
            self.node_list[initial_infected]['infect_time'] = times[0]
            self.node_list[initial_infected]['viral_loads'] = viral_load(time_steps) #might need to change this based on literature....
            self.node_list[initial_infected]['remove_time'] = len(self.node_list[initial_infected]['viral_loads'])

        if initial_recovered is None:
            initial_recovered = []

        I = set([]) # list of infected nodes at time t
        R = set([]) # list of recovered nodes at time t

        S = set([])
        for ts, t in enumerate(times):
            #account for a static network
            print('Simulation time : ' + str(t) + ' (' + str(ts)+ ' of ' + str(len(times)) + ').', end = ' ')
            index = t
            if self.netType == 'static':
                index = 0

            for n in self.node_list.keys(): #update iterator

                if self.node_list[n]['status'] == 'I':
                    # check if this is the last day its infectious
                    #print(t, n, ts - times.index(self.node_list[n]['infect_time']), len(self.node_list[n]['viral_loads']) )
                    if(ts - times.index(self.node_list[n]['infect_time'])) >= len(self.node_list[n]['viral_loads']):
                        # no longer infectious. remove from I and add to R
                        I.remove(n)
                        self.node_list[n]['status'] = 'R'
                        R.add(n)
                    else: # otherwise add it to I
                        I.add(n)
                elif self.node_list[n]['status'] == 'R': # think about SIR recovery.
                    R.add(n)
            print('There are currently ' + str(N - len(I) - len(R)) + ' susecptible, ' + str(len(I)) + ' infected, ' + str(len(R)) + ' recovered (out of ' + str(N) + ').')

            if len(R) == N or len(I) == 0:
                break
            # infect nodes
            for infected_node in I:
                prob_of_infection = 0.0
                neighbors, sus_neighbors = get_neighbors(self.edge_list[index], self.node_list, infected_node)
                #print(self.node_list[infected_node], self.node_list[infected_node]['status'])
                if self.contagionModel == 'VL':
                    # -> calculate viral viral_load at time t (t - 'infected_time')
                    #print(infected_node, self.node_list.keys())
                    vl_time = int(ts - times.index(self.node_list[infected_node]['infect_time']))
                    #print(ts, times.index(self.node_list[n]['infect_time']))
                    #print(t, vl_time, self.node_list[infected_node]['viral_loads'])
                    vl_current = self.node_list[infected_node]['viral_loads'][vl_time] #[t - self.node_list[infected_node]

                    # ->calculate probability of infection: infectiousness was taken to be proportional to the logarithm of viral load in excess of 106 cp/ml {Larremore 2020}
                    prob_of_infection = vl_prob(vl_current)
                    print( '\t Node ' + str(infected_node) + ' is infected (probability of spread: ' + str(prob_of_infection) + '; viral load: ' + str(vl_current)+ ') and has ' + str(len(sus_neighbors)) + ' susecptible neighbors.')
                else:
                    #else -> use default SIR probability
                    prob_of_infection = self.SIR_prob
                    print( '\t Node ' + str(infected_node) + ' is infected (probability of spread: ' + str(prob_of_infection) + ') and has ' + str(len(sus_neighbors)) + ' susecptible neighbors.')

                num_infected_by_inf = 0
                if len(sus_neighbors) > 0:
                    for neighbor in sus_neighbors: # of i at time t

                        if toPrintNodeUpdates:
                            print('\t \t Node ' + str(infected_node) + ' comes into contact with node ' + str(neighbor) + ' who', end = ' ')
                        if self.node_list[neighbor]['status'] == 'S':

                            # do edge duration if using
                            if useEdgeDuration:
                                duration = get_edge_duration(self.edge_list, edge, t)
                                prob_of_infection = prob_with_edge_duration(prob_of_infection, duration)

                            #infect with probablility p
                            to_infect = random.random()
                            #print(to_infect, 'node: ', neighbor, prob_of_infection)
                            if to_infect < prob_of_infection:
                                # update status, infect_time at time t
                                self.node_list[neighbor]['status'] = 'I'
                                self.node_list[neighbor]['infect_time'] = t
                                self.node_list[neighbor]['viral_loads'] = viral_load(time_steps)
                                self.node_list[neighbor]['remove_time'] = ts + len(self.node_list[neighbor]['viral_loads'])
                                if toPrintNodeUpdates:
                                    print(' is infected.')
                                num_infected_by_inf = num_infected_by_inf + 1
                            else:
                                if toPrintNodeUpdates:
                                    print(' is not infected.')
                        elif  self.node_list[neighbor]['status'] == 'I':
                            if toPrintNodeUpdates:
                                print(' was already infected.')
                        else:
                            if toPrintNodeUpdates:
                                print(' is recovered.')
                    print('\t \t Node ' + str(infected_node) + ' infected ' + str(num_infected_by_inf) + ' other nodes (' + str((num_infected_by_inf/len(neighbors))*100)+ '% of its neighbors).')

                else:
                    print('\t \t There are no nodes to infect (number of susecptible neighbors is 0).')
                #fix percent error
                print()
            print('-------------------------------------------------------------------------------')
            print('-------------------------------------------------------------------------------')

            drawContagion_nx(self.edge_list, self.node_list, index, t, times, exp_name = exp_name, pos = self.pos, contagionModel = self.contagionModel, SIR_prob = self.SIR_prob)

        for n_final in self.node_list.keys():
            if n_final not in I and n_final not in R:
                S.add(n_final)
        return np.array(times), np.array(S), np.array(I), np.array(R)

    def setPos(self, k = 2.5):
        G_temp = nx.Graph()
        for t in self.edge_list.keys():
            #all_edges_inTime.add(edge_list[t])
            G_temp.add_edges_from(self.edge_list[t])

        self.pos = nx.spring_layout(G_temp, k = k)
