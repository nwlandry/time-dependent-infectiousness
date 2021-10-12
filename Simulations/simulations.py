from collections import defaultdict
import numpy as np
import random


def SIR_model_static_network(G, gamma, beta, dt=1, initial_infecteds=None, tmin=0, tmax=np.inf):
    state = defaultdict( lambda : "S" )
    new_state = state.copy()
    for infected in initial_infecteds:
        state[infected] = 'I'

    n = G.number_of_nodes()
    t = tmin
    times = [tmin]
    I = [len(initial_infecteds)]
    S = [n - I[0]]
    R = [0]

    while t < tmax and I[-1] != 0:
        S.append(S[-1])
        I.append(I[-1])
        R.append(R[-1])

        for node in G.nodes:
            if state[node] == "I":
                # heal
                if random.random() <= gamma*dt:
                    new_state[node] = "R"
                    R[-1] += 1
                    I[-1] += -1
                else:
                    new_state[node] = "I"
            elif state[node] == "S":
                # infect by neighbors
                for nbr in G.neighbors(node):
                    if state[nbr] == "I" and random.random() <= beta*dt:
                        new_state[node] = "I"
                        S[-1] += -1
                        I[-1] += 1
                        break
                else:
                    new_state[node] == "S"
        state = new_state.copy()
        t += dt
        times.append(t)

    return np.array(times), np.array(S), np.array(I), np.array(R)

def SIR_model_temporal_network(Glist, n, gamma, beta, dt=1, initial_infecteds=None, tmin=0, tmax=np.inf):
    state = defaultdict( lambda : "S" )
    new_state = state.copy()
    for infected in initial_infecteds:
        state[infected] = 'I'

    t = tmin
    times = [tmin]
    I = [len(initial_infecteds)]
    S = [n - I[0]]
    R = [0]

    index = 0

    while t < tmax and I[-1] != 0 and index < len(Glist):
        S.append(S[-1])
        I.append(I[-1])
        R.append(R[-1])
        G = Glist[index]

        for node in G.nodes:
            if state[node] == "I":
                # heal
                if random.random() <= gamma*dt:
                    new_state[node] = "R"
                    R[-1] += 1
                    I[-1] += -1
                else:
                    new_state[node] = "I"
            elif state[node] == "S":
                # infect by neighbors
                for nbr in G.neighbors(node):
                    if state[nbr] == "I" and random.random() <= beta*dt:
                        new_state[node] = "I"
                        S[-1] += -1
                        I[-1] += 1
                        break
                else:
                    new_state[node] == "S"
        state = new_state.copy()
        t += dt
        times.append(t)
        index += 1
    return np.array(times), np.array(S), np.array(I), np.array(R)


def VL_model_static_network(G, beta, tauR, dt=1, initial_infecteds=None, tmin=0, tmax=np.inf, return_infection_distribution=False):
    infected_time = dict()
    state = defaultdict( lambda : "S" )
    new_state = state.copy()
    for infected in initial_infecteds:
        state[infected] = 'I'
        infected_time[infected] = tmin

    n = G.number_of_nodes()
    t = tmin
    times = [tmin]
    I = [len(initial_infecteds)]
    S = [n - I[0]]
    R = [0]

    if return_infection_distribution:
        infection_distribution = np.zeros((round((tmax - tmin)/dt)+1, round(max(tauR)/dt)+1))
        infection_distribution[0, 0] = I[-1]

    while t < tmax and I[-1] != 0:
        S.append(S[-1])
        I.append(I[-1])
        R.append(R[-1])

        for node in G.nodes:
            if state[node] == "I":
                # heal
                if t - infected_time[node] >= tauR[node]:
                    new_state[node] = "R"
                    R[-1] += 1
                    I[-1] += -1
                else:
                    new_state[node] = "I"
            elif state[node] == "S":
                # infect by neighbors
                for nbr in G.neighbors(node):
                    
                    if state[nbr] == "I" and random.random() <= beta[node](t - infected_time[nbr])*dt:
                        new_state[node] = "I"
                        infected_time[node] = t + dt
                        S[-1] += -1
                        I[-1] += 1
                        break
                else:
                    new_state[node] == "S"
                    
        
        state = new_state.copy()
        t += dt
        times.append(t)

        if return_infection_distribution:
            for node in state:
                if state[node] == "I":
                    try:
                        infection_distribution[round((t - tmin)/dt), round((t - infected_time[node])/dt)] += 1
                    except:
                        pass

    if return_infection_distribution:
        return np.array(times), np.array(S), np.array(I), np.array(R), infection_distribution
    else:
        return np.array(times), np.array(S), np.array(I), np.array(R)


def VL_model_temporal_network(Glist, n, beta, tauR, dt=1, initial_infecteds=None, tmin=0, tmax=np.inf, return_infection_distribution=False):
    infected_time = dict()
    state = defaultdict( lambda : "S" )
    new_state = state.copy()
    for infected in initial_infecteds:
        state[infected] = 'I'
        infected_time[infected] = tmin

    t = tmin
    times = [tmin]
    I = [len(initial_infecteds)]
    S = [n - I[0]]
    R = [0]

    index = 0

    if return_infection_distribution:
        infection_distribution = np.zeros((min(round((tmax - tmin)/dt)+1, len(Glist)+1), round(max(tauR)/dt)+1))
        infection_distribution[index, 0] = I[-1]

    while t < tmax and I[-1] != 0 and index < len(Glist):
        S.append(S[-1])
        I.append(I[-1])
        R.append(R[-1])
        G = Glist[index]

        for node in G.nodes:
            if state[node] == "I":
                # heal
                if t - infected_time[node] >= tauR[node]:
                    new_state[node] = "R"
                    R[-1] += 1
                    I[-1] += -1
                else:
                    new_state[node] = "I"
            elif state[node] == "S":
                # infect by neighbors
                for nbr in G.neighbors(node):
                    if state[nbr] == "I" and random.random() <= beta[node](t - infected_time[nbr])*dt:
                        new_state[node] = "I"
                        infected_time[node] = t + dt
                        S[-1] += -1
                        I[-1] += 1
                        break
                else:
                    new_state[node] == "S"
        state = new_state.copy()
        t += dt
        times.append(t)
        index += 1
        if return_infection_distribution:
            for node in state:
                if state[node] == "I":
                    try:
                        infection_distribution[index, round((t - infected_time[node])/dt)] += 1
                    except:
                        pass

    if return_infection_distribution:
        return np.array(times), np.array(S), np.array(I), np.array(R), infection_distribution
    else:
        return np.array(times), np.array(S), np.array(I), np.array(R)

def generatePowerLawDegreeSequence(n, minDegree, maxDegree, exponent):
    degrees = np.round(invCDFPowerLaw(np.random.rand(n), minDegree, maxDegree, exponent)).astype(int)
    if sum(degrees) % 2 != 0:
        degrees[random.randrange(n)] += 1
    return degrees

def invCDFPowerLaw(u, minDegree, maxDegree, exponent):
    return (minDegree**(1-exponent) + u*(maxDegree**(1-exponent) - minDegree**(1-exponent)))**(1/(1-exponent))