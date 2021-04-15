import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from numpy.linalg import inv

def betaVL(t, threshold, maxRate, timeToMaxRate):
    rate = math.e/timeToMaxRate*np.multiply(t, np.exp(-t/timeToMaxRate))
    rate[np.where(rate < threshold)] = 0
    return maxRate*rate

def betaConstant(t, rate):
    return rate*np.ones(len(t))

def viralLoadModelFullyMixed(t, X, beta, dt):
    S = X[0]
    I = X[1:-1]
    dXdt = np.zeros(len(X))
    dXdt[0] = -S*np.sum(np.multiply(beta, I))
    dXdt[1] = S*np.sum(np.multiply(beta, I)) - I[0]/dt
    dXdt[2:-1] = (I[:-1] - I[1:])/dt
    dXdt[-1] = I[-1]/dt
    return dXdt

def viralLoadModelDegreeBased(t, X, P, beta, dt):
    k = len(P)
    dXdt = np.zeros(len(X))
    S = X[:k]
    I = X[k:-k]
    # R = X[-k:]
    dXdt[k:-k] += -I/dt
    dXdt[2*k:] += I/dt

    for index in range(len(beta)):
        dXdt[:k] += -np.multiply(S, beta[index]*P.dot(I[k*index:k*(index+1)]))
        dXdt[k:2*k] += np.multiply(S, beta[index]*P.dot(I[k*index:k*(index+1)]))
    return dXdt

def generateConfigurationModelP(degreeSequence):
    k, counts = np.unique(degreeSequence, return_counts=True) # get unique k values
    p = counts/len(degreeSequence)
    meanDegree = np.mean(degreeSequence)
    return np.outer(k,np.multiply(k, p))/(meanDegree*len(degreeSequence))

def SIRModelFullyMixed(t, X, beta, gamma):
    S = X[0]
    I = X[1]
    R = X[2]
    dXdt = np.zeros(len(X))
    dXdt[0] = -beta*S*I
    dXdt[1] = beta*S*I - gamma*I
    dXdt[2] = gamma*I
    return dXdt

def SIRModelDegreeBased(t, X, P, beta, gamma):
    k = len(P)
    S = X[:k]
    I = X[k:2*k]
    # R = X[-k:]
    dXdt = np.zeros(len(X))
    dXdt[:k] = -beta*np.multiply(S, P.dot(I))
    dXdt[k:2*k] = beta*np.multiply(S, P.dot(I)) - gamma*I
    dXdt[-k:] = gamma*I
    return dXdt

def calculateCriticalMax(nInfectiousStates, tStates, threshold, dt, tolerance=0.000001):
    T = np.zeros((nInfectiousStates, nInfectiousStates))
    S = np.diag(-np.ones(nInfectiousStates), k=0)/dt + np.diag(np.ones(nInfectiousStates-1), k=-1)/dt

    pMin = 0
    T[0, :] = betaVL(tStates, threshold, pMin)
    l = np.linalg.eigvals(-np.matmul(T,inv(S)))
    lowerRho = np.max(np.abs(l))

    pMax = 1.0
    T[0, :] = betaVL(tStates, threshold, pMax)
    l = np.linalg.eigvals(-np.matmul(T,inv(S)))
    upperRho = np.max(np.abs(l))

    if upperRho < 1.0:
        print("Uhhhhh.")
        return
    maxIterations = 1000
    iterations = 0
    while upperRho - lowerRho > tolerance and iterations <= maxIterations:
        pNew = 0.5*(pMin + pMax)
        T[0, :] = betaVL(tStates, threshold, pNew)
        l = np.linalg.eigvals(-np.matmul(T,inv(S)))
        newRho = np.max(np.abs(l))
        if newRho < 1.0:
            lowerRho = newRho
            pMin = pNew
        else:
            upperRho = newRho
            pMax = pNew
        iterations += 1
    return pMin


def generatePowerLawDegreeSequence(n, minDegree, maxDegree, exponent):
    return np.round(invCDFPowerLaw(np.random.rand(n), minDegree, maxDegree, exponent))
    # for i in range(n):
    #     u = random.uniform(0, 1)
    #     degreeSequence[i] = round(self.invCDFPowerLaw(u, minDegree, maxDegree, exponent))
    # return degreeSequence

def invCDFPowerLaw(u, minDegree, maxDegree, exponent):
    return (minDegree**(1-exponent) + u*(maxDegree**(1-exponent) - minDegree**(1-exponent)))**(1/(1-exponent))
