import random
import numpy as np
import matplotlib.pyplot as plt
import utilities

n = 100000
nbins = 100
dt = 0.1
degreeDistribution = np.random.randint(low=50, high=150, size=(1000))
degreeDistribution = utilities.generatePowerLawDegreeSequence(n, 45, 1000, 3)
beta = utilities.betaVL(np.linspace(0, 21, 100), 0, 1, 4)
bAvg = np.mean(beta)
meanDegree = np.mean(degreeDistribution)

plt.figure()
# fixed random
secondaryInfections = np.zeros(n)
for i in range(n):
    k = meanDegree
    b = bAvg
    secondaryInfections[i] = np.random.binomial(k, b)
plt.subplot(141)
plt.hist(secondaryInfections, bins=nbins, density=True)

# contact random
secondaryInfections = np.zeros(n)
for i in range(n):
    k = random.choice(degreeDistribution)
    b = bAvg
    secondaryInfections[i] = np.random.binomial(k, b, size=None)
plt.subplot(142)
plt.hist(secondaryInfections, bins=nbins, density=True)

# vl random
secondaryInfections = np.zeros(n)
for i in range(n):
    k = meanDegree
    b = random.choice(beta)*dt
    secondaryInfections[i] = np.random.binomial(k, b, size=None)

plt.subplot(143)
plt.hist(secondaryInfections, bins=nbins, density=True)

# vl and contact random
secondaryInfections = np.zeros(n)
for i in range(n):
    k = random.choice(degreeDistribution)
    b = random.choice(beta)*dt
    secondaryInfections[i] = np.random.binomial(k, b, size=None)

plt.subplot(144)
plt.hist(secondaryInfections, bins=nbins, density=True)
plt.tight_layout()
plt.show()
