import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from numpy.linalg import inv
from utilities import *

contacts = np.arange(1, 1000, 10)
beta = np.arange(0, 1, 0.01)

ccdf  = np.zeros((len(contacts), len(beta)))
thresholds = [10, 50, 100, 500]

plt.figure()
plotIndex = 1
for threshold in thresholds:
    for i in range(len(contacts)):
        for j in range(len(beta)):
            ccdf[i,j] = binomialCCDF(contacts[i], beta[j], threshold)

    plt.subplot(2,2, plotIndex)
    plt.title(r"$P(\nu \geq t), \ t=$" + str(threshold))
    c = plt.imshow(np.flipud(ccdf), vmin=0, vmax=1, extent=(min(beta), max(beta), min(contacts), max(contacts)), aspect="auto")
    if plotIndex == 3:
        plt.xlabel("Infection probability")
        plt.ylabel("Number of contacts")
    else:
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plotIndex += 1

plt.colorbar(c)
plt.tight_layout()
plt.savefig("Figures/superspreading_events.pdf", dpi=1000)
plt.show()
