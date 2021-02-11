import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from matplotlib import cm
import utilities

tmax = 1500
nStates = 10
tStates = np.linspace(0, 20, nStates-1)
b = utilities.betaVL(tStates, 0.2, 0.0002269744873046875)
bConst = utilities.betaConstant(tStates, np.mean(b))

N = 1000
initial_states = np.zeros(nStates)
initial_states[1] = 1
initial_states[0] = N - np.sum(initial_states[1:])

sol1 = solve_ivp(utilities.f, (0, tmax), initial_states, t_eval=np.arange(0, tmax, 0.1), args=(b,))
t1 = sol1.t
y1 = sol1.y.T

sol2 = solve_ivp(utilities.f, (0, tmax), initial_states, t_eval=np.arange(0, tmax, 0.1), args=(bConst,))
t2 = sol2.t
y2 = sol2.y.T


plt.figure()
plt.plot(t1, y1[:, 0], label="S")
for i in np.arange(1, nStates, 10, dtype=int):
    plt.plot(t1, y1[:, i], label="I"+ str(i))
plt.plot(t1, y1[:, -1], label="I"+ str(nStates-1))
plt.ylabel("Number of people", fontsize=14)
plt.xlabel("Time (days)", fontsize=14)
plt.legend()
plt.show()

plt.figure()
plt.subplot(311)
plt.title(r"$\beta(t)=\frac{e}{4} t e^{-t/4}I_{\beta(t)\geq 0.2}$")
plt.imshow(y1.T, cmap=cm.coolwarm, aspect="auto", extent=(0, tmax, 0, nStates))
plt.subplot(312)
plt.title(r"$\beta(t)=c$")
plt.imshow(y2.T, cmap=cm.coolwarm, aspect="auto", extent=(0, tmax, 0, nStates))
plt.subplot(313)
plt.title("Difference")
plt.imshow(y2.T - y1.T, aspect="auto", extent=(0, tmax, 0, nStates))
plt.xlabel("Time (days)")
plt.ylabel("State Number")
plt.tight_layout()
plt.show()
