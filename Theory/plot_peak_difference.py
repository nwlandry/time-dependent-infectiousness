import matplotlib.pyplot as plt
import shelve
import numpy as np

with shelve.open("Theory/peak_difference") as data:
    num_states = data["num-states"]
    time_SIR = data["time-SIR"]
    magnitude_SIR = data["mag-SIR"]
    time_VL_const = data["time-VL-const"]
    magnitude_VL_const = data["mag-VL-const"]
    time_VL_gamma = data["time-VL-gamma"]
    magnitude_VL_gamma = data["mag-VL-gamma"]
   

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.semilogx(num_states, time_VL_gamma, 'ks-', linewidth=2, label="VL model, " + r"$\beta(\tau)=\beta_{\Gamma}(\tau)$")
plt.plot(num_states, time_VL_const, 'ko-', linewidth=2, label="VL model, " + r"$\beta(\tau)=\beta_{const}(\tau)$")
plt.plot(num_states, time_SIR, 'k--', label="SIR model")
plt.xlabel("Number of infectious states", fontsize=14)
plt.ylabel("Time of infectious peak", fontsize=14)
ymin, ymax = plt.gca().get_ylim()
yPos = ymin + 0.15*(ymax-ymin)
plt.text(5, yPos, "(a)", fontsize=16)

plt.subplot(1, 2, 2)
plt.semilogx(num_states, magnitude_VL_gamma, 'ks-', linewidth=2, label="VL model, " + r"$\beta(\tau)=\beta_{\Gamma}(\tau)$")
plt.plot(num_states, magnitude_VL_const, 'ko-', linewidth=2, label="VL model, " + r"$\beta(\tau)=\beta_{const}(\tau)$")
plt.plot(num_states, magnitude_SIR, 'k--', label="SIR model")
plt.xlabel("Number of infectious states", fontsize=14)
plt.ylabel("Magnitude of infectious peak", fontsize=14)
plt.legend(fontsize=12, loc=(0.3, 0.2))
ymin, ymax = plt.gca().get_ylim()
yPos = ymin + 0.15*(ymax-ymin)
plt.text(5, yPos, "(b)", fontsize=16)

plt.tight_layout()
plt.savefig("Figures/peak_difference.png", dpi=600)
plt.savefig("Figures/peak_difference.pdf")
plt.show()
