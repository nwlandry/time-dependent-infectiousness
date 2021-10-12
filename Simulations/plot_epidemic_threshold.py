import matplotlib.pyplot as plt
import shelve

with shelve.open("Simulations/numerical_threshold") as data:
    R0 = data["R0"]
    extent_SIR = data["extent-SIR-static"]
    extent_VL_const = data["extent-VL-const-static"]
    extent_VL_gamma = data["extent-VL-gamma-static"]

plt.figure()
plt.subplot(211)
plt.plot(R0, extent_SIR, 'ko', label="SIR (Simulation)")
plt.plot(R0, extent_VL_const, 'k^', label=r"VL ($\beta_{const})$ (Simulation)")
plt.plot(R0, extent_VL_gamma, 'ks', label=r"VL ($\beta_{\Gamma})$ (Simulation)")
ymin, ymax = plt.gca().get_ylim()
yPos = ymin + 0.3*(ymax-ymin)
plt.text(0.1, yPos, "(a)", fontsize=14)

with shelve.open("Simulations/numerical_threshold") as data:
    R0 = data["R0"]
    extent_SIR = data["extent-SIR-temporal"]
    extent_VL_const = data["extent-VL-const-temporal"]
    extent_VL_gamma = data["extent-VL-gamma-temporal"]

plt.subplot(212)
plt.plot(R0, extent_SIR, 'ko', label="SIR (Simulation)")
plt.plot(R0, extent_VL_const, 'k^', label=r"VL ($\beta_{const})$ (Simulation)")
plt.plot(R0, extent_VL_gamma, 'ks', label=r"VL ($\beta_{\Gamma})$ (Simulation)")
plt.xlabel(r"$R_0$", fontsize=14)
plt.ylabel("Fraction infected", fontsize=14)
ymin, ymax = plt.gca().get_ylim()
yPos = ymin + 0.3*(ymax-ymin)
plt.text(0.1, yPos, "(b)", fontsize=14)
plt.legend()
plt.tight_layout()

plt.savefig("Figures/extent_vs_R0.pdf", dpi=1000)
plt.savefig("Figures/extent_vs_R0.png", dpi=1000)
plt.show()