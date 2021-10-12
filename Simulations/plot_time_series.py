import matplotlib.pyplot as plt
import shelve

with shelve.open("Simulations/numerical_sims") as data:
    t_SIR_sim = data["t-SIR-static"]
    S_SIR_sim = data["S-SIR-static"]
    I_SIR_sim = data["I-SIR-static"]
    R_SIR_sim = data["R-SIR-static"]

    t_VL_const_sim = data["t-VL-const-static"]
    S_VL_const_sim = data["S-VL-const-static"]
    I_VL_const_sim = data["I-VL-const-static"]
    R_VL_const_sim = data["R-VL-const-static"]

    t_VL_gamma_sim = data["t-VL-gamma-static"]
    S_VL_gamma_sim = data["S-VL-gamma-static"]
    I_VL_gamma_sim = data["I-VL-gamma-static"]
    R_VL_gamma_sim = data["R-VL-gamma-static"]


plt.figure()
plt.subplot(211)
plt.plot(t_SIR_sim, I_SIR_sim, 'k-', linewidth=2, label="SIR (Simulation)")
plt.plot(t_VL_const_sim, I_VL_const_sim, 'k--', linewidth=2, label=r"VL ($\beta_{const})$ (Simulation)")
plt.plot(t_VL_gamma_sim, I_VL_gamma_sim, 'k-.', linewidth=2, label=r"VL ($\beta_{\Gamma})$ (Simulation)")
ymin, ymax = plt.gca().get_ylim()
yPos = ymin + 0.6*(ymax-ymin)
plt.text(0, yPos, "(a)", fontsize=14)
plt.legend()

with shelve.open("Simulations/numerical_sims") as data:
    t_SIR_sim = data["t-SIR-temporal"]
    S_SIR_sim = data["S-SIR-temporal"]
    I_SIR_sim = data["I-SIR-temporal"]
    R_SIR_sim = data["R-SIR-temporal"]

    t_VL_const_sim = data["t-VL-const-temporal"]
    S_VL_const_sim = data["S-VL-const-temporal"]
    I_VL_const_sim = data["I-VL-const-temporal"]
    R_VL_const_sim = data["R-VL-const-temporal"]

    t_VL_gamma_sim = data["t-VL-gamma-temporal"]
    S_VL_gamma_sim = data["S-VL-gamma-temporal"]
    I_VL_gamma_sim = data["I-VL-gamma-temporal"]
    R_VL_gamma_sim = data["R-VL-gamma-temporal"]

plt.subplot(212)
plt.plot(t_SIR_sim, I_SIR_sim, 'k-', linewidth=2, label="SIR (Simulation)")
plt.plot(t_VL_const_sim, I_VL_const_sim, 'k--', linewidth=2, label=r"VL ($\beta_{const})$ (Simulation)")
plt.plot(t_VL_gamma_sim, I_VL_gamma_sim, 'k-.', linewidth=2, label=r"VL ($\beta_{\Gamma})$ (Simulation)")
ymin, ymax = plt.gca().get_ylim()
yPos = ymin + 0.6*(ymax-ymin)
plt.text(0, yPos, "(b)", fontsize=14)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Fraction infected", fontsize=14)
plt.tight_layout()
plt.savefig("Figures/infection_curves.pdf", dpi=1000)
plt.savefig("Figures/infection_curves.png", dpi=1000)
plt.show()