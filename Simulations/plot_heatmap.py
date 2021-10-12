import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.ndimage
import shelve
import numpy as np
import matplotlib.gridspec as gridspec

with shelve.open("Simulations/heatmap") as data:
    infection_distribution_static = data["infection-distribution-static"]
    infection_distribution_temporal = data["infection-distribution-temporal"]
    tmax = data["tmax"]
    tauR = data["tauR"]
    dt = data["dt"]
extent = (0, tmax, 0, tauR)

is_rasterized = False

# Coordinates of the line we'd like to sample along
x1 = 40
line1 = [(x1, 0), (x1, tauR-dt)]

x2 = 60
line2 = [(x2, 0), (x2, tauR-dt)]

x3 = 80
line3 = [(x3, 0), (x3, tauR-dt)]

lines = [line1, line2, line3]

cross_section_list = list()
x_list = list()
y_list = list()

for line in lines:
    # Convert the line to pixel/index coordinates
    x_world, y_world = np.array(line).T
    x_list.append(x_world)
    y_list.append(y_world)

    col = infection_distribution_temporal.shape[0] * (x_world - 0) / (tmax-dt)
    row = infection_distribution_temporal.shape[1] * (y_world - 0) / tauR

    # Interpolate the line at "num" points...
    num = 1000
    row, col = [np.linspace(item[0], item[1], num) for item in [row, col]]

    # Extract the values along the line, using cubic interpolation
    cross_section = scipy.ndimage.map_coordinates(np.flipud(infection_distribution_temporal.T), np.vstack((row, col)))
    normalized_cross_section = cross_section/(np.mean(cross_section)*tauR)
    cross_section_list.append(normalized_cross_section)



fig = plt.figure(figsize=(10, 8.5))

gs = gridspec.GridSpec(3, 1, height_ratios = [1, 1, 1.5])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
gs_inner = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[2], width_ratios = [0.5, 1, 1, 1, 0.75])
ax31 = fig.add_subplot(gs_inner[1])
ax32 = fig.add_subplot(gs_inner[2])
ax33 = fig.add_subplot(gs_inner[3])


heatmap1 = ax1.imshow(np.flipud(infection_distribution_static.T), extent=extent, rasterized=is_rasterized, vmin=0, vmax=250)
ax1.set_xticks([0, 25, 50, 75, 100, 125])
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='1%', pad=0.05)
cbar = fig.colorbar(heatmap1, cax=cax, orientation='vertical')
cbar.set_label('Number of individuals', rotation=270, labelpad=10)

heatmap2 = ax2.imshow(np.flipud(infection_distribution_temporal.T), extent=extent, rasterized=is_rasterized, vmin=0, vmax=350)
ax2.set_xticks([0, 25, 50, 75, 100, 125])
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='1%', pad=0.05)
cbar = fig.colorbar(heatmap2, cax=cax, orientation='vertical')
cbar.set_label('Number of individuals', rotation=270, labelpad=10)

ax2.set_xlabel(r"$t$", fontsize=20)
ax2.set_ylabel(r"$\tau$", fontsize=20)

ax2.plot(x_list[0], y_list[0] + dt/2, 'wo-', linewidth=2, rasterized=is_rasterized)
ax2.plot(x_list[1], y_list[1] + dt/2, 'wo-', linewidth=2, rasterized=is_rasterized)
ax2.plot(x_list[2], y_list[2] + dt/2, 'wo-', linewidth=2, rasterized=is_rasterized)

ax31.plot(np.linspace(tauR, 0, len(cross_section_list[0])), cross_section_list[0], 'k', linewidth=2, rasterized=True)
ax31.set_ylim([0, 0.125])
ax31.set_xlabel(r"$\tau$", fontsize=16)
ax31.set_ylabel("Probability", fontsize=16)

ax32.plot(np.linspace(tauR, 0, num), cross_section_list[1], 'k', linewidth=2, rasterized=True)
ax32.set_ylim([0, 0.125])
ax32.tick_params(labelbottom=False, labelleft=False)

ax33.plot(np.linspace(tauR, 0, num), cross_section_list[2], 'k', linewidth=2, rasterized=True)
ax33.set_ylim([0, 0.125])
ax33.tick_params(labelbottom=False, labelleft=False)
plt.tight_layout()

plt.savefig("Figures/heatmap.pdf", dpi=1000)
plt.savefig("Figures/heatmap.png", dpi=1000)
plt.show()