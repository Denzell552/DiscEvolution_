import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from DiscEvolution.constants import *

fp = "Winter_2026/Data/wind_model/test3_alpha_psi=1000.json"

with open (fp, 'r') as f:
    data = json.load(f)

gas_density = data['Sigma_G']
dust_density = data['Sigma_dust']
pebble_density = data['Sigma_pebbles']
radius = data['R']
size = data['pebble_size']
peb_v = data['pebble_drift_velocity']
dust_v = data['dust_drift_velocity']
time = data['time']
Mdot, alpha, psi, Rd = data['parameters']['Mdot'], data['parameters']['alpha'], data['parameters']['psi'], data['parameters']['Rd']
Mdot_evol = data['Mdot']
Mtot_evol = data['Mtot']

fig, ax = plt.subplots(3,2,figsize=(20,24))

color1 = iter(plt.cm.Blues(np.linspace(0.4, 1, 6)))
color2 = iter(plt.cm.Oranges(np.linspace(0.4, 1, 6)))
color3 = iter(plt.cm.Greys(np.linspace(0.4, 1, 6)))
color4 = iter(plt.cm.Purples(np.linspace(0.4, 1, 6)))
color5 = iter(plt.cm.Reds(np.linspace(0.4, 1, 6)))
color6 = iter(plt.cm.Greys(np.linspace(0.4, 1, 6)))
color7 = iter(plt.cm.Greens(np.linspace(0.4, 1, 6)))

for t in range(len(time)):
    
    ax[0,0].loglog(radius, gas_density[t], color=next(color1), label=f'{time[t]:.2f} Myr')
    ax[0,1].loglog(radius, pebble_density[t], color=next(color2))
    ax[1,0].loglog(radius, dust_density[t], color=next(color3))


ax[0,0].set_xlabel('Radius (AU)', fontsize=20)
ax[0,0].set_ylabel('Gas Surface Density ($g/cm^2$)', fontsize=20)
ax[0,0].set_title('Gas Surface Density Evolution', fontsize=25)

ax[0,1].set_xlabel('Radius (AU)', fontsize=20)
ax[0,1].set_ylabel('Surface Density ($g/cm^2$)', fontsize=20)
ax[0,1].set_title('Pebble Surface Density Evolution', fontsize=25)
ax[0,1].set_ylim(1e-5, 1e3)

ax[1,0].set_xlabel('Radius (AU)', fontsize=20)
ax[1,0].set_ylabel('Surface Density ($g/cm^2$)', fontsize=20)
ax[1,0].set_title('Dust Surface Density Evolution', fontsize=25)
ax[1,0].set_ylim(1e-7, 5e3)

ax[1,1].plot(time, Mdot_evol, marker='o', color='gray')
ax[1,1].set_xlabel('Time (Myr)', fontsize=20)
ax[1,1].set_ylabel('Mdot ($M_{sun}/yr$)', fontsize=20)
ax[1,1].set_title('Accretion Rate Evolution', fontsize=25)

ax[2,0].plot(time, Mtot_evol, marker='o', color='blue')
ax[2,0].set_xlabel('Time (Myr)', fontsize=20)
ax[2,0].set_ylabel('Mtot ($M_{sun}$)', fontsize=20)
ax[2,0].set_title('Total Disk Mass Evolution', fontsize=25)

ax[2,1].set_axis_off()

plt.figtext(0.5, 0.01, f"Mdot={Mdot:.3e}Msun/yr, alpha={alpha:.0e}, psi={psi}, Rd={Rd:.2f}AU", ha="center", fontsize=16)
plt.tight_layout(pad=3.5)

for row in range(len(ax)):
    for column in range(len(ax[row])):
        ax[row][column].legend(fontsize=12)
        ax[row][column].grid(True)
        ax[row][column].tick_params(axis='both', which='major', labelsize=18)

fig.savefig(f"Winter_2026/Figs/wind_model/test3_alpha_psi={psi}.png")