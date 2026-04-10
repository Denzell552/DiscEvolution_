import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from DiscEvolution.constants import *
import pandas as pd

fp = "Winter_2026/Data/wind_model/test_bump_1Myrs_vfrag=50_psi=10.json"
fp2 = "Winter_2026/Data/wind_model/test_1Myrs_vfrag=50_psi=10.json"

suriano = pd.read_csv('Winter_2026/Data/wind_model/Suriano_Data.csv', header=None, names=['x','y'])

with open (fp, 'r') as f:
    data = json.load(f)

gas_density = np.array(data['Sigma_G'])
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
viscosity = np.array(data['viscosity']) * AU**2 * yr
pressure = data['Pressure']
gradient = data['pressure_gradient']
pebble_flux = data['pebble_flux']
stokes = data['stokes_number']


with open (fp2, 'r') as f:
    data_nobump = json.load(f)

pebble_density_nobump = data_nobump['Sigma_pebbles']
dust_density_nobump = data_nobump['Sigma_dust']
size_nobump = data_nobump['pebble_size']


fig, ax = plt.subplots(4,2,figsize=(20,30))

color1 = plt.cm.Blues(np.linspace(0.4, 1, 9))
color2 = plt.cm.Oranges(np.linspace(0.4, 1, 9))
color3 = plt.cm.Greys(np.linspace(0.4, 1, 9))
color4 = plt.cm.Purples(np.linspace(0.4, 1, 9))
color5 = plt.cm.Reds(np.linspace(0.4, 1, 9))
color6 = plt.cm.Greys(np.linspace(0.4, 1, 9))
color7 = plt.cm.Greens(np.linspace(0.4, 1, 9))

'''
for t in range(len(time)):
    
    ax[0,0].loglog(radius, gas_density[t], color=next(color1), label=f'{time[t]:.2f} Myr')
    ax[0,1].loglog(radius, pebble_density[t], color=next(color2))
    ax[1,0].loglog(radius, dust_density[t], color=next(color3))
    ax[2,1].loglog(radius, viscosity[t], color=next(color4))
    ax[3,0].loglog(radius, size[t], color=next(color5))
    ax[1,1].semilogx(radius, gradient[t], color=next(color6))
    ax[2,0].loglog(radius, stokes[t][1], color=next(color7))
'''

ax[0,0].loglog(radius, gas_density[-1], color=color1[-1], label=f'{time[-1]:.2f} Myr')
ax[0,0].set_xlabel('Radius (AU)', fontsize=20)
ax[0,0].set_ylabel('Gas Surface Density ($g/cm^2$)', fontsize=20)
ax[0,0].set_title('Gas Surface Density Evolution', fontsize=25)

ax[0,1].loglog(radius, pebble_density[-1], color=color2[-1], label='with bump')
ax[0,1].loglog(radius, pebble_density_nobump[-1], color='black', linestyle='--', label='no bump')
ax[0,1].set_xlabel('Radius (AU)', fontsize=20)
ax[0,1].set_ylabel('Surface Density ($g/cm^2$)', fontsize=20)
ax[0,1].set_title('Pebble Surface Density Evolution', fontsize=25)
ax[0,1].set_ylim(1e-5, 1e3)

ax[1,0].loglog(radius, dust_density[-1], color=color3[-1], label='with bump')
ax[1,0].loglog(radius, dust_density_nobump[-1], color='navy', linestyle='--', label='no bump')
ax[1,0].set_xlabel('Radius (AU)', fontsize=20)
ax[1,0].set_ylabel('Surface Density ($g/cm^2$)', fontsize=20)
ax[1,0].set_title('Dust Surface Density Evolution', fontsize=25)
ax[1,0].set_ylim(1e-7, 5e3)

ax[1,1].semilogx(radius, gradient[-1], color=color6[-1])
ax[1,1].set_xlabel('Radius (AU)', fontsize=20)
ax[1,1].set_ylabel('Pressure Gradient', fontsize=20)
ax[1,1].set_title('Pressure Gradient Evolution', fontsize=25)

ax[2,0].loglog(radius, stokes[1][-1], color='blue')
ax[2,0].set_xlabel('Radius (AU)', fontsize=20)
ax[2,0].set_ylabel('Stokes Number', fontsize=20)
ax[2,0].set_title('Stokes Number Evolution', fontsize=25)

ax[2,1].loglog(radius, viscosity[-1], color=color4[-1])
ax[2,1].set_xlabel('Radius (AU)', fontsize=20)
ax[2,1].set_ylabel('Viscosity (cm²/s)', fontsize=20)
ax[2,1].set_title('Viscosity Evolution', fontsize=25)

ax[3,0].loglog(radius, size[-1], color=color5[-1], label='With Bump')
ax[3,0].loglog(radius, size_nobump[-1], color='black', linestyle='--', label='No Bump')
ax[3,0].set_xlabel('Radius (AU)', fontsize=20)
ax[3,0].set_ylabel('Pebble Size (cm)', fontsize=20)
ax[3,0].set_title('Pebble Size Evolution', fontsize=25)

ax[3,1].plot(suriano['x'], suriano['y'], label='Suriano (2020) Data', color='black', linewidth=2)
ax[3,1].plot(radius, gas_density[1]/gas_density[0], color='green')
ax[3,1].set_xlabel('Radius (AU)', fontsize=20)
ax[3,1].set_ylabel('$\\Sigma_{gas} / \\Sigma_{gas,0}$', fontsize=20)
ax[3,1].set_title('Gas Surface Density Ratio', fontsize=25)
ax[3,1].set_ylim(0, 8)
ax[3,1].set_xlim(0, 20)

plt.figtext(0.5, 0.01, f"Mdot={Mdot:.3e}Msun/yr, alpha={alpha:.0e}, psi={psi}, Rd={Rd:.2f}AU", ha="center", fontsize=16)
plt.tight_layout(pad=3.5)

for row in range(len(ax)):
    for column in range(len(ax[row])):
        ax[row][column].legend(fontsize=12)
        ax[row][column].grid(True)
        ax[row][column].tick_params(axis='both', which='major', labelsize=18)

fig.savefig(f"Winter_2026/Figs/wind_model/test_bump_1Myrs_vfrag=50_psi={psi}.png")


# Animated pebble flux evolution over all available timesteps
if len(time) > 1:
    fig_flux, ax_flux = plt.subplots(figsize=(10, 6))

    # Avoid invalid log limits if any timestep has very small/large values.
    y_min = np.nanmin(pebble_flux)
    y_max = np.nanmax(pebble_flux)
    if y_min <= 0:
        y_min = 1e-30

    (flux_line,) = ax_flux.loglog(radius[:-1], pebble_flux[0], color='tab:red', lw=2)
    ax_flux.set_xlabel('Radius (AU)', fontsize=14)
    ax_flux.set_ylabel('Pebble Flux ($M_{Earth}/yr$)', fontsize=14)
    ax_flux.set_title(f'Pebble Flux Evolution | t = {time[0]:.2f} Myr', fontsize=16)
    ax_flux.set_xlim(np.min(radius[:-1]), np.max(radius[:-1]))
    ax_flux.set_ylim(y_min, y_max)
    ax_flux.grid(True, which='both', alpha=0.3)

    def update(frame):
        flux_line.set_ydata(pebble_flux[frame])
        ax_flux.set_title(f'Pebble Flux Evolution | t = {time[frame]:.2f} Myr', fontsize=16)
        return (flux_line,)

    ani = FuncAnimation(fig_flux, update, frames=len(time), interval=200, blit=True, repeat=True)
    ani.save(f"Winter_2026/Figs/wind_model/pebble_flux_1Myrs_bump_vfrag=50_animation_psi={psi}.gif", writer=PillowWriter(fps=2))
    plt.close(fig_flux)
