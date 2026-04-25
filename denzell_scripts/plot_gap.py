import json
import numpy as np
import matplotlib.pyplot as plt
from DiscEvolution.constants import *

file_path1 = "denzell_scripts/Data_Updated/wind_disks/wind_bump_alpha=1.0e-03_psi=100.0.json"

with open (file_path1, 'r') as fp1:
    data = json.load(fp1)

radius = np.array(data['R'])
gas_density = data['Sigma_G']
dust_density = data['Sigma_dust']
dust_flux = np.array(data['dust_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_density = data['Sigma_pebbles']
Mdot, alpha, Mtot, Rd = data['parameters']['Mdot'], data['parameters']['alpha'], data['parameters']['Mtot'], data['parameters']['Rd']
pebble_flux = np.array(data['pebble_flux']) * 3.15e7 / Mearth # to Mearth/year
size = data['pebble_size']
time = data['time']
gas_velocity = np.array(data['gas_velocity']) * AU * yr / 3.15e7
dust_velocity = data['dust_velocity']
pebble_velocity = data['pebble_velocity']
dust_drift_velocity = data['dust_drift_velocity']
pebble_drift_velocity = data['pebble_drift_velocity']
pressure = data['Pressure']
gradient = data['pressure_gradient']
stokes = data['stokes_number']
Mdot_evolv = data['Mdot']
Mdot_ss = data['Mdot_ss']
viscosity = data['viscosity']
temperature = data['temperature']
#inner_edge_idx = data['gap_profile']['inner_edge_idx']
#outer_edge_idx = data['gap_profile']['outer_edge_idx']
frag_limit = data['frag_limit']
drift_limit = data['drift_limit']

fig, ax = plt.subplots(8, 2, figsize=(20,48))

color1 = iter(plt.cm.Blues(np.linspace(0.4, 1, 9)))
color2 = iter(plt.cm.Oranges(np.linspace(0.4, 1, 9)))
color3 = iter(plt.cm.Greys(np.linspace(0.4, 1, 9)))
color4 = iter(plt.cm.Purples(np.linspace(0.4, 1, 9)))
color5 = iter(plt.cm.Reds(np.linspace(0.4, 1, 9)))
color6 = iter(plt.cm.Greys(np.linspace(0.4, 1, 9)))
color7 = iter(plt.cm.Greens(np.linspace(0.4, 1, 9)))
color8 = iter(plt.cm.Oranges(np.linspace(0.4, 1, 9)))
color9 = iter(plt.cm.Purples(np.linspace(0.4, 1, 9)))
color10 = iter(plt.cm.Blues(np.linspace(0.4, 1, 9)))
color11 = iter(plt.cm.Reds(np.linspace(0.4, 1, 9)))
color12 = iter(plt.cm.Greys(np.linspace(0.4, 1, 9)))
color13 = iter(plt.cm.Greens(np.linspace(0.4, 1, 9)))
color14 = iter(plt.cm.Oranges(np.linspace(0.4, 1, 9)))



for t in range(len(time)):
    if time[t] % 0.25 == 0:
        ax[0,0].loglog(radius, gas_density[t], color=next(color1), label=f'{time[t]:.2f} Myrs')
        ax[0,1].loglog(radius, size[t], color=next(color2))
        ax[1,0].loglog(radius, dust_density[t], color=next(color3))
        ax[1,1].loglog(radius, pebble_density[t], color=next(color4))
        ax[2,0].loglog(radius[:-1], dust_flux[t], color=next(color5))
        ax[2,1].loglog(radius[:-1], pebble_flux[t], color=next(color6))
        ax[3,0].loglog(radius, dust_drift_velocity[t], color=next(color7))
        ax[3,1].loglog(radius, pebble_drift_velocity[t], color=next(color8))
        ax[4,0].loglog(radius[:-1], dust_velocity[t], color=next(color9))
        ax[4,1].loglog(radius[:-1], pebble_velocity[t], color=next(color10))
        ax[5,0].semilogx(radius, gradient[t], color=next(color11))
        ax[5,1].loglog(radius, stokes[t][1], color=next(color12))
        ax[7,0].loglog(radius, frag_limit[t], color=next(color13))
        ax[7,1].loglog(radius, drift_limit[t], color=next(color14))
  

# Gas density
ax[0,0].set_title('Gas Surface Density', fontsize=25)
ax[0,0].set_xlabel('Radius [AU]', fontsize=20)
ax[0,0].set_ylabel('Surface Density $[g/cm^2]$', fontsize=20)

# Pebble size
ax[0,1].set_title('Pebble Size', fontsize=25)
ax[0,1].set_xlabel('Radius [AU]', fontsize=20)
ax[0,1].set_ylabel('Size [cm]', fontsize=20)

# Dust density
ax[1,0].set_title('Dust Surface Density', fontsize=25)
ax[1,0].set_xlabel('Radius [AU]', fontsize=20)
ax[1,0].set_ylabel('Surface Density $[g/cm^2]$', fontsize=20)
#ax[1,0].set_ylim(1e-5, 1e3)
ax[1,0].set_yscale('symlog', linthresh=1e-5)
ax[1,0].axvline(radius[np.argmin(np.abs(radius-0.5))], color='green')
ax[1,0].axvline(radius[np.argmin(np.abs(radius-20))], color='blue')

# Pebble density
ax[1,1].set_title('Pebble Surface Density', fontsize=25)
ax[1,1].set_xlabel('Radius [AU]', fontsize=20)
ax[1,1].set_ylabel('Surface Density $[g/cm^2]$', fontsize=20)
#ax[1,1].set_ylim(1e-5, 1e3)
ax[1,1].set_yscale('symlog', linthresh=1e-5)
ax[1,1].axvline(radius[np.argmin(np.abs(radius-0.5))], color='green')
ax[1,1].axvline(radius[np.argmin(np.abs(radius-20))], color='blue')

# Dust flux
ax[2,0].set_title('Dust Flux', fontsize=25)
ax[2,0].set_xlabel('Radius [AU]', fontsize=20)
ax[2,0].set_ylabel('Flux $[M_{\\oplus}/yr]$', fontsize=20)
#ax[2,0].set_ylim(1e10, 1e19)
ax[2,0].set_yscale('symlog', linthresh=1e-10)
ax[2,0].axvline(radius[np.argmin(np.abs(radius-0.5))], color='green')
ax[2,0].axvline(radius[np.argmin(np.abs(radius-20))], color='blue')

# Pebble flux
ax[2,1].set_title('Pebble Flux', fontsize=25)
ax[2,1].set_xlabel('Radius [AU]', fontsize=20)
ax[2,1].set_ylabel('Flux $[M_{\\oplus}/yr]$', fontsize=20)
#ax[2,1].set_ylim(1e14, 1e19)
ax[2,1].set_yscale('symlog', linthresh=1e-10)
ax[2,1].axvline(radius[np.argmin(np.abs(radius-0.5))], color='green')
ax[2,1].axvline(radius[np.argmin(np.abs(radius-20))], color='blue')

# Dust drift velocity
ax[3,0].set_title('Dust Drift Velocity', fontsize=25)
ax[3,0].set_xlabel('Radius [AU]', fontsize=20)
ax[3,0].set_ylabel('Velocity $[cm/s]$', fontsize=20)
ax[3,0].set_yscale('symlog', linthresh=1e-10)

# Pebble drift velocity
ax[3,1].set_title('Pebble Drift Velocity', fontsize=25)
ax[3,1].set_xlabel('Radius [AU]', fontsize=20)
ax[3,1].set_ylabel('Velocity $[cm/s]$', fontsize=20)
ax[3,1].set_yscale('symlog', linthresh=1e-10)

# dust velocity
ax[4,0].set_title('Dust Velocity', fontsize=25)
ax[4,0].set_xlabel('Radius [AU]', fontsize=20)
ax[4,0].set_ylabel('Velocity $[cm/s]$', fontsize=20)
ax[4,0].set_yscale('symlog', linthresh=1e-3)

# pebble velocity
ax[4,1].set_title('Pebble Velocity', fontsize=25)
ax[4,1].set_xlabel('Radius [AU]', fontsize=20)
ax[4,1].set_ylabel('Velocity $[cm/s]$', fontsize=20)
ax[4,1].set_yscale('symlog', linthresh=1e-3)

# pressure gradient
ax[5,0].set_title('Pressure Gradient', fontsize=25)
ax[5,0].set_xlabel('Radius [AU]', fontsize=20)
ax[5,0].set_ylabel('Gradient', fontsize=20)

# pebble stokes number
ax[5,1].set_title('Pebble Stokes Number', fontsize=25)
ax[5,1].set_xlabel('Radius [AU]', fontsize=20)
ax[5,1].set_ylabel('Stokes Number', fontsize=20)

# Mdot
ax[6,0].plot(time, Mdot_evolv, color='blue', label='Mdot from Gas Velocity')
ax[6,0].plot(time, Mdot_ss, color='orange', label='Steady State Mdot')
ax[6,0].set_title('Mass Accretion Rate', fontsize=25)
ax[6,0].set_xlabel('Time [Myrs]', fontsize=20)
ax[6,0].set_ylabel('$\dot{M}$ [$M_{\odot}/yr$]', fontsize=20)

# growth limit compare 2
ax[6,1].loglog(radius, frag_limit[15], color='blue', label='Fragmentation Limit')
ax[6,1].loglog(radius, drift_limit[15], color='orange', label='Drift Limit')
ax[6,1].set_title('Growth Limit at 1.875Myrs', fontsize=25)
ax[6,1].set_xlabel('Radius [AU]', fontsize=20)
ax[6,1].set_ylabel('Size [cm]', fontsize=20)
ax[6,1].set_yscale('symlog', linthresh=1e-3)

# fragmentation limitax[7,0].loglog(radius, frag_limit3[22], color='blue', label='Fragmentation Limit')
ax[7,0].set_title('Fragmentation Limit', fontsize=25)
ax[7,0].set_xlabel('Radius [AU]', fontsize=20)
ax[7,0].set_ylabel('Size [cm]', fontsize=20)
ax[7,0].set_yscale('symlog', linthresh=1e-3)

# drfit limit
ax[7,1].set_title('Drift Limit', fontsize=25)
ax[7,1].set_xlabel('Radius [AU]', fontsize=20)
ax[7,1].set_ylabel('Size [cm]', fontsize=20)
ax[7,1].set_yscale('symlog', linthresh=1e-3)

plt.figtext(0.5, 0.003, f"Mdot={Mdot:.3e}Msun/yr, alpha={alpha:.0e}, Mtot={Mtot:.3e}Msun, Rd={Rd:.2f}AU", ha="center", fontsize=16)
plt.tight_layout(pad=3.5)

for row in range(len(ax)):
    for column in range(len(ax[row])):
        ax[row][column].legend(fontsize=15)
        ax[row][column].grid(True)
        ax[row][column].tick_params(axis='both', which='major', labelsize=18)
        plt.setp(ax[row][column].spines.values(), linewidth=2)


plt.savefig('denzell_scripts/Figs_Updated/wind_disks/wind_bump_alpha=1.0e-03_psi=100.0.png')

