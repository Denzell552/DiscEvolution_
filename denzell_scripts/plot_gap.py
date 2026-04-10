import json
import numpy as np
import matplotlib.pyplot as plt
from DiscEvolution.constants import *

file_path1 = "denzell_scripts/Data_Updated/flux_vs_alpha/test_nogap_alpha=1.0e-03.json"

with open (file_path1, 'r') as fp1:
    data3 = json.load(fp1)

radius = np.array(data3['R'])
gas_density3 = data3['Sigma_G']
dust_density3 = data3['Sigma_dust']
dust_flux3 = np.array(data3['dust_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_density3 = data3['Sigma_pebbles']
Mdot, alpha, Mtot, Rd = data3['parameters']['Mdot'], data3['parameters']['alpha'], data3['parameters']['Mtot'], data3['parameters']['Rd']
pebble_flux3 = np.array(data3['pebble_flux']) * 3.15e7 / Mearth # to Mearth/year
size3 = data3['pebble_size']
time = data3['time']
gas_velocity3 = np.array(data3['gas_velocity']) * AU * yr / 3.15e7
dust_velocity3 = data3['dust_velocity']
pebble_velocity3 = data3['pebble_velocity']
dust_drift_velocity3 = data3['dust_drift_velocity']
pressure3 = data3['Pressure']
gradient = data3['pressure_gradient']
stokes3 = data3['stokes_number']
Mdot_evolv3 = data3['Mdot']
viscosity3 = data3['viscosity']
temperature3 = data3['temperature']
#inner_edge_idx3 = data3['gap_profile']['inner_edge_idx']
#outer_edge_idx3 = data3['gap_profile']['outer_edge_idx']

fig, ax = plt.subplots(6, 2, figsize=(20,40))

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

for t in range(len(time)):
    if time[t] % 0.25 == 0:
        ax[0,0].loglog(radius, gas_density3[t], color=next(color1), label=f'{time[t]:.2f} Myrs')
        ax[0,1].loglog(radius, size3[t], color=next(color2))
        ax[1,0].loglog(radius, dust_density3[t], color=next(color3))
        ax[1,1].loglog(radius, pebble_density3[t], color=next(color4))
        ax[2,0].loglog(radius[:-1], dust_flux3[t], color=next(color5))
        ax[2,1].loglog(radius[:-1], pebble_flux3[t], color=next(color6))
        ax[3,0].loglog(radius[:-1], dust_velocity3[t], color=next(color7))
        ax[3,1].loglog(radius[:-1], pebble_velocity3[t], color=next(color8))
        ax[4,0].loglog(radius, dust_drift_velocity3[t], color=next(color9))
        ax[4,1].semilogx(radius, temperature3[t], color=next(color10))
        ax[5,0].loglog(radius, stokes3[t][0], color=next(color11))
        ax[5,1].loglog(radius, stokes3[t][1], color=next(color12))

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
#ax[1,0].axvline(radius[inner_edge_idx3], color='green')
#ax[1,0].axvline(radius[outer_edge_idx3]+6, color='blue')

# Pebble density
ax[1,1].set_title('Pebble Surface Density', fontsize=25)
ax[1,1].set_xlabel('Radius [AU]', fontsize=20)
ax[1,1].set_ylabel('Surface Density $[g/cm^2]$', fontsize=20)
#ax[1,1].set_ylim(1e-5, 1e3)
ax[1,1].set_yscale('symlog', linthresh=1e-5)
#ax[1,1].axvline(radius[inner_edge_idx3], color='green')
#ax[1,1].axvline(radius[outer_edge_idx3]+6, color='blue')

# Dust flux
ax[2,0].set_title('Dust Flux', fontsize=25)
ax[2,0].set_xlabel('Radius [AU]', fontsize=20)
ax[2,0].set_ylabel('Flux $[M_{\\oplus}/yr]$', fontsize=20)
#ax[2,0].set_ylim(1e10, 1e19)
ax[2,0].set_yscale('symlog', linthresh=1e-10)
#ax[2,0].axvline(radius[inner_edge_idx3], color='green')
#ax[2,0].axvline(radius[outer_edge_idx3]+6, color='blue')

# Pebble flux
ax[2,1].set_title('Pebble Flux', fontsize=25)
ax[2,1].set_xlabel('Radius [AU]', fontsize=20)
ax[2,1].set_ylabel('Flux $[M_{\\oplus}/yr]$', fontsize=20)
#ax[2,1].set_ylim(1e14, 1e19)
ax[2,1].set_yscale('symlog', linthresh=1e-10)
#ax[2,1].axvline(radius[inner_edge_idx3], color='green')
#ax[2,1].axvline(radius[outer_edge_idx3]+6, color='blue')

# Dust velocity
ax[3,0].set_title('Dust Velocity', fontsize=25)
ax[3,0].set_xlabel('Radius [AU]', fontsize=20)
ax[3,0].set_ylabel('Velocity $[cm/s]$', fontsize=20)
ax[3,0].set_yscale('symlog')

# Pebble velocity
ax[3,1].set_title('Pebble Velocity', fontsize=25)
ax[3,1].set_xlabel('Radius [AU]', fontsize=20)
ax[3,1].set_ylabel('Velocity $[cm/s]$', fontsize=20)
ax[3,1].set_yscale('symlog')

# Dust drift velocity
ax[4,0].set_title('Dust Drift Velocity', fontsize=25)
ax[4,0].set_xlabel('Radius [AU]', fontsize=20)
ax[4,0].set_ylabel('Velocity $[cm/s]$', fontsize=20)
ax[4,0].set_yscale('symlog', linthresh=1e-10)

# Mdot evolution 
#ax[4,1].loglog(time, Mdot_evolv3, marker='o', color='black')
ax[4,1].set_title('Temperature', fontsize=25)
ax[4,1].set_xlabel('Radius [AU]', fontsize=20)
ax[4,1].set_ylabel('Temperature [K]', fontsize=20)
#ax[4,1].set_yscale('symlog', linthresh=1e-3)

# dust stokes number
ax[5,0].set_title('Dust Stokes Number', fontsize=25)
ax[5,0].set_xlabel('Radius [AU]', fontsize=20)
ax[5,0].set_ylabel('Stokes Number', fontsize=20)

# pebble stokes number
ax[5,1].set_title('Pebble Stokes Number', fontsize=25)
ax[5,1].set_xlabel('Radius [AU]', fontsize=20)
ax[5,1].set_ylabel('Stokes Number', fontsize=20)

plt.figtext(0.5, 0.003, f"Mdot={Mdot:.3e}Msun/yr, alpha={alpha:.0e}, Mtot={Mtot:.3e}Msun, Rd={Rd:.2f}AU", ha="center", fontsize=16)
plt.tight_layout(pad=3.5)

for row in range(len(ax)):
    for column in range(len(ax[row])):
        ax[row][column].legend(fontsize=12)
        ax[row][column].grid(True)
        ax[row][column].tick_params(axis='both', which='major', labelsize=18)
        plt.setp(ax[row][column].spines.values(), linewidth=2)


plt.savefig('denzell_scripts/Figs_Updated/flux_vs_alpha/test_nogap_alpha=1.0e-03.png')

