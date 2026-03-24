import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from DiscEvolution.constants import *

fp = "Winter_2026/Data/Stammler2023rep/rep_test4_vfrag=100.0.json"
fp_nogap = "Winter_2026/Data/Stammler2023rep/rep_nogap_test4_vfrag=100.0.json"

with open (fp, 'r') as f:
    data = json.load(f)

gas_density = data['Sigma_G']
dust_density = data['Sigma_dust']
pebble_density = data['Sigma_pebbles']
radius = data['R']
size = data['pebble_size']
peb_v = data['pebble_velocity']
time = data['time']
time_yr = np.array(time) * 1e6
frag_v = data['frag_velocity']
pebble_flux_15 = data['pebble_flux_15']
pebble_flux_2 = data['pebble_flux_2']
total_flux = data['total_flux']
Mdot, alpha, Mtot, Rd = data['parameters']['Mdot'], data['parameters']['alpha'], data['parameters']['Mtot'], data['parameters']['Rd']
accreted_mass = -np.array(data['total_accreted_mass_fraction'])
total_grain_mass = np.array(data['total_grain_mass'])

with open (fp_nogap, 'r') as f:
    data_nogap = json.load(f)

pebble_flux_15_nogap = data_nogap['pebble_flux_15']
pebble_flux_2_nogap = data_nogap['pebble_flux_2']
accreted_mass_nogap = -np.array(data_nogap['total_accreted_mass_fraction'])
total_grain_mass_nogap = np.array(data_nogap['total_grain_mass'])


fig, ax = plt.subplots(4,2,figsize=(20,32))

color1 = iter(plt.cm.Blues(np.linspace(0.4, 1, 9)))
color2 = iter(plt.cm.Oranges(np.linspace(0.4, 1, 9)))
color3 = iter(plt.cm.Greys(np.linspace(0.4, 1, 9)))
color4 = iter(plt.cm.Purples(np.linspace(0.4, 1, 9)))
color5 = iter(plt.cm.Reds(np.linspace(0.4, 1, 9)))
color6 = iter(plt.cm.Greys(np.linspace(0.4, 1, 9)))
color7 = iter(plt.cm.Greens(np.linspace(0.4, 1, 9)))

# points where to plot
time_Myr = np.array(time)
idx0 = np.argmin(np.abs(time_Myr - 0.05))
idx1 = np.argmin(np.abs(time_Myr - 0.25))
idx2 = np.argmin(np.abs(time_Myr - 0.5))
idx3 = np.argmin(np.abs(time_Myr - 0.75))
idx4 = np.argmin(np.abs(time_Myr - 1.0))
idx5 = np.argmin(np.abs(time_Myr - 5))
idx6 = np.argmin(np.abs(time_Myr - 10))

for t in range(len(time_yr)):
    
    if t == 0 or t == idx0 or t == idx1 or t == idx2 or t == idx3 or t == idx4 or t == idx5 or t == idx6:
        ax[0,0].loglog(radius, gas_density[t], color=next(color1), label=f'{time[t]:.2f} Myrs')
        ax[2,0].loglog(radius, pebble_density[t], color=next(color2))
        ax[2,1].loglog(radius, dust_density[t], color=next(color4))
        ax[3,0].loglog(radius, size[t], color=next(color5))


ax[0,0].set_xlabel('Radius (AU)', fontsize=20)
ax[0,0].set_ylabel('Gas Surface Density ($g/cm^2$)', fontsize=20)
ax[0,0].set_title('Gas Surface Density Evolution', fontsize=25)
ax[0,0].axvline(radius[np.argmin(np.abs(np.array(radius) - 15))], color='Green', linestyle='--', label='15 AU')
ax[0,0].axvline(radius[np.argmin(np.abs(np.array(radius) - 2))], color='#A8B504', linestyle='--', label='2 AU')

ax[0,1].loglog(time_yr, accreted_mass/total_grain_mass[0], color='blue', label='Accreted Mass w Planet')
ax[0,1].loglog(time_yr, accreted_mass_nogap/total_grain_mass_nogap[0], color='blue', linestyle='--', label='Accreted Mass w/o Planet')
ax[0,1].loglog(time_yr, total_grain_mass/total_grain_mass[0], color='orange', label='Total Grain Mass w Planet')
ax[0,1].loglog(time_yr, total_grain_mass_nogap/total_grain_mass_nogap[0], color='orange', linestyle='--', label='Total Grain Mass w/o Planet')
ax[0,1].set_xlabel('Time (yrs)', fontsize=20)
ax[0,1].set_ylabel('Mass Fraction', fontsize=20)
ax[0,1].set_title('Accreted Grains vs Total Grains left', fontsize=25)
ax[0,1].set_yscale('linear')
ax[0,1].set_ylim(5e-3, 1.1)
ax[0,1].set_xlim(10**3, 10**7)

ax[1,0].loglog(time_yr[::5], pebble_flux_15[::5], color='Green', label='r = 15 AU')
ax[1,0].loglog(time_yr[::5], pebble_flux_2[::5], color='#A8B504', label='r = 2 AU')
ax[1,0].loglog(time_yr[::5], pebble_flux_15_nogap[::5], color='Green', linestyle='--')
ax[1,0].loglog(time_yr[::5], pebble_flux_2_nogap[::5], color='#A8B504', linestyle='--')
ax[1,0].set_xlabel('Time (yrs)', fontsize=20)
ax[1,0].set_ylabel('Pebble Flux ($M_{earth}/yr$)', fontsize=20)
ax[1,0].set_title('Pebble Flux Evolution at 15AU and 2AU', fontsize=25)
ax[1,0].tick_params(axis='y', which='minor', labelsize=18)
ax[1,0].set_ylim(1e-8, 1e-2)
ax[1,0].set_xlim(10**3, 10**7)

ax[1,1].loglog(time_yr, accreted_mass/total_grain_mass[0], color='blue')
ax[1,1].loglog(time_yr, accreted_mass_nogap/total_grain_mass_nogap[0], color='blue', linestyle='--')
ax[1,1].set_xlabel('Time (yrs)', fontsize=20)
ax[1,1].set_ylabel('Accreted Mass Fraction', fontsize=20)
ax[1,1].set_title('Fraction of Total Mass Accreted', fontsize=25)
ax[1,1].axhline(1, color='black', linestyle='--')
ax[1,1].set_ylim(5e-3, 2e0)
ax[1,1].set_xlim(10**3, 10**7)

ax[2,0].set_xlabel('Radius (AU)', fontsize=20)
ax[2,0].set_ylabel('Pebble Surface Density ($g/cm^2$)', fontsize=20)
ax[2,0].set_title('Pebble Surface Density Evolution', fontsize=25)
ax[2,0].set_ylim(1e-6, 1e3)

ax[2,1].set_xlabel('Radius (AU)', fontsize=20)
ax[2,1].set_ylabel('Dust Surface Density ($g/cm^2$)', fontsize=20)
ax[2,1].set_title('Dust Surface Density Evolution', fontsize=25)
ax[2,1].set_ylim(1e-6, 1e3)

ax[3,0].set_xlabel('Radius (AU)', fontsize=20)
ax[3,0].set_ylabel('Pebble Size (cm)', fontsize=20)
ax[3,0].set_title('Pebble Size Evolution', fontsize=25)
ax[3,0].set_ylim(1e-6, 1e3)

size_avg = []
size_max = []
for t in range(len(time_yr)):
    #average pebble size in first 30 AU only
    idx = np.argmin(np.abs(np.array(radius) - 30))
    s_avg = np.array(size[t][:idx]).sum(0) / len(size[t][:idx])
    size_avg.append(s_avg)

    # maximum pebble size
    s_max = np.array(size[t]).max()
    size_max.append(s_max)

ax[3,1].loglog(time_yr, size_avg, color='purple', label='Average Pebble Size')
ax[3,1].loglog(time_yr, size_max, color='red', label='Max Pebble Size')
ax[3,1].set_xlabel('Time (yrs)', fontsize=20)
ax[3,1].set_ylabel('Pebble Size (cm)', fontsize=20)
ax[3,1].set_title('Total Pebble Size Evolution', fontsize=25)
ax[3,1].set_xlim(10**3, 10**7)

for row in range(len(ax)):
    for column in range(len(ax[row])):
        ax[row][column].legend(fontsize=12)
        ax[row][column].grid(True)
        ax[row][column].tick_params(axis='both', which='major', labelsize=18)

custom_handles = [
    Line2D([0], [0], color='Green', linestyle='-', label='r = 15 AU'),
    Line2D([0], [0], color='#A8B504', linestyle='-', label='r = 2 AU'),
    Line2D([0], [0], color='black', linestyle='--', label='No planet'),
    Line2D([0], [0], color='black', linestyle='-', label='With a planet'),
]
ax[1,0].legend(handles=custom_handles, loc='upper right', fontsize=14)

plt.figtext(0.5, 0.01, f"Mdot={Mdot:.3e}Msun/yr, alpha={alpha:.0e}, Mtot={Mtot:.3e}Msun, Rd={Rd}AU, frag_velocity={frag_v[0]:.1f}m/s", ha="center", fontsize=12)

plt.tight_layout(pad=3.5)
fig.savefig(f"Winter_2026/Figs/Stammler2023rep/rep_test4.png")