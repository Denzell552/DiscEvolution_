import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from DiscEvolution.constants import *

file_path1 = "denzell_scripts/Data_Updated/flux_vs_alpha/planetgap_Mp=1_alpha=1.0e-03.json"
file_path2 = "denzell_scripts/Data_Updated/flux_vs_alpha/planetgap_Mp=1_alpha=1.0e-04.json"
file_path3 = "denzell_scripts/Data_Updated/flux_vs_alpha/nogap_alpha=1.0e-03.json"
file_path4 = "denzell_scripts/Data_Updated/flux_vs_alpha/nogap_alpha=1.0e-04.json"

# getting data from alpha = 1e-3 
with open (file_path1, 'r') as fp1:
    data3 = json.load(fp1)

radius = np.array(data3['R'])
gas_density3 = data3['Sigma_G']
dust_density3 = data3['Sigma_dust']
dust_flux3 = np.array(data3['dust_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_density3 = data3['Sigma_pebbles']
Mdot, alpha, Mtot, Rd = data3['parameters']['Mdot'], data3['parameters']['alpha'], data3['parameters']['Mtot'], data3['parameters']['Rd']
pebble_flux3 = np.array(data3['pebble_flux']) * 3.15e7 / Mearth # to Mearth/year
time = data3['time']
inner_edge_idx3 = data3['gap_profile']['inner_edge_idx']
outer_edge_idx3 = data3['gap_profile']['outer_edge_idx']

# getting data from alpha = 1e-4
with open (file_path2, 'r') as fp2:
    data4 = json.load(fp2)

gas_density4 = data4['Sigma_G']
dust_density4 = data4['Sigma_dust']
dust_flux4 = np.array(data4['dust_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_density4 = data4['Sigma_pebbles']
pebble_flux4 = np.array(data4['pebble_flux']) * 3.15e7 / Mearth # to Mearth/year
inner_edge_idx4 = data4['gap_profile']['inner_edge_idx']
outer_edge_idx4 = data4['gap_profile']['outer_edge_idx']

# getting data from alpha = 1e-3 (no gap)
with open (file_path3, 'r') as fp3:
    data3_nogap = json.load(fp3)

dust_flux3_nogap = np.array(data3_nogap['dust_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_flux3_nogap = np.array(data3_nogap['pebble_flux']) * 3.15e7 / Mearth # to Mearth/year

# getting data from alpha = 1e-4 (no gap)
with open (file_path4, 'r') as fp4:
    data4_nogap = json.load(fp4)

dust_flux4_nogap = np.array(data4_nogap['dust_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_flux4_nogap = np.array(data4_nogap['pebble_flux']) * 3.15e7 / Mearth # to Mearth/year

# calculating the fraction of flux through the gap at each time step for both alpha values
dust_flux_frac3 = []
dust_flux_frac3_nogap = []
dust_flux_frac4 = []
dust_flux_frac4_nogap = []
pebble_flux_frac3 = []
pebble_flux_frac3_nogap = []
pebble_flux_frac4 = []
pebble_flux_frac4_nogap = []
time_clipped = []

# plotting comparison figure
fig, ax = plt.subplots(2, 2, figsize=(20,16))

color1 = iter(plt.cm.Purples(np.linspace(0.4, 1, 9)))
color2 = iter(plt.cm.Reds(np.linspace(0.4, 1, 9)))

for t in range(len(time)):
    if time[t] % 0.25 == 0:
        dust_flux_frac3.append(np.abs(dust_flux3[t][inner_edge_idx3] / dust_flux3[t][np.argmin(np.abs(radius - (radius[outer_edge_idx3]+6)))]))
        dust_flux_frac3_nogap.append(np.abs(dust_flux3_nogap[t][inner_edge_idx3] / dust_flux3_nogap[t][np.argmin(np.abs(radius - (radius[outer_edge_idx3]+6)))]))
        dust_flux_frac4.append(np.abs(dust_flux4[t][inner_edge_idx4] / dust_flux4[t][outer_edge_idx4]))
        dust_flux_frac4_nogap.append(np.abs(dust_flux4_nogap[t][inner_edge_idx4] / dust_flux4_nogap[t][outer_edge_idx4]))
        pebble_flux_frac3.append(np.abs(pebble_flux3[t][inner_edge_idx3] / pebble_flux3[t][np.argmin(np.abs(radius - (radius[outer_edge_idx3]+6)))]))
        pebble_flux_frac3_nogap.append(np.abs(pebble_flux3_nogap[t][inner_edge_idx3] / pebble_flux3_nogap[t][np.argmin(np.abs(radius - (radius[outer_edge_idx3]+6)))]))
        pebble_flux_frac4.append(np.abs(pebble_flux4[t][inner_edge_idx4] / pebble_flux4[t][outer_edge_idx4]))
        pebble_flux_frac4_nogap.append(np.abs(pebble_flux4_nogap[t][inner_edge_idx4] / pebble_flux4_nogap[t][outer_edge_idx4]))
        time_clipped.append(time[t])

        ax[0,0].loglog(radius, pebble_density3[t], color=next(color1), label=f'{time[t]:.2f} Myrs')
        ax[0,1].loglog(radius, pebble_density4[t], color=next(color2))


# Pebble density alpha = 1e-3
ax[0,0].set_title('$\\alpha=10^{-3}$', fontsize=25)
ax[0,0].set_xlabel('Radius (AU)', fontsize=20)
ax[0,0].set_ylabel('$\\Sigma [g/cm^2]$', fontsize=20)
ax[0,0].axvline(radius[inner_edge_idx3], color='black')
ax[0,0].axvline(radius[outer_edge_idx3]+5, color='black')
ax[0,0].set_yscale('symlog', linthresh=1e-5)

# Pebble density alpha = 1e-4
ax[0,1].set_title('$\\alpha=10^{-4}$', fontsize=25)
ax[0,1].set_xlabel('Radius (AU)', fontsize=20)
ax[0,1].set_ylabel('$\\Sigma [g/cm^2]$', fontsize=20)
ax[0,1].axvline(radius[inner_edge_idx4], color='black')
ax[0,1].axvline(radius[outer_edge_idx4], color='black')
ax[0,1].set_yscale('symlog', linthresh=1e-5)

# Dust flux fraction
ax[1,0].semilogy(time_clipped, dust_flux_frac3, color='blue', label=r'$\alpha=10^{-3}$')
ax[1,0].semilogy(time_clipped, dust_flux_frac4, color='green', label=r'$\alpha=10^{-4}$')
ax[1,0].semilogy(time_clipped, dust_flux_frac3_nogap, color='blue', linestyle='dashed', label=r'$\alpha=10^{-3}$ no gap', alpha=0.7)
ax[1,0].semilogy(time_clipped, dust_flux_frac4_nogap, color='green', linestyle='dashed', label=r'$\alpha=10^{-4}$ no gap', alpha=0.7)
ax[1,0].set_title('Dust Flux Fraction', fontsize=25)
ax[1,0].set_xlabel('Time (Myrs)', fontsize=20)
ax[1,0].set_ylabel('$\\phi_{inner} / \\phi_{outer}$', fontsize=20)

# Pebble flux fraction
ax[1,1].semilogy(time_clipped, pebble_flux_frac3, color='blue')
ax[1,1].semilogy(time_clipped, pebble_flux_frac4, color='green')
ax[1,1].semilogy(time_clipped, pebble_flux_frac3_nogap, color='blue', linestyle='dashed', alpha=0.7)
ax[1,1].semilogy(time_clipped, pebble_flux_frac4_nogap, color='green', linestyle='dashed', alpha=0.7)
ax[1,1].set_title('Pebble Flux Fraction', fontsize=25)
ax[1,1].set_xlabel('Time (Myrs)', fontsize=20)
ax[1,1].set_ylabel('$\\phi_{inner} / \\phi_{outer}$', fontsize=20)

plt.tight_layout(pad=3.5)

for row in range(len(ax)):
    for column in range(len(ax[row])):
        ax[row][column].legend(fontsize=14)
        ax[row][column].grid(True)
        ax[row][column].tick_params(axis='both', which='major', labelsize=19)
        plt.setp(ax[row][column].spines.values(), linewidth=2)

custom_handles = [
    Line2D([0], [0], color='blue', linestyle='-', label=r'$\alpha=10^{-3}$'),
    Line2D([0], [0], color='green', linestyle='-', label=r'$\alpha=10^{-4}$'),
    Line2D([0], [0], color='black', linestyle='-', label='With Gap'),
    Line2D([0], [0], color='black', linestyle='--', label='No Gap'),
]
ax[1,0].legend(handles=custom_handles, loc='lower right', fontsize=14)

plt.savefig('denzell_scripts/Figs_Updated/flux_vs_alpha/alpha_comparison.png')

