import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from DiscEvolution.constants import *

file_path1 = "denzell_scripts/Data_Updated/frag_limit/planetgap_vfrag=10_Mp=2_alpha=1.0e-04.json"
file_path2 = "denzell_scripts/Data_Updated/frag_limit/vfrag=10_alpha=1.0e-04.json"

with open (file_path1, 'r') as fp1:
    data1 = json.load(fp1)

radius = np.array(data1['R'])
dust_flux1 = np.array(data1['dust_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_flux1 = np.array(data1['pebble_flux']) * 3.15e7 / Mearth # to Mearth/year
time = data1['time']
inner_edge_idx = data1['gap_profile']['inner_edge_idx']
outer_edge_idx = data1['gap_profile']['outer_edge_idx']

with open (file_path2, 'r') as fp2:
    data2 = json.load(fp2)
dust_flux2 = np.array(data2['dust_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_flux2 = np.array(data2['pebble_flux']) * 3.15e7 / Mearth # to Mearth/year
   

# calculating the fraction of flux through the gap at each time step for both alpha values
dust_flux_frac = []
pebble_flux_frac = []
dust_flux_frac_nogap = []
pebble_flux_frac_nogap = []
time_clipped = []

# plotting comparison figure
fig, ax = plt.subplots(1, 2, figsize=(25, 8))


for t in range(len(time)):
    if time[t] % 0.25 == 0:
        dust_flux_frac.append(np.abs(dust_flux1[t][inner_edge_idx] / dust_flux1[t][np.argmin(np.abs(radius - (radius[outer_edge_idx])))]))
        dust_flux_frac_nogap.append(np.abs(dust_flux2[t][inner_edge_idx] / dust_flux2[t][np.argmin(np.abs(radius - (radius[outer_edge_idx])))]))

        pebble_flux_frac.append(np.abs(pebble_flux1[t][inner_edge_idx] / pebble_flux1[t][np.argmin(np.abs(radius - (radius[outer_edge_idx])))]))
        pebble_flux_frac_nogap.append(np.abs(pebble_flux2[t][inner_edge_idx] / pebble_flux2[t][np.argmin(np.abs(radius - (radius[outer_edge_idx])))]))

        time_clipped.append(time[t])

# plotting dust flux fraction comparison
ax[0].semilogy(time_clipped, dust_flux_frac, color='blue', label='Mp=$1 M_J$')
ax[0].semilogy(time_clipped, dust_flux_frac_nogap, linestyle='--', color='black', label='No Gap', alpha=0.7)
ax[0].set_xlabel('Time (Myrs)', fontsize=27)
ax[0].set_title('Dust Flux Fraction', fontsize=25)
ax[0].set_ylabel('$\\phi_{inner} / \\phi_{outer}$', fontsize=25)

# plotting pebble flux fraction comparison
ax[1].semilogy(time_clipped, pebble_flux_frac, color='blue')
ax[1].semilogy(time_clipped, pebble_flux_frac_nogap, linestyle='--', color='black', alpha=0.7)
ax[1].set_xlabel('Time (Myrs)', fontsize=27)
ax[1].set_title('Pebble Flux Fraction', fontsize=25)

plt.tight_layout(pad=3.5)

for column in range(len(ax)):
    ax[column].legend(fontsize=17)
    ax[column].grid(True)
    ax[column].tick_params(axis='both', which='major', labelsize=23)
    plt.setp(ax[column].spines.values(), linewidth=2)

plt.savefig('denzell_scripts/Figs_Updated/frag_limit/strong_gap2.png')


