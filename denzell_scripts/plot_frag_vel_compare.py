import json
import numpy as np
import matplotlib.pyplot as plt
import math

fp1 = "denzell_scripts/Data_Updated/frag_limit/planetgap_vfrag=10_Mp=1_alpha=1.0e-03.json"
fp2 = "denzell_scripts/Data_Updated/frag_limit/planetgap_vfrag=7_Mp=1_alpha=1.0e-03.json"
fp3 = "denzell_scripts/Data_Updated/frag_limit/planetgap_vfrag=4_Mp=1_alpha=1.0e-03.json"
fp4 = "denzell_scripts/Data_Updated/frag_limit/planetgap_vfrag=1_Mp=1_alpha=1.0e-03.json"

with open (fp1, 'r') as f:
    data1 = json.load(f)

gas_density1 = data1['Sigma_G']
dust_density1 = data1['Sigma_dust']
pebble_density1 = data1['Sigma_pebbles']
radius = data1['R']
size1 = data1['pebble_size']
peb_v1 = data1['pebble_velocity']
time = data1['time']
dust_v1 = data1['dust_velocity']
frag_v1 = 10
drift_limit1 = data1['drift_limit']
frag_limit1 = data1['frag_limit']

with open (fp2, 'r') as f:
    data2 = json.load(f)
gas_density2 = data2['Sigma_G']
dust_density2 = data2['Sigma_dust']
pebble_density2 = data2['Sigma_pebbles']
size2 = data2['pebble_size']
peb_v2 = data2['pebble_velocity']
dust_v2 = data2['dust_velocity']
frag_v2 = 7
frag_limit2 = data2['frag_limit']
drift_limit2 = data2['drift_limit']

with open (fp3, 'r') as f:
    data3 = json.load(f)
gas_density3 = data3['Sigma_G']
dust_density3 = data3['Sigma_dust']
pebble_density3 = data3['Sigma_pebbles']
size3 = data3['pebble_size']
peb_v3 = data3['pebble_velocity']
dust_v3 = data3['dust_velocity']
frag_v3 = 4
frag_limit3 = data3['frag_limit']
drift_limit3 = data3['drift_limit']

with open (fp4, 'r') as f:
    data4 = json.load(f)
gas_density4 = data4['Sigma_G']
dust_density4 = data4['Sigma_dust']
pebble_density4 = data4['Sigma_pebbles']
size4 = data4['pebble_size']
peb_v4 = data4['pebble_velocity']
dust_v4 = data4['dust_velocity']
frag_v4 = 1
frag_limit4 = data4['frag_limit']
drift_limit4 = data4['drift_limit']

fig, ax = plt.subplots(2, 2, figsize=(25,16))

color1 = 'green'
color2 = 'blue'
color3 = 'red'
color4 = 'grey'

ax[0,0].loglog(radius, dust_density1[-1], color=color1, label=f'Frag Velocity = 10 m/s')
ax[0,0].loglog(radius, dust_density2[-1], color=color2, label=f'Frag Velocity = 7 m/s')
ax[0,0].loglog(radius, dust_density3[-1], color=color3, label=f'Frag Velocity = 4 m/s')
ax[0,0].loglog(radius, dust_density4[-1], color=color4, label=f'Frag Velocity = 1 m/s')
ax[0,0].set_xlabel('Radius [AU]', fontsize=25)
ax[0,0].set_ylabel('Surface Density [$g/cm^2$]', fontsize=25)
ax[0,0].set_title('Dust Surface Density', fontsize=27)
ax[0,0].set_yscale('symlog', linthresh=1e-4)

ax[1,1].loglog(radius, size1[-1], color=color1)
ax[1,1].loglog(radius, size2[-1], color=color2)
ax[1,1].loglog(radius, size3[-1], color=color3)
ax[1,1].loglog(radius, size4[-1], color=color4)
ax[1,1].set_xlabel('Radius [AU]', fontsize=25)
ax[1,1].set_ylabel('Pebble Size [cm]', fontsize=25)
ax[1,1].set_title('Pebble Size', fontsize=27)

ax[1,0].loglog(radius, frag_limit1[-1], color=color1, label='Fragmentation Limit')
ax[1,0].loglog(radius, frag_limit2[-1], color=color2)
ax[1,0].loglog(radius, frag_limit3[-1], color=color3)
ax[1,0].loglog(radius, frag_limit4[-1], color=color4)
ax[1,0].loglog(radius, drift_limit1[-1], color=color1, linestyle='--', label='Drift Limit')
ax[1,0].loglog(radius, drift_limit2[-1], color=color2, linestyle='--')
ax[1,0].loglog(radius, drift_limit3[-1], color=color3, linestyle='--')
ax[1,0].loglog(radius, drift_limit4[-1], color=color4, linestyle='--')
ax[1,0].set_xlabel('Radius [AU]', fontsize=25)
ax[1,0].set_ylabel('Velocity [m/s]', fontsize=25)
ax[1,0].set_title('Growth Limits', fontsize=27)
ax[1,0].set_ylim(1e-5, 1e3)

ax[0,1].loglog(radius, pebble_density1[-1], color=color1)
ax[0,1].loglog(radius, pebble_density2[-1], color=color2)
ax[0,1].loglog(radius, pebble_density3[-1], color=color3)
ax[0,1].loglog(radius, pebble_density4[-1], color=color4)
ax[0,1].set_xlabel('Radius [AU]', fontsize=25)
ax[0,1].set_title('Pebble Surface Density', fontsize=27)
ax[0,1].set_yscale('symlog', linthresh=1e-4)


# create legend for fragmentation and drift limits
for row in range(len(ax)):
    for column in range(len(ax[row])):
        ax[row][column].legend(fontsize=17)
        ax[row][column].grid(True)
        ax[row][column].tick_params(axis='both', which='major', labelsize=23)
        plt.setp(ax[row][column].spines.values(), linewidth=2)


legend = ax[1,0].legend(loc='upper right', fontsize=17)
handles = legend.legend_handles
symbols = ['-', '--', ':', 'o']
colors = ['black', 'black', 'black', 'black']

for i, handle in enumerate(handles):
    handle.set_color(colors[i])
    if i==3:
        handle.set_marker(symbols[i])
    else:
        handle.set_linestyle(symbols[i])


plt.tight_layout(pad=3.5)


fig.savefig(f"denzell_scripts/Figs_Updated/frag_limit/frag_compare.png")