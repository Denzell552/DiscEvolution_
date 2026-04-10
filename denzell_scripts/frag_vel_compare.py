import json
import numpy as np
import matplotlib.pyplot as plt
import math

fp1 = "Winter_2026/Data/Stammler2023rep/frag_nodrift_complete_compare_f=10.0.json"
fp2 = "Winter_2026/Data/Stammler2023rep/frag_nodrift_complete_compare_f=7.0.json"
fp3 = "Winter_2026/Data/Stammler2023rep/frag_nodrift_complete_compare_f=4.0.json"
fp4 = "Winter_2026/Data/Stammler2023rep/frag_nodrift_complete_compare_f=1.0.json"

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
frag_v1 = data1['frag_velocity']
Mdot, alpha, Mtot, Rd = data1['parameters']['Mdot'], data1['parameters']['alpha'], data1['parameters']['Mtot'], data1['parameters']['Rd']
drift_limit1 = data1['drift_limit']
frag_limit1 = data1['frag_limit']
drifttime_limit1 = data1['drifttime_limit']

with open (fp2, 'r') as f:
    data2 = json.load(f)
gas_density2 = data2['Sigma_G']
dust_density2 = data2['Sigma_dust']
pebble_density2 = data2['Sigma_pebbles']
size2 = data2['pebble_size']
peb_v2 = data2['pebble_velocity']
dust_v2 = data2['dust_velocity']
frag_v2 = data2['frag_velocity']
frag_limit2 = data2['frag_limit']
drift_limit2 = data2['drift_limit']
drifttime_limit2 = data2['drifttime_limit']

with open (fp3, 'r') as f:
    data3 = json.load(f)
gas_density3 = data3['Sigma_G']
dust_density3 = data3['Sigma_dust']
pebble_density3 = data3['Sigma_pebbles']
size3 = data3['pebble_size']
peb_v3 = data3['pebble_velocity']
dust_v3 = data3['dust_velocity']
frag_v3 = data3['frag_velocity']
frag_limit3 = data3['frag_limit']
drift_limit3 = data3['drift_limit']
drifttime_limit3 = data3['drifttime_limit']

with open (fp4, 'r') as f:
    data4 = json.load(f)
gas_density4 = data4['Sigma_G']
dust_density4 = data4['Sigma_dust']
pebble_density4 = data4['Sigma_pebbles']
size4 = data4['pebble_size']
peb_v4 = data4['pebble_velocity']
dust_v4 = data4['dust_velocity']
frag_v4 = data4['frag_velocity']
frag_limit4 = data4['frag_limit']
drift_limit4 = data4['drift_limit']
drifttime_limit4 = data4['drifttime_limit']

fig, ax = plt.subplots(1, 3, figsize=(30,8))

color1 = 'green'
color2 = 'blue'
color3 = 'red'
color4 = 'grey'

ax[0].loglog(radius[-1], dust_density1[-1], color=color1, label=f'Frag Velocity = 10 m/s')
ax[0].loglog(radius[-1], dust_density2[-1], color=color2, label=f'Frag Velocity = 7 m/s')
ax[0].loglog(radius[-1], dust_density3[-1], color=color3, label=f'Frag Velocity = 4 m/s')
ax[0].loglog(radius[-1], dust_density4[-1], color=color4, label=f'Frag Velocity = 1 m/s')
ax[0].set_xlabel('Radius (AU)', fontsize=20)
ax[0].set_ylabel('Surface Density ($g/cm^2$)', fontsize=20)
ax[0].set_title('Dust Surface Density Evolution', fontsize=25)
ax[0].set_ylim(1e-7, 1e2)

ax[1].loglog(radius[-1], size1[-1], color=color1)
ax[1].loglog(radius[-1], size2[-1], color=color2)
ax[1].loglog(radius[-1], size3[-1], color=color3)
ax[1].loglog(radius[-1], size4[-1], color=color4)
ax[1].set_xlabel('Radius (AU)', fontsize=20)
ax[1].set_ylabel('Pebble Size (cm)', fontsize=20)
ax[1].set_title('Pebble Size Evolution', fontsize=25)

ax[2].loglog(radius[-1], pebble_density1[-1], color=color1)
ax[2].loglog(radius[-1], pebble_density2[-1], color=color2)
ax[2].loglog(radius[-1], pebble_density3[-1], color=color3)
ax[2].loglog(radius[-1], pebble_density4[-1], color=color4)
ax[2].set_xlabel('Radius (AU)', fontsize=20)
ax[2].set_ylabel('Surface Density ($g/cm^2$)', fontsize=20)
ax[2].set_title('Pebble Surface Density Evolution', fontsize=25)
ax[2].set_ylim(1e-4, 1e2)

'''
ax[1,0].loglog(radius[-1], frag_limit1[-1], color=color1, label='Turbulent Limit')
ax[1,0].loglog(radius[-1], frag_limit2[-1], color=color2)
ax[1,0].loglog(radius[-1], frag_limit3[-1], color=color3)
ax[1,0].loglog(radius[-1], frag_limit4[-1], color=color4)

ax[1,0].loglog(radius[-1], drift_limit1[-1], color=color1, linestyle='--', label='Drift Limit')
ax[1,0].loglog(radius[-1], drift_limit2[-1], color=color2, linestyle='--')
ax[1,0].loglog(radius[-1], drift_limit3[-1], color=color3, linestyle='--')
ax[1,0].loglog(radius[-1], drift_limit4[-1], color=color4, linestyle='--')

ax[1,0].loglog(radius[-1], drifttime_limit1[-1], color=color1, linestyle='--', label='Drift Timescale Limit')
ax[1,0].loglog(radius[-1], drifttime_limit2[-1], color=color2, linestyle='--')
ax[1,0].loglog(radius[-1], drifttime_limit3[-1], color=color3, linestyle='--')
ax[1,0].loglog(radius[-1], drifttime_limit4[-1], color=color4, linestyle='--')
ax[1,0].set_ylim(1e-5, 1e4)

ax[1,0].set_xlabel('Radius (AU)', fontsize=20)
ax[1,0].set_ylabel('Pebble Size (cm)', fontsize=20)
ax[1,0].set_title('Growth Limits', fontsize=25)


# plotting intersection points of fragmentation and drift limits to show where the dominant growth barrier changes
diff1 = np.abs(np.array(frag_limit1[-1][:400]) - np.array(drift_limit1[-1][:400]))
idx1 = np.argmin(diff1)
ax[1,0].plot(radius[-1][idx1], drift_limit1[-1][idx1], color=color1, markersize=7, marker='o', label='Dominant Limit Switch')

diff2 = np.abs(np.array(frag_limit2[-1][:500]) - np.array(drift_limit2[-1][:500]))
idx2 = np.argmin(diff2)
ax[1,0].plot(radius[-1][idx2], drift_limit2[-1][idx2], color=color2, markersize=7, marker='o')

diff3 = np.abs(np.array(frag_limit3[-1][:600]) - np.array(drift_limit3[-1][:600]))
idx3 = np.argmin(diff3)
ax[1,0].plot(radius[-1][idx3], drift_limit3[-1][idx3], color=color3, markersize=7, marker='o')

diff4 = np.abs(np.array(frag_limit4[-1]) - np.array(drift_limit4[-1]))
idx4 = np.argmin(diff4)
ax[1,0].plot(radius[-1][idx4], drift_limit4[-1][idx4], color=color4, markersize=7, marker='o')
'''

plt.figtext(0.5, 0.01, f"Mdot={Mdot:.3e}Msun/yr, alpha={alpha:.0e}, Mtot={Mtot:.3e}Msun, Rd={Rd}AU, Mp=0.2994Mjup", ha="center", fontsize=14)

# create legend for fragmentation and drift limits
for row in range(len(ax)):
        ax[row].legend(fontsize=14)
        ax[row].grid(True)
        ax[row].tick_params(axis='both', which='major', labelsize=18)

'''
legend = ax[1,0].legend(loc='upper right', fontsize=14)
handles = legend.legend_handles
symbols = ['-', '--', ':', 'o']
colors = ['black', 'black', 'black', 'black']

for i, handle in enumerate(handles):
    handle.set_color(colors[i])
    if i==3:
        handle.set_marker(symbols[i])
    else:
        handle.set_linestyle(symbols[i])
'''

plt.tight_layout(pad=3.5)


fig.savefig(f"Winter_2026/Figs/Stammler2023rep/frag_vel_nodrift_complete_compare.png")