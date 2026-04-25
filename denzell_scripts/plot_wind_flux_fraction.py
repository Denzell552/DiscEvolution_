import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from DiscEvolution.constants import *

file_path1 = "denzell_scripts/Data_Updated/wind_disks/wind_bump_alpha=1.0e-03_psi=1.0.json"
file_path2 = "denzell_scripts/Data_Updated/wind_disks/wind_bump_alpha=1.0e-03_psi=10.0.json"
file_path3 = "denzell_scripts/Data_Updated/wind_disks/wind_bump_alpha=1.0e-03_psi=100.0.json"
file_path4 = "denzell_scripts/Data_Updated/wind_disks/wind_alpha=1.0e-03_psi=1.0.json"
file_path5 = "denzell_scripts/Data_Updated/wind_disks/wind_alpha=1.0e-03_psi=10.0.json"
file_path6 = "denzell_scripts/Data_Updated/wind_disks/wind_alpha=1.0e-03_psi=100.0.json"

with open (file_path1, 'r') as fp1:
    data1 = json.load(fp1)

radius = np.array(data1['R'])
dust_flux1 = np.array(data1['dust_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_flux1 = np.array(data1['pebble_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_density1 = data1['Sigma_pebbles']

time = data1['time']


with open (file_path2, 'r') as fp2:
    data2 = json.load(fp2)
dust_flux2 = np.array(data2['dust_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_flux2 = np.array(data2['pebble_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_density2 = data2['Sigma_pebbles']
   
with open (file_path3, 'r') as fp3:
    data3 = json.load(fp3)
dust_flux3 = np.array(data3['dust_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_flux3 = np.array(data3['pebble_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_density3 = data3['Sigma_pebbles']

with open (file_path4, 'r') as fp4:
    data4 = json.load(fp4)
dust_flux1_nobump = np.array(data4['dust_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_flux1_nobump = np.array(data4['pebble_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_density1_nobump = data4['Sigma_pebbles']

with open (file_path5, 'r') as fp5:
    data5 = json.load(fp5)
dust_flux2_nobump = np.array(data5['dust_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_flux2_nobump = np.array(data5['pebble_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_density2_nobump = data5['Sigma_pebbles']

with open (file_path6, 'r') as fp6:
    data6 = json.load(fp6)
dust_flux3_nobump = np.array(data6['dust_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_flux3_nobump = np.array(data6['pebble_flux']) * 3.15e7 / Mearth # to Mearth/year
pebble_density3_nobump = data6['Sigma_pebbles']

# calculating the fraction of flux through the gap at each time step for both alpha values
dust_flux_frac1 = []
pebble_flux_frac1 = []

dust_flux_frac2 = []
pebble_flux_frac2 = []

dust_flux_frac3 = []
pebble_flux_frac3 = []

dust_flux_frac1_nobump = []
pebble_flux_frac1_nobump = []

dust_flux_frac2_nobump = []
pebble_flux_frac2_nobump = []

dust_flux_frac3_nobump = []
pebble_flux_frac3_nobump = []

time_clipped = []

# plotting comparison figure
fig, ax = plt.subplots(1, 2, figsize=(25, 8))

inner_edge_idx = np.argmin(np.abs(radius - 0.5))
outer_edge_idx = np.argmin(np.abs(radius - 20))


for t in range(len(time)):
    if time[t] % 0.25 == 0:

        # flux comparisons
        dust_flux_frac1.append(np.abs(dust_flux1[t][inner_edge_idx] / dust_flux1[t][np.argmin(np.abs(radius - (radius[outer_edge_idx])))]))
        dust_flux_frac2.append(np.abs(dust_flux2[t][inner_edge_idx] / dust_flux2[t][np.argmin(np.abs(radius - (radius[outer_edge_idx])))]))
        dust_flux_frac3.append(np.abs(dust_flux3[t][inner_edge_idx] / dust_flux3[t][np.argmin(np.abs(radius - (radius[outer_edge_idx])))]))

        pebble_flux_frac1.append(np.abs(pebble_flux1[t][inner_edge_idx] / pebble_flux1[t][np.argmin(np.abs(radius - (radius[outer_edge_idx])))]))
        pebble_flux_frac2.append(np.abs(pebble_flux2[t][inner_edge_idx] / pebble_flux2[t][np.argmin(np.abs(radius - (radius[outer_edge_idx])))]))
        pebble_flux_frac3.append(np.abs(pebble_flux3[t][inner_edge_idx] / pebble_flux3[t][np.argmin(np.abs(radius - (radius[outer_edge_idx])))]))

        time_clipped.append(time[t])

# plotting dust flux fraction comparison
ax[0].semilogy(time_clipped, dust_flux_frac1, color='blue', label='psi=1')
ax[0].semilogy(time_clipped, dust_flux_frac2, color='green', label='psi=10')
ax[0].semilogy(time_clipped, dust_flux_frac3, color='red', label='psi=100')
ax[0].set_xlabel('Time (Myrs)', fontsize=27)
ax[0].set_title('Dust Flux Fraction', fontsize=25)
ax[0].set_ylabel('$\\phi_{inner} / \\phi_{outer}$', fontsize=25)

# plotting pebble flux fraction comparison
ax[1].semilogy(time_clipped, pebble_flux_frac1, color='blue')
ax[1].semilogy(time_clipped, pebble_flux_frac2, color='green')
ax[1].semilogy(time_clipped, pebble_flux_frac3, color='red')
ax[1].set_xlabel('Time (Myrs)', fontsize=27)
ax[1].set_title('Pebble Flux Fraction', fontsize=25)

plt.tight_layout(pad=3.5)

for column in range(len(ax)):
    ax[column].legend(fontsize=17)
    ax[column].grid(True)
    ax[column].tick_params(axis='both', which='major', labelsize=23)
    plt.setp(ax[column].spines.values(), linewidth=2)



#plt.savefig('denzell_scripts/Figs_Updated/wind_disks/flux_compare.png')

fig2 = plt.figure(figsize=(25, 16))
gs = fig2.add_gridspec(2, 4, wspace=0.2, hspace=0.2)

ax00 = fig2.add_subplot(gs[0, 0:2])
ax01 = fig2.add_subplot(gs[0, 2:4])
ax10 = fig2.add_subplot(gs[1, 1:3])

color1 = iter(plt.cm.Blues(np.linspace(0.4, 1, 9)))
color2 = iter(plt.cm.Oranges(np.linspace(0.4, 1, 9)))
color3 = iter(plt.cm.Greens(np.linspace(0.4, 1, 9)))


for t in range(len(time)):
    if time[t] % 0.25 == 0:
        # density comparisons
        ax00.loglog(radius, pebble_density1[t], color=next(color1), label=f'{time[t]:.2f}Myr')
        ax01.loglog(radius, pebble_density2[t], color=next(color2))
        ax10.loglog(radius, pebble_density3[t], color=next(color3))

#ax00.loglog(radius, pebble_density1[-1], color='blue', label='With Rings')
#ax00.loglog(radius, pebble_density1_nobump[-1], color='black', linestyle='dashed', label='No Rings')
ax00.set_title('$\\psi=1$', fontsize=27)
ax00.set_xlabel('Radius [AU]', fontsize=25)
ax00.set_ylabel('$\\Sigma [g/cm^2]$', fontsize=25)
ax00.set_yscale('symlog', linthresh=1e-5)
ax00.grid(True)
ax00.tick_params(axis='both', which='major', labelsize=23)
plt.setp(ax00.spines.values(), linewidth=2)
ax00.legend(fontsize=17)

#ax01.loglog(radius, pebble_density2[-1], color='blue')
#ax01.loglog(radius, pebble_density2_nobump[-1], color='black', linestyle='dashed')
ax01.set_title('$\\psi=10$', fontsize=27)
ax01.set_xlabel('Radius [AU]', fontsize=25)
ax01.set_yscale('symlog', linthresh=1e-5)
ax01.grid(True)
ax01.tick_params(axis='both', which='major', labelsize=23)
plt.setp(ax01.spines.values(), linewidth=2)

#ax10.loglog(radius, pebble_density3[-1], color='blue')
#ax10.loglog(radius, pebble_density3_nobump[-1], color='black', linestyle='dashed')
ax10.set_title('$\\psi=100$', fontsize=27)
ax10.set_xlabel('Radius [AU]', fontsize=25)
ax10.set_ylabel('$\\Sigma [g/cm^2]$', fontsize=25)
ax10.set_yscale('symlog', linthresh=1e-5)
ax10.grid(True)
ax10.tick_params(axis='both', which='major', labelsize=23)
plt.setp(ax10.spines.values(), linewidth=2)


plt.savefig('denzell_scripts/Figs_Updated/wind_disks/density_compare.png')
