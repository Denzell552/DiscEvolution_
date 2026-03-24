import json
import numpy as np
import matplotlib.pyplot as plt

fp1 = 'Winter_2026/Data/Weber2018rep/rep_size=0.01_q=-0.5_p=-0.5_Mp=2Mj_alpha=3.0e-03_Mdot=1.0e-07.json'
fp2 = 'Winter_2026/Data/Weber2018rep/rep_size=0.03_q=-0.5_p=-0.5_Mp=2Mj_alpha=3.0e-03_Mdot=1.0e-07.json'
fp3 = 'Winter_2026/Data/Weber2018rep/rep_size=0.10_q=-0.5_p=-0.5_Mp=2Mj_alpha=3.0e-03_Mdot=1.0e-07.json'
fp4 = 'Winter_2026/Data/Weber2018rep/rep_size=0.30_q=-0.5_p=-0.5_Mp=2Mj_alpha=3.0e-03_Mdot=1.0e-07.json'
fp5 = 'Winter_2026/Data/Weber2018rep/rep_size=1.00_q=-0.5_p=-0.5_Mp=2Mj_alpha=3.0e-03_Mdot=1.0e-07.json'

#fp6 = 'Winter_2026/Data/Weber2018rep/rep_nogap_size=0.01_q=-0.5_p=-0.5_Mp=2Mj_alpha=3.0e-03_Mdot=1.0e-07.json'


# extracting densities from json files

# densities with gap
with open (fp1, 'r') as f1:
    data1 = json.load(f1)
gas_density = np.array(data1['Sigma_G'][-1])
pebble_density1 = np.array(data1['Sigma_dust'][-1])
r = np.array(data1['R/Rp'][-1])
orbits = np.array(data1['orbits'])
dust_flux1 = np.array(data1['dust_flux_fraction'])
gas_density_0 = np.array(data1['Sigma_G'][0])

with open (fp2, 'r') as f2:
    data2 = json.load(f2)
pebble_density2 = np.array(data2['Sigma_dust'][-1])
dust_flux2 = np.array(data2['dust_flux_fraction'])

with open (fp3, 'r') as f3:
    data3 = json.load(f3)
pebble_density3 = np.array(data3['Sigma_dust'][-1])
dust_flux3 = np.array(data3['dust_flux_fraction'])

with open (fp4, 'r') as f4:
    data4 = json.load(f4)
pebble_density4 = np.array(data4['Sigma_dust'][-1])
dust_flux4 = np.array(data4['dust_flux_fraction'])

with open (fp5, 'r') as f5:
    data5 = json.load(f5)
pebble_density5 = np.array(data5['Sigma_dust'][-1])
dust_flux5 = np.array(data5['dust_flux_fraction'])

'''
# densityies with no gap
with open (fp6, 'r') as f6:
    data6 = json.load(f6)
gas_density_0 = np.array(data6['Sigma_G'][0])
'''

# plotting
fig, ax = plt.subplots(2, 1, figsize=(10,16))

# for grid slices
x1 = 26
x2 = 116

c = iter(plt.cm.viridis(np.linspace(0.4, 1, 6)))

ax[0].semilogy(r[x1:x2], gas_density[x1:x2] / gas_density_0[x1:x2], color='Purple', label='Gas')
ax[0].semilogy(r[x1:x2], pebble_density1[x1:x2] / gas_density_0[x1:x2], color=next(c), label='0.01cm')
ax[0].semilogy(r[x1:x2], pebble_density2[x1:x2] / gas_density_0[x1:x2], color=next(c), label='0.03cm')
ax[0].semilogy(r[x1:x2], pebble_density3[x1:x2] / gas_density_0[x1:x2], color=next(c), label='0.1cm')
ax[0].semilogy(r[x1:x2], pebble_density4[x1:x2] / gas_density_0[x1:x2], color=next(c), label='0.3cm')
ax[0].semilogy(r[x1:x2], pebble_density5[x1:x2] / gas_density_0[x1:x2], color=next(c), label='1.0cm')
ax[0].set_xlabel('r', fontsize=20)
ax[0].set_ylabel('$\\Sigma / \\Sigma_{g,0}$', fontsize=23)
ax[0].set_title('Mp = 2Mj (q ≈ 5e-4)', fontsize=25)
ax[0].grid(True)
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax[0].legend(fontsize=16)

color = iter(plt.cm.gist_earth(np.linspace(0, 0.9, 5)))

ax[1].plot(orbits[3:], dust_flux1[3:], marker='o', color=next(color), label='0.01cm')
ax[1].plot(orbits[3:], dust_flux2[3:], marker='o', color=next(color), label='0.03cm')
ax[1].plot(orbits[3:], dust_flux3[3:], marker='o', color=next(color), label='0.1cm')
ax[1].plot(orbits[3:], dust_flux4[3:], marker='o', color=next(color), label='0.3cm')
ax[1].plot(orbits[3:], dust_flux5[3:], marker='o', color=next(color), label='1.0cm')
ax[1].set_xlabel('Number of Orbits', fontsize=20)
ax[1].set_ylabel('$\\phi_{inner} / \\phi_{outer}$', fontsize=23)
#ax_flux.set_title('Dust Flux Fraction for Different Grain Sizes at \n ~20000 orbits (~0.24Myrs)for a Planetary Gap with Mp = 0.5Mj (q~5e-4)', fontsize=16)
ax[1].grid(True)
ax[1].tick_params(axis='both', which='major', labelsize=16)
ax[1].legend(fontsize=16)
#ax[1].set_ylim(0, 1)
ax[1].set_xlim(10000,21000)

plt.tight_layout(pad=3.5)

fig.savefig(f"Winter_2026/Figs/Weber2018rep/weber2018rep_q=-0.5_p=-0.5_Mp=2Mj_alpha=3.0e-03_Mdot=1.0e-07.png")
