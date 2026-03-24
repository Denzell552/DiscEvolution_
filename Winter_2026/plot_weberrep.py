import json
import numpy as np
import matplotlib.pyplot as plt

fp1 = 'Winter_2026/Data/rep_size=1.00_q=-0.5_p=-0.5_Mp=2Mj_alpha=3.0e-03_Mdot=1.0e-07.json'
fp2 = 'Winter_2026/Data/rep_size=1.00_q=-0.5_p=-0.5_Mp=0.5Mj_alpha=3.0e-03_Mdot=1.0e-07.json'


with open (fp1, 'r') as f:
    data1 = json.load(f)
dust1 = data1['Sigma_dust']
flux1 = data1['dust_flux']
radius1 = data1['R']
orbits1 = data1['orbits']

with open (fp2, 'r') as f:
    data2 = json.load(f)
dust2 = data2['Sigma_dust']
flux2 = data2['dust_flux']
radius2 = data2['R']
orbits2 = data2['orbits']

fig, ax = plt.subplots(2,2,figsize=(20,14))

color1 = iter(plt.cm.Greens(np.linspace(0.4, 1, 9)))
color2 = iter(plt.cm.Oranges(np.linspace(0.4, 1, 9)))

for t in range(len(dust1)):
    ax[0,0].loglog(radius1[t], dust1[t], color=next(color1), label=f'{orbits1[t]:.1f}')

for t in range(len(dust2)):
    ax[0,1].loglog(radius2[t], dust2[t], label=f'{orbits2[t]:.1f}', color=next(color2))

ax[0,0].set_xlabel('Radius (AU)', fontsize=14)
ax[0,0].set_ylabel('Dust Surface Density ($g/cm^2$)', fontsize=14)
ax[0,0].set_title('Dust Surface Density Evolution for \n a Planetary Gap with Mp = 2Mj (q~5e-4)', fontsize=16)
ax[0,0].grid(True)
ax[0,0].tick_params(axis='both', which='major', labelsize=13)
ax[0,0].legend(fontsize=12)

ax[0,1].set_xlabel('Radius (AU)', fontsize=14)
ax[0,1].set_ylabel('Dust Surface Density ($g/cm^2$)', fontsize=14)
ax[0,1].set_title('Dust Surface Density Evolution for \n a Planetary Gap with Mp = 0.5Mj (q~5e-4)', fontsize=16)
ax[0,1].grid(True)
ax[0,1].tick_params(axis='both', which='major', labelsize=13)
ax[0,1].legend(fontsize=12)

ax[1,0].plot(orbits1, flux1, marker='o')
ax[1,0].set_xlabel('Number of Orbits', fontsize=14)
ax[1,0].set_ylabel('$\\phi_{inner} / \\phi_{outer}$', fontsize=14)
ax[1,0].set_title('Dust Flux Fraction for Different Grain Sizes at \n ~20000 orbits (~0.24Myrs)for a Planetary Gap with Mp = 2Mj (q~5e-4)', fontsize=16)
ax[1,0].grid(True)
ax[1,0].tick_params(axis='both', which='major', labelsize=13)
ax[1,0].legend(fontsize=12)

ax[1,1].plot(orbits2, flux2, marker='o')
ax[1,1].set_xlabel('Number of Orbits', fontsize=14)
ax[1,1].set_ylabel('$\\phi_{inner} / \\phi_{outer}$', fontsize=14)
ax[1,1].set_title('Dust Flux Fraction for Different Grain Sizes at \n ~20000 orbits (~0.24Myrs)for a Planetary Gap with Mp = 0.5Mj (q~5e-4)', fontsize=16)
ax[1,1].grid(True)
ax[1,1].tick_params(axis='both', which='major', labelsize=13)
ax[1,1].legend(fontsize=12)

plt.tight_layout(pad=3.5)
plt.savefig('Winter_2026/Figs/rep_compare.png')




