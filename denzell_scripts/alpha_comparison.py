import json
import numpy as np
import matplotlib.pyplot as plt
from DiscEvolution.constants import *

file_path1 = "denzell_scripts/Data_Updated/test_vfrag=1_Rstar=2.5_planetgap_Mp=1_alpha=1.0e-03.json"
file_path2 = "denzell_scripts/Data_Updated/test_planetgap_Mp=1_alpha=1e-04.json"

# getting data from alpha = 1e-3 
with open (file_path1, 'r') as fp1:
    data3 = json.load(fp1)

radius = data3['R']
gas_density3 = data3['Sigma_G']
dust_density3 = data3['Sigma_dust']
dust_flux3 = np.array(data3['dust_flux']) 
pebble_density3 = data3['Sigma_pebbles']
Mdot, alpha, Mtot, Rd = data3['parameters']['Mdot'], data3['parameters']['alpha'], data3['parameters']['Mtot'], data3['parameters']['Rd']
pebble_flux3 = np.array(data3['pebble_flux']) 
time = data3['time']
inner_edge_idx3 = data3['gap_profile']['inner_edge_idx']
outer_edge_idx3 = data3['gap_profile']['outer_edge_idx']

# getting data from alpha = 1e-4
with open (file_path2, 'r') as fp2:
    data4 = json.load(fp2)

gas_density4 = data4['Sigma_G']
dust_density4 = data4['Sigma_dust']
dust_flux4 = np.array(data4['dust_flux'])
pebble_density4 = data4['Sigma_pebbles']
pebble_flux4 = np.array(data4['pebble_flux'])
inner_edge_idx4 = data4['gap_profile']['inner_edge_idx']
outer_edge_idx4 = data4['gap_profile']['outer_edge_idx']

# plotting comparison figures
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
        ax[2,0].loglog(radius[:-1], np.abs(dust_flux3[t]), color=next(color5))
        ax[2,1].loglog(radius[:-1], np.abs(pebble_flux3[t]), color=next(color6))
        ax[3,0].loglog(radius[:-1], dust_velocity3[t], color=next(color7))
        ax[3,1].loglog(radius[:-1], pebble_velocity3[t], color=next(color8))
        ax[4,0].loglog(radius, viscosity3[t], color=next(color9))
        #ax[4,1].semilogx(radius, gradient[t], color=next(color10))
        ax[5,0].loglog(radius, stokes3[t][0], color=next(color11))
        ax[5,1].loglog(radius, stokes3[t][1], color=next(color12))


plt.tight_layout(pad=3.5)

for row in range(len(ax)):
    for column in range(len(ax[row])):
        ax[row][column].legend(fontsize=12)
        ax[row][column].grid(True)
        ax[row][column].tick_params(axis='both', which='major', labelsize=18)
        plt.setp(ax[row][column].spines.values(), linewidth=2)


plt.savefig('denzell_scripts/Figs_Updated/alpha_comparison_vfrag=1_Rstar=2.5.png')

