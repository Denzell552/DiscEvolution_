import json
import numpy as np
import matplotlib.pyplot as plt
from DiscEvolution.constants import *

file_path1 = "denzell_scripts/Data_Updated/flux_vs_alpha/planetgap_Mp=1_alpha=1.0e-03.json"

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
pebble_drift_velocity3 = data3['pebble_drift_velocity']
pressure3 = data3['Pressure']
gradient = data3['pressure_gradient']
stokes3 = data3['stokes_number']
Mdot_evolv3 = data3['Mdot']
viscosity3 = data3['viscosity']
temperature3 = data3['temperature']
inner_edge_idx3 = data3['gap_profile']['inner_edge_idx']
outer_edge_idx3 = data3['gap_profile']['outer_edge_idx']
frag_limit3 = data3['frag_limit']
drift_limit3 = data3['drift_limit']

fig, ax = plt.subplots(1, 1, figsize=(12.5,8))


# growth limit compare 1
ax.loglog(radius, frag_limit3[2], color='blue', label='Fragmentation Limit')
ax.loglog(radius, drift_limit3[2], color='orange', label='Drift Limit')
ax.loglog(radius, frag_limit3[12], color='blue', linestyle='--', alpha=0.8)
ax.loglog(radius, drift_limit3[12], color='orange', linestyle='--', alpha=0.8)
ax.set_title('Growth Limits for $\\alpha=10^{-3}$', fontsize=25)
ax.set_xlabel('Radius [AU]', fontsize=20)
ax.set_ylabel('Size [cm]', fontsize=20)
ax.set_yscale('symlog', linthresh=1e-3)
ax.legend(fontsize=18)
ax.grid(True)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.setp(ax.spines.values(), linewidth=2)



plt.savefig('denzell_scripts/Figs_Updated/flux_vs_alpha/test_planetgap_Mp=1_alpha=1.0e-03.png')

