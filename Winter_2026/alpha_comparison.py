import json
import numpy as np
import matplotlib.pyplot as plt

file_path1 = "Winter_2026/Data/Gap_Mp=2Mj_alpha=1.0e-03_M=5.0e-01_Rd=5.0e+01.json"
file_path2 = "Winter_2026/Data/Gap_Mp=2Mj_alpha=1.0e-04_M=5.0e-01_Rd=5.0e+01.json"

# getting data from alpha = 1e-3 
# only plotting data from the last time step
with open (file_path1, 'r') as alpha3:
    data3 = json.load(alpha3)

R3 = np.array(data3['R'][-1])
sigmaG3 = np.array(data3['Sigma_G'][-1])
sigmaD3 = np.array(data3['Sigma_dust'][-1])
sigmaP3 = np.array(data3['Sigma_pebbles'][-1])
sizeP3 = np.array(data3['pebble size'][-1])
pressure3 = np.array(data3['pressure'][-1])
gradP3 = np.array(data3['pressure gradient'][-1])

# getting data from alpha = 1e-4
with open (file_path2, 'r') as alpha4:
    data4 = json.load(alpha4)

R4 = np.array(data4['R'][-1])
sigmaG4 = np.array(data4['Sigma_G'][-1])
sigmaD4 = np.array(data4['Sigma_dust'][-1])
sigmaP4 = np.array(data4['Sigma_pebbles'][-1])
sizeP4 = np.array(data4['pebble size'][-1])
pressure4 = np.array(data4['pressure'][-1])
gradP4 = np.array(data4['pressure gradient'][-1])

# plotting comparison figures
fig, axes = plt.subplots(3, 2, figsize=(20,18))
fig.suptitle('Comparison of Disk Properties for α = $10^{-3}$ (blue) and $10^{-4}$ (red) at 1 Myr \n for a Planetary Gap with Mp = 2Mj', fontsize=20)

# Gas density
axes[0][0].loglog(R3, sigmaG3, color='blue')
axes[0][0].loglog(R4, sigmaG4, color='red')
axes[0][0].set_title('Gas Surface Density', fontsize=17)
axes[0][0].set_xlabel('Radius (AU)', fontsize=15)
axes[0][0].set_ylabel('Surface Density $(g/cm^3)$', fontsize=15)
axes[0][0].grid(True)

# Pebble Size
axes[0][1].loglog(R3, sizeP3, color='blue')
axes[0][1].loglog(R4, sizeP4, color='red')
axes[0][1].set_title('Pebble Size', fontsize=17)
axes[0][1].set_xlabel('Radius (AU)', fontsize=15)
axes[0][1].set_ylabel('Size (cm)', fontsize=15)
axes[0][1].grid(True)

# Dust Surface Density
axes[1][0].loglog(R3, sigmaD3, color='blue')
axes[1][0].loglog(R4, sigmaD4, color='red')
axes[1][0].set_title('Dust Surface Density', fontsize=17)
axes[1][0].set_xlabel('Radius (AU)', fontsize=15)
axes[1][0].set_ylabel('Surface Density $(g/cm^3)$', fontsize=15)
axes[1][0].set_ylim(10**-6, 10**3)
axes[1][0].grid(True)

# Pebble Surface Density
axes[1][1].loglog(R3, sigmaP3, color='blue')
axes[1][1].loglog(R4, sigmaP4, color='red')
axes[1][1].set_title('Pebble Surface Density', fontsize=17)
axes[1][1].set_xlabel('Radius (AU)', fontsize=15)
axes[1][1].set_ylabel('Surface Density $(g/cm^3)$', fontsize=15)
axes[1][1].set_ylim(10**-4, 10**3)
axes[1][1].grid(True)

# Pressure gradient
axes[2][0].semilogx(R3, gradP3, color='blue')
axes[2][0].semilogx(R4, gradP4, color='red')
axes[2][0].set_title('Pressure Gradient', fontsize=17)
axes[2][0].set_xlabel('Radius (AU)', fontsize=15)
axes[2][0].set_ylabel('dP/dr', fontsize=15)
axes[2][0].grid(True)

# presssure
axes[2][1].loglog(R3, pressure3, color='blue')
axes[2][1].loglog(R4, pressure4, color='red')
axes[2][1].set_title('Pressure Profile', fontsize=17)
axes[2][1].set_xlabel('Radius (AU)', fontsize=15)
axes[2][1].set_ylabel('Pressure (Pa)', fontsize=15)
axes[2][1].grid(True)

for row in range(len(axes)):
    for column in range(len(axes[row])):

        axes[row][column].tick_params(labelsize=14)
        axes[row][column].minorticks_off()

plt.tight_layout(pad=3.5)

plt.savefig('Winter_2026/Figs/alpha_comparison_Mp=2Mj.png')

