import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

# import data
kd_03Mj = pd.read_csv('Winter_2026/Data/Kanagawa_depth_0.3Mj.csv', header=None, names=['x','y'])
kd_1Mj = pd.read_csv('Winter_2026/Data/Kanagawa_depth_1Mj.csv', header=None, names=['x','y'])

with open('Winter_2026/Data/Duffell2019_gap_0.3Mj.json', 'r') as f1:
    duffell_03Mj = json.load(f1)

with open('Winter_2026/Data/Duffell2019_gap_1Mj.json', 'r') as f2:
    duffell_1Mj = json.load(f2)

with open('Winter_2026/Data/Gaussian**2_gap_0.3Mj.json', 'r') as f3:
    gaussian2_03Mj = json.load(f3)

with open('Winter_2026/Data/Gaussian**2_gap_1Mj.json', 'r') as f4:
    gaussian2_1Mj = json.load(f4)
with open('Winter_2026/Data/Gaussian**4_gap_0.3Mj.json', 'r') as f5:
    gaussian4_03Mj = json.load(f5)

with open('Winter_2026/Data/Gaussian**4_gap_1Mj.json', 'r') as f6:
    gaussian4_1Mj = json.load(f6)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(kd_03Mj['x'], kd_03Mj['y'], label='Kanagawa (2016) Data', color='black', linewidth=2)
axes[0].plot(duffell_03Mj['R/Rp'][:85], duffell_03Mj['depth'][:85], label='Duffell (2019) Model', color='green', linestyle='dashed')
axes[0].plot(gaussian2_03Mj['R/Rp'][:85], gaussian2_03Mj['depth'][:85], label='Gaussian (**2) Model', color='orange', linestyle='dashed')
axes[0].plot(gaussian4_03Mj['R/Rp'][:85], gaussian4_03Mj['depth'][:85], label='Gaussian (**4) Model', color='red', linestyle='dashed')
axes[0].set_xlabel('R/Rp', fontsize=14)
axes[0].set_ylabel('Gap Depth ($\\Sigma_{gap} / \\Sigma_0$)', fontsize=14)
axes[0].set_title('Gap Depth Comparison for 0.3 Mj Planet', fontsize=16)
axes[0].legend(fontsize=9)
axes[0].grid(True)

axes[1].plot(kd_1Mj['x'], kd_1Mj['y'], label='Kanagawa (2016) Data', color='black', linewidth=2)
axes[1].plot(duffell_1Mj['R/Rp'][:85], duffell_1Mj['depth'][:85], label='Duffell (2019) Model', color='green', linestyle='dashed')
axes[1].plot(gaussian2_1Mj['R/Rp'][:85], gaussian2_1Mj['depth'][:85], label='Gaussian (**2) Model', color='orange', linestyle='dashed')
axes[1].plot(gaussian4_1Mj['R/Rp'][:85], gaussian4_1Mj['depth'][:85], label='Gaussian (**4) Model', color='red', linestyle='dashed')
axes[1].set_xlabel('R/Rp', fontsize=14)
axes[1].set_title('Gap Depth Comparison for 1 Mj Planet', fontsize=16)
axes[1].legend(fontsize=9)
axes[1].grid(True)

plt.figtext(0.5, 0.01, f"Rp=4.5AU, h0=0.035, hp/Rp=0.05, alpha=10$^{{-3}}$", ha="center", fontsize=12)


plt.tight_layout(pad=3.5)

fig.savefig('Winter_2026/Figs/gap_depth_comparison.png')