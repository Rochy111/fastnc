"""Milliseconds to read every variable of a file, raw (mask and scale off):
netCDF4-python, h5py and fastnc's Dataset.load(), on every logical CPU.
The numbers are read from docs/benchmarks.json, written by tools/bench.py
(files in the OS cache, best of 5 runs, the readers alternated run by run).
h5py does not read netCDF-3, so WRF d02 has no h5py bar. The chart is
drawn with fastplot."""
import json
import os

import numpy as np

import fastplot.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, '..', 'benchmarks.json')) as f:
    rows = json.load(f)['rows']

labels = {'WRF d01 netCDF4': 'WRF d01\nnetCDF-4', 'WRF d02 netCDF3': 'WRF d02\nnetCDF-3',
          'MODIS 1 km L1B': 'MODIS\n1 km L1B', 'MODIS 250 m L1B': 'MODIS\n250 m L1B',
          'VIIRS I-band': 'VIIRS\nI-band', 'Metop-SG VII': 'Metop-SG\nVII',
          'AVHRR HRPT': 'AVHRR\nHRPT', 'RCM SAR winds': 'RCM\nSAR winds',
          'Ku scatterometer': 'Ku-band\nscatterometer', 'GOES-19 ABI RadF C02': 'GOES-19\nABI C02',
          'GOES-19 ABI RadF C13': 'GOES-19\nABI C13'}
readers = [('netCDF4', 'netCDF4-python', '#cacfd2'), ('h5py', 'h5py', '#909497'),
           ('fastnc', 'fastnc', '#1b4f72')]

fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150)
x = np.arange(len(rows))
w = 0.27
for k, (key, name, colour) in enumerate(readers):
    ms = np.array([r['all_raw'].get(key, np.nan) for r in rows], float)
    pos = x + (k - 1) * w
    ok = np.isfinite(ms)
    ax.bar(pos[ok], ms[ok], width=w * 0.92, color=colour, label=name)
    for px, v in zip(pos[ok], ms[ok]):
        ax.text(px, v * 1.1, format(v, '.0f') if v >= 10 else format(v, '.1f'), ha='center',
                fontsize=6.5, color='#333333')

ax.set_xlim(-0.6, len(rows) - 0.4)
ax.set_yscale('log')
ax.set_ylim(1, 1e4)
ax.set_xticks(x, [labels.get(r['file'], r['file']) for r in rows], fontsize=8)
ax.set_ylabel('ms per file (log scale)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linewidth=0.3)
ax.set_axisbelow(True)
ax.legend(loc='upper right', ncol=3, frameon=False, fontsize=8.5)
ax.set_title('Reading every variable of a file, raw', loc='left', fontsize=11)
fig.tight_layout()

fig.savefig(os.path.join(HERE, '..', 'images', 'speed_chart.png'))
