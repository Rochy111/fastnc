"""GOES-19's full disk in the clean longwave window (ABI band 13, 10.3 um),
read with fastnc's netCDF4-style Dataset: the 12-bit radiances come back
masked and scaled, as netCDF4-python gives them, and the brightness
temperature follows from the Planck coefficients stored in the file."""
import os

import numpy as np

import fastnc
import fastplot.colors as mcolors
import fastplot.pyplot as plt

PATH = 'OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc'

with fastnc.Dataset(PATH) as ds:
    rad = ds['Rad'][:]                              # 5424 x 5424, masked off the disk
    fk1, fk2, bc1, bc2 = (float(ds[name][...])
                          for name in ('planck_fk1', 'planck_fk2', 'planck_bc1', 'planck_bc2'))
    start = ds.time_coverage_start

rad = rad.reshape(1808, 3, 1808, 3).mean(axis=(1, 3))     # 2 km pixels to 6 km
bt = (fk2 / np.ma.log(fk1 / rad + 1) - bc1) / bc2           # brightness temperature, K

# Grey for the surface and most cloud, colour for tops colder than about -40 C.
lo, hi = 180.0, 310.0
stops = [(180, '#ffffff'), (189, '#fdae6b'), (197, '#e7298a'), (205, '#6a51a3'),
         (215, '#2171b5'), (225, '#9ecae1'), (233, '#f2f2f2'), (300, '#262626'),
         (310, '#0b0b0b')]
cmap = mcolors.LinearSegmentedColormap.from_list(
    'ir', [((t - lo) / (hi - lo), c) for t, c in stops], bad='#000000')

fig = plt.figure(figsize=(8, 8), dpi=125, facecolor='black')
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(bt, cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
ax.set_axis_off()
fig.text(0.02, 0.975, 'GOES-19 ABI band 13, 10.3 µm', color='#e6e6e6', fontsize=11, va='top')
fig.text(0.02, 0.948, '%s %s UTC' % (start[:10], start[11:16]), color='#a6a6a6', fontsize=9,
         va='top')
fig.text(0.98, 0.025, 'read with fastnc, drawn with fastplot', color='#808080', fontsize=8,
         ha='right')

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images',
                         'goes_full_disk.png'), facecolor='black')
