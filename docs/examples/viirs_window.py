"""The eye of an eastern Pacific hurricane at 375 m: a 1024 x 1024 window of
NOAA-20 VIIRS band I5 (11.45 um), read with fastnc's h5py-style File. The
variable is stored in strips of 32 rows, so the window costs 32 of its 202
chunks. Counts become brightness temperatures through the lookup table
shipped beside them."""
import os

import numpy as np

import fastnc
import fastplot.colors as mcolors
import fastplot.pyplot as plt

PATH = 'VJ102IMG_NRT.A2026265.2006.021.nc'
ROWS, COLS = slice(1408, 2432), slice(2688, 3712)

with fastnc.File(PATH) as f:
    counts = f['observation_data/I05'][ROWS, COLS]              # raw uint16, as h5py gives
    lut = f['observation_data/I05_brightness_temperature_lut'][:]
    when = f.attrs['time_coverage_start'].decode()

bt = np.ma.masked_array(lut[counts], counts > 65527) - 273.15    # flags sit above 65527

stops = [(-90, '#ffffff'), (-82, '#fdae6b'), (-75, '#e7298a'), (-67, '#6a51a3'),
         (-58, '#2171b5'), (-48, '#9ecae1'), (-40, '#f2f2f2'), (27, '#262626'), (35, '#0b0b0b')]
cmap = mcolors.LinearSegmentedColormap.from_list(
    'ir', [((t + 90) / 125, c) for t, c in stops])

km = 0.375 * counts.shape[1]
fig, ax = plt.subplots(figsize=(8, 6.6), dpi=150)
im = ax.imshow(bt, cmap=cmap, vmin=-90, vmax=35, extent=(0, km, km, 0),
               interpolation='nearest')
ax.set_xlabel('km along scan')
ax.set_ylabel('km along track')
fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03, label='Brightness temperature (°C)')
ax.set_title('NOAA-20 VIIRS I5, 375 m', loc='left', fontsize=11)
ax.set_title('%s %s UTC' % (when[:10], when[11:16]), loc='right', fontsize=9, color='#555555')

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images',
                         'viirs_window.png'), bbox_inches='tight')
