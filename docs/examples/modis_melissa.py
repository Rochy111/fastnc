"""Hurricane Melissa a few hours before landfall in Jamaica, in MODIS band 31
(11 um) from Terra's 1 km L1B granule, read with fastnc's h5py-style File.
The scaled integers become radiances with the file's radiance_scales and
radiance_offsets, then brightness temperatures by inverting Planck's law.
Latitude and longitude are stored every 5th pixel; they are interpolated
linearly to every pixel, scan by scan."""
import os

import numpy as np

import fastnc
import fastplot.crs as ccrs
import fastplot.pyplot as plt

PATH = 'MOD021KM.A2025301.1445.007.2025301204859.nc'
SWATH = 'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/'

with fastnc.File(PATH) as f:
    ev = f[SWATH + 'Data Fields/EV_1KM_Emissive']
    band = ev.attrs['band_names'].decode().split(',').index('31')
    si = ev[band]                                           # (2040, 1354) uint16
    scale = ev.attrs['radiance_scales'][band]
    offset = ev.attrs['radiance_offsets'][band]
    lat5 = f[SWATH + 'Geolocation Fields/Latitude'][()]     # (408, 271)
    lon5 = f[SWATH + 'Geolocation Fields/Longitude'][()]

radiance = np.ma.masked_greater(si, 32767) * scale - scale * offset   # W m-2 um-1 sr-1
wl = 11.03                                                              # um
bt = 1.4387752e4 / (wl * np.ma.log(1 + 1.191042e8 / (wl ** 5 * radiance))) - 273.15


def to_1km(a):
    """5 km geolocation (rows 2 and 7 of each 10-row scan, every 5th column
    from 2) to every 1 km pixel, extrapolating to the scan's edges."""
    g = a.reshape(-1, 2, a.shape[1])                     # scans, 2 rows, columns
    t = (np.arange(10) - 2) / 5.0
    rows = (g[:, :1] + (g[:, 1:] - g[:, :1]) * t[None, :, None]).reshape(-1, a.shape[1])
    x = np.clip((np.arange(1354) - 2) / 5.0, 0, a.shape[1] - 1)
    i = np.minimum(x.astype(int), a.shape[1] - 2)
    w = x - i
    return rows[:, i] * (1 - w) + rows[:, i + 1] * w


lat, lon = to_1km(lat5), to_1km(lon5)

fig = plt.figure(figsize=(9, 7.2), dpi=135)
ax = fig.add_subplot(projection=ccrs.Mercator(central_longitude=-80))
ax.set_extent([-93, -67, 13, 34], crs=ccrs.PlateCarree())
mesh = ax.pcolormesh(lon, lat, bt, cmap='Greys', vmin=-90, vmax=35,
                     transform=ccrs.PlateCarree())
ax.coastlines(linewidth=0.6, color='#c0392b')
grid = ax.gridlines(draw_labels=True, linewidth=0.25, color='#999999')
grid.top_labels = grid.right_labels = False
fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.03, label='Brightness temperature (°C)')
ax.set_title('Hurricane Melissa, Terra MODIS band 31', loc='left', fontsize=11)
ax.set_title('14:45 UTC 28 Oct 2025', loc='right', fontsize=9, color='#555555')

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images',
                         'modis_melissa.png'), bbox_inches='tight')
