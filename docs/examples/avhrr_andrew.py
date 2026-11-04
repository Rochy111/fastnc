"""Hurricane Andrew over the Bahamas on 23 August 1992, the day before it
struck south Florida: NOAA-12 AVHRR channel 4 (10.8 um) from a 1 km HRPT
pass, read with fastnc's netCDF4-style Dataset. The channel comes back in
degrees Celsius, masked and scaled, and every pixel has its own latitude
and longitude."""
import os

import fastnc
import fastplot.colors as mcolors
import fastplot.crs as ccrs
import fastplot.pyplot as plt

PATH = 'NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc'     # day 236 of 1992, 12:31-12:43 UTC

with fastnc.Dataset(PATH) as ds:
    t = ds['avhrr_ch4'][:]                               # degrees C, scale_factor applied
    lat, lon = ds['lat'][:], ds['lon'][:]

stops = [(-90, '#ffffff'), (-82, '#fdae6b'), (-75, '#e7298a'), (-67, '#6a51a3'),
         (-58, '#2171b5'), (-48, '#9ecae1'), (-40, '#f2f2f2'), (27, '#262626'), (35, '#0b0b0b')]
cmap = mcolors.LinearSegmentedColormap.from_list(
    'ir', [((v + 90) / 125, c) for v, c in stops])

fig = plt.figure(figsize=(10, 6.4), dpi=130)
ax = fig.add_subplot(projection=ccrs.Mercator(central_longitude=-75))
ax.set_extent([-83, -68, 21.5, 30], crs=ccrs.PlateCarree())
mesh = ax.pcolormesh(lon, lat, t, cmap=cmap, vmin=-90, vmax=35, transform=ccrs.PlateCarree())
ax.coastlines(linewidth=0.6, color='#f1c40f')
grid = ax.gridlines(draw_labels=True, linewidth=0.25, color='#777777')
grid.top_labels = grid.right_labels = False
fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.03, label='Brightness temperature (°C)')
ax.set_title('Hurricane Andrew, NOAA-12 AVHRR channel 4', loc='left', fontsize=11)
ax.set_title('12:31 UTC 23 Aug 1992', loc='right', fontsize=9, color='#555555')

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images',
                         'avhrr_andrew.png'), bbox_inches='tight')
