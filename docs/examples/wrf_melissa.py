"""Hurricane Melissa south of Jamaica in a 13.5 km WRF run: 10 m wind from
a wrfout file, opened through xarray with fastnc as the engine. XLAT and
XLONG come along as coordinates, as they do with the netcdf4 engine."""
import os

import numpy as np
import xarray as xr

import fastplot.crs as ccrs
import fastplot.pyplot as plt

PATH = 'wrfout_d01_2025-10-26_06_20_00.nc'

with xr.open_dataset(PATH, engine='fastnc') as ds:
    sfc = ds[['U10', 'V10']].isel(Time=0).load()
    start = ds.attrs['START_DATE']

u, v = sfc.U10.values, sfc.V10.values
lat, lon = sfc.XLAT.values, sfc.XLONG.values
speed = np.hypot(u, v)
when = sfc.XTIME.dt.strftime('%H:%M UTC %d %b %Y').item()

fig = plt.figure(figsize=(10, 7.6), dpi=150)
ax = fig.add_subplot(projection=ccrs.Mercator(central_longitude=-74.65))
ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
mesh = ax.pcolormesh(lon, lat, speed, cmap='YlOrRd', vmin=0, vmax=32,
                     transform=ccrs.PlateCarree())
k = (slice(3, None, 8), slice(3, None, 8))
ax.quiver(lon[k], lat[k], u[k], v[k], color='#222222', scale=600, width=0.0018,
          transform=ccrs.PlateCarree())
ax.coastlines(linewidth=0.6, color='#1b4f72')
grid = ax.gridlines(draw_labels=True, linewidth=0.25, color='#999999')
grid.top_labels = grid.right_labels = False
fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.03, extend='max', label='10 m wind (m s$^{-1}$)')
ax.set_title('Hurricane Melissa, WRF 13.5 km', loc='left', fontsize=11)
ax.set_title('%s (run from %s UTC)' % (when, start[:13].replace('_', ' ')), loc='right',
             fontsize=8.5, color='#555555')

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images',
                         'wrf_melissa.png'), bbox_inches='tight')
