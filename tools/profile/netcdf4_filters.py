from common import DATA, fast_thread, path, real_files
import os, sys, time
if len(sys.argv)>1: os.environ['HDF5_PLUGIN_PATH'] = sys.argv[1]
import netCDF4
print('HDF5_PLUGIN_PATH =', os.environ.get('HDF5_PLUGIN_PATH'))
p = path('wrfout_d01_2025-10-26_06_20_00.nc')
f = netCDF4.Dataset(p); v = f['T']
for m in ['filters']:
    t=time.perf_counter(); r=v.filters(); print(m, (time.perf_counter()-t)*1e3, 'ms', r)
for name in ['has_zstd_filter','has_bzip2_filter','has_blosc_filter','has_szip_filter']:
    fn = getattr(f, name, None)
    if fn: t=time.perf_counter(); r=fn(); print(name, r, '%.2f ms'%((time.perf_counter()-t)*1e3))
