from common import DATA, fast_thread, path, real_files
import os, time, h5py, netCDF4, numpy as np, ctypes
fast_thread()
cases = [('wrfout_d01_2025-10-26_06_20_00.nc','T2',None),('wrfout_d01_2025-10-26_06_20_00.nc','T',None),
         ('wrfout_d01_2025-10-26_06_20_00.nc','T',np.s_[0,0]),
         ('VJ102IMG_NRT.A2026265.2006.021.nc','observation_data/I05',None),
         ('VJ102IMG_NRT.A2026265.2006.021.nc','observation_data/I05',np.s_[3000:3256,3000:3256]),
         ('MOD02QKM.A2025301.1445.007.2025301204859.nc','HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_250_RefSB',np.s_[0]),
         ('MOD02QKM.A2025301.1445.007.2025301204859.nc','HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_250_RefSB',np.s_[0,4000:4256,2000:2256]),
         ('W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20260922172210_G_O_20260922163559_20260922163659_C_N_T__.nc','data/measurement_data/vii_10690',None),
         ('wrfout_d02_2024-11-06_01_30_00','T',None),('wrfout_d02_2024-11-06_01_30_00','T2',None)]
def b(fn,n=5):
    r=[]
    for _ in range(n): t=time.perf_counter(); fn(); r.append(time.perf_counter()-t)
    return min(r)*1e3
for f,v,sl in cases:
    sl2 = sl if sl is not None else Ellipsis
    def h():
        with h5py.File(path(f),'r') as x: x[v][sl2]
    def n(ms):
        with netCDF4.Dataset(path(f)) as x:
            x.set_auto_maskandscale(ms)
            g=x
            for part in v.split('/')[:-1]: g=g[part]
            g.variables[v.split('/')[-1]][sl2]
    try: th=b(h)
    except Exception: th=float('nan')
    print(f'{f[:18]:18s} {v.split("/")[-1]:14s} {str(sl):40s} h5py {th:7.1f} ms  nc4 raw {b(lambda:n(False)):7.1f} ms  nc4 m+s {b(lambda:n(True)):7.1f} ms')
# metop-sg: are the row chunks back to back in the file?
with h5py.File(path(cases[7][0]),'r') as x:
    d=x[cases[7][1]]; L=[]; d.id.chunk_iter(L.append)
    offs=np.array([c.byte_offset for c in L]); sz=np.array([c.size for c in L])
    print('metop chunks back-to-back:', bool(np.all(offs[1:]==offs[:-1]+sz[:-1])), 'nchunks', len(L))
    print('attrs', dict(d.attrs))
with h5py.File(path('VJ102IMG_NRT.A2026265.2006.021.nc'),'r') as x:
    print('VIIRS I05 attrs', {k:(v, getattr(v,'dtype',None)) for k,v in x['observation_data/I05'].attrs.items()})
