"""Everyday reads through fastnc's public API (raw values), against h5py
and netCDF4-python: open + one variable, a window, one level."""
import sys, time
import numpy as np, h5py, netCDF4
from common import fast_thread, path
sys.path.insert(0, path('.'))
import fastnc

fast_thread()
cases = [('wrfout_d01_2025-10-26_06_20_00.nc', 'T2', ()), ('wrfout_d01_2025-10-26_06_20_00.nc', 'T', ()),
         ('wrfout_d01_2025-10-26_06_20_00.nc', 'T', (0, 0)),
         ('VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I05', ()),
         ('VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I05', np.s_[3000:3256, 3000:3256]),
         ('MOD02QKM.A2025301.1445.007.2025301204859.nc', 'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_250_RefSB', (0,)),
         ('MOD02QKM.A2025301.1445.007.2025301204859.nc', 'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_250_RefSB', np.s_[0, 4000:4256, 2000:2256]),
         ('W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20260922172210_G_O_20260922163559_20260922163659_C_N_T__.nc', 'data/measurement_data/vii_10690', ()),
         ('20260904I1_ku_outer_lfm_VV_L2_pass13_combined.nc', 'Ze', ())]


def best(fn, n=7):
    r = []
    for _ in range(n):
        t = time.perf_counter(); fn(); r.append(time.perf_counter() - t)
    return min(r) * 1e3


for f, v, sl in cases:
    p = path(f)
    def ours():
        with fastnc.Dataset(p) as d:
            x = d[v]; x.set_auto_maskandscale(False); x[sl]
    def ours_h5():
        with fastnc.File(p) as d:
            d[v][sl]
    def h5():
        with h5py.File(p, 'r') as d:
            d[v][sl]
    def nc():
        with netCDF4.Dataset(p) as d:
            x = d[v]; x.set_auto_maskandscale(False); x[sl]
    ours(); ours_h5()
    a, b, c, e = best(ours), best(ours_h5), best(h5), best(nc)
    print('%-18s %-14s %-34s fastnc.Dataset %7.2f ms  fastnc.File %7.2f ms | h5py %7.2f  netCDF4 %7.2f  (x%.1f / x%.1f)' % (
        f[:18], v.split('/')[-1][:14], str(sl)[:34], a, b, c, e, c / b, e / a))
