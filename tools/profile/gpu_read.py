"""GPU reads (raw values to a cupy array), honestly timed: first call
(page-locking the host buffer, cupy start-up) and warm; against fastnc on
the CPU followed by cupy.asarray, and h5py followed by cupy.asarray.
Checks the device values equal the CPU's."""
import sys, time
import numpy as np, cupy as cp, h5py
from common import fast_thread, path
sys.path.insert(0, path('.'))
import fastnc

fast_thread()
cases = [('VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I05', ()),
         ('VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I01', ()),
         ('wrfout_d01_2025-10-26_06_20_00.nc', 'T', ()),
         ('MOD021KM.A2025301.1445.007.2025301204859.nc', 'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive', ()),
         ('MOD02QKM.A2025301.1445.007.2025301204859.nc', 'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_250_RefSB', ())]


def timed(fn, n=5):
    cp.cuda.Device().synchronize()
    t = time.perf_counter(); out = fn(); cp.cuda.Device().synchronize(); first = time.perf_counter() - t
    r = []
    for _ in range(n):
        t = time.perf_counter(); out = fn(); cp.cuda.Device().synchronize(); r.append(time.perf_counter() - t)
    return first * 1e3, min(r) * 1e3, out


for f, v, sl in cases:
    p = path(f)
    d = fastnc.Dataset(p); var = d[v]; var.set_auto_maskandscale(False)
    hf = h5py.File(p, 'r')
    g1, g, dev = timed(lambda: var.read(sl, device='gpu'))
    c1, c, _ = timed(lambda: cp.asarray(var.read(sl)))
    h1, h, _ = timed(lambda: cp.asarray(hf[v][sl]), 3)
    ok = np.array_equal(cp.asnumpy(dev), var.read(sl))
    mb = dev.nbytes / 1e6
    print('%-12s %-16s %6.0f MB  gpu path %6.1f ms (first %6.1f) | fastnc cpu+upload %6.1f | h5py+upload %6.1f   equal=%s' % (
        f[:12], v.split('/')[-1][:16], mb, g, g1, c, h, ok))
