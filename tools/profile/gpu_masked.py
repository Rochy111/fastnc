"""Masked/scaled reads to the GPU: fastnc's GPU path (CPU decode, packed
integers over PCIe, mask/scale on the card) against fastnc on the CPU +
upload of the float result and mask, and netCDF4-python + upload.
Warm (best of 5), synchronised; values checked equal to the CPU path."""
import sys, time
import numpy as np, cupy as cp, netCDF4
from common import fast_thread, path
sys.path.insert(0, path('.'))
import fastnc

fast_thread()
fastnc.warm()
cases = [('VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I05'),
         ('VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I01'),
         ('MOD021KM.A2025301.1445.007.2025301204859.nc', 'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_RefSB'),
         ('extra/OR_ABI-L1b-RadF-M6C02_G19_s20263021200207_e20263021209516_c20263021209551.nc', 'Rad'),
         ('extra/OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc', 'Rad'),
         ('W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20260922172210_G_O_20260922163559_20260922163659_C_N_T__.nc', 'data/measurement_data/vii_10690')]


def sync():
    cp.cuda.Device().synchronize()


def up(res):
    if np.ma.isMaskedArray(res):
        return cp.asarray(res.data), cp.asarray(np.ma.getmaskarray(res))
    return cp.asarray(res)


def best(fn, n=5):
    r = []
    for _ in range(n):
        sync(); t = time.perf_counter(); out = fn(); sync(); r.append(time.perf_counter() - t)
    return min(r) * 1e3, out


for f, v in cases:
    p = path(f)
    d = fastnc.Dataset(p); var = d[v]
    nd = netCDF4.Dataset(p); nv = nd[v]
    var.read(Ellipsis, device='gpu')
    tg, g = best(lambda: var.read(Ellipsis, device='gpu'))
    tc, _ = best(lambda: up(var[...]))
    tn, _ = best(lambda: up(nv[...]), 3)
    cpu = var[...]
    gh = g.get() if hasattr(g, 'get') and not isinstance(g, cp.ndarray) else cp.asnumpy(g)
    same = (np.ma.getdata(gh).tobytes() == np.ma.getdata(cpu).tobytes()
            and np.array_equal(np.ma.getmaskarray(gh), np.ma.getmaskarray(cpu)))
    print('%-14s %-16s %s  gpu path %6.1f ms | fastnc cpu+upload %6.1f | netCDF4+upload %7.1f   equal=%s' % (
        f[:14], v.split('/')[-1][:16], str(np.ma.getdata(cpu).dtype), tg, tc, tn, same))
