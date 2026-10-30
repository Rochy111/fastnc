"""Fused masked reads (fastnc._fused) vs the usual read then mask/scale:
same result (values, mask, fill value), and the times."""
import sys, time
from common import fast_thread, path
sys.path.insert(0, path('.'))
import numpy as np, fastnc
from fastnc import _dataset
fastnc.warm(); fast_thread()


def same(a, b):
    if np.ma.isMaskedArray(a) != np.ma.isMaskedArray(b): return 'type'
    if np.ma.isMaskedArray(a):
        if (np.ma.getmask(a) is np.ma.nomask) != (np.ma.getmask(b) is np.ma.nomask): return 'nomask'
        if np.ma.getmaskarray(a).tobytes() != np.ma.getmaskarray(b).tobytes(): return 'mask'
        if repr(a.fill_value) != repr(b.fill_value) or type(a.fill_value) != type(b.fill_value): return 'fill'
    x, y = np.ma.getdata(a), np.ma.getdata(b)
    if x.dtype != y.dtype or x.shape != y.shape: return 'dtype/shape'
    return '' if x.tobytes() == y.tobytes() else 'data'


def best(fn, k=5):
    r = []
    for _ in range(k):
        t = time.perf_counter(); fn(); r.append(time.perf_counter() - t)
    return min(r) * 1e3


CASES = [('wrfout_d01_2025-10-26_06_20_00.nc', 'T'), ('wrfout_d01_2025-10-26_06_20_00.nc', 'QVAPOR'),
         ('VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I05'),
         ('VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I05_quality_flags'),
         ('extra/OR_ABI-L1b-RadF-M6C02_G19_s20263021200207_e20263021209516_c20263021209551.nc', 'Rad'),
         ('extra/OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc', 'Rad'),
         ('NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc', 'avhrr_ch4'),
         ('rcm2-sclna-owi-cm-20261014t010230-20261014t010349-00003-xxxxx_ll_gd.nc', 'wind_speed')]
for f, v in CASES:
    d = fastnc.Dataset(path(f))
    x = d[v]
    for key in [Ellipsis, (slice(None), slice(3, 1500, 2)), (0,)]:
        _dataset.FUSED = False; a = x[key]
        _dataset.FUSED = True; b = x[key]
        _dataset.FUSED = False; t0 = best(lambda: x[key])
        _dataset.FUSED = True; t1 = best(lambda: x[key])
        print('%-26s %-22s %-26s %-5s unfused %7.2f fused %7.2f ms' % (
            f[-26:], v[-22:], str(key)[:26], same(a, b) or 'same', t0, t1), flush=True)
