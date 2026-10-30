"""device='gpu' reads (raw and masked) at several band sizes."""
import sys, time
from common import fast_thread, path
sys.path.insert(0, path('.'))
import numpy as np, cupy as cp, fastnc
from fastnc import _gpu
fastnc.warm(); fast_thread()
CASES = [('wrfout_d01_2025-10-26_06_20_00.nc', 'T'),
         ('VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I05'),
         ('MOD021KM.A2025301.1445.007.2025301204859.nc', 'MODIS_SWATH_Type_L1B/Data_Fields/EV_1KM_Emissive'),
         ('extra/OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc', 'Rad'),
         ('extra/OR_ABI-L1b-RadF-M6C02_G19_s20263021200207_e20263021209516_c20263021209551.nc', 'Rad'),
         ('NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc', 'avhrr_ch4')]
def best(fn, k=5):
    r = []
    for _ in range(k):
        cp.cuda.Device().synchronize()
        t = time.perf_counter(); fn(); cp.cuda.Device().synchronize(); r.append(time.perf_counter() - t)
    return min(r) * 1e3
sizes = [int(s) << 20 for s in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['32', '8', '4', '2'])]
for f, v in CASES:
    d = fastnc.Dataset(path(f))
    try:
        x = d[v]
    except Exception:
        x = d['/'.join(['HDFEOS', 'SWATHS'] + v.split('/'))] if False else None
    if x is None:
        continue
    row = []
    for m in (False, True):
        x.set_auto_maskandscale(m)
        row.append('cpu %6.1f' % best(lambda: x[...]))
        for s in sizes:
            _gpu.PIPE_BAND_BYTES = s
            row.append('%dM %6.1f' % (s >> 20, best(lambda: x.read(Ellipsis, device='gpu'))))
    print('%-26s %-14s %s' % (f[-26:], v[-14:], ' | '.join(row)), flush=True)
