"""The output arena's effect on repeated reads (a loop over files: each
result dropped before the next read), arena on vs off, alternated."""
import sys, time
import numpy as np
from common import fast_thread, path
sys.path.insert(0, path('.'))
import fastnc
from fastnc import _arena

fast_thread()
fastnc.warm()
cases = [('VIIRS raw, all variables', 'VJ102IMG_NRT.A2026265.2006.021.nc', None, False),
         ('VIIRS masked, all variables', 'VJ102IMG_NRT.A2026265.2006.021.nc', None, True),
         ('WRF d02 netCDF3, all', 'wrfout_d02_2024-11-06_01_30_00', None, False),
         ('WRF d01 netCDF4, all', 'wrfout_d01_2025-10-26_06_20_00.nc', None, False),
         ('GOES RadF C02 Rad masked', 'extra/OR_ABI-L1b-RadF-M6C02_G19_s20263021200207_e20263021209516_c20263021209551.nc', 'Rad', True)]


def run(p, var, masked):
    with fastnc.Dataset(p) as d:
        d.set_auto_maskandscale(masked)
        if var:
            x = d[var][...]
        else:
            x = d.load(recursive=True)
    del x


budget = _arena.BUDGET
for label, f, var, masked in cases:
    p = path(f)
    run(p, var, masked)
    on, off = [], []
    for _ in range(5):
        _arena.BUDGET = budget
        t = time.perf_counter(); run(p, var, masked); on.append(time.perf_counter() - t)
        _arena.BUDGET = 0
        _arena.release()
        t = time.perf_counter(); run(p, var, masked); off.append(time.perf_counter() - t)
    _arena.BUDGET = budget
    print('%-30s arena %7.1f ms   fresh memory %7.1f ms   (%.2fx)' % (label, min(on) * 1e3, min(off) * 1e3, min(off) / min(on)))
