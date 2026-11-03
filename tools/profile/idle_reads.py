"""A read straight after another (the way benchmarks loop) against a read
after the machine has been quiet (the way a program reads): do parked
cores and sleeping threads cost anything?

    python tools/profile/idle_reads.py [seconds idle ...]
"""
import sys
import time

from common import fast_thread, path

sys.path.insert(0, path('.'))
import numpy as np                                  # noqa: E402

import fastnc                                       # noqa: E402

fastnc.warm()
fast_thread()
CASES = [('MODIS 1 km', 'MOD021KM.A2025301.1445.007.2025301204859.nc',
          'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive'),
         ('VIIRS I05', 'VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I05'),
         ('GOES C02', 'extra/OR_ABI-L1b-RadF-M6C02_G19_s20263021200207_e20263021209516_c20263021209551.nc', 'Rad'),
         ('GOES C13', 'extra/OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc', 'Rad'),
         ('WRF d01 T', 'wrfout_d01_2025-10-26_06_20_00.nc', 'T')]


def timed(x):
    t = time.perf_counter()
    x[...]
    return (time.perf_counter() - t) * 1e3


def main():
    gaps = [float(a) for a in sys.argv[1:]] or [0.05, 0.3, 1.0]
    print('%-12s %9s ' % ('read', 'loop') + ' '.join('%6.2fs idle' % g for g in gaps) + '   (ms: median of 7)')
    for name, f, v in CASES:
        d = fastnc.Dataset(path(f))
        x = d[v]
        x.set_auto_maskandscale(False)
        for _ in range(3):
            x[...]
        loop = np.median([timed(x) for _ in range(9)])
        row = []
        for g in gaps:
            r = []
            for _ in range(7):
                time.sleep(g)
                r.append(timed(x))
            row.append(np.median(r))
        print('%-12s %9.1f ' % (name, loop) + ' '.join('%11.1f' % t for t in row), flush=True)
        d.close()


if __name__ == '__main__':
    main()
