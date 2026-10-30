"""Real reads at several settings of the work worth one more thread
(_pool._WORK: what fastnc.tune() measures)."""
import sys
import time

from common import fast_thread, path

sys.path.insert(0, path('.'))
import fastnc                                       # noqa: E402
from fastnc import _pool                            # noqa: E402

fastnc.warm()
fast_thread()
CASES = [('wrfout_d01_2025-10-26_06_20_00.nc', 'T', Ellipsis),
         ('wrfout_d01_2025-10-26_06_20_00.nc', 'T', (0, slice(10, 20), slice(50, 80), slice(60, 90))),
         ('wrfout_d01_2025-10-26_06_20_00.nc', 'HGT', Ellipsis),
         ('VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I05', (slice(3000, 3512), slice(3000, 3512))),
         ('VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I05', (slice(None), 100)),
         ('NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc', 'avhrr_ch4', Ellipsis),
         ('rcm2-sclna-owi-cm-20261014t010230-20261014t010349-00003-xxxxx_ll_gd.nc', 'wind_speed', Ellipsis),
         ('20260904I1_ku_outer_lfm_VV_L2_pass13_combined.nc', 'Ze', Ellipsis),
         ('extra/OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc', 'Rad',
          (slice(1000, 1500), slice(1000, 1500)))]


def best(fn, k=40):
    r = []
    for _ in range(k):
        t = time.perf_counter()
        fn()
        r.append(time.perf_counter() - t)
    return min(r) * 1e6


vals = [int(v) << 10 for v in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['32', '96', '192', '378'])]
print('%-22s %-30s ' % ('file', 'read') + ' '.join('%7dK' % (v >> 10) for v in vals) + '   (us)')
for f, v, key in CASES:
    d = fastnc.Dataset(path(f))
    d.set_auto_maskandscale(False)
    x = d[v]
    row = []
    for w in vals:
        _pool._WORK = w
        row.append(best(lambda: x[key]))
    print('%-22s %-30s ' % (f[-22:], (v[-12:] + ' ' + str(key)[:16])) + ' '.join('%8.0f' % t for t in row), flush=True)
