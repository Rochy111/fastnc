"""Single-variable and whole-file reads (raw, warm) at several PAR_MIN
(the compressed size from which one chunk is inflated on many threads)."""
import sys, time
from common import fast_thread, path
sys.path.insert(0, path('.'))
import fastnc
from fastnc import _chunks
fastnc.warm(); fast_thread()
CASES = [('wrfout_d01_2025-10-26_06_20_00.nc', 'T'), ('wrfout_d01_2025-10-26_06_20_00.nc', None),
         ('VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I05'), ('VJ102IMG_NRT.A2026265.2006.021.nc', None),
         ('MOD021KM.A2025301.1445.007.2025301204859.nc', None),
         ('NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc', 'avhrr_ch4'), ('NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc', None),
         ('rcm2-sclna-owi-cm-20261014t010230-20261014t010349-00003-xxxxx_ll_gd.nc', 'wind_speed'),
         ('extra/OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc', 'Rad')]
def best(fn, k=7):
    r = []
    for _ in range(k):
        t = time.perf_counter(); fn(); r.append(time.perf_counter() - t)
    return min(r) * 1e3
vals = [int(v) << 10 for v in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['4096', '1024', '512'])]
for f, v in CASES:
    d = fastnc.Dataset(path(f)); d.set_auto_maskandscale(False)
    row = []
    for pm in vals:
        _chunks.PAR_MIN = pm
        row.append(best(lambda: d[v][...]) if v else best(lambda: d.load(recursive=True), 3))
    print('%-34s %-22s ' % (f[-34:], v or '(load)') + '  '.join('%7.2f' % x for x in row))
