"""MODIS reads with the chunk engine as it is against a copy of
fastnc/_chunks.py with big chunks inflated by fastnc._pinflate (the
proposed integration), A/B alternated: median times and the ratio of each
pair (median, min-max). Values checked against h5py first.

    python tools/profile/pinflate_read.py PATCHED_CHUNKS_PY [reps]
"""
import importlib.util
import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from common import fast_thread  # noqa: E402
import fastnc  # noqa: E402
from fastnc import _chunks, _h5read  # noqa: E402

DATA = os.environ.get('FASTNC_DATA', r'C:\Users\Roch\Desktop\Python\fastnc')
QKM = os.path.join(DATA, 'MOD02QKM.A2025301.1445.007.2025301204859.nc')
KM = os.path.join(DATA, 'MOD021KM.A2025301.1445.007.2025301204859.nc')
BAND = 'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_250_RefSB'


def load_patched(path):
    spec = importlib.util.spec_from_file_location('fastnc._chunks_par', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['fastnc._chunks_par'] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    patched = load_patched(sys.argv[1])
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    fast_thread()
    _chunks.warm()
    patched.warm()
    f = fastnc.File(QKM)
    d = f[BAND]
    reads = [
        ('QKM EV_250_RefSB[0]', lambda: d[0], lambda h: h[BAND][0]),
        ('QKM EV_250_RefSB[:]', lambda: d[...], lambda h: h[BAND][...]),
        ('QKM EV_250_RefSB[0, :256, :256]', lambda: d[0, :256, :256], lambda h: h[BAND][0, :256, :256]),
        ('QKM EV_250_RefSB[1, 4000:4256, 1000:1256]', lambda: d[1, 4000:4256, 1000:1256],
         lambda h: h[BAND][1, 4000:4256, 1000:1256]),
    ]
    whole = []
    for p in (KM, QKM):
        ds = fastnc.Dataset(p)
        ds.set_auto_maskandscale(False)
        whole.append(('%s whole file (load, raw)' % os.path.basename(p)[:8],
                      (lambda ds=ds: ds.load(recursive=True)), None))
    # values first
    for mod in (_chunks, patched):
        _h5read._chunks = mod
        with h5py.File(QKM, 'r') as h:
            for name, fn, ref in reads:
                assert np.array_equal(fn(), ref(h)), (name, mod.__name__)
    print('values identical to h5py, both engines')
    print('%-44s %10s %10s   %s' % ('read', 'engine ms', 'patched ms', 'speedup median (min-max)'))
    for name, fn, _ in reads + whole:
        ta, tb, r = [], [], []
        for _ in range(reps):
            _h5read._chunks = _chunks
            t = time.perf_counter()
            fn()
            a = time.perf_counter() - t
            _h5read._chunks = patched
            t = time.perf_counter()
            fn()
            b = time.perf_counter() - t
            ta.append(a)
            tb.append(b)
            r.append(a / b)
        print('%-44s %10.1f %10.1f   %5.2fx (%.2f-%.2f)' % (
            name, 1e3 * np.median(ta), 1e3 * np.median(tb), np.median(r), min(r), max(r)))
    _h5read._chunks = _chunks


if __name__ == '__main__':
    main()
