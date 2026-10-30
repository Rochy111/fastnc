"""The proposed chunk engine (a copy of fastnc/_chunks.py with big chunks
inflated by fastnc._pinflate) against the engine as it is, on every
variable of the real files: whole-file loads (all chunks of a file in one
batch), each variable read alone, and random windows of the variables
with chunks over 1 MB (partial reads stop early). Values must be equal
(the engine itself is checked against h5py by read_vs_h5py.py).

    python tools/check/pinflate_engine.py PATCHED_CHUNKS_PY [windows per variable]
"""
import glob
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import fastnc  # noqa: E402
from fastnc import _chunks, _h5read  # noqa: E402

DATA = os.environ.get('FASTNC_DATA', r'C:\Users\Roch\Desktop\Python\fastnc')
FILES = ['MOD021KM.A2025301.1445.007.2025301204859.nc',
         'MOD02QKM.A2025301.1445.007.2025301204859.nc',
         'VJ102IMG_NRT.A2026265.2006.021.nc',
         'wrfout_d01_2025-10-26_06_20_00.nc',
         'NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc']


def load_patched(path):
    spec = importlib.util.spec_from_file_location('fastnc._chunks_par', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['fastnc._chunks_par'] = mod
    spec.loader.exec_module(mod)
    return mod


def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def main():
    patched = load_patched(sys.argv[1])
    nwin = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    _chunks.warm()
    patched.warm()
    rng = np.random.default_rng(0)
    bad = 0
    paths = [os.path.join(DATA, f) for f in FILES] + sorted(glob.glob(os.path.join(DATA, 'extra', '*.nc')))
    for p in paths:
        ds = fastnc.Dataset(p)
        ds.set_auto_maskandscale(False)
        res = {}
        for mod in (_chunks, patched):
            _h5read._chunks = mod
            res[mod] = ds.load(recursive=True)
        nvar, nw = len(res[_chunks]), 0
        ok = same(res[_chunks], res[patched])
        f = fastnc.File(p)
        big = []

        def visit(name, obj):
            if isinstance(obj, fastnc._h5py.Dataset) and obj.chunks and obj.shape:
                if int(np.prod(obj.chunks)) * obj.dtype.itemsize > 1 << 20:
                    big.append(name)
        f.visititems(visit)
        for name in big:
            d = f[name]
            keys = [Ellipsis, (0,) * (len(d.shape) - 2) + (slice(None),) * min(2, len(d.shape))]
            for _ in range(nwin):
                key = []
                for n in d.shape:
                    a = int(rng.integers(0, n))
                    b = int(rng.integers(a + 1, n + 1))
                    key.append(slice(a, b, int(rng.choice([1, 1, 1, 2, 3]))))
                keys.append(tuple(key))
            for key in keys:
                vals = []
                for mod in (_chunks, patched):
                    _h5read._chunks = mod
                    vals.append(d[key])
                nw += 1
                if not same(*vals):
                    ok = False
                    print('  MISMATCH', name, key)
        _h5read._chunks = _chunks
        bad += not ok
        print('%-48s %4d variables loaded, %4d reads of %d big variables: %s'
              % (os.path.basename(p)[:48], nvar, nw, len(big), 'identical' if ok else 'DIFFERENT'))
    print('ALL IDENTICAL' if not bad else '%d FILES DIFFER' % bad)
    return bad


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
