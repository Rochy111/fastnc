"""fastnc's numba bzip2 decoder against Python's bz2 (libbzip2), one
thread, on chunk-like data: equal output, and MB/s of output for each
(best of 5, alternating).

    python tools/check/filters_hdf_bzip2.py
"""
import bz2
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'profile'))
from common import fast_thread                                    # noqa: E402

from fastnc import _bzip2                                          # noqa: E402


def main():
    fast_thread()
    r = np.random.default_rng(0)
    side = 1024
    x = np.cumsum(np.cumsum(r.standard_normal((side, side)), 0), 1).astype('f4')
    cases = {
        'f4 field 1MB': x[:256].tobytes(),
        'f4 field 4MB': x.tobytes(),
        'i2 smooth': (x[:512] * 10).astype('i2').tobytes(),
        'text-like': bytes(r.choice(list(b'abcdefgh  \n'), 1 << 20).astype('u1')),
        'random 256k': r.integers(0, 256, 1 << 18, dtype=np.uint8).tobytes(),
        'zeros 1MB': bytes(1 << 20),
        'runs': np.repeat(np.arange(200, dtype='u1'), 5000).tobytes(),
    }
    ws = np.zeros(_bzip2.workspace_size(0), np.uint8)
    for name, raw in cases.items():
        for lev in (9, 1):
            z = bz2.compress(raw, lev)
            src = np.frombuffer(z, np.uint8)
            dst = np.zeros(len(raw) + 16, np.uint8)
            e = _bzip2.decode(src, 0, src.size, dst, 0, len(raw), None, ws)
            ok = e == len(raw) and dst[:len(raw)].tobytes() == raw
            tn = tb = 1e9
            for k in range(5):
                t = time.perf_counter()
                _bzip2.decode(src, 0, src.size, dst, 0, len(raw), None, ws)
                tn = min(tn, time.perf_counter() - t)
                t = time.perf_counter()
                bz2.decompress(z)
                tb = min(tb, time.perf_counter() - t)
            print('%-14s level %d ratio %.2f  fastnc %7.1f MB/s  bz2 %7.1f MB/s  %.2fx %s' % (
                name, lev, len(z) / len(raw), len(raw) / tn / 1e6, len(raw) / tb / 1e6, tb / tn,
                'equal' if ok else 'DIFFERENT (%s)' % e))


if __name__ == '__main__':
    main()
