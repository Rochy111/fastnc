"""Speed of fastnc._pinflate on one big MODIS chunk against the serial
inflate (fastnc._inflate.zlib_inflate, one thread), A/B alternated since
other work may share the machine: per thread count, the median time and
the ratio serial / parallel of each adjacent pair (median, min-max).

    python tools/profile/pinflate_speed.py [file] [chunk index] [reps]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'check'))
from common import fast_thread  # noqa: E402
from fastnc import _inflate, _pinflate  # noqa: E402
from pinflate_real import DATA, deflate_chunks  # noqa: E402


def main():
    f = sys.argv[1] if len(sys.argv) > 1 else 'MOD02QKM.A2025301.1445.007.2025301204859.nc'
    ci = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    fast_thread()
    p = os.path.join(DATA, f)
    raw = np.fromfile(p, np.uint8)
    chunks = sorted(deflate_chunks(p, 1 << 20), key=lambda c: -c[1])
    off, sz, csz, name = chunks[ci]
    print('%s %s: %.1f MB -> %.1f MB' % (f[:24], name.split('/')[-1], sz / 1e6, csz / 1e6))
    out = np.zeros(csz, np.uint8)
    ref = np.zeros(csz, np.uint8)
    tab, iws = _inflate.workspace()

    def serial():
        return _inflate.zlib_inflate(raw, off, off + sz, ref, 0, csz, csz, tab, iws)

    def par(n):
        return _pinflate.inflate_parallel(raw, off, off + sz, out, 0, csz, nthreads=n)
    assert serial() == csz
    for n in (2, 4, 8, 16, 24):
        assert par(n) == csz and np.array_equal(out, ref)
    print('threads   serial ms   parallel ms   speedup median (min-max)   GB/s out')
    for n in (2, 4, 8, 16, 24):
        ts, tp, ratio = [], [], []
        for _ in range(reps):
            t = time.perf_counter()
            serial()
            a = time.perf_counter() - t
            t = time.perf_counter()
            par(n)
            b = time.perf_counter() - t
            ts.append(a)
            tp.append(b)
            ratio.append(a / b)
        print('%7d   %9.1f   %11.2f   %6.2fx (%.2f-%.2f)   %8.1f' % (
            n, 1e3 * np.median(ts), 1e3 * np.median(tp), np.median(ratio), min(ratio), max(ratio),
            csz / np.median(tp) / 1e9))


if __name__ == '__main__':
    main()
