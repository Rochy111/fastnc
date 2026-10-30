"""Where fastnc._pinflate's work goes on one big chunk: each part decoded
alone on this thread (time, output, share of 16-bit symbols), the chain
walk, the copy / marker step -- against the serial inflate of the same
chunk. Work, not wall time: the parallel time is about the slowest part
plus the chain plus the copy step spread over the threads.

    python tools/profile/pinflate_steps.py [file] [chunk index] [threads]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'check'))
from common import fast_thread  # noqa: E402
from fastnc import _inflate, _pinflate as P  # noqa: E402
from pinflate_real import DATA, deflate_chunks  # noqa: E402


def main():
    f = sys.argv[1] if len(sys.argv) > 1 else 'MOD02QKM.A2025301.1445.007.2025301204859.nc'
    ci = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    fast_thread()
    p = os.path.join(DATA, f)
    raw = np.fromfile(p, np.uint8)
    chunks = sorted(deflate_chunks(p, 1 << 20), key=lambda c: -c[1])
    off, sz, csz, name = chunks[ci]
    print('%s %s: %.1f MB -> %.1f MB, %d threads' % (f[:24], name.split('/')[-1], sz / 1e6,
                                                    csz / 1e6, T))
    out = np.zeros(csz, np.uint8)
    tab, iws = _inflate.workspace()
    best = 9e9
    for _ in range(5):
        t = time.perf_counter()
        _inflate.zlib_inflate(raw, off, off + sz, out, 0, csz, csz, tab, iws)
        best = min(best, time.perf_counter() - t)
    print('serial inflate: %.1f ms (%.0f MB/s out)' % (best * 1e3, csz / best / 1e6))
    info = []
    P.inflate_parallel(raw, off, off + sz, out, 0, csz, nthreads=T, info=info)
    walls = []
    for _ in range(9):
        info = []
        t = time.perf_counter()
        P.inflate_parallel(raw, off, off + sz, out, 0, csz, nthreads=T, info=info)
        walls.append((time.perf_counter() - t, info[0][6], info[0][7]))
    w = sorted(walls)[len(walls) // 2]
    print('parallel wall: %.2f ms (steps A+B %.2f ms, C %.2f ms, rest %.2f ms)' % (
        w[0] * 1e3, w[1] * 1e3, w[2] * 1e3, (w[0] - w[1] - w[2]) * 1e3))
    PT, RES, SEG, used, nslab, ssize = info[0][:6]
    SI = info[-1]
    print('parts %d, slabs used %d of %d (%.1f MB each), redone %d' % (
        len(PT), used, nslab, ssize / 1e6, SI[0, P.SI_NREDO]))
    # each part again, alone, timed
    lo, pend = off + 2, off + sz
    B8 = np.empty(ssize * nslab, np.uint8)
    W16 = B8.view(np.uint16)
    slab = np.zeros(1, np.int64)
    res = np.zeros(P.NRES, np.int64)
    seg = np.zeros((P.MAXSEG, P.NG), np.int64)
    tot_spec = 0.0
    tot16 = tot8 = 0
    rows = []
    for t in range(len(PT)):
        a, b = PT[t, P.PT_A], PT[t, P.PT_B]
        tm = 9e9
        for _ in range(3):
            slab[0] = 0
            t0 = time.perf_counter()
            if t == 0:
                P._decode_true(raw, lo, pend, 0, a, b, out, 0, 0, csz, csz, tab, iws, res)
            else:
                P._part(raw, lo, pend, a, b, P.FINDMAX, B8, W16, seg, slab, 0, ssize, nslab,
                        tab, iws, res)
            tm = min(tm, time.perf_counter() - t0)
        if t == 0:
            n16, n8 = 0, res[P.R_O]
        else:
            n16 = sum(seg[k, P.G_N] for k in range(res[P.R_NSEG]) if seg[k, P.G_TYP] == 1)
            n8 = sum(seg[k, P.G_N] for k in range(res[P.R_NSEG]) if seg[k, P.G_TYP] == 2)
            tot_spec += tm
            tot16 += n16
            tot8 += n8
        rows.append((tm, n16, n8))
    tms = np.array([r[0] for r in rows])
    print('part times ms: part0 %.2f, others median %.2f, max %.2f; sum %.1f ms = %.2fx serial'
          % (tms[0] * 1e3, np.median(tms[1:]) * 1e3, tms[1:].max() * 1e3, tms.sum() * 1e3,
             tms.sum() / best))
    print('speculative parts: %.1f MB, %.0f%% as 16-bit symbols, %.0f MB/s out per thread'
          % ((tot16 + tot8) / 1e6, 100 * tot16 / max(1, tot16 + tot8),
             (tot16 + tot8) / tot_spec / 1e6))
    for t, (tm, n16, n8) in enumerate(rows):
        if '-v' in sys.argv:
            print('  part %2d %6.2f ms  %7d sym  %8d bytes  %5.0f MB/s' % (
                t, tm * 1e3, n16, n8, (n16 + n8) / tm / 1e6))


if __name__ == '__main__':
    main()
