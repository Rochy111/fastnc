"""What inflate's time is made of, without hardware counters: each sampled
chunk timed on its own, its symbols counted, and a least-squares fit of
time against literals, matches, match bytes, near matches, switches
between literals and matches and blocks, for fastnc's inflate and
libdeflate. The coefficients, in ns, are the cost of one of each.

    python tools/profile/inflate_costs.py [file-substring ...]
"""
import mmap
import sys
import time

import numpy as np
from common import fast_thread, path
from stream_stats import (S_BLOCK0, S_BLOCK1, S_BLOCK2, S_D1, S_DNEAR, S_LIT, S_LIT8, S_MATCH, S_MBYTES,
                          S_TRANS, N, walk)

import fastnc                                       # noqa: E402
from fastnc import _hdf5, _inflate, _nc4, _zlib     # noqa: E402

fast_thread()
FILES = ['wrfout_d01_2025-10-26_06_20_00.nc', 'VJ102IMG_NRT.A2026265.2006.021.nc',
         'NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc',
         'extra/OR_ABI-L1b-RadF-M6C02_G19_s20263021200207_e20263021209516_c20263021209551.nc',
         'extra/OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc']
NAMES = ['literal', 'match', 'match byte', 'near match', 'd=1 match', 'switch', 'block', 'output byte']


def sample(f, most=600):
    fh = open(path(f), 'rb')
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    src = np.frombuffer(mm, np.uint8)
    nc = _nc4.NC4File(_hdf5.H5File(mm), src)
    out = []

    def walkg(g):
        for v in g.variables.values():
            ds = v.ds
            if ds.type.dtype is None or not ds.shape or ds.layout.cls != 2:
                continue
            fs = [x.id for x in ds.filters]
            if fs not in ([1], [2, 1], [1, 3], [2, 1, 3]):
                continue
            idx = v._reader.index
            cb = int(np.prod(idx.chunks)) * ds.type.size
            cut = 4 if fs[-1] == 3 else 0
            out.extend((a, s - cut, cb) for a, s in zip(idx.addr.tolist(), idx.size.tolist()))
        for s in g.groups.values():
            walkg(s)
    walkg(nc.root)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(out), min(most, len(out)), replace=False)
    return src, [out[i] for i in pick]


def solve(A, b):
    """Gaussian elimination with pivoting (no LAPACK: np.linalg ends the
    process in some numpy builds)."""
    A, b = A.copy(), b.copy()
    n = len(b)
    for i in range(n):
        p = i + int(np.argmax(np.abs(A[i:, i])))
        A[[i, p]], b[[i, p]] = A[[p, i]], b[[p, i]]
        for j in range(i + 1, n):
            f = A[j, i] / A[i, i]
            A[j, i:] -= f * A[i, i:]
            b[j] -= f * b[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - (A[i, i + 1:] * x[i + 1:]).sum()) / A[i, i]
    return x


def fit(X, y):
    keep = X.sum(axis=0) > 0
    w = 1 / np.maximum(y, 1.0)                      # relative errors: chunks of all sizes count
    Z = X[:, keep] * w[:, None]
    s = Z.max(axis=0)
    Z = Z / s
    k = Z.shape[1]
    G = np.array([[(Z[:, i] * Z[:, j]).sum() for j in range(k)] for i in range(k)])
    c = solve(G, np.array([(Z[:, i] * y * w).sum() for i in range(k)])) / s
    full = np.zeros(X.shape[1])
    full[keep] = c
    pred = (X * full).sum(axis=1)
    return full, 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    want = sys.argv[1:]
    fastnc.warm()
    tab, iws = _inflate.workspace()
    tab2, iws2 = _inflate.workspace()
    fp, dec = _zlib.native()
    iws2[_inflate.NATIVE:_inflate.NATIVE + 2] = fp, dec
    rows, tn, tl = [], [], []
    per_file = []
    for f in FILES:
        if want and not any(w in f for w in want):
            continue
        src, ch = sample(f)
        out = np.empty(max(c[2] for c in ch) + 64, np.uint8)
        start = len(rows)
        for a, s, cb in ch:
            st = np.zeros(N, np.int64)
            walk(src[a + 2:a + s - 4], st)
            blocks = st[S_BLOCK0] + st[S_BLOCK1] + st[S_BLOCK2]
            bytes_out = st[S_LIT] + st[S_MBYTES]
            if bytes_out != cb:
                continue
            rows.append([st[S_LIT], st[S_MATCH], st[S_MBYTES], st[S_DNEAR], st[S_D1], st[S_TRANS], blocks, cb])
            b1 = b2 = 1e9
            for _ in range(5):
                t = time.perf_counter()
                _inflate.zlib_inflate(src, a, a + s, out, 0, cb, cb, tab, iws)
                t1 = time.perf_counter()
                _inflate.zlib_inflate(src, a, a + s, out, 0, cb, cb, tab2, iws2)
                t2 = time.perf_counter()
                b1 = min(b1, t1 - t)
                b2 = min(b2, t2 - t1)
            tn.append(b1 * 1e9)
            tl.append(b2 * 1e9)
        per_file.append((f[-24:], start, len(rows)))
        print(f[-24:], len(rows) - start, 'chunks', flush=True)
    X = np.array(rows, float)
    y1, y2 = np.array(tn), np.array(tl)
    for name, y in (('fastnc inflate', y1), ('libdeflate', y2)):
        c, r2 = fit(X, y)
        print('\n%s: R2 %.3f' % (name, r2))
        tot = (X * c).sum(axis=0)
        for n, v, t in zip(NAMES, c, tot):
            print('  %-12s %8.2f ns each  %5.1f%% of the time' % (n, v, 100 * t / y.sum()))
    for name, a, b in per_file:
        print('%-26s fastnc %8.1f ms  libdeflate %8.1f ms' % (name, y1[a:b].sum() / 1e6, y2[a:b].sum() / 1e6))


if __name__ == '__main__':
    main()
