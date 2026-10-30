"""Unshuffle speed (GB/s of output, one thread) for element sizes 2, 4, 8
and chunks of 32 kB - 4 MB, against numpy's transpose-copy and a plain
copy (np.copyto) of the same bytes. Best of `--reps` passes over a set
of chunks larger than the L2 cache (as the chunks of a read would be,
coming straight from inflate they would be in cache: `--hot` reuses one).

    python tools/check/filters_hdf_shuffle_bench.py [--hot] [--reps 7]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'profile'))
from common import fast_thread                                    # noqa: E402

from fastnc import _shuffle                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hot', action='store_true')
    ap.add_argument('--reps', type=int, default=7)
    a = ap.parse_args()
    fast_thread()
    print('%8s %6s %10s %10s %10s' % ('chunk', 'esize', 'fastnc', 'numpy', 'copy'))
    for n in (32 << 10, 256 << 10, 1 << 20, 4 << 20):
        nchunks = 1 if a.hot else max(1, (64 << 20) // n)
        src = np.random.default_rng(0).integers(0, 256, n * nchunks, dtype=np.uint8)
        dst = np.empty_like(src)
        for es in (2, 4, 8):
            _shuffle.unshuffle_into(src, 0, n, es, dst, 0)
            t_ours = t_np = t_cp = 1e9
            for r in range(a.reps):
                t = time.perf_counter()
                for c in range(nchunks):
                    _shuffle.unshuffle_into(src, c * n, n, es, dst, c * n)
                t_ours = min(t_ours, time.perf_counter() - t)
                t = time.perf_counter()
                for c in range(nchunks):
                    d = dst[c * n:(c + 1) * n].reshape(-1, es)
                    np.copyto(d, src[c * n:(c + 1) * n].reshape(es, -1).T)
                t_np = min(t_np, time.perf_counter() - t)
                t = time.perf_counter()
                for c in range(nchunks):
                    np.copyto(dst[c * n:(c + 1) * n], src[c * n:(c + 1) * n])
                t_cp = min(t_cp, time.perf_counter() - t)
            tot = n * nchunks / 1e9
            print('%7dk %6d %7.1f GB/s %7.1f GB/s %7.1f GB/s' % (n >> 10, es, tot / t_ours, tot / t_np,
                                                              tot / t_cp))


if __name__ == '__main__':
    main()
