"""Fuzz the HDF5 filter decoders longer than the test suite does: damaged
chunks from the committed files and random parameters, many rounds.
Every call must return an error or an output inside the range given, and
leave the guard bytes around the output alone. Run it with numba's bounds
checks on, and a cache of its own, so that any read or write outside an
array is an IndexError rather than silence:

    set NUMBA_BOUNDSCHECK=1& set NUMBA_CACHE_DIR=%TEMP%\\fastnc_bc
    python tools/check/filters_hdf_fuzz.py [rounds=2000] [seed=1]

(the 8-byte loads and stores of fastnc/_intrin.py are not checked; every
one of them sits behind an explicit range test.)
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))

import test_filters_hdf as T                                       # noqa: E402
from fastnc import _nbit, _scaleoffset, _shuffle, _szip           # noqa: E402


def mutate(r, c):
    kind = r.integers(0, 6)
    c = c.copy()
    if kind == 0 and c.size:
        for _ in range(r.integers(1, 6)):
            c[r.integers(0, c.size)] ^= np.uint8(1 << r.integers(0, 8))
    elif kind == 1 and c.size:
        k = r.integers(1, 9)
        c[r.integers(0, c.size, k)] = r.integers(0, 256, k)
    elif kind == 2:
        c = c[:r.integers(0, c.size + 1)]
    elif kind == 3:
        c = r.integers(0, 256, r.integers(0, 2 * c.size + 8), dtype=np.uint8)
    elif kind == 4 and c.size > 4:
        i = r.integers(4, c.size)
        c[i:] = r.integers(0, 256, c.size - i, dtype=np.uint8)
    elif kind == 5 and c.size > 8:                  # a run of zeros or ones
        i = r.integers(0, c.size - 4)
        c[i:i + r.integers(1, 64)] = 0 if r.random() < 0.5 else 255
    return c


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    r = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    cases = T._fuzz_inputs()
    print('bounds checks %s; %d chunks' % ('ON' if os.environ.get('NUMBA_BOUNDSCHECK') == '1' else 'off',
                                          len(cases)))
    t = time.perf_counter()
    nerr = nout = 0
    pad = 64
    for i in range(rounds):
        mod, cd, chunk, nbytes = cases[i % len(cases)]
        c = mutate(r, chunk)
        src = np.concatenate([np.full(pad, 0xA5, np.uint8), c, np.full(pad, 0xA5, np.uint8)])
        dst = np.full(nbytes + 2 * pad, 0x5A, np.uint8)
        ws = np.zeros(max(1, mod.workspace_size(nbytes)), np.uint8)
        end = mod.decode(src, pad, pad + c.size, dst, pad, pad + nbytes, cd, ws)
        assert end < 0 or pad <= end <= pad + nbytes, (mod.__name__, end)
        assert (dst[:pad] == 0x5A).all() and (dst[pad + nbytes:] == 0x5A).all(), mod.__name__
        nerr += end < 0
        nout += end >= 0
    src = r.integers(0, 256, 5000, dtype=np.uint8)
    for i in range(rounds):                         # random parameters
        mod = (_szip, _nbit, _scaleoffset, _shuffle)[i % 4]
        n = int(r.integers(0, 40))
        vals = r.integers(0, 70, n) if r.random() < 0.7 else r.integers(0, 2 ** 32, n)
        cd = np.array([int(r.integers(1, 9)), 2000, n] + list(vals), np.int64)
        if mod is _nbit and n:
            cd[3] = n
        src[:4] = np.frombuffer(int(r.integers(0, 2500)).to_bytes(4, 'little'), np.uint8)
        dst = np.full(2200, 0x5A, np.uint8)
        end = mod.decode(src, 0, int(r.integers(0, src.size)), dst, 100, 2100, cd, np.zeros(2000, np.uint8))
        assert end < 0 or 100 <= end <= 2100
        assert (dst[:100] == 0x5A).all() and (dst[2100:] == 0x5A).all()
    print('%d damaged chunks (%d errors, %d decoded to something) and %d parameter sets: ok, %.1f s'
          % (rounds, nerr, nout, rounds, time.perf_counter() - t))


if __name__ == '__main__':
    main()
