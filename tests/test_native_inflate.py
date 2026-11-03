"""zlib_inflate through libdeflate (the workspace's native slots set) gives
what fastnc's own inflate gives, stream for stream -- including the ones
libdeflate rejects or decodes to another length, where the own inflate
answers."""
import zlib

import numpy as np
import pytest

import fastnc
from fastnc import _inflate, _zlib


def _native():
    fastnc.warm()
    fp, dec = _zlib.native()
    if not fp:
        pytest.skip('no libdeflate here')
    return fp, dec


def _run(z, cap, native, stop=None):
    tab, iws = _inflate.workspace()
    if native:
        iws[_inflate.NATIVE:_inflate.NATIVE + 2] = native
    out = np.full(cap + 16, 0xAB, np.uint8)
    r = _inflate.zlib_inflate(z, 0, z.size, out, 0, cap, cap if stop is None else stop, tab, iws)
    return r, out.tobytes()


def _streams():
    rng = np.random.default_rng(5)
    out = []
    for n in (0, 1, 100, 5000, 300000):
        for kind in ('random', 'runs', 'text'):
            if kind == 'random':
                raw = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
            elif kind == 'runs':
                raw = np.repeat(rng.integers(0, 4, max(1, n // 50), dtype=np.uint8), 50)[:n].tobytes()
            else:
                raw = (b'netCDF fastnc deflate ' * (n // 22 + 1))[:n]
            for level in (0, 1, 6, 9):
                out.append((raw, zlib.compress(raw, level)))
    return out


def test_same_as_own_inflate():
    nat = _native()
    for raw, z in _streams():
        zz = np.frombuffer(z, np.uint8)
        n = len(raw)
        cases = [(zz, n), (zz, n + 7), (zz, max(0, n - 1))]
        bad = bytearray(z)
        if len(bad) > 6:
            bad[-1] ^= 0x55                         # Adler-32 wrong
            cases.append((np.frombuffer(bytes(bad), np.uint8), n))
        cases.append((np.frombuffer(z + b'trailing', np.uint8), n))
        if len(z) > 4:
            cases.append((zz[:len(z) // 2], n))     # cut short
        for src, cap in cases:
            got, want = _run(src, cap, nat), _run(src, cap, None)
            # (a stream that failed: what it left in the output is nobody's)
            assert got[0] == want[0] and (got[0] < 0 or got == want), (n, cap)
            if cap > 2:
                stop = cap // 2                     # early stop: never native
                assert _run(src, cap, nat, stop) == _run(src, cap, None, stop)


def test_blosc_sized_workspace_ignores_slots():
    """A workspace no longer than IWS_SIZE (blosc's slice of its own,
    unzeroed buffer) is never read past its end for the native slots."""
    _native()
    raw = bytes(range(256)) * 100
    z = np.frombuffer(zlib.compress(raw), np.uint8)
    tab, iws = _inflate.workspace()
    short = np.full(_inflate.IWS_SIZE, 123456789, np.int64)     # (garbage where slots would be)
    out = np.zeros(len(raw), np.uint8)
    assert _inflate.zlib_inflate(z, 0, z.size, out, 0, len(raw), len(raw), tab, short) == len(raw)
    assert out.tobytes() == raw
