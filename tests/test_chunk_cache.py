"""Small reads of shuffled chunks go through the decoded-chunk cache
(_chunks._cached): the bytes are those of the uncached read, for points,
windows across chunk edges, strides and integer indexes, first read and
repeated, and with the cache evicting (a tiny CACHE_BYTES)."""
import glob
import os

import numpy as np
import pytest

import fastnc
from fastnc import _chunks, _h5py

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FILES = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, '*.nc'))
               + glob.glob(os.path.join(DATA, '*.h5')) if not os.path.basename(p).startswith('nc3'))


def _datasets(f):
    out = []
    def visit(name, obj):
        if (isinstance(obj, _h5py.Dataset) and obj.chunks and obj.shape and obj.dtype.kind != 'O'
                and obj.dtype.itemsize > 1 and obj.shuffle):
            out.append(name)
    f.visititems(visit)
    return out


def _keys(shape, chunks):
    rank = len(shape)
    keys = [tuple(0 for _ in shape), tuple(n - 1 for n in shape),
            tuple(min(c, n - 1) for c, n in zip(chunks, shape))]
    # a small window straddling the first chunk edge of every dimension
    keys.append(tuple(slice(max(0, min(c, n) - 2), min(n, c + 2)) for c, n in zip(chunks, shape)))
    if rank >= 2:
        keys.append((0,) + tuple(slice(None, None, 3) if c > 3 else slice(None) for c in chunks[1:]))
        keys.append(tuple(slice(1, 4) for _ in shape))
    return keys


@pytest.mark.parametrize('name', FILES)
@pytest.mark.parametrize('cache_bytes', [_chunks.CACHE_BYTES, 1 << 12])
def test_cached_equals_uncached(name, cache_bytes, monkeypatch):
    with fastnc.File(os.path.join(DATA, name)) as f:
        names = _datasets(f)
        if not names:
            pytest.skip('no shuffled datasets')
        want = {}
        monkeypatch.setattr(_chunks, 'CACHE_BYTES', 0)              # no cache: the usual read
        for n in names:
            ds = f[n]
            for k in _keys(ds.shape, ds.chunks):
                want[n, repr(k)] = np.asarray(ds[k])
        monkeypatch.setattr(_chunks, 'CACHE_BYTES', cache_bytes)
        used = []
        real = _chunks._cached
        monkeypatch.setattr(_chunks, '_cached', lambda *a: used.append(real(*a)) or used[-1])
        for rep in range(3):                        # (kept from the second read on)
            for n in names:
                ds = f[n]
                for k in _keys(ds.shape, ds.chunks):
                    got = np.asarray(ds[k])
                    w = want[n, repr(k)]
                    assert got.dtype == w.dtype and got.shape == w.shape, (n, k)
                    assert got.tobytes() == w.tobytes(), (n, k, rep)
        if cache_bytes > 1 << 12:
            assert any(used), 'no read went through the cache'
