"""The paths reads take before numba's kernels are loaded (a fresh process)
give the kernels' values exactly: chunks inflated through _zlib, chunk
grids of unfiltered datasets gathered at once (forced on for every size
here), mask / scale by numpy -- every variable of the corpus, whole and in
windows and strides."""
import glob
import os

import numpy as np
import pytest

import fastnc
from fastnc import _chunks, _zlib

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FILES = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, '*.nc'))
               + glob.glob(os.path.join(DATA, '*.h5')))


def _keys(shape):
    out = [Ellipsis]
    if len(shape) >= 1 and shape[0] > 3:
        out.append((slice(1, None, 2),))
    if len(shape) >= 2 and shape[0] > 2 and shape[1] > 4:
        out.append((slice(1, -1), slice(2, None, 3)))
    return out


class _Raised:
    def __init__(self, e):
        self.what = (type(e).__name__, str(e))

    def __eq__(self, other):
        return isinstance(other, _Raised) and self.what == other.what


def _same(a, b):
    if isinstance(a, _Raised) or isinstance(b, _Raised):
        return a == b
    if np.ma.isMaskedArray(a) or np.ma.isMaskedArray(b):
        if not (np.ma.isMaskedArray(a) and np.ma.isMaskedArray(b)):
            return False
        if np.ma.getmask(a) is np.ma.nomask or np.ma.getmask(b) is np.ma.nomask:
            if (np.ma.getmask(a) is np.ma.nomask) != (np.ma.getmask(b) is np.ma.nomask):
                return False
        elif np.ma.getmaskarray(a).tobytes() != np.ma.getmaskarray(b).tobytes():
            return False
        fa, fb = a.fill_value, b.fill_value
        if type(fa) is not type(fb) or np.asarray(fa).tobytes() != np.asarray(fb).tobytes():
            return False
    a, b = np.ma.getdata(a), np.ma.getdata(b)
    if isinstance(a, np.ndarray) and a.dtype.kind == 'O':
        return a.shape == b.shape and all(
            np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
            for x, y in zip(a.ravel(), b.ravel()))
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _reads(path):
    out = []
    with fastnc.Dataset(path) as ds:
        for g in ds._walk():
            for name, v in g.variables.items():
                full = ('%s/%s' % (g.path.rstrip('/'), name)).lstrip('/')
                for k in _keys(v.shape):
                    for m in (False, True):
                        v.set_auto_maskandscale(m)
                        try:
                            out.append((full, k, m, v[k]))
                        except Exception as e:      # (as netCDF4-python raises, in places)
                            out.append((full, k, m, _Raised(e)))
    return out


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('name', FILES)
def test_without_kernels_same_values(name, split, monkeypatch):
    """(split: the staged grid's chunks inflated, and placed, on many
    threads however small)"""
    path = os.path.join(DATA, name)
    fastnc.warm()
    want = _reads(path)                             # (the kernels: checked against the references)
    monkeypatch.setattr(_chunks, '_READY', __import__('threading').Event())
    monkeypatch.setattr(_chunks, 'IMAGE', False)
    monkeypatch.setattr(_chunks, 'LOAD_AFTER', float('inf'))
    monkeypatch.setattr(_chunks, 'GRID_MIN', 1)
    if split:
        monkeypatch.setattr(_chunks, 'ZGRID_SOLO', 0)
        monkeypatch.setattr(_chunks, 'ZGRID_BAND', 1)
    got = _reads(path)
    assert [r[:3] for r in got] == [r[:3] for r in want]
    bad = [r[:3] for r, w in zip(got, want) if not _same(r[3], w[3])]
    assert not bad, bad[:5]


def test_zlib_inflate_into():
    import zlib
    raw = np.random.default_rng(1).integers(0, 7, 300000, dtype=np.uint8).tobytes()
    z = np.frombuffer(zlib.compress(raw, 6), np.uint8)
    out = np.zeros(len(raw) + 10, np.uint8)
    assert _zlib.inflate_into(z, 0, z.size, out, 5, len(raw)) == len(raw)
    assert out[5:5 + len(raw)].tobytes() == raw
    assert _zlib.inflate_into(z, 0, z.size, out, 0, len(raw) - 1) == -1     # too small: -1
    with pytest.raises(zlib.error):
        _zlib.inflate_into(np.frombuffer(b'\x78\x9c garbage', np.uint8), 0, 10, out, 0, 100)
