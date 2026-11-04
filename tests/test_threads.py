"""Many user threads reading one open file at once (dask's threaded
scheduler, a user's own pool) get the same values as one thread: the
shared pool, per-thread scratch, cached plans and the output arena."""
import os
import threading

import numpy as np
import pytest

import fastnc
from helpers import _bytes

DATA =os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FILES = ['nc4.nc', 'nc4_classic.nc', 'nc3_64bit.nc', 'idx_latest.h5', 'filters.h5']


def _reads(path):
    """(name, key, masked) reads of a file's variables: whole and windows."""
    ds = fastnc.Dataset(path)
    out = []
    for g in ds._walk():
        for name, v in g.variables.items():
            full = ('%s/%s' % (g.path.rstrip('/'), name)).lstrip('/')
            try:
                v[...]
            except NotImplementedError:
                continue
            out.append((full, Ellipsis, True))
            out.append((full, Ellipsis, False))
            if v.ndim >= 1 and v.shape[0] > 2:
                out.append((full, slice(1, None, 2), True))
    ds.close()
    return out


def _value(ds, name, key, masked):
    v = ds[name]
    v.set_auto_maskandscale(masked)
    return v[key]


def _same(a, b):
    if np.ma.isMaskedArray(a) or np.ma.isMaskedArray(b):
        return (np.array_equal(np.ma.getmaskarray(a), np.ma.getmaskarray(b))
                and _bytes(np.ma.getdata(a)) == _bytes(np.ma.getdata(b)))
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == 'O':
        return a.shape == b.shape and all(
            np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
            for x, y in zip(a.ravel(), b.ravel()))
    return a.dtype == b.dtype and a.shape == b.shape and _bytes(a) == _bytes(b)  # (no compound padding)


@pytest.mark.parametrize('name', FILES)
def test_threads_read_one_file(name):
    path = os.path.join(DATA, name)
    reads = _reads(path)
    with fastnc.Dataset(path) as ds:
        want = [_value(ds, *r) for r in reads]
    # (mask/scale flags are per-variable state, as in netCDF4-python: one
    # Dataset per setting, set before the threads start, shared by them all)
    with fastnc.Dataset(path) as masked, fastnc.Dataset(path) as raw:
        raw.set_auto_maskandscale(False)
        errors = []
        bad = []

        def worker(k):
            try:
                for rep in range(3):
                    for i in range(k, len(reads), 2):       # overlapping halves
                        name, key, m = reads[i]
                        got = (masked if m else raw)[name][key]
                        if not _same(got, want[i]):
                            bad.append(reads[i])
            except Exception as e:                          # pragma: no cover
                errors.append(e)
        ts = [threading.Thread(target=worker, args=(k % 2,)) for k in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    assert not errors, errors[0]
    assert not bad, bad[:3]


def test_threads_load_and_read():
    path = os.path.join(DATA, 'idx_latest.h5')
    with fastnc.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        want = ds.load(recursive=True)
    with fastnc.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        results = [None] * 6

        def worker(k):
            results[k] = ds.load(recursive=True)
        ts = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    for r in results:
        assert r.keys() == want.keys()
        for k in want:
            assert _same(r[k], want[k]), k


@pytest.mark.parametrize('name', FILES)
def test_threads_read_one_file_with_kernels(name):
    """The same with numba's kernels loaded (small reads above may all have
    gone through the zlib path)."""
    fastnc.warm()
    test_threads_read_one_file(name)
