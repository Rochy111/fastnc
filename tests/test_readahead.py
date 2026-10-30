"""Readahead across variables (fastnc._readahead): whatever order and mix
of reads a program makes, each read returns what it returns without it.
Thresholds lowered so that the corpus' small variables drive it."""
import glob
import os
import random
import threading

import numpy as np
import pytest

import fastnc
from fastnc import _readahead

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FILES = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, '*.nc'))
               + glob.glob(os.path.join(DATA, '*.h5')))


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(_readahead, 'BIG', 0)
    monkeypatch.setattr(_readahead, 'START', 4096)
    monkeypatch.setattr(_readahead, 'MAX', 1 << 16)


def _enc(x):
    if x is np.ma.masked:
        return 'masked'
    if np.ma.isMaskedArray(x):
        m = x._mask
        return ('ma', x.dtype.str, x.shape, np.ma.getdata(x).tobytes(),
                None if m is np.ma.nomask else np.asarray(m).tobytes(), repr(x.fill_value))
    a = np.asarray(x)
    return (type(x).__name__, a.dtype.str, a.shape,
            repr(a.tolist()) if a.dtype.hasobject else a.tobytes())


def _plan(ds, seed):
    """(group path, variable, key, masked) reads: whole in order, with
    partial reads, repeats and jumps mixed in."""
    rnd = random.Random(seed)
    out = []
    for g in ds._walk():
        names = list(g.variables)
        for i, n in enumerate(names):
            out.append((g.path, n, Ellipsis, True))
            r = rnd.random()
            if r < 0.15:
                out.append((g.path, n, (slice(0, 1),), False))
            elif r < 0.25 and i > 1:
                out.append((g.path, names[rnd.randrange(i)], Ellipsis, False))
            elif r < 0.3:
                out.append((g.path, n, Ellipsis, False))
    return out


def _run(path, reads, on):
    _readahead.ENABLED = on
    try:
        res = []
        with fastnc.Dataset(path) as ds:
            groups = {g.path: g for g in ds._walk()}
            for gp, n, k, m in reads:
                v = groups[gp].variables[n]
                v.set_auto_maskandscale(m)
                try:
                    res.append(_enc(v[k]))
                except Exception as e:
                    res.append(('raised', type(e).__name__, str(e)))
        return res
    finally:
        _readahead.ENABLED = True


@pytest.mark.parametrize('seed', [0, 1])
@pytest.mark.parametrize('name', FILES)
def test_same_values_any_order(name, seed, small):
    path = os.path.join(DATA, name)
    with fastnc.Dataset(path) as ds:
        reads = _plan(ds, seed)
    assert _run(path, reads, True) == _run(path, reads, False)


def test_window_is_used(small, monkeypatch):
    """Reading a file's variables in order is served from the window."""
    monkeypatch.setattr(_readahead, 'START', 1 << 20)
    served = []
    real = _readahead.Readahead.take

    def take(self, v):
        r = real(self, v)
        served.append(r is not None)
        return r
    monkeypatch.setattr(_readahead.Readahead, 'take', take)
    with fastnc.Dataset(os.path.join(DATA, 'idx_latest.h5')) as ds:
        ds.set_auto_maskandscale(False)
        for v in ds.variables.values():
            v[...]
    assert sum(served) >= 3


def test_threads_share_a_group(small):
    """Several threads reading the same group's variables whole at once."""
    path = os.path.join(DATA, 'nc4.nc')
    with fastnc.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        names = list(ds.variables)
        want = {n: _enc(ds.variables[n][...]) for n in names}
    with fastnc.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        bad = []

        def worker(k):
            for rep in range(5):
                for n in names[k % 2::1 + k % 2]:
                    if _enc(ds.variables[n][...]) != want[n]:
                        bad.append(n)
        ts = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    assert not bad, bad[:5]
