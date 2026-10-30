"""Masked / scaled reads fused with the decoding (fastnc._fused) give what
the usual read then _maskscale gives -- values, mask, fill value, types --
for every variable of the corpus, whole and in windows, strides and with
integer indexes (the size threshold lowered so that small arrays take it)."""
import glob
import os
import warnings

import numpy as np
import pytest

import fastnc
from fastnc import _dataset, _fused, _maskscale

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FILES = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, '*.nc'))
               + glob.glob(os.path.join(DATA, '*.h5')))


def _keys(shape):
    out = [Ellipsis]
    if len(shape) >= 1 and shape[0] > 3:
        out += [(slice(1, None, 2),), (slice(2, -1),)]
    if len(shape) >= 2 and shape[0] > 2 and shape[1] > 4:
        out += [(slice(1, -1), slice(2, None, 3)), (1,), (slice(None), 2)]
    if len(shape) >= 3:
        out.append((0, slice(1, None), slice(None, None, 2)))
    return out


def _outcome(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            r = fn()
        except Exception as e:
            return ('raised', type(e).__name__, str(e)), [str(x.message) for x in w]
    if np.ma.isMaskedArray(r):
        m = r._mask
        enc = ('ma', r.dtype.str, r.shape, np.ma.getdata(r).tobytes(),
               None if m is np.ma.nomask else m.tobytes(), type(r.fill_value).__name__,
               np.asarray(r.fill_value).tobytes())
    else:
        a = np.asarray(r)
        enc = (type(r).__name__, a.dtype.str, a.shape,
               repr(a.tolist()) if a.dtype.hasobject else a.tobytes())
    return enc, [str(x.message) for x in w]


@pytest.mark.parametrize('threads', [1, 3])
@pytest.mark.parametrize('name', FILES)
def test_fused_equals_usual(name, threads, monkeypatch):
    fastnc.warm()
    monkeypatch.setattr(_maskscale, 'FAST_MIN_SIZE', 1)
    bad = []
    with fastnc.Dataset(os.path.join(DATA, name)) as ds:
        for g in ds._walk():
            for vname, v in g.variables.items():
                for k in _keys(v.shape):
                    monkeypatch.setattr(_dataset, 'FUSED', False)
                    want = _outcome(lambda: v.read(k, threads=threads))
                    monkeypatch.setattr(_dataset, 'FUSED', True)
                    got = _outcome(lambda: v.read(k, threads=threads))
                    if got != want:
                        bad.append((g.path, vname, k))
    assert not bad, bad[:5]


def _attr_sets(v, x):
    """Mask / scale attributes for a variable of x's values (some of them hit)."""
    dt = x.dtype
    f4, f8 = np.float32, np.float64
    flat = x.reshape(-1)
    a, b = (flat[3], flat[-2]) if flat.size > 3 else (flat[0], flat[0])
    lo, hi = (min(a, b), max(a, b))
    sets = [{'scale_factor': f4(0.5)},
            {'scale_factor': f8(0.1), 'add_offset': f8(3)},
            {'scale_factor': f4(2), 'add_offset': f4(-1), '_FillValue': dt.type(a)},
            {'add_offset': f4(1.5), 'missing_value': dt.type(b)},
            {'scale_factor': f4(0.25), 'valid_range': np.array([lo, hi], dt)},
            {'scale_factor': f8(3), 'valid_min': dt.type(lo), '_FillValue': dt.type(b)}]
    if dt.kind == 'i':
        sets.append({'_Unsigned': 'true', 'scale_factor': f4(1.5), '_FillValue': dt.type(a)})
    return sets


@pytest.mark.parametrize('name', FILES)
def test_fused_attribute_combinations(name, monkeypatch):
    """Every chunked numeric variable of the corpus (its types, chunkings,
    filters, byte orders) under each of a set of mask / scale attributes."""
    fastnc.warm()
    monkeypatch.setattr(_maskscale, 'FAST_MIN_SIZE', 1)
    calls = []
    real = _fused._run
    monkeypatch.setattr(_fused, '_run', lambda *a: calls.append(1) or real(*a))
    bad = []
    with fastnc.Dataset(os.path.join(DATA, name)) as ds:
        for g in ds._walk():
            for vname, v in g.variables.items():
                bv = v._bv
                r = getattr(bv, '_reader', None)
                if (r is None or not isinstance(bv.dtype, np.dtype) or bv.dtype.kind not in 'iuf'
                        or r.ds.layout.cls != 2 or not v.shape):
                    continue
                v.set_auto_maskandscale(False)
                try:
                    x = v[...]
                except Exception:
                    continue
                v.set_auto_maskandscale(True)
                if x.size == 0:
                    continue
                keep = bv._attrs
                try:
                    for attrs in _attr_sets(v, np.asarray(x)):
                        bv._attrs = dict(attrs)
                        for k in _keys(v.shape):
                            for th in (1, 2):
                                monkeypatch.setattr(_dataset, 'FUSED', False)
                                want = _outcome(lambda: v.read(k, threads=th))
                                monkeypatch.setattr(_dataset, 'FUSED', True)
                                got = _outcome(lambda: v.read(k, threads=th))
                                if got != want:
                                    bad.append((g.path, vname, sorted(attrs), k, th))
                finally:
                    bv._attrs = keep
    assert not bad, bad[:5]
    if name in ('nc4.nc', 'nc4_classic.nc', 'filters.h5', 'idx_latest.h5'):
        assert calls                                # (these files have such variables)


@pytest.mark.parametrize('name', FILES)
def test_load_with_fused_equals_plain(name, monkeypatch):
    """Group.load() taking big scaled variables through the fused path
    (every size here) gives what it gives without."""
    fastnc.warm()
    monkeypatch.setattr(_maskscale, 'FAST_MIN_SIZE', 1)
    monkeypatch.setattr(_dataset, 'FUSED_LOAD_MIN', 0)
    path = os.path.join(DATA, name)
    out = {}
    for fused in (False, True):
        monkeypatch.setattr(_dataset, 'FUSED', fused)
        with fastnc.Dataset(path) as ds:
            try:
                got = ds.load(recursive=True, threads=2)
            except Exception as e:
                got = {'raised': type(e).__name__}
            out[fused] = {k: _outcome(lambda v=v: v)[0] for k, v in got.items()}
    assert out[True] == out[False]


@pytest.mark.parametrize('name', FILES)
def test_read_lut_equals_lookup_of_raw(name):
    """_fused.read_lut (table looked up while decoding) == the table looked
    up in the raw read, for every chunked 8 / 16-bit variable and selection."""
    fastnc.warm()
    rng = np.random.default_rng(3)
    ran = 0
    bad = []
    with fastnc.Dataset(os.path.join(DATA, name)) as ds:
        for g in ds._walk():
            for vname, v in g.variables.items():
                bv = v._bv
                if not isinstance(bv.dtype, np.dtype) or bv.dtype.kind not in 'iu' \
                        or bv.dtype.itemsize > 2 or not v.shape:
                    continue
                n = 1 << (8 * bv.dtype.itemsize)
                for lut in (rng.integers(0, 1 << 32, n, dtype=np.uint32).view(np.float32),
                            rng.integers(0, 1 << 63, n, dtype=np.uint64).view(np.float64)):
                    for k in _keys(v.shape):
                        v.set_auto_maskandscale(False)
                        got = _fused.read_lut(v, k if isinstance(k, tuple) else (k,), lut,
                                              threads=1, min_size=1)
                        if got is None:
                            continue
                        ran += 1
                        raw = np.asarray(v[k])
                        want = lut.view('u%d' % lut.itemsize)[raw.view('u%d' % raw.itemsize)]
                        if got.shape != raw.shape or got.view(want.dtype).tobytes() != want.tobytes():
                            bad.append((g.path, vname, k))
    assert not bad, bad[:5]
    if name in ('nc4.nc', 'idx_latest.h5', 'filters.h5'):
        assert ran
