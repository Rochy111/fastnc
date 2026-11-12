"""What Variable.read takes besides a key: per-read mask / scale / nan in
place of the variable's settings, `out=` (values written into an array of
the caller's), and set_auto_nan (masked floats as NaN in plain arrays).
Each is checked against the plain reads it stands for, which
test_references holds to netCDF4-python's."""
import os
import pickle
import threading
import warnings

import numpy as np
import pytest

import fastnc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
FILES = ['nc4.nc', 'nc3_classic.nc', 'nc3_cdf5.nc', 'hdf4/comp.hdf', 'hdf4/layout.hdf', 'hdf4/types.hdf']
KEYS = [Ellipsis, (slice(1, 4),), (0,), (slice(None, None, 2), slice(3, 9)), (-1, slice(2, 7, 3)),
        ([0, 2],)]


def variables(d):
    def walk(g):
        for v in g.variables.values():
            yield v
        for s in g.groups.values():
            yield from walk(s)
    return [v for v in walk(d) if v.shape and isinstance(v.datatype, np.dtype) and v.dtype.kind in 'iuf']


def same(a, b):
    """Equal results: type, dtype, mask and the values not masked."""
    if type(a) is not type(b) or np.asarray(a).dtype != np.asarray(b).dtype:
        return False
    ma, mb = np.ma.getmaskarray(a), np.ma.getmaskarray(b)
    da, db = np.asarray(np.ma.getdata(a)), np.asarray(np.ma.getdata(b))
    return np.array_equal(ma, mb) and da[~ma].tobytes() == db[~mb].tobytes()


def fits(v, key):
    try:
        v[key]
        return True
    except IndexError:
        return False


@pytest.fixture(params=FILES)
def ds(request):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with fastnc.Dataset(os.path.join(DATA, request.param)) as d:
            yield d


def test_per_read_flags_are_the_setters(ds):
    for v in variables(ds):
        for key in KEYS:
            if not fits(v, key):
                continue
            want = {}
            for m in (True, False):
                for s in (True, False):
                    v.set_auto_mask(m)
                    v.set_auto_scale(s)
                    want[m, s] = v[key]
            v.set_auto_maskandscale(True)
            for (m, s), w in want.items():
                assert same(v.read(key, mask=m, scale=s), w), (v.name, key, m, s)
            assert v._mask and v._scale                     # (the variable as it was)


def test_per_read_flags_on_threads():
    """A raw read and a masked one of the same variable at once, many times."""
    with fastnc.Dataset(os.path.join(DATA, 'nc4.nc')) as d:
        v = d.variables['packed']
        raw, masked = v.read(mask=False, scale=False), v[...]
        bad = []

        def run(i):
            for _ in range(200):
                if i % 2:
                    r = v.read(mask=False, scale=False)
                    ok = type(r) is np.ndarray and r.tobytes() == raw.tobytes()
                else:
                    ok = same(v[...], masked)
                if not ok:
                    bad.append(i)
        ts = [threading.Thread(target=run, args=(i,)) for i in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert not bad


def test_nan(ds):
    for v in variables(ds):
        for key in KEYS:
            if not fits(v, key):
                continue
            m = v[key]
            got = v.read(key, nan=True)
            if m is np.ma.masked:                           # (one value, masked)
                want_nan = v.read(key, mask=False).dtype.kind == 'f'
                assert (isinstance(got, np.floating) and np.isnan(got)) if want_nan else got is m
            elif isinstance(m, np.ma.MaskedArray) and m.dtype.kind == 'f':
                assert type(got) is np.ndarray and got.dtype == m.dtype, (v.name, key)
                mask = np.ma.getmaskarray(m)
                assert np.array_equal(np.isnan(got) & ~np.isnan(np.ma.getdata(m)), mask & ~np.isnan(np.ma.getdata(m)))
                assert np.isnan(got[mask]).all()
                assert got[~mask].tobytes() == np.ma.getdata(m)[~mask].tobytes()
            else:
                assert same(got, m), (v.name, key)          # (integers, plain arrays: unchanged)
            assert same(v.read(key, nan=True, mask=False), v.read(key, mask=False))


def test_nan_switches_and_load():
    p = os.path.join(DATA, 'nc4.nc')
    with fastnc.Dataset(p) as d:
        v = d.variables['packed']
        assert isinstance(v[...], np.ma.MaskedArray)
        v.set_auto_nan(True)
        a = v[...]
        assert type(a) is np.ndarray and np.isnan(a).sum() == 4
        v.set_auto_nan(False)
        d.set_auto_nan(True)
        assert type(d.variables['missing'][...]) is np.ndarray
        assert isinstance(d.variables['ubyte_as_signed'][...], np.ma.MaskedArray)     # (integers)
        got = d.load(['packed', 'missing', 'temp'])
        assert all(type(x) is np.ndarray for x in got.values())
        assert np.isnan(got['missing']).sum() == 3
        back = pickle.loads(pickle.dumps(d))
        try:
            assert back._auto_nan and type(back.variables['missing'][...]) is np.ndarray
        finally:
            back.close()
        bv = pickle.loads(pickle.dumps(d.variables['missing']))
        try:
            assert type(bv[...]) is np.ndarray
        finally:
            bv.group().close()


def test_file_switches_survive_pickling():
    with fastnc.Dataset(os.path.join(DATA, 'nc4.nc')) as d:
        d.set_auto_maskandscale(False)
        back = pickle.loads(pickle.dumps(d))
        try:
            assert back.variables['packed'][...].dtype == np.int16    # (raw, as set)
        finally:
            back.close()


def test_nan_of_one_masked_value():
    with fastnc.Dataset(os.path.join(DATA, 'nc4.nc')) as d:
        v = d.variables['missing']
        m = v[...]
        i, j = np.argwhere(np.ma.getmaskarray(m))[0]
        assert v[i, j] is np.ma.masked
        x = v.read((int(i), int(j)), nan=True)
        assert isinstance(x, np.float64) and np.isnan(x)


def test_out(ds):
    for v in variables(ds):
        for key in KEYS:
            if not fits(v, key):
                continue
            for flags in ({'mask': False, 'scale': False}, {}):
                want = v.read(key, **flags)
                if want is np.ma.masked or np.ndim(want) == 0:
                    continue
                data = np.ma.getdata(want)
                out = np.full(data.shape, 7, data.dtype)
                got = v.read(key, out=out, **flags)
                assert same(got, want), (v.name, key, flags)
                if isinstance(want, np.ma.MaskedArray):
                    assert np.shares_memory(got, out) or out.size == 0
                else:
                    assert got is out
                assert out.tobytes() == np.asarray(data).tobytes() or isinstance(want, np.ma.MaskedArray)


def test_out_refused():
    with fastnc.Dataset(os.path.join(DATA, 'nc4.nc')) as d:
        v = d.variables['packed']
        with pytest.raises(ValueError):
            v.read(out=np.empty(v.shape, np.float64))       # (the read gives float32)
        with pytest.raises(ValueError):
            v.read(out=np.empty((2, 2), np.float32))
        frozen = np.empty(v.shape, np.int16)
        frozen.flags.writeable = False
        with pytest.raises(TypeError):
            v.read(out=frozen, mask=False, scale=False)
        with pytest.raises(ValueError):
            v.read(out=np.empty(v.shape, np.int16), device='gpu')


def test_out_is_where_the_decoder_writes():
    """A raw read of a box goes into out with no array of its own."""
    from fastnc import _arena
    with fastnc.Dataset(os.path.join(DATA, 'nc4.nc')) as d:
        v = d.variables['temp']
        out = np.empty(v.shape, v.dtype)
        v.read(out=out, mask=False)
        made = []
        real = _arena.empty
        _arena.empty = lambda *a, **k: made.append(a) or real(*a, **k)
        try:
            assert v.read(out=out, mask=False) is out
        finally:
            _arena.empty = real
        assert not made
