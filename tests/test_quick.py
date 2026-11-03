"""Small reads answered by a dataset's own function, and links found by name:
whatever the layout and the index, the answer is the one the whole list or the
general path gives, h5py's to the element."""
import os
import random

import numpy as np
import pytest

import fastnc
from fastnc import _chunks, _hdf5

h5py = pytest.importorskip('h5py')


@pytest.fixture(scope='module')
def path(tmp_path_factory):
    f = str(tmp_path_factory.mktemp('quick') / 'quick.h5')
    rng = np.random.default_rng(5)
    a = rng.integers(-30000, 30000, (40, 30, 20))
    with h5py.File(f, 'w') as h:
        shuffled = dict(compression='gzip', shuffle=True)
        h.create_dataset('deflate', data=a.astype('f4'), chunks=(8, 10, 5), **shuffled)
        h.create_dataset('whole', data=a.astype('i4'), chunks=(40, 30, 20), **shuffled)
        h.create_dataset('rows', data=a.astype('i2'), chunks=(4, 30, 20), **shuffled)
        h.create_dataset('bigzip', data=a.astype('>i4'), chunks=(8, 10, 5), **shuffled)
        h.create_dataset('noshuffle', data=a.astype('i4'), chunks=(8, 10, 5), compression='gzip')
        h.create_dataset('bare', data=a.astype('f8'), chunks=(8, 10, 5))
        h.create_dataset('barebig', data=a.astype('>f4'), chunks=(8, 10, 5))
        h.create_dataset('barefew', data=a.astype('i2'), chunks=(20, 15, 20))     # (under 16 chunks)
        h.create_dataset('flat', data=a.astype('u2'))
        h.create_dataset('flatbig', data=a.astype('>i4'))
        h.create_dataset('tiny', data=np.arange(6, dtype='i4'))                   # (compact below)
        h.create_dataset('one', data=np.arange(50, dtype='f8'), chunks=(7,), compression='gzip', shuffle=True)
        h.create_dataset('long', data=np.arange(7000, dtype='f8'), chunks=(70,), compression='gzip', shuffle=True)
        # chunks not stored: fill values between them
        for name, kw in (('sparse', shuffled), ('baresparse', {})):
            d = h.create_dataset(name, (40, 30), 'i4', chunks=(10, 10), fillvalue=-7, **kw)
            d[12:18, 3:7] = 5
        d = h.create_dataset('lonely', (40, 30), 'i4', chunks=(10, 10), fillvalue=-7)
        d[20:30, 20:30] = np.arange(100).reshape(10, 10)                          # (one chunk, not the first)
    with h5py.File(f, 'r+') as h:
        s = h5py.h5s.create_simple((5,))
        p = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        p.set_layout(h5py.h5d.COMPACT)
        d = h5py.h5d.create(h.id, b'compact', h5py.h5t.NATIVE_INT32, s, dcpl=p)
        d.write(h5py.h5s.ALL, h5py.h5s.ALL, np.arange(5, dtype='i4'))
    return f


def keys(shape, rng, n):
    """Indexes of every kind this path takes or passes on."""
    def one(s):
        k = rng.randrange(9)
        if k < 3:
            return rng.randrange(-s, s)
        if k == 3:
            return np.int64(rng.randrange(s))
        a, b = sorted((rng.randrange(s), rng.randrange(s + 1)))
        if k == 4:
            return slice(None)
        if k == 5:
            return slice(a, b + 1, rng.randrange(1, 4))
        if k == 6:
            return slice(a - s, b)
        return slice(a, b + 1)
    yield ()
    yield slice(None)
    yield 0
    for _ in range(n):
        m = rng.randrange(1, len(shape) + 1)
        k = tuple(one(s) for s in shape[:m])
        yield k[0] if m == 1 and rng.randrange(2) else k


@pytest.mark.parametrize('name', ['deflate', 'whole', 'rows', 'bigzip', 'noshuffle', 'bare', 'barebig', 'barefew',
                                  'flat', 'flatbig', 'tiny', 'one', 'sparse', 'baresparse', 'lonely', 'compact'])
def test_as_h5py(path, name):
    _chunks.release_cache()
    rng = random.Random(name)
    with fastnc.File(path) as f, h5py.File(path, 'r') as h:
        x, y = f[name], h[name]
        for key in keys(x.shape, rng, 300):
            want = y[key]
            for _ in range(3):                      # (through the path, kept, from the cache)
                got = x[key]
                assert type(got) is type(want), key
                assert np.shape(got) == np.shape(want), key
                assert np.asarray(got).dtype == np.asarray(want).dtype.newbyteorder('='), key
                assert np.array_equal(got, want), key


@pytest.fixture(scope='module')
def nc(tmp_path_factory):
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path_factory.mktemp('quicknc') / 'quick.nc')
    rng = np.random.default_rng(9)
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('t', 30)
        d.createDimension('y', 24)
        d.createDimension('x', 16)
        d.createDimension('c', 4)
        a = rng.integers(0, 30000, (30, 24, 16))
        for name, dt, kw, attrs in (
                ('plain', 'f4', dict(zlib=True, shuffle=True, chunksizes=(6, 8, 8)), {}),
                ('fill', 'i2', dict(zlib=True, shuffle=True, chunksizes=(6, 8, 8), fill_value=-1), {}),
                ('packed', 'i2', dict(zlib=True, shuffle=True, chunksizes=(6, 24, 16), fill_value=-1),
                 dict(scale_factor=0.5, add_offset=3.0)),
                ('range', 'i4', dict(chunksizes=(6, 8, 8)), dict(valid_min=100, valid_max=29000, missing_value=7)),
                ('flat', 'f8', dict(contiguous=True), dict(scale_factor=2.0)),
                ('big', 'i4', dict(endian='big', zlib=True, shuffle=True, chunksizes=(6, 8, 8)), {})):
            v = d.createVariable(name, dt, ('t', 'y', 'x'), **kw)
            for k, x in attrs.items():
                v.setncattr(k, np.array(x, 'f4' if isinstance(x, float) else dt))
            v.set_auto_maskandscale(False)
            b = a.copy()
            b[::5, 3] = -1
            v[...] = b.astype(dt)
        v = d.createVariable('uns', 'i2', ('t', 'y', 'x'), zlib=True, shuffle=True, chunksizes=(6, 8, 8))
        v._Unsigned = 'true'
        v.set_auto_maskandscale(False)
        v[...] = (a - 15000).astype('i2')
        v = d.createVariable('chars', 'S1', ('t', 'c'))
        v._Encoding = 'utf-8'
        v.set_auto_chartostring(False)
        v[:] = np.array([list('ab%02d' % i) for i in range(30)], 'S1')
    return p


@pytest.mark.parametrize('name', ['plain', 'fill', 'packed', 'range', 'flat', 'big', 'uns', 'chars'])
def test_as_netcdf4(nc, name):
    netCDF4 = pytest.importorskip('netCDF4')
    _chunks.release_cache()
    rng = random.Random(name)
    with fastnc.Dataset(nc) as a, netCDF4.Dataset(nc) as b:
        x, y = a.variables[name], b.variables[name]
        for key in keys(x.shape, rng, 150):
            if isinstance(key, tuple) and not key:
                continue
            try:
                want = y[key]
            except IndexError:                      # (some empty slices: its own rule)
                continue
            for _ in range(3):
                got = x[key]
                assert type(got) is type(want), key
                assert np.shape(got) == np.shape(want), key
                assert np.asarray(got).dtype == np.asarray(want).dtype.newbyteorder('='), key
                if np.ma.isMaskedArray(want):
                    assert np.array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(want)), key
                    assert np.array_equal(np.ma.getdata(got)[~np.ma.getmaskarray(got)],
                                          np.ma.getdata(want)[~np.ma.getmaskarray(want)]), key
                else:
                    assert np.array_equal(got, want), key


@pytest.mark.parametrize('kw', [{}, {'compression': 'gzip', 'shuffle': True}, {'compression': 'gzip'}],
                         ids=['plain', 'shuffled', 'deflated'])
def test_across_chunks_with_integers_anywhere(tmp_path, kw):
    """Integers before, between and after the slices, for rows that repeat."""
    import h5py
    p = str(tmp_path / 'a.h5')
    x = np.arange(12 * 14 * 16 * 6, dtype='f4').reshape(12, 14, 16, 6)
    with h5py.File(p, 'w') as f:
        f.create_dataset('v', data=x, chunks=(4, 5, 3, 2), **kw)
    keys = [(3, 4), (3,), (3, 4, 5), (-1, 4), (3, Ellipsis, 2), (Ellipsis, 2), (Ellipsis, 2, 1), (slice(None), 4), (slice(None), 4, 5),
            (slice(None), 4, slice(None), 1), (3, slice(1, 9), 5), (3, slice(1, 9), 5, 1), (3, slice(1, 9), -2, slice(1, 5)),
            (slice(2, 7), 4, slice(1, 9), 1), (slice(2, 7), slice(3, 12), 5, slice(1, 5)), (slice(1, 9),), (2, 3, slice(2, 13))]
    with fastnc.File(p) as f:
        v = f['v']
        for _ in range(3):
            for k in keys:
                got = v[k]
                assert got.dtype == x.dtype and got.shape == x[k].shape and np.array_equal(got, x[k]), k
        for i in range(12):                             # (the same slices, a row after another)
            assert np.array_equal(v[i, :, 5], x[i, :, 5])
            assert np.array_equal(v[:, 7, i], x[:, 7, i])
            assert np.array_equal(v[1, :, i, :], x[1, :, i, :])


def test_taken_by_the_function(path):
    """The cases it is here for do not go to the general path."""
    with fastnc.File(path) as f:
        for name, key in (('bare', 3), ('bare', (3, 4, 2)), ('barebig', (3, 4)), ('flat', 7), ('flat', (1, 2, 3)),
                          ('flatbig', 7), ('compact', 2), ('barefew', (1, 2)), ('barefew', 30),
                          ('bare', (slice(0, 20), 3, 2))):              # (the last over chunks)
            x = f[name]
            x[key]
            x[key]                                                      # (small: made by the second)
            assert x._quick and x._quick(key) is not None, (name, key)


def test_results_are_their_own(path):
    """Nothing handed out is a view of the file or of a kept chunk."""
    with fastnc.File(path) as f:
        for name, key in (('flat', 3), ('bare', 3), ('deflate', (1, 2)), ('deflate', (slice(0, 20), 3))):
            x = f[name]
            for _ in range(3):
                got = x[key]
            first = got.copy()
            got[...] = 0
            assert np.array_equal(x[key], first), name


def test_wrong_index_says_so(path):
    with fastnc.File(path) as f:
        x = f['bare']
        x[0]
        for key in (40, -41, (0, 30), (0, 0, 0, 0)):
            with pytest.raises(IndexError):
                x[key]


def test_release_forgets(path):
    with fastnc.File(path) as f:
        x = f['rows']
        for _ in range(3):
            x[1]
        fastnc.release()
        assert x[1].shape == (30, 20)


@pytest.fixture(scope='module')
def series(tmp_path_factory):
    """A chunk a time step: a time series goes through all of them. With a
    missing stretch, big-endian values, and a variable too big to keep."""
    p = str(tmp_path_factory.mktemp('series') / 'series.h5')
    rng = np.random.default_rng(1)
    a = rng.normal(0, 9, (60, 40, 30)).astype('f4')
    with h5py.File(p, 'w') as h:
        z = dict(compression='gzip', shuffle=True, chunks=(1, 40, 30))
        h.create_dataset('t', data=a, **z)
        h.create_dataset('big', data=a.astype('>f8'), **z)
        d = h.create_dataset('holes', a.shape, 'f4', fillvalue=-9, **z)
        d[10:50] = a[10:50]
    return p


@pytest.mark.parametrize('name', ['t', 'big', 'holes'])
def test_series_kept_give_what_they_gave(series, name):
    from fastnc import _hold
    with fastnc.File(series) as f, h5py.File(series, 'r') as h:
        x, y = f[name], h[name]
        for j, i in ((3, 4), (5, 6), (7, 8), (39, 29)):             # (held from the second)
            assert np.array_equal(x[:, j, i], y[:, j, i]), (j, i)
        assert x._reader.index in _hold._held
        for key in ((slice(5, 50, 3), 2, 3), (slice(0, 60), slice(1, 4), 6), (Ellipsis, 3), (7, 2, 3), (7,), (slice(None),)):
            got, want = x[key], y[key]
            assert np.array_equal(got, want) and np.shape(got) == np.shape(want), key
        got = x[:, 1, 1]
        got[:] = 0                                                  # (its own, not the region)
        assert np.array_equal(x[:, 1, 1], y[:, 1, 1])
    assert not _hold._held                                          # (let go with the file)


def test_series_masked_as_netcdf4(tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path / 'm.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('t', 50)
        d.createDimension('y', 20)
        d.createDimension('x', 20)
        v = d.createVariable('v', 'i2', ('t', 'y', 'x'), zlib=True, chunksizes=(1, 20, 20), fill_value=-1)
        v.scale_factor = 0.25
        v.set_auto_maskandscale(False)
        b = np.arange(20000, dtype='i2').reshape(50, 20, 20)
        b[::3, 2, 2] = -1
        v[...] = b
    with fastnc.Dataset(p) as a, netCDF4.Dataset(p) as n:
        x, y = a.variables['v'], n.variables['v']
        for j, i in ((1, 1), (2, 2), (3, 3), (2, 2)):
            got, want = x[:, j, i], y[:, j, i]
            assert np.ma.isMaskedArray(got) and got.dtype == want.dtype
            assert np.array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(want))
            assert np.array_equal(got.compressed(), want.compressed())


@pytest.mark.parametrize('name', ['t', 'big', 'holes'])
def test_time_steps_handed_over(series, name):
    """A chunk decoded ahead goes to the read that wants all of it, with no
    copy: whatever is done to it then is nobody else's business."""
    _chunks.release_cache()
    with fastnc.File(series) as f, h5py.File(series, 'r') as h:
        x, y = f[name], h[name]
        steps = []
        for i in range(60):
            v = x[i]
            assert np.array_equal(v, y[i]) and v.dtype == y[i].dtype.newbyteorder('='), i
            steps.append(v)
        for i, v in enumerate(steps):                               # (none overlaps another)
            assert np.array_equal(v, y[i])
            v[...] = -1
        assert not any(k[1] in x._reader.index.fresh for k in _chunks._CACHE)
        for i in (0, 7, 30, 59, 7):                                 # (and none is what is kept)
            assert np.array_equal(x[i], y[i]), i
    assert not _chunks._CACHE                                       # (let go with the file)


def test_part_of_a_chunk_decoded_ahead_leaves_it(series):
    """A partial read keeps a view of a chunk, which must then stay shared."""
    _chunks.release_cache()
    with fastnc.File(series) as f, h5py.File(series, 'r') as h:
        x, y = f['t'], h['t']
        for i in range(7):
            x[i]
        fresh = sorted(x._reader.index.fresh)
        assert fresh
        i = fresh[0][0]
        assert np.array_equal(x[i, 2], y[i, 2])                     # (partial: now seen)
        v = x[i]                                                    # (all of it: must be a copy)
        v[...] = -1
        assert np.array_equal(x[i, 2], y[i, 2])


@pytest.fixture(scope='module')
def grid(tmp_path_factory):
    p = str(tmp_path_factory.mktemp('grid') / 'g.h5')
    x = np.arange(40 * 120 * 160, dtype='f4').reshape(40, 120, 160)
    with h5py.File(p, 'w') as f:
        f.create_dataset('t', data=x, chunks=(1, 120, 160), compression='gzip', compression_opts=1)
    return p, x


def test_places_near_each_other_keep_a_part(grid):
    """A sixteenth of the grid, not all of it, and every read from it right."""
    from fastnc import _hold
    p, want = grid
    with fastnc.File(p) as f:
        x = f['t']
        for j, i in ((60, 80), (61, 81), (62, 79), (59, 80)):
            assert np.array_equal(x[:, j, i], want[:, j, i])
        h = next(iter(_hold._held.values()))
        assert h.values.nbytes * 8 <= want.nbytes
        assert np.array_equal(x[3:9, 61, 80], want[3:9, 61, 80])


def test_a_read_past_the_edge_grows_what_is_kept(grid):
    from fastnc import _hold
    p, want = grid
    with fastnc.File(p) as f:
        x = f['t']
        for j, i in ((60, 80), (61, 81), (62, 82)):
            assert np.array_equal(x[:, j, i], want[:, j, i])
        small = next(iter(_hold._held.values())).values.nbytes
        for j, i in ((5, 5), (119, 159), (0, 159), (119, 0), (61, 81)):
            assert np.array_equal(x[:, j, i], want[:, j, i])
        assert next(iter(_hold._held.values())).values.nbytes > small


def test_places_far_apart_keep_it_all_at_once(grid):
    from fastnc import _hold
    p, want = grid
    with fastnc.File(p) as f:
        x = f['t']
        for j, i in ((5, 5), (110, 150), (60, 10), (20, 120)):
            assert np.array_equal(x[:, j, i], want[:, j, i])
        assert next(iter(_hold._held.values())).values.nbytes == want.nbytes


def test_rows_keep_it_all(grid):
    """A sweep along an axis would otherwise pass over the chunks once for each
    time the region grew, to save memory only if it stopped early."""
    from fastnc import _hold
    p, want = grid
    with fastnc.File(p) as f:
        x = f['t']
        for j in (60, 62, 61, 63):
            assert np.array_equal(x[:, j, :], want[:, j, :])
        assert next(iter(_hold._held.values())).values.nbytes == want.nbytes


@pytest.mark.parametrize('dt', ['u2', '>u2', 'f4', '>f4', '>f8', 'i1', '>c16'])
@pytest.mark.parametrize('chunks', [(1, 40), (3, 7), (30, 1), (2, 2)], ids=str)
def test_windows_over_plain_chunks(tmp_path, dt, chunks):
    """Copied by a kernel with nothing to plan, in all shapes of chunk."""
    p = str(tmp_path / 'w.h5')
    x = (np.arange(30 * 40).reshape(30, 40) % 251).astype(dt)
    with h5py.File(p, 'w') as f:
        f.create_dataset('x', data=x, chunks=chunks)
    with fastnc.File(p) as f:
        d = f['x']
        d[0:3, 0:3]
        for key in ((slice(2, 29), slice(1, 39)), (slice(None), slice(None)), (5, slice(0, 40)), (slice(0, 30), 7),
                    (slice(0, 30), slice(5, 6)), (slice(-9, -1), slice(-30, None)), (slice(3, 4), slice(0, 40))):
            for _ in range(3):
                got = d[key]
                assert got.dtype == x[key].dtype.newbyteorder('=') and got.shape == x[key].shape
                assert np.array_equal(got, x[key]), key


def test_windows_in_three_dimensions_and_holes(tmp_path):
    p = str(tmp_path / 'w.h5')
    x = np.arange(4 * 9 * 10, dtype='i4').reshape(4, 9, 10)
    with h5py.File(p, 'w') as f:
        f.create_dataset('x', data=x, chunks=(1, 3, 2))
        h = f.create_dataset('h', shape=(20, 20), dtype='f4', chunks=(2, 2), fillvalue=-1)
        h[0:6, 0:6] = 1
    with fastnc.File(p) as f, h5py.File(p) as r:
        d = f['x']
        for key in ((slice(None), slice(1, 8), slice(1, 9)), (2, slice(1, 8), slice(1, 9)), (slice(1, 3), 4, slice(None))):
            d[key]
            assert np.array_equal(d[key], x[key]), key
        e, g = f['h'], r['h']
        for _ in range(3):
            assert np.array_equal(e[1:11, 1:11], g[1:11, 1:11])         # (unwritten chunks: fill)


def test_a_chunk_past_the_end_of_a_cut_file_is_not_read_into_memory(tmp_path):
    p = str(tmp_path / 'c.h5')
    with h5py.File(p, 'w') as f:
        f.create_dataset('x', data=np.arange(4000, dtype='i4').reshape(200, 20), chunks=(1, 20))
    with open(p, 'r+b') as f:
        f.truncate(os.path.getsize(p) - 1000)
    with fastnc.File(p) as f:
        try:
            v = f['x'][0:200, 0:20]
        except (OSError, ValueError, IndexError):
            return
        assert v.shape == (200, 20)


def test_v1_lookups_find_each_chunk_and_no_other(tmp_path):
    p = str(tmp_path / 'b.h5')
    x = np.arange(600 * 600, dtype='i4').reshape(600, 600)
    with h5py.File(p, 'w') as f:
        d = f.create_dataset('x', shape=x.shape, dtype='i4', chunks=(2, 2), fillvalue=-7)
        d[:400, :600] = x[:400]
    with fastnc.File(p) as f:
        d = f['x']
        r = random.Random(0)
        for _ in range(3000):
            y, i = r.randrange(600), r.randrange(600)
            assert d[y, i] == (x[y, i] if y < 400 else -7)


def test_nothing_kept_past_the_budget(series, monkeypatch):
    from fastnc import _arena, _hold
    monkeypatch.setattr(_arena, 'BUDGET', 1 << 10)
    with fastnc.File(series) as f, h5py.File(series, 'r') as h:
        x, y = f['t'], h['t']
        for j in range(4):
            assert np.array_equal(x[:, j, 1], y[:, j, 1])
        assert not _hold._held


@pytest.mark.parametrize('kw', [dict(chunks=(10, 10), compression='gzip'), dict(chunks=(10, 10)), {}])
@pytest.mark.parametrize('opener', ['File', 'Dataset'])
def test_close_lets_go_of_the_file(tmp_path, kw, opener):
    """What still holds a dataset does not hold the file: it can go, here and
    now; and a read after close says why, whatever went through it before."""
    p = str(tmp_path / 'gone.h5')
    with h5py.File(p, 'w') as h:
        h.create_dataset('x', data=np.arange(10000, dtype='i4').reshape(100, 100), **kw)
        h['x'].attrs['a'] = 1
    if opener == 'File':
        f = fastnc.File(p)
        x = f['x']
    else:
        f = fastnc.Dataset(p)
        x = f.variables['x']
    x.set_auto_maskandscale(False) if opener == 'Dataset' else None
    for k in (3, (1, 2), (slice(0, 30), 4)):
        for _ in range(3):
            np.asarray(x[k])
    f.close()
    f.close()                                       # (twice: as before)
    os.remove(p)
    assert not os.path.exists(p)
    with pytest.raises(ValueError):
        x[50]


def test_what_is_kept_goes_when_unused(series, monkeypatch):
    """Output blocks and regions, once nothing has wanted them for a while."""
    import gc
    import time
    from fastnc import _arena, _hold
    with fastnc.File(series) as f:
        x = f['t']
        for j in range(3):
            x[:, j, 1]
        assert _hold._held
        a = _arena.empty((1 << 21,), 'u1')
        del a
        gc.collect()
        assert _arena.held() > 0
        # (only now: with it short all along, a 50 ms stall between the reads,
        # on a loaded machine, let go what the asserts above look for)
        monkeypatch.setattr(_arena, 'IDLE', 0.05)
        time.sleep(0.1)
        _arena._tick()                              # (what the timer calls)
        assert not _hold._held and not _arena.held()
        assert np.array_equal(x[:, 2, 1], x[:, 2, 1])   # (and reads go on)


def test_release_lets_go(series):
    from fastnc import _hold
    with fastnc.File(series) as f:
        x = f['t']
        for j in range(3):
            x[:, j, 1]
        assert _hold._held
        fastnc.release()
        assert not _hold._held
        assert x[:, 3, 1].shape == (60,)


@pytest.fixture(scope='module')
def indexed(tmp_path_factory):
    """Each kind of chunk index that is looked into, with chunks not stored:
    a v1 B-tree, a v2 B-tree, a fixed array, paged and not; filtered or not."""
    p = str(tmp_path_factory.mktemp('index') / 'index.h5')
    a = np.random.default_rng(2).integers(-9, 9, (240, 160)).astype('i4')
    with h5py.File(p, 'w', libver='latest') as h:
        z = dict(compression='gzip', shuffle=True)
        for tag, kw in (('zip', z), ('raw', {})):
            d = h.create_dataset('btree2' + tag, (240, 160), 'i4', chunks=(4, 4), maxshape=(None, None), fillvalue=-99, **kw)
            d[::1, :80] = a[:, :80]
            d = h.create_dataset('array' + tag, (240, 160), 'i4', chunks=(4, 4), maxshape=(None, 160), fillvalue=-99, **kw)
            d[:, :80] = a[:, :80]
            d = h.create_dataset('swapped' + tag, (240, 160), 'i4', chunks=(4, 4), maxshape=(240, None), fillvalue=-99, **kw)
            d[:, :80] = a[:, :80]
            d = h.create_dataset('fixed' + tag, data=a, chunks=(4, 4), **kw)
            d = h.create_dataset('gaps' + tag, (240, 160), 'i4', chunks=(4, 4), fillvalue=-99, **kw)
            d[100:200, 40:90] = a[100:200, 40:90]
    q = p + '.old'
    with h5py.File(q, 'w') as h:
        for tag, kw in (('zip', dict(compression='gzip', shuffle=True)), ('raw', {})):
            d = h.create_dataset('tree' + tag, (240, 160), 'i4', chunks=(4, 4), fillvalue=-99, **kw)
            d[100:200, 40:90] = a[100:200, 40:90]
            h.create_dataset('full' + tag, data=a, chunks=(4, 4), **kw)
    return p, q


NAMES = ['btree2zip', 'btree2raw', 'arrayzip', 'arrayraw', 'swappedzip', 'swappedraw', 'fixedzip', 'fixedraw',
         'gapszip', 'gapsraw']
OLD = ['treezip', 'treeraw', 'fullzip', 'fullraw']


@pytest.mark.parametrize('name', NAMES + OLD)
def test_chunks_found_as_the_table_has_them(indexed, name):
    from fastnc import _h5read
    p = indexed[1 if name in OLD else 0]
    with fastnc.File(p) as f:
        r = f[name]._reader
        i = r.index
        assert isinstance(i, _h5read.LazyIndex) and i.seen == {}
        find = i._find
        found = {}
        for y in range(0, 240, 4):
            for x in range(0, 160, 4):
                got = find((y, x))
                if got is not None:
                    found[y, x] = got
        table = {tuple(o): (a, s, m) for o, a, s, m in zip(i.offsets.tolist(), i.addr.tolist(), i.size.tolist(), i.mask.tolist())}
        assert found == table
        assert i.seen is None                       # (asked for the table: it is made)


@pytest.mark.parametrize('name', NAMES + OLD)
def test_read_by_looking_as_h5py(indexed, name):
    p = indexed[1 if name in OLD else 0]
    rng = random.Random(name)
    _chunks.release_cache()
    with fastnc.File(p) as f, h5py.File(p, 'r') as h:
        x, y = f[name], h[name]
        for key in keys(x.shape, rng, 200):
            if isinstance(key, tuple) and not key:
                continue
            for _ in range(2):
                assert np.array_equal(x[key], y[key]), key
        assert np.array_equal(x[()], y[()])


def test_big_extensible_array(tmp_path):
    """Data blocks in pages, in super blocks: each with a bitmap of its own."""
    p = str(tmp_path / 'ea.h5')
    with h5py.File(p, 'w', libver='latest') as h:
        d = h.create_dataset('x', (6000, 40), 'i4', chunks=(1, 40), maxshape=(None, 40), compression='gzip')
        d[...] = np.arange(240000, dtype='i4').reshape(6000, 40)
    with fastnc.File(p) as f, h5py.File(p, 'r') as h:
        x, y = f['x'], h['x']
        for k in (0, 5999, 3000, (slice(2990, 3010), 3)):
            assert np.array_equal(x[k], y[k]), k
        assert np.array_equal(x[()], y[()])


def test_page_bits_of_many_data_blocks(tmp_path):
    """Super blocks of many data blocks of several pages: bits that run on from
    one data block to the next, some set, some not, each chunk where h5py says."""
    p = str(tmp_path / 'bits.h5')
    n = 300000
    with h5py.File(p, 'w', libver='latest') as h:
        d = h.create_dataset('x', (n,), 'i4', chunks=(1,), maxshape=(None,), fillvalue=-5)
        d[:150000] = np.arange(150000)
        d[220000:220100] = 8
        d[n - 50:] = 9
    with fastnc.File(p) as f, h5py.File(p, 'r') as h:
        x, y = f['x'], h['x']
        r = x._reader
        find = r.f.chunk_finder(r.ds)
        for i in list(range(0, n, 977)) + [149999, 150000, 220050, n - 1]:
            got = find((i,))
            info = y.id.get_chunk_info_by_coord((i,))
            assert (got[0] if got else None) == info.byte_offset, i
        assert np.array_equal(x[()], y[()])
        i = r.index
        assert len(i.addr) == y.id.get_num_chunks()


@pytest.mark.parametrize('maxshape', [None, (None, 40)])
def test_pages_never_written_are_fill_values(tmp_path, maxshape):
    """Their elements are not on disk: not chunks at address zero."""
    p = str(tmp_path / 'pages.h5')
    with h5py.File(p, 'w', libver='latest') as h:
        d = h.create_dataset('x', (6000, 40), 'i4', chunks=(1, 40), maxshape=maxshape, fillvalue=-5)
        d[5500:5510] = 7
        d[10:20] = 8
    with fastnc.File(p) as f, h5py.File(p, 'r') as h:
        x, y = f['x'], h['x']
        for k in (slice(None), 3000, 15, 5505, slice(2990, 3010), slice(5490, 5520)):
            assert np.array_equal(x[k], y[k]), k


@pytest.mark.parametrize('kw', [dict(compression='gzip'), dict(compression='gzip', shuffle=True),
                                dict(compression='gzip', shuffle=True, fletcher32=True)])
@pytest.mark.parametrize('dt', ['i4', '>i4', 'f8', 'i1'])
def test_pixels_here_and_there(tmp_path, kw, dt):
    """Small chunks decoded one at a time, as h5py gives them: whatever the
    filters, the byte order, one read after another out of order; and a chunk
    not stored."""
    p = str(tmp_path / 'p.h5')
    a = np.random.default_rng(4).integers(-100, 100, (400, 300))
    with h5py.File(p, 'w') as h:
        d = h.create_dataset('x', (400, 300), dt, chunks=(5, 5), fillvalue=-3, **kw)
        d[:300] = a[:300].astype(dt)
    rng = random.Random(0)
    with fastnc.File(p) as f, h5py.File(p, 'r') as h:
        x, y = f['x'], h['x']
        for _ in range(600):
            k = (rng.randrange(400), rng.randrange(300))
            got, want = x[k], y[k]
            assert got == want and got.dtype == want.dtype.newbyteorder('='), k
        for k in ((slice(10, 13), slice(10, 13)), (7,), (390,), (slice(295, 305), 2)):
            assert np.array_equal(x[k], y[k]), k


@pytest.mark.parametrize('kw', [{}, dict(compression='gzip'), dict(compression='gzip', shuffle=True),
                                dict(compression='lzf'), dict(fletcher32=True)])
@pytest.mark.parametrize('dt', ['>i4', '>f8', '>u2', '<i4'])
def test_fill_values_in_the_order_of_the_data(tmp_path, kw, dt):
    """Chunks never written, beside chunks that were: what is big-endian on
    disk comes out native, the fill value with it."""
    p = str(tmp_path / 'f.h5')
    with h5py.File(p, 'w') as h:
        d = h.create_dataset('x', (40, 30), dt, chunks=(10, 10), fillvalue=3, **kw)
        d[10:20] = 7
    with fastnc.File(p) as f, h5py.File(p, 'r') as h:
        x, y = f['x'], h['x']
        for k in ((), (35, 5), (slice(5, 15),), (slice(None), 3), (slice(0, 10),)):
            for _ in range(4):                      # (through the cache, too)
                assert np.array_equal(x[k], y[k]), k
    _chunks.release_cache()
    with fastnc.File(p) as f, h5py.File(p, 'r') as h:
        assert np.array_equal(f['x'][()], h['x'][()])


@pytest.mark.parametrize('kw', [dict(compression='gzip'), dict(compression='gzip', shuffle=True)])
@pytest.mark.parametrize('dt', ['i4', '>i4', '>f8', 'u1'])
@pytest.mark.parametrize('shape,chunks', [((500,), (500,)), ((500,), (200,)), ((30, 40), (30, 40)), ((30, 40), (17, 40)),
                                          ((30, 40), (16, 21)), ((1,), (1,))])
def test_small_datasets_whole(tmp_path, kw, dt, shape, chunks):
    """Whole small datasets decoded on the spot, edges of chunks and all; with
    chunks not stored, the general path."""
    p = str(tmp_path / 's.h5')
    a = np.random.default_rng(6).integers(0, 200, shape)
    with h5py.File(p, 'w') as h:
        h.create_dataset('x', data=a.astype(dt), chunks=chunks, **kw)
        d = h.create_dataset('gap', shape, dt, chunks=chunks, fillvalue=5, **kw)
        d[(0,) * len(shape)] = 1
    with fastnc.File(p) as f, h5py.File(p, 'r') as h, fastnc.Dataset(p) as n:
        for k in ('x', 'gap'):
            want = h[k][()]
            got = f[k][()]
            assert np.array_equal(got, want) and got.dtype == want.dtype.newbyteorder('=') and got.shape == want.shape, k
            assert np.array_equal(f[k][()], want), k
            assert np.array_equal(n.variables[k][...], want), k


@pytest.mark.parametrize('preload', [False, True])
def test_masked_in_place_as_xarray(tmp_path, preload):
    """Big floats with fill values through xarray: what its own coders give,
    masked directly in what was read, or in a copy of what is kept."""
    xr = pytest.importorskip('xarray')
    netCDF4 = pytest.importorskip('netCDF4')
    from fastnc import _xarray
    p = str(tmp_path / 'x.nc')
    rng = np.random.default_rng(8)
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('y', 700)
        d.createDimension('x', 600)
        a = rng.normal(0, 1, (700, 600)).astype('f4')
        a[::7, ::3] = -9
        a[::11, ::5] = 1e20
        a[::13, ::4] = np.nan
        v = d.createVariable('both', 'f4', ('y', 'x'), zlib=True, chunksizes=(70, 600), fill_value=-9)
        v.missing_value = np.float32(1e20)
        v.set_auto_maskandscale(False)
        v[...] = a
        v = d.createVariable('one', 'f8', ('y', 'x'), fill_value=-9.0)
        v.set_auto_maskandscale(False)
        v[...] = a.astype('f8')
        v = d.createVariable('scaled', 'f4', ('y', 'x'), fill_value=-9)
        v.scale_factor = np.float32(2)
        v.set_auto_maskandscale(False)
        v[...] = a
    with xr.open_dataset(p, engine='netcdf4') as w, xr.open_dataset(p, engine=_xarray.FastncBackendEntrypoint, preload=preload) as g:
        for name in ('both', 'one', 'scaled'):
            for _ in range(2):                      # (a second read: nothing kept was changed)
                x, y = g[name].values, w[name].values
                assert x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True), name
            assert np.array_equal(g[name].isel(x=3).values, w[name].isel(x=3).values, equal_nan=True), name


def test_a_damaged_chunk_is_reported(tmp_path):
    p = str(tmp_path / 'bad.h5')
    with h5py.File(p, 'w') as h:
        d = h.create_dataset('x', data=np.random.default_rng(0).integers(0, 9, (300, 300)), chunks=(5, 5),
                             compression='gzip')
        at = d.id.get_chunk_info(70).byte_offset
    with open(p, 'r+b') as o:
        o.seek(at)
        o.write(b'\xff\xff\xff\xff')
    with fastnc.File(p) as f:
        with pytest.raises(Exception):
            f['x'][(70 // 60) * 5, (70 % 60) * 5]


def test_first_read_does_not_read_the_index(indexed):
    with fastnc.File(indexed[1]) as f:
        for name in OLD:
            x = f[name]
            x[50, 50]
            x[101:103, 41:43]
            assert x._reader.index.seen is not None, name


def test_few_chunks_planned_the_same(path):
    """Plain integers or arrays, the same plan, whatever the box."""
    rng = random.Random(11)
    fields = ('addr', 'size', 'mask', 'cstart', 'count', 'ostart', 'needend')
    tried = 0
    with fastnc.File(path) as f:
        for name in ('deflate', 'whole', 'rows', 'bare', 'barefew', 'sparse', 'baresparse', 'lonely', 'one'):
            r = f[name]._reader
            index, shape = r.index, r.ds.shape
            for _ in range(400):
                sel = []
                for n in shape:
                    a, b = sorted((rng.randrange(n), rng.randrange(n + 1)))
                    sel.append((a, b + 1, rng.choice((1, 1, 1, 2, 3))))
                if rng.randrange(3) == 0:
                    sel = [(s, e, 1) for s, e, _ in sel]
                rows = index.near(sel)
                off = index.offsets if rows is None else index.offsets[rows]
                if len(off) > 8 or not (index.complete or all(s[2] == 1 for s in sel)):
                    continue
                a = _chunks._few_plan(index, shape, sel, off, rows)
                b = _chunks._many_plan(index, shape, sel, off, rows)
                for k in fields:
                    x, y = getattr(a, k), getattr(b, k)
                    assert x.dtype == y.dtype and x.shape == y.shape, (name, sel, k)
                    assert np.array_equal(x, y), (name, sel, k)
                assert a.fill_needed == b.fill_needed, (name, sel)
                tried += 1
    assert tried > 500


def test_walking_decodes_ahead(path, monkeypatch):
    """Going along a dataset, the chunks decoded at a time keep growing: a
    run of them does not start over at the end of each."""
    _chunks.release_cache()
    calls = []
    real = _chunks.read_batch

    def count(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(_chunks, 'read_batch', count)
    with fastnc.File(path) as f:
        x = f['long']
        y = np.array([x[i] for i in range(7000)])
    assert np.array_equal(y, np.arange(7000, dtype='f8'))
    assert len(calls) < 20                          # (of 100 chunks)


@pytest.mark.parametrize('libver', ['earliest', 'latest'])
@pytest.mark.parametrize('order', [False, True])
@pytest.mark.parametrize('count', [3, 7, 40, 1200])
def test_links_by_name(tmp_path, libver, order, count):
    """Symbol tables, links in the header, dense links, both B-tree depths:
    each name found alone as in the list of all, and none that is not there."""
    p = str(tmp_path / 'links.h5')
    with h5py.File(p, 'w', libver=libver, track_order=order) as h:
        for i in range(count):
            h['v%05d' % (i * 11 % count)] = i
        h['soft'] = h5py.SoftLink('/v00001')
        h['ext'] = h5py.ExternalLink('other.h5', '/x')
        h['g'] = h5py.Empty('f4')
    f = fastnc.File(p)._f
    whole = _hdf5.Group(f, f.root.address, '/').links
    g = _hdf5.Group(f, f.root.address, '/')
    for k, v in whole.items():
        assert g.link(k) == v, k
    for k in ('nothing', 'A', 'zzzz', 'v', 'v000000'):
        assert g.link(k) is None, k
    assert g._links is None                         # (and it read no list to say so)


def test_name_not_ascii(tmp_path):
    p = str(tmp_path / 'u.h5')
    with h5py.File(p, 'w') as h:
        h['café'] = 1
    with fastnc.File(p) as f:
        assert f['café'][()] == 1
        assert 'cafè' not in f


def test_dense_attribute_by_hash(tmp_path):
    p = str(tmp_path / 'a.h5')
    with h5py.File(p, 'w', libver='latest', track_order=True) as h:
        d = h.create_dataset('x', data=1)
        for i in range(60):
            d.attrs['a%03d' % i] = i
    f = fastnc.File(p)._f
    b = fastnc.File(p)
    x = b['x']
    for i in range(60):
        assert f.find_attribute(x._header, 'a%03d' % i) is not None
    assert f.find_attribute(x._header, 'a999') is None
    assert list(x.attrs) == ['a%03d' % i for i in range(60)]
