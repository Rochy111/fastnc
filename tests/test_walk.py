"""Going through a dataset a piece at a time: the chunk grid, reads answered
from decoded chunks with no plan, chunks decoded ahead of a read, h5py-style
loops with readahead. Whatever the order, each read gives what it gives alone."""
import glob
import os
import random

import numpy as np
import pytest

import fastnc
from fastnc import _chunks, _pool, _readahead

netCDF4 = pytest.importorskip('netCDF4')
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FILES = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, '*.nc')) + glob.glob(os.path.join(DATA, '*.h5')))


@pytest.fixture(scope='module')
def path(tmp_path_factory):
    """Rows of a chunk, a chunk a row, chunks of two dimensions, a missing
    stretch (fill values), and big-endian values."""
    f = str(tmp_path_factory.mktemp('walk') / 'walk.nc')
    rng = np.random.default_rng(3)
    with netCDF4.Dataset(f, 'w') as d:
        d.createDimension('t', 120)
        d.createDimension('y', 90)
        d.createDimension('x', 70)
        for name, chunks, dt, kw in (('rows', (8, 90, 70), 'f4', {}), ('one', (1, 90, 70), 'i2', {}),
                                     ('tiles', (8, 30, 20), 'f8', {}), ('big', (8, 45, 70), 'i4', {'endian': 'big'})):
            v = d.createVariable(name, dt, ('t', 'y', 'x'), zlib=True, shuffle=True, chunksizes=chunks, **kw)
            v.set_auto_maskandscale(False)
            v[...] = np.round(rng.normal(100, 20, (120, 90, 70)))
        v = d.createVariable('holes', 'f4', ('t', 'y', 'x'), zlib=True, shuffle=True, chunksizes=(8, 90, 70),
                             fill_value=-1.0)
        v[20:60] = 5.0
        for name, kw, dt in (('bare', dict(chunksizes=(8, 30, 20)), 'f4'), ('bare1', dict(chunksizes=(1, 90, 70)), 'u2'),
                             ('barebig', dict(chunksizes=(8, 30, 20), endian='big'), 'i4'),
                             ('flat', dict(contiguous=True), 'i2'), ('flatbig', dict(contiguous=True, endian='big'), 'f8')):
            v = d.createVariable(name, dt, ('t', 'y', 'x'), **kw)
            v.set_auto_maskandscale(False)
            v[...] = np.round(rng.normal(1000, 200, (120, 90, 70)))
        v = d.createVariable('barehole', 'i4', ('t', 'y', 'x'), chunksizes=(8, 30, 20), fill_value=-7)
        v[16:32, 30:60, 20:40] = 11
    return f


NAMES = ('rows', 'one', 'tiles', 'big', 'holes', 'bare', 'bare1', 'barebig', 'flat', 'flatbig', 'barehole')


def keys(n, how):
    if how == 'forward':
        return list(range(n))
    if how == 'backward':
        return list(range(n - 1, -1, -1))
    if how == 'every 7th':
        return list(range(0, n, 7))
    if how == 'random':
        return random.Random(5).sample(range(n), n)
    return [k for i in range(0, n, 16) for k in range(min(i + 3, n) - 1, i - 1, -1)]    # to and fro


@pytest.mark.parametrize('how', ['forward', 'backward', 'every 7th', 'random', 'to and fro'])
@pytest.mark.parametrize('name', NAMES)
def test_rows_in_any_order(path, name, how):
    _chunks.release_cache()
    with netCDF4.Dataset(path) as d:
        d.set_auto_maskandscale(False)
        want = d[name][...]
    with fastnc.File(path) as f:
        v = f[name]
        dtype = v[0].dtype
        for k in keys(want.shape[0], how):
            a = v[k]
            assert a.dtype == dtype                 # (cached or not)
            assert np.array_equal(a, want[k]), k


@pytest.mark.parametrize('name', NAMES)
def test_rows_cut_across(path, name):
    """Reads going along another dimension, and of parts of chunks."""
    _chunks.release_cache()
    with netCDF4.Dataset(path) as d:
        d.set_auto_maskandscale(False)
        want = d[name][...]
    with fastnc.Dataset(path) as d:
        d.set_auto_maskandscale(False)
        v = d[name]
        for j in range(0, 90, 3):
            assert np.array_equal(v[:, j], want[:, j]), j
            assert np.array_equal(v[3, j, 2:60:3], want[3, j, 2:60:3]), j
        for k in range(0, 112, 4):
            assert np.array_equal(v[k:k + 8, 5:40, ::2], want[k:k + 8, 5:40, ::2]), k


@pytest.mark.parametrize('name', NAMES)
def test_inside_one_chunk_at_every_offset(path, name):
    """Every kind of small read, near the edges of chunks and of the data."""
    with netCDF4.Dataset(path) as d:
        d.set_auto_maskandscale(False)
        want = d[name][...]
    keys = [(0, 0, 0), (119, 89, 69), (7, 29, 19), (8, 30, 20), (5, Ellipsis), (119,), (slice(0, 8),),
            (slice(112, 120),), (3, slice(0, 30), slice(0, 20)), (3, slice(29, 31), slice(19, 21)),
            (slice(1, 8, 3), slice(0, 30, 7), slice(2, 20, 5)), (slice(0, 1), slice(5, 6), slice(7, 8)),
            (slice(8, 16), slice(30, 60), slice(20, 40)), (slice(16, 32), slice(30, 60), slice(20, 40)),
            (4, 10, slice(None)), (slice(None), 10, 10), (slice(116, 120), 89)]
    with fastnc.File(path) as f:
        v = f[name]
        for k in keys:
            a = v[k]
            assert np.array_equal(a, want[k]), k
            assert np.shape(a) == np.shape(want[k]), k
    with fastnc.Dataset(path) as d:
        d.set_auto_maskandscale(False)
        v = d[name]
        for k in keys:
            assert np.array_equal(v[k], want[k]), k


def test_uncompressed_reads_take_the_short_way(path):
    with fastnc.File(path) as f:
        r = f['bare']._reader
        s = np.zeros((1, 30, 20), np.float32)
        assert _chunks._plain(r.ds, r.index, r.src, [(3, 4, 1), (0, 30, 1), (0, 20, 1)], s)
        # (more than a chunk, or filters: the long way)
        assert not _chunks._plain(r.ds, r.index, r.src, [(7, 9, 1), (0, 30, 1), (0, 20, 1)], np.zeros((2, 30, 20)))
        z = f['rows']._reader
        assert not _chunks._plain(z.ds, z.index, z.src, [(3, 4, 1), (0, 90, 1), (0, 70, 1)], np.zeros((1, 90, 70)))
        h = f['barehole']._reader                   # (a chunk never written)
        assert not _chunks._plain(h.ds, h.index, h.src, [(0, 1, 1), (0, 1, 1), (0, 1, 1)], np.zeros((1, 1, 1)))


def test_chunks_are_decoded_ahead(path):
    _chunks.release_cache()
    with fastnc.File(path) as f:
        v = f['one']
        for k in range(12):
            v[k]
        assert max(key[1][0] for key in _chunks._CACHE) > 11    # (chunks not read yet)
        assert _chunks._CACHE_HELD[0] == sum(h.nbytes for h in _chunks._CACHE.values())
    _chunks.release_cache()


def test_random_reads_decode_nothing_ahead(path):
    _chunks.release_cache()
    with fastnc.File(path) as f:
        v = f['rows']
        keys = (3, 100, 40, 77, 12)
        for k in keys:
            v[k]
        C = v._reader.index.cshape
        assert {g[0] for _, g, _ in _chunks._CACHE} <= {k // C[0] for k in keys}
    _chunks.release_cache()


def test_cache_stays_within_budget(path, monkeypatch):
    monkeypatch.setattr(_chunks, 'CACHE_BYTES', 1 << 21)
    _chunks.release_cache()
    with fastnc.File(path) as f:
        v = f['one']
        for k in range(120):
            v[k]
        assert _chunks._CACHE_HELD[0] <= _chunks._hw.scaled(1 << 21)
    _chunks.release_cache()


@pytest.mark.parametrize('name', NAMES)
def test_grid_table_finds_the_same_chunks(path, name):
    with fastnc.File(path) as f:
        r = f[name]._reader
        index, shape = r.index, r.ds.shape
        for sel in ([(0, 8, 1), (0, 90, 1), (0, 70, 1)], [(3, 4, 1), (10, 20, 1), (5, 6, 1)],
                    [(5, 90, 7), (1, 80, 3), (2, 69, 5)], [(119, 120, 1), (89, 90, 1), (69, 70, 1)]):
            if index.near(sel) is None:
                continue
            plain = index.near
            index.near = lambda s: None
            whole = _chunks._plan(index, shape, sel)
            index.near = plain
            mine = _chunks._plan(index, shape, sel)
            assert sorted(whole.addr.tolist()) == sorted(mine.addr.tolist())
            order = np.argsort(whole.addr), np.argsort(mine.addr)
            for a in ('size', 'needend'):
                assert np.array_equal(getattr(whole, a)[order[0]], getattr(mine, a)[order[1]])
            for a in ('cstart', 'count', 'ostart'):
                assert np.array_equal(getattr(whole, a)[order[0]], getattr(mine, a)[order[1]])
            assert whole.fill_needed == mine.fill_needed


@pytest.mark.parametrize('name', FILES)
def test_h5py_loops_read_the_same(name, monkeypatch):
    """A visititems loop and a loop over a group, readahead on (and made eager)
    and off: the same values and types."""
    monkeypatch.setattr(_readahead, 'BIG', 0)
    p = os.path.join(DATA, name)
    with open(p, 'rb') as f:
        if f.read(3) == b'CDF':
            pytest.skip('netCDF-3: no HDF5')

    def read():
        got = {}
        with fastnc.File(p) as f:
            def visit(n, o):
                if isinstance(o, fastnc._h5py.Dataset):
                    got[n] = o[()]
            f.visititems(visit)
            for k in f:
                o = f[k]
                if isinstance(o, fastnc._h5py.Dataset):
                    got['again ' + k] = o[...]
        return got
    with_it = read()
    monkeypatch.setattr(_readahead, 'ENABLED', False)
    without = read()
    assert with_it.keys() == without.keys()
    for k, a in with_it.items():
        b = without[k]
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype, k
            if a.dtype.kind not in 'OV':
                assert a.tobytes() == b.tobytes(), k


def test_a_mistaken_pattern_stops_reading_ahead(path, monkeypatch):
    monkeypatch.setattr(_readahead, 'BIG', 0)
    with fastnc.File(path) as f:
        for name in ('rows', 'one', 'tiles', 'big', 'holes', 'one', 'rows', 'big', 'tiles', 'rows', 'one', 'big'):
            f[name][()]
        ahead = next(iter(f._aheads.values()))
        assert ahead.missed <= _readahead.MISSES + 1


def test_threads_for_memory_work():
    _pool._CORES, _pool._WORK = 8, 1 << 15
    assert _pool.memory_threads(1 << 20, 24) == 1
    assert _pool.memory_threads(1 << 24, 24) == 8               # (16 MB: as the bytes say, 8)
    assert _pool.memory_threads(1 << 30, 24) == 8               # one bus
    assert _pool.memory_threads(1 << 30, 3) == 3
    assert _pool.memory_threads(1 << 20, 24, 8) == 4            # (work that does more per byte)
    _pool._CORES = _pool._WORK = 0
