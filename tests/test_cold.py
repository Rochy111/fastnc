"""Reads of files on storage that makes each request dear: compressed chunks
staged in a few big reads, not faulted in through the mapping; the values
must not notice."""
import numpy as np
import pytest

import fastnc
from fastnc import _chunks, _pio

netCDF4 = pytest.importorskip('netCDF4')


@pytest.fixture
def values():
    rng = np.random.default_rng(5)
    return np.round(rng.normal(280, 8, (60, 300, 400)).astype(np.float32), 1)


@pytest.fixture
def path(tmp_path, values):
    f = str(tmp_path / 'cold.nc')
    with netCDF4.Dataset(f, 'w') as d:
        for n, s in zip('tyx', values.shape):
            d.createDimension(n, s)
        v = d.createVariable('T', 'f4', ('t', 'y', 'x'), zlib=True, shuffle=True, chunksizes=(10, 100, 100),
                             fill_value=-999.0)
        v[...] = values
        s = d.createVariable('packed', 'i2', ('t', 'y', 'x'), zlib=True, chunksizes=(20, 150, 200))
        s.scale_factor = 0.01
        s.add_offset = 280.0
        s[...] = values
    return f


@pytest.fixture
def slow(monkeypatch):
    if not _chunks.loaded():
        pytest.skip('without kernels chunks go through _read_python, which does not stage')
    calls = []

    def always(src, addr, threads):
        calls.append(len(addr))
        return True
    monkeypatch.setattr(_pio, 'slow', always)
    return calls


def test_staged_batch_reads_the_same(path, values, slow):
    with fastnc.Dataset(path) as d:
        d.set_auto_maskandscale(False)
        a = d['T'][...]
        b = d.load()
    assert slow
    assert np.array_equal(a, values)
    assert np.array_equal(b['T'], values)


def test_staged_masked_reads_the_same(path, slow):
    with netCDF4.Dataset(path) as d:
        want = d['packed'][...]
    with fastnc.Dataset(path) as d:
        got = d['packed'][...]
    assert slow
    assert got.dtype == want.dtype
    assert np.array_equal(np.ma.getdata(got), np.ma.getdata(want))
    assert np.array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(want))


def test_cached_pages_are_not_slow(path):
    """Pages in the OS cache take microseconds (a thousand threads: a margin
    for a busy machine)."""
    src = np.zeros(1 << 20, np.uint8)
    assert _pio.slow(src, np.arange(12, dtype=np.int64) * 4096, 1000) is False


def test_small_batch_never_probes(path, monkeypatch):
    def never(*a):
        raise AssertionError('probed a small read')
    monkeypatch.setattr(_pio, 'slow', never)
    with fastnc.Dataset(path) as d:
        d['T'][0, 0, :10]


@pytest.mark.parametrize('threads, seconds, want', [(24, 2e-5, False), (24, 1e-2, True), (1, 2e-5, True),
                                                    (1, 2e-6, False)])
def test_threshold_follows_threads(path, threads, seconds, want, monkeypatch):
    """A fault costs staging's copy of 64 kB per thread faulting at once."""
    class Clock:
        now = 0.0

        @classmethod
        def perf_counter(cls):
            cls.now += seconds
            return cls.now
    monkeypatch.setattr(_pio, 'time', Clock)
    src = np.zeros(1 << 20, np.uint8)
    assert _pio.slow(src, np.arange(12, dtype=np.int64) * 4096, threads) is want
