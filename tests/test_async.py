"""read_async / load_async: the same results as the reads they stand for,
errors raised from result(), many at once, a file closed meanwhile."""
import os
import threading

import numpy as np
import pytest

import fastnc

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
WAIT = 900      # (a read's first use of a codec compiles it, minutes on a cold CI runner; a hang still ends)


def same(a, b):
    if type(a) is not type(b) or np.asarray(a).dtype != np.asarray(b).dtype:
        return False
    ma, mb = np.ma.getmaskarray(a), np.ma.getmaskarray(b)
    return np.array_equal(ma, mb) and np.asarray(np.ma.getdata(a))[~ma].tobytes() == \
        np.asarray(np.ma.getdata(b))[~mb].tobytes()


@pytest.mark.parametrize('name', ['nc4.nc', 'nc3_classic.nc', 'hdf4/comp.hdf'])
def test_read_async_as_read(name):
    with fastnc.Dataset(os.path.join(DATA, name)) as d:
        futs = []
        for v in d.variables.values():
            if v.shape and isinstance(v.datatype, np.dtype):
                for key, kw in ((Ellipsis, {}), ((slice(1, None),), {}), (Ellipsis, {'mask': False}),
                                (Ellipsis, {'nan': True})):
                    futs.append((v, key, kw, v.read_async(key, **kw)))
        for v, key, kw, f in futs:
            assert same(f.result(timeout=WAIT), v.read(key, **kw)), (v.name, key, kw)
        got = d.load_async().result(timeout=WAIT)
        want = d.load()
        assert list(got) == list(want)
        assert all(same(got[k], want[k]) for k in got
                   if isinstance(want[k], np.ndarray) and want[k].dtype.kind in 'iufbS')


def test_errors_come_from_result():
    with fastnc.Dataset(os.path.join(DATA, 'nc4.nc')) as d:
        f = d.variables['temp'].read_async((99,))
        with pytest.raises(IndexError):
            f.result(timeout=WAIT)


def test_file_read_async():
    with fastnc.File(os.path.join(DATA, 'layouts.h5')) as f:
        names = [n for n in f if hasattr(f[n], 'shape') and f[n].shape]
        futs = [(n, f[n].read_async()) for n in names]
        for n, fu in futs:
            assert np.array_equal(fu.result(timeout=WAIT), f[n][()], equal_nan=True), n


def test_closed_while_reading():
    """A file closed with reads of it in flight: each read either answers or
    raises ValueError (closed), never anything worse."""
    for name in ('nc4.nc', 'nc3_classic.nc', 'hdf4/comp.hdf'):
        for _ in range(20):
            d = fastnc.Dataset(os.path.join(DATA, name))
            futs = [v.read_async() for v in d.variables.values() if v.shape][:8]
            d.close()
            for f in futs:
                try:
                    f.result(timeout=WAIT)
                except ValueError as e:
                    assert 'closed' in str(e), (name, e)
    for _ in range(20):
        f5 = fastnc.File(os.path.join(DATA, 'layouts.h5'))
        futs = [f5[n].read_async() for n in ('contiguous', 'bigendian', 'compact')]
        f5.close()
        for f in futs:
            try:
                f.result(timeout=WAIT)
            except ValueError as e:
                assert 'closed' in str(e), e


def test_closed_under_a_read_whatever_it_raised(monkeypatch):
    """A read the file was closed under says so, whatever went wrong inside
    (numba typing the file's bytes once let go raised its own TypingError,
    on CI); and no kernel is handed bytes that are gone."""
    from fastnc import _chunks
    with pytest.raises(ValueError):
        _chunks.read_batch([], None, None)

    class Odd(Exception):
        pass
    d = fastnc.Dataset(os.path.join(DATA, 'nc4.nc'))
    v = d.variables['temp']

    def boom(*a, **k):
        d.close()
        raise Odd('typing None')
    monkeypatch.setattr(type(v), '_values', boom)
    with pytest.raises(ValueError, match='closed'):
        v.read()
    f5 = fastnc.File(os.path.join(DATA, 'layouts.h5'))
    ds = f5['contiguous']

    def boom5(*a, **k):
        f5.close()
        raise Odd('typing None')
    monkeypatch.setattr(type(ds), '_read', boom5)
    with pytest.raises(ValueError, match='closed'):
        ds[()]


def test_workers_are_fastnc_threads():
    with fastnc.Dataset(os.path.join(DATA, 'nc4.nc')) as d:
        name = d.variables['temp'].read_async().result() is not None and \
            [t.name for t in threading.enumerate() if t.name.startswith('fastnc-read')]
        assert name
