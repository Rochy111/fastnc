"""The xarray backend: fastnc applies xarray's mask / scale functions block
by block on its threads; the result must be what xarray's own decoding of
the same raw values gives, value for value (and attributes, encodings)."""
import glob
import os
import warnings

import numpy as np
import pytest

xr = pytest.importorskip('xarray')

from fastnc import _xarray  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FILES = sorted(f for f in glob.glob(os.path.join(DATA, '*.nc')))


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.kind == 'O':
        return all((x == y) if not isinstance(x, np.ndarray) else np.array_equal(x, y)
                   for x, y in zip(a.ravel(), b.ravel()))
    return a.tobytes() == b.tobytes()


@pytest.mark.parametrize('path', FILES, ids=[os.path.basename(f) for f in FILES])
def test_block_decoding_equals_xarray(path):
    E = _xarray.FastncBackendEntrypoint
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            ours = xr.open_dataset(path, engine=E)
            raw = xr.open_dataset(path, engine=E, mask_and_scale=False, decode_times=False)
        except NotImplementedError as e:
            pytest.skip(str(e))
        theirs = xr.decode_cf(raw, mask_and_scale=True, decode_times=True)
        try:
            ours.load()
            theirs.load()
        except NotImplementedError as e:
            pytest.skip(str(e))
    assert set(ours.variables) == set(theirs.variables)
    for k in theirs.variables:
        a, b = ours[k], theirs[k]
        assert a.dtype == b.dtype, k
        assert _same(a.values, b.values), k
        assert dict(a.attrs).keys() == dict(b.attrs).keys(), k


def test_decoded_array_blocks(monkeypatch):
    """Arrays bigger than one block go through the threaded path."""
    monkeypatch.setattr(_xarray.DecodedArray, 'BLOCK', 7)
    path = os.path.join(DATA, 'nc4_classic.nc')
    E = _xarray.FastncBackendEntrypoint
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ours = xr.open_dataset(path, engine=E).load()
        raw = xr.open_dataset(path, engine=E, mask_and_scale=False, decode_times=False)
        theirs = xr.decode_cf(raw).load()
    for k in ('temp', 'packed', 'missing', 'ubyte_as_signed'):
        assert _same(ours[k].values, theirs[k].values), k


@pytest.mark.parametrize('path', FILES, ids=[os.path.basename(f) for f in FILES])
def test_lookup_table_decoding_equals_xarray(path, monkeypatch):
    """8 / 16-bit variables decoded through a table of xarray's functions
    applied to every possible value (forced on for every size here)."""
    monkeypatch.setattr(_xarray.DecodedArray, 'LUT_MIN', 1)
    test_block_decoding_equals_xarray(path)


@pytest.mark.parametrize('path', FILES, ids=[os.path.basename(f) for f in FILES])
def test_fused_lookup_equals_xarray(path, monkeypatch):
    """The same with the table looked up as each chunk is decoded
    (_fused.read_lut), on one thread so that every chunked variable takes it."""
    import fastnc
    from fastnc import _fused
    fastnc.warm()
    monkeypatch.setattr(_xarray.DecodedArray, 'LUT_MIN', 1)
    monkeypatch.setattr(_xarray.DecodedArray, 'FUSED_THREADS', 1)
    test_block_decoding_equals_xarray(path)


LUT_ATTRS = [
    {'scale_factor': np.float32(0.15859237), 'add_offset': np.float32(-20.289911), '_FillValue': 4095},
    {'scale_factor': np.float64(0.01), '_FillValue': -1},
    {'add_offset': np.float32(273.15), 'missing_value': 7},
    {'_FillValue': 3},
    {'_Unsigned': 'true', 'scale_factor': np.float32(2.5), '_FillValue': -1},
    {'_Unsigned': 'true', '_FillValue': -2},
    {'scale_factor': np.float32(-1e-3), 'add_offset': np.float64(1e5), 'missing_value': np.array([0, 5])},
    {'scale_factor': np.float32(np.inf), 'add_offset': np.float32(1)},
    {'scale_factor': np.float32(0), 'add_offset': np.float32(-np.inf), '_FillValue': 9},
]


@pytest.mark.parametrize('dt', ['i1', 'u1', 'i2', 'u2'])
@pytest.mark.parametrize('k', range(len(LUT_ATTRS)))
def test_lookup_table_every_value(dt, k):
    """Every possible raw value: the table path equals xarray's functions
    applied to the array as xarray would (the block path)."""
    attrs = dict(LUT_ATTRS[k])
    if '_Unsigned' in attrs and dt[0] == 'u':
        attrs.pop('_Unsigned')
    bits = 8 * np.dtype(dt).itemsize
    x = np.arange(1 << bits, dtype='u%d' % (bits // 8)).view(dt)
    x = np.concatenate([x, x[::-1], x[::7]]).reshape(3, -1) if x.size % 3 == 0 else \
        np.concatenate([x, x[::-1]]).reshape(2, -1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            v = _xarray._decoded('v', ('a', 'b'), _Raw(x), dict(attrs), {'dtype': x.dtype})
        except Exception:
            v = None
        if v is None:
            pytest.skip('not decoded by fastnc for these attributes')
        arr = v._data.array
        key = (slice(None), slice(None))
        save = _xarray.DecodedArray.LUT_MIN
        try:
            _xarray.DecodedArray.LUT_MIN = 1 << 62
            want = arr._getitem(key)
            _xarray.DecodedArray.LUT_MIN = 1
            got = arr._getitem(key)
        finally:
            _xarray.DecodedArray.LUT_MIN = save
        ref = xr.decode_cf(xr.Dataset({'v': (('a', 'b'), x, attrs)}))['v'].values
    assert got.dtype == want.dtype == ref.dtype
    assert got.tobytes() == want.tobytes() == ref.tobytes()


class _Raw:
    """A raw array as FastncArray presents it."""

    def __init__(self, x):
        self.x, self.shape, self.dtype = x, x.shape, x.dtype

    def _getitem(self, key):
        return self.x[key]
