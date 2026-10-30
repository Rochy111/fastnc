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
