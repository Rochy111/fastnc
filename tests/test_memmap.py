"""memmap(): a variable's stored values viewed in place, without a copy,
where it lies in one uncompressed piece; equal to the raw read everywhere it
is given, refused (ValueError) everywhere else."""
import glob
import os

import numpy as np
import pytest

import fastnc
from fastnc import _h5py

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NC = sorted(glob.glob(os.path.join(DATA, '*.nc'))) + sorted(glob.glob(os.path.join(DATA, 'hdf4', '*.hdf')))
H5 = sorted(glob.glob(os.path.join(DATA, '*.h5'))) + [os.path.join(DATA, 'nc4.nc')]


def _equal(view, raw):
    raw = np.asarray(raw)
    if view.dtype.names:                    # (field by field: an aligned record's padding is not values)
        return view.shape == raw.shape and all(_equal(view[n], raw[n]) for n in view.dtype.names)
    if view.dtype.kind in 'SV':
        return view.shape == raw.shape and view.astype(raw.dtype).tobytes() == raw.tobytes()
    return view.shape == raw.shape and np.array_equal(view, raw, equal_nan=view.dtype.kind == 'f')


def _walk(g):
    yield from g.variables.values()
    for s in g.groups.values():
        yield from _walk(s)


@pytest.mark.parametrize('path', NC, ids=os.path.basename)
def test_dataset_memmap(path):
    given = 0
    with fastnc.Dataset(path) as d:
        for v in _walk(d):
            try:
                m = v.memmap()
            except ValueError:
                continue
            given += 1
            assert not m.flags.writeable
            assert _equal(m, v.read(mask=False, scale=False)), v.name
            if m.ndim and m.shape[0] > 1:
                assert _equal(m[1:, ...], v.read(slice(1, None), mask=False, scale=False)), v.name
    if os.path.basename(path).startswith(('nc3', 'layout')):
        assert given                                        # (netCDF-3: every variable lies so)


@pytest.mark.parametrize('path', H5, ids=os.path.basename)
def test_file_memmap(path):
    with fastnc.File(path) as f:
        names = []
        f.visititems(lambda n, o: names.append(n) if isinstance(o, _h5py.Dataset) else None)
        for n in names:
            d = f[n]
            try:
                m = d.memmap()
            except ValueError:
                continue
            assert d.chunks is None and not m.flags.writeable
            assert m.dtype.newbyteorder('=') == d.dtype.newbyteorder('=') or d.dtype.kind == 'V', n
            assert _equal(m, d[()]), n


def test_netcdf3_every_variable_and_after_close():
    p = os.path.join(DATA, 'nc3_classic.nc')
    d = fastnc.Dataset(p)
    views = {}
    for name, v in d.variables.items():
        views[name] = (v.memmap(), v.read(mask=False, scale=False))
    assert any(v.isunlimited() for v in d.variables.values())     # (record variables, strided)
    d.close()
    for name, (m, raw) in views.items():
        assert _equal(m, raw), name                                  # (still readable: the mapping stays)
    with pytest.raises(ValueError):
        d.variables['temp'].memmap()


def test_isunlimited_on_netcdf3():
    with fastnc.Dataset(os.path.join(DATA, 'nc3_classic.nc')) as d:
        for v in d.variables.values():
            assert v.isunlimited() == any(d.dimensions[n].isunlimited() for n in v.dimensions), v.name
        assert any(v.isunlimited() for v in d.variables.values())


def test_refused_where_compressed():
    with fastnc.Dataset(os.path.join(DATA, 'nc4.nc')) as d:
        v = d.variables['temp']
        if v.filters().get('zlib'):
            with pytest.raises(ValueError):
                v.memmap()
