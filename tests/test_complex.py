"""Dataset(path, auto_complex=True): complex numbers stored as a compound of
two floats {r, i} or as a last dimension of length 2 (`ri`, `complex`) read
as numpy complex arrays, as netCDF4-python 1.7 reads them; against the stored
parts read without it, and against netCDF4-python where installed."""
import os
import pickle

import numpy as np
import pytest

import fastnc

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'complex')
CONV = os.path.join(DATA, 'conventions.nc')
H5PY = os.path.join(DATA, 'h5py.h5')
COMPLEX = {'dim_fill': 'c8', 'dim_scaled': 'c16', 'cmp_float': 'c8', 'cmp_double': 'c16', 'dim_scalar': 'c8',
           'dim_short': 'c8', 'not_prefix': 'c8'}       # (a last dimension 'r': the start of 'ri' counts)
# nc-complex takes these too, wrongly: a complex dimension not last (netCDF4-python then
# drops the last dimension, and reads the wrong values) and an integer variable (which
# it then cannot read)
NOT_TAKEN = ('not_first', 'not_int_dim')
KEYS = [Ellipsis, 0, -1, slice(None, None, 2), [0, 1], (slice(1, None), Ellipsis)]


def _keys(shape):
    return [Ellipsis] if not shape else [k for k in KEYS if not isinstance(k, list) or shape[0] > 1]


def _same(a, b):
    if np.asarray(a).dtype.names:
        return type(a) is type(b) and np.asarray(a).tolist() == np.asarray(b).tolist()
    return (type(a) is type(b) and np.asarray(a).dtype == np.asarray(b).dtype
            and np.array_equal(np.ma.getmaskarray(a), np.ma.getmaskarray(b)) and bool(np.ma.allequal(a, b)))


def _parts(v, key):
    """The complex values of plain variable v (auto_complex off) for key,
    put together here from the stored parts."""
    if v.dtype.names:
        s = np.asarray(v.read(key, mask=False, scale=False))
        return s[v.dtype.names[0]] + 1j * s[v.dtype.names[1]].astype(s[v.dtype.names[0]].dtype)
    k = key if isinstance(key, tuple) else (key,)
    p = np.asarray(v.read(k + (slice(None),) if Ellipsis not in k else k, mask=False, scale=False))
    return p[..., 0] + 1j * p[..., 1]


@pytest.mark.parametrize('path', [CONV, H5PY], ids=os.path.basename)
def test_values_as_the_stored_parts(path):
    with fastnc.Dataset(path) as plain, fastnc.Dataset(path, auto_complex=True) as d:
        assert d.auto_complex and not plain.auto_complex
        for name, v in d.variables.items():
            p = plain.variables[name]
            if v.dtype.kind != 'c':
                assert v.dtype == p.dtype and v.shape == p.shape, name
                continue
            assert name in COMPLEX or path == H5PY, name
            assert v.auto_complex
            if p.dtype.names is None:
                assert v.shape == p.shape[:-1] and v.dimensions == p.dimensions[:-1], name
            for key in _keys(v.shape):
                raw = np.asarray(v.read(key, mask=False, scale=False))
                want = _parts(p, key)
                assert raw.dtype == v.dtype and raw.dtype.isnative, name
                assert np.array_equal(raw, want.astype(v.dtype)), (name, key)


def test_types_and_shapes():
    with fastnc.Dataset(CONV, auto_complex=True) as d:
        for name, dt in COMPLEX.items():
            assert d[name].dtype == np.dtype(dt), name
        for name in ('not_ints', 'not_order', 'not_mixed') + NOT_TAKEN:
            assert d[name].dtype.kind != 'c', name
        assert d['dim_fill'].dimensions == ('t', 'x') and d['dim_fill'].chunking() == [1, 3]
        assert d['dim_scalar'].shape == () and d['dim_scalar'][...] == 1 + 2j
        t = d['cmp_double'].datatype
        assert t.name == 'complex' and t.dtype.names == ('f0', 'f1')
        assert d['dim_scaled'].datatype == np.dtype('c16')
    with fastnc.Dataset(CONV) as d:
        assert d['cmp_float'].dtype.names == ('Real', 'Imag')       # (off: the compound as stored)


def test_masked_and_scaled_as_netcdf4():
    with fastnc.Dataset(CONV, auto_complex=True) as d:
        a = d['dim_fill'][:]                        # (records past those written: fill values)
        assert a.shape == (3, 3) and np.array_equal(a[2].data, np.full(3, -9 - 9j, 'c8'))
        assert np.array_equal(d['dim_scaled'][:], (np.arange(0, 6, 2) * 2 + 1) + 1j * (np.arange(1, 6, 2) * 2))
        c = d['cmp_float'][:]                       # (a compound: never masked)
        assert type(c) is np.ndarray and np.array_equal(c, [1 + 2j, 3 + 4j, 5 + 6j])


def test_against_netcdf4():
    netCDF4 = pytest.importorskip('netCDF4')
    if not hasattr(netCDF4.Dataset, 'auto_complex'):
        pytest.skip('netCDF4-python before 1.7')
    for path in (CONV, H5PY):
        a, b = netCDF4.Dataset(path, auto_complex=True), fastnc.Dataset(path, auto_complex=True)
        try:
            for name, x in a.variables.items():
                if name in NOT_TAKEN:
                    continue
                y = b[name]
                assert (x.dtype, x.dimensions, x.shape) == (y.dtype, y.dimensions, y.shape), name
                for key in _keys(x.shape):
                    for ms in (True, False):
                        x.set_auto_maskandscale(ms)
                        y.set_auto_maskandscale(ms)
                        assert _same(x[key], y[key]), (name, key, ms)
        finally:
            a.close()
            b.close()


@pytest.mark.parametrize('dim,taken', [('r', True), ('ri', True), ('c', True), ('comp', True), ('complex', True),
                                       ('_pfnc_complex', True), ('R', False), ('rx', False), ('complexes', False),
                                       ('i', False), ('_pfnc', True), ('_complex', False)])
def test_dimension_names_as_nc_complex(tmp_path, dim, taken):
    """Every start of 'complex', 'ri' or '_pfnc_complex', case counting, as nc-complex
    compares them; against netCDF4-python where it has auto_complex."""
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path / 'names.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('x', 3)
        d.createDimension(dim, 2)
        v = d.createVariable('v', 'f8', ('x', dim))
        v[:] = np.arange(6.0).reshape(3, 2)
        w = d.createVariable('w', 'f8', (dim, 'x'))      # (not last: never)
        w[:] = np.arange(6.0).reshape(2, 3)
    with fastnc.Dataset(p, auto_complex=True) as b:
        assert (b['v'].dtype.kind == 'c') == taken
        assert b['w'].dtype == np.dtype('f8')
        if taken:
            assert b['v'].dimensions == ('x',) and np.array_equal(b['v'][:], [1j, 2 + 3j, 4 + 5j])
        if hasattr(netCDF4.Dataset, 'auto_complex'):
            with netCDF4.Dataset(p, auto_complex=True) as a:
                x, y = a['v'], b['v']
                assert (x.dtype, x.dimensions, x.shape) == (y.dtype, y.dimensions, y.shape)
                assert _same(x[:], y[:])


def test_other_reads():
    with fastnc.Dataset(CONV, auto_complex=True) as d:
        got = d.load()
        for name, v in d.variables.items():
            assert _same(got[name], v[...]), name
        out = np.empty((3, 3), 'c8')
        assert d['dim_fill'].read(out=out, mask=False) is out
        assert np.array_equal(out, d['dim_fill'].read(mask=False))
        assert np.array_equal(d['cmp_float'].read_async().result(), d['cmp_float'][:])
        with pytest.raises(ValueError):
            d['cmp_float'].memmap()                 # (the stored values are not complex ones)
        e = pickle.loads(pickle.dumps(d))
        assert e.auto_complex and e['cmp_double'].dtype == np.dtype('c16')
        e.close()
        assert d.tocdl() == fastnc.Dataset(CONV).tocdl()  # (the file's header, as ncdump prints it)
