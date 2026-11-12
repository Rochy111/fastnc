"""fastnc.Dataset's netCDF4-python methods besides reading: isopen(),
get_variables_by_attributes(), get_dims(), get_fill_value(), quantization(),
getValue() and the mask / scale / always_mask / chartostring attributes."""
import os
import warnings

import numpy as np
import pytest

import fastnc

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FILES = ['nc4.nc', 'nc3_classic.nc', 'nc4_classic.nc', 'nc3_cdf5.nc', 'hdf4/types.hdf']


def test_isopen_called_and_not():
    d = fastnc.Dataset(os.path.join(DATA, 'nc4.nc'))
    assert d.isopen() is True and d.isopen
    d.close()
    assert d.isopen() is False and not d.isopen


def test_switches_as_attributes():
    with fastnc.Dataset(os.path.join(DATA, 'nc4.nc')) as d:
        v = d.variables['packed']
        assert (v.mask, v.scale, v.always_mask, v.chartostring) == (True, True, True, True)
        v.set_auto_mask(False)
        v.set_always_mask(False)
        assert (v.mask, v.scale, v.always_mask) == (False, True, False)


def test_get_dims_and_get_value():
    with fastnc.Dataset(os.path.join(DATA, 'nc4.nc')) as d:
        v = d.variables['temp']
        dims = v.get_dims()
        assert [x.name for x in dims] == list(v.dimensions)
        assert all(x is d.dimensions[x.name] for x in dims)
        deep = d['sub/deep'] if 'deep' in d['sub'].groups else None
        if deep is not None:
            for w in deep.variables.values():
                assert [x.name for x in w.get_dims()] == list(w.dimensions)
        s = d.variables['scalar']
        assert s.getValue() == s[...]
        with pytest.raises(IndexError):
            v.getValue()


@pytest.mark.parametrize('fn', FILES + ['groups_new.h5', 'filters_hdf/nbit.h5', 'complex/anonymous.h5'])
def test_repr_as_netcdf4(fn):
    """print() of datasets, groups, variables and dimensions as netCDF4-python
    prints them (byte order aside: fastnc's dtypes are native)."""
    netCDF4 = pytest.importorskip('netCDF4')
    import re
    p = os.path.join(DATA, fn)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a, b = netCDF4.Dataset(p), fastnc.Dataset(p)
    try:
        def walk(g):
            yield g
            for s in g.groups.values():
                yield from walk(s)
        for ga, gb in zip(walk(a), walk(b)):
            pairs = [(ga, gb)] + [(ga.variables[n], gb.variables[n]) for n in ga.variables] + \
                [(ga.dimensions[n], gb.dimensions[n]) for n in ga.dimensions]
            for x, y in pairs:
                want = re.sub(r'>([iuf][1248])\b', lambda m: str(np.dtype('<' + m.group(1))),
                              repr(x).replace('netCDF4.', 'fastnc.'))
                assert repr(y) == want
    finally:
        a.close()
        b.close()


def test_hidden_as_netcdf_c(tmp_path):
    """Datasets netCDF-c does not show: HDF5 array types, and compounds of a
    compound or enum it has no named type for (h5py's bool is one); the
    same through a lone lookup as through the whole list."""
    h5py = pytest.importorskip('h5py')
    p = str(tmp_path / 'hidden.h5')
    pair = np.dtype([('q', '<i4'), ('r', '<u2')])
    with h5py.File(p, 'w') as f:
        f.create_dataset('array', (3,), dtype=np.dtype(('<i2', (2,))))
        f['bools'] = np.zeros(3, [('a', '?'), ('b', 'i4')])
        f['nested'] = np.zeros(3, [('a', 'i4'), ('n', pair)])
        f['plain'] = np.zeros(3, [('a', 'i4'), ('e', '<u4', (3,)), ('s', 'S3')])
        g = f.create_group('g')
        g['named'] = pair
        g['nested_named'] = np.zeros(3, [('a', 'i4'), ('n', pair)])
    shown = {'/': ['plain'], '/g': ['nested_named']}
    with fastnc.Dataset(p) as d:
        for name in ('array', 'bools', 'nested', 'plain'):
            assert (name in d.variables) == (name in shown['/'])     # (a lone lookup first)
        assert list(d.variables) == shown['/'] and list(d['g'].variables) == shown['/g']
    with fastnc.Dataset(p) as d:
        with pytest.raises(IndexError):
            d['array']


def test_filter_queries_and_chunk_cache():
    with fastnc.Dataset(os.path.join(DATA, 'nc4.nc')) as d:
        assert d.has_blosc_filter() and d.has_zstd_filter() and d.has_bzip2_filter() and d.has_szip_filter()
        assert d['sub'].has_zstd_filter()
        v = d.variables['temp']
        assert v.set_var_chunk_cache(size=1 << 20, nelems=521, preemption=0.75) is None
        assert np.ma.allequal(v[:], fastnc.Dataset(os.path.join(DATA, 'nc4.nc'))['temp'][:])


def test_get_variables_by_attributes():
    with fastnc.Dataset(os.path.join(DATA, 'nc4.nc')) as d:
        with_units = d.get_variables_by_attributes(units=lambda u: u is not None)
        assert with_units and all('units' in v.ncattrs() for v in with_units)
        assert d.get_variables_by_attributes(name='temp') == [d.variables['temp']]
        assert d.get_variables_by_attributes(nope='x') == []


@pytest.mark.parametrize('fn', FILES)
def test_as_netcdf4(fn):
    netCDF4 = pytest.importorskip('netCDF4')
    p = os.path.join(DATA, fn)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a, b = netCDF4.Dataset(p), fastnc.Dataset(p)
        try:
            for sel in ({'units': lambda u: u is not None}, {'units': lambda u: u is None},
                        {'axis': 'X'}, {'_FillValue': lambda f: f is not None}):
                assert [v.name for v in a.get_variables_by_attributes(**sel)] == \
                    [v.name for v in b.get_variables_by_attributes(**sel)], sel
            for n, va in a.variables.items():
                vb = b.variables[n]
                assert [x.name for x in va.get_dims()] == [x.name for x in vb.get_dims()]
                fa, fb = va.get_fill_value(), vb.get_fill_value()
                if getattr(fa, 'dtype', np.dtype('=i1')).byteorder != '>':   # (netCDF4 swaps HDF4's)
                    assert repr(fa) == repr(fb), n
                if fn.endswith('.nc'):
                    assert va.quantization() == vb.quantization(), n
                if not va.dimensions:
                    assert repr(va.getValue()) == repr(vb.getValue()), n
        finally:
            a.close()
            b.close()
