"""MFDataset against netCDF4-python's MFDataset on files it wrote, and the
module-level helpers (default_fillvals, chartostring, num2date ...)."""
import os
import pickle
import warnings

import numpy as np
import pytest

import fastnc

netCDF4 = pytest.importorskip('netCDF4')

KEYS = [Ellipsis, slice(None), 0, -1, 7, slice(3, 17), slice(None, None, 3), slice(25, 5, -2),
        [0, 4, 9, 23], np.arange(30) % 2 == 0, (slice(2, 22), 1), (slice(None), slice(1, 3)),
        (12, slice(None, None, -1)), (Ellipsis, 2)]


@pytest.fixture(scope='module')
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp('mf')
    out = []
    rng = np.random.default_rng(4)
    for n, fmt in enumerate(['NETCDF4_CLASSIC', 'NETCDF3_CLASSIC', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT_OFFSET']):
        p = str(d / ('part%d.nc' % n))
        with netCDF4.Dataset(p, 'w', format=fmt) as f:
            f.createDimension('time', None)
            f.createDimension('x', 4)
            f.title = 'part %d' % n
            t = f.createVariable('time', 'f8', ('time',))
            t.units = 'hours since 2000-01-01'
            v = f.createVariable('v', 'i2', ('time', 'x'), fill_value=-999)
            v.scale_factor = 0.5
            v.add_offset = 1.0
            w = f.createVariable('w', 'f4', ('time', 'x'))
            c = f.createVariable('const', 'f4', ('x',))
            m = [9, 5, 10, 6][n]
            t[:] = np.arange(m) + 10 * n
            vals = rng.integers(-100, 100, (m, 4)).astype('i2')
            vals[1, 2] = -999
            v[:m] = vals * 0.5 + 1.0
            w[:m] = rng.normal(size=(m, 4)).astype('f4')
            c[:] = np.arange(4) + n
        out.append(p)
    return out


def same(a, b):
    if type(a) is not type(b) or np.asarray(a).dtype != np.asarray(b).dtype or np.shape(a) != np.shape(b):
        return False
    if isinstance(a, np.ma.MaskedArray) and repr(a.fill_value) != repr(b.fill_value):
        return False
    ma, mb = np.ma.getmaskarray(a), np.ma.getmaskarray(b)
    da, db = np.asarray(np.ma.getdata(a)), np.asarray(np.ma.getdata(b))
    return np.array_equal(ma, mb) and da[~ma].tobytes() == db[~mb].tobytes()


def test_as_netcdf4(files):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a, b = netCDF4.MFDataset(files), fastnc.MFDataset(files)
        try:
            assert list(a.variables) == list(b.variables)
            assert {k: len(d) for k, d in a.dimensions.items()} == {k: len(d) for k, d in b.dimensions.items()}
            assert [d.isunlimited() for d in a.dimensions.values()] == [d.isunlimited() for d in b.dimensions.values()]
            assert a.ncattrs() == b.ncattrs() and a.title == b.title
            for n in a.variables:
                va, vb = a.variables[n], b.variables[n]
                assert va.shape == vb.shape and va.dimensions == vb.dimensions and va.dtype == vb.dtype, n
                assert va.ncattrs() == vb.ncattrs(), n
                for key in KEYS:
                    try:
                        want = va[key]
                    except Exception:
                        continue
                    assert same(vb[key], want), (n, key)
            for flag in (False, True):
                a.set_auto_maskandscale(flag)
                b.set_auto_maskandscale(flag)
                assert same(b.variables['v'][:], a.variables['v'][:])
        finally:
            a.close()
            b.close()


def test_glob_master_exclude_and_errors(files):
    pattern = os.path.join(os.path.dirname(files[0]), 'part*.nc')
    with fastnc.MFDataset(pattern, exclude=['w'], master_file=files[2]) as m:
        assert 'w' not in m.variables and m.title == 'part 2'
        assert m.variables['time'].shape == (30,)
        assert np.array_equal(m.variables['const'][:], np.arange(4) + 2)      # (the master's)
        assert m.isopen()
    assert not m.isopen()
    with pytest.raises(OSError):
        fastnc.MFDataset(os.path.join(os.path.dirname(files[0]), 'none*.nc'))
    with pytest.raises(OSError):
        fastnc.MFDataset(files, aggdim='nope')
    with fastnc.MFDataset(files) as m:
        with pytest.raises(IndexError):
            m.variables['v'][30]
        m.set_auto_nan(True)
        x = m.variables['v'][:]
        assert type(x) is np.ndarray and np.isnan(x).sum() == 4


def test_module_helpers():
    assert fastnc.default_fillvals == netCDF4.default_fillvals
    c = np.array([[b'a', b'b', b''], [b'x', b'', b'']], 'S1')
    assert np.array_equal(fastnc.chartostring(c), netCDF4.chartostring(c))
    pytest.importorskip('cftime')
    d = fastnc.num2date([0, 24], 'hours since 2000-01-01')
    assert list(d) == list(netCDF4.num2date([0, 24], 'hours since 2000-01-01'))
    assert list(fastnc.date2num(d, 'days since 2000-01-01')) == [0, 1]
