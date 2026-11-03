"""Opening reads only what is asked for: one variable alone, the group's other
variables, dimensions and sub-groups when something wants them. What comes back
must not depend on that order, whatever is asked first: names, order, shapes,
dimensions (phony ones keep their numbers), attributes, values."""
import glob
import os

import numpy as np
import pytest

import fastnc
from fastnc import _dataset, _readahead

netCDF4 = pytest.importorskip('netCDF4')
h5py = pytest.importorskip('h5py')

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FILES = sorted(glob.glob(os.path.join(DATA, '*.nc')) + glob.glob(os.path.join(DATA, '*.h5')))


def describe(d):
    """Everything about a file's structure, with the groups walked in order."""
    out = {}
    for g in d._walk():
        out[g.path] = dict(
            dims={k: (len(v), v.isunlimited()) for k, v in g.dimensions.items()},
            groups=list(g.groups),
            vars={k: (v.dimensions, v.shape, str(v.dtype), v.ncattrs()) for k, v in g.variables.items()})
    return out


@pytest.mark.parametrize('path', FILES, ids=os.path.basename)
def test_structure_does_not_depend_on_what_is_asked_first(path):
    with fastnc.Dataset(path) as d:
        want = describe(d)
        names = list(d.variables)
    for order in (names, names[::-1]):
        with fastnc.Dataset(path) as d:
            for n in order[:5]:                     # (one by one, each alone)
                d[n]
            assert describe(d) == want
    with fastnc.Dataset(path) as d:                 # dimensions asked for first
        d.dimensions
        assert describe(d) == want


@pytest.mark.parametrize('path', FILES, ids=os.path.basename)
def test_a_variable_found_alone_reads_the_same(path):
    with fastnc.Dataset(path) as whole:
        names = list(whole.variables)
        for n in names:
            with fastnc.Dataset(path) as d:
                a, b = d[n], whole[n]
                assert a.shape == b.shape and a.dtype == b.dtype and a.ncattrs() == b.ncattrs()
                try:
                    x = a[...]
                except Exception as e:
                    with pytest.raises(type(e)):
                        b[...]
                    continue
                y = b[...]
                if isinstance(x, np.ndarray) and x.dtype.kind in 'iuf':
                    assert np.ma.getdata(x).tobytes() == np.ma.getdata(y).tobytes()


@pytest.fixture(scope='module')
def path(tmp_path_factory):
    p = str(tmp_path_factory.mktemp('lazy') / 'x.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('t', None)
        d.createDimension('y', 4)
        d.createDimension('x', 5)
        d.createDimension('no_coord', 3)
        d.createVariable('t', 'f8', ('t',))[:] = np.arange(6.0)      # a coordinate: a scale
        d.createVariable('y', 'f4', ('y',))[:] = np.arange(4.0)
        d.createVariable('a', 'f4', ('t', 'y', 'x'))[:] = np.ones((6, 4, 5))
        d.createVariable('short', 'f4', ('t', 'y'))[0:2] = np.ones((2, 4))
        d.createVariable('other', 'i2', ('no_coord',))[:] = [1, 2, 3]
        g = d.createGroup('g')
        g.createDimension('z', 2)
        g.createVariable('b', 'f4', ('z',))[:] = [7, 8]
    return p


def test_it_is_still_a_dict(path):
    with fastnc.Dataset(path) as d, netCDF4.Dataset(path) as r:
        d['a']
        v = d.variables
        assert isinstance(v, dict) and not isinstance(v, type(None))
        assert list(v) == list(r.variables)         # the file's order, not the order asked
        assert list(v.keys()) == list(r.variables.keys())
        assert len(v) == len(r.variables)
        assert 'a' in v and 'nothing' not in v and 'g' not in v
        assert v.get('nothing') is None and v.get('nothing', 5) == 5
        assert dict(v).keys() == r.variables.keys()
        assert sorted(v) == sorted(r.variables)
        assert list(reversed(v)) == list(reversed(list(r.variables)))
        with pytest.raises(KeyError):
            v['nothing']


@pytest.mark.parametrize('name', ['a', 'nothing', 'g', 'no_coord', 't', 'y', 'short'])
def test_each_name_alone_as_in_netcdf4(path, name):
    with fastnc.Dataset(path) as d, netCDF4.Dataset(path) as r:
        assert (name in d.variables) == (name in r.variables)


def test_a_group_or_a_dimension_is_no_variable(path):
    with fastnc.Dataset(path) as d:
        assert 'g' not in d.variables
    with fastnc.Dataset(path) as d:
        assert 'no_coord' not in d.variables        # a scale with no variable behind it
    with fastnc.Dataset(path) as d:
        assert d['g'].path == '/g' and d['g/b'][...].tolist() == [7, 8]


def test_the_same_object_before_and_after_the_scan(path):
    with fastnc.Dataset(path) as d:
        v = d['a']
        v.set_auto_mask(False)
        list(d.variables)
        assert d['a'] is v and d.variables['a'] is v
        assert not isinstance(d['a'][...], np.ma.MaskedArray)


def test_an_empty_or_scalar_variable_keeps_its_object(tmp_path):
    p = str(tmp_path / 'x.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('n', 0)
        d.createDimension('m', 3)
        d.createVariable('empty', 'f4', ('n',))
        d.createVariable('scalar', 'f4', ())[:] = 2.0
        d.createVariable('other', 'f4', ('m',))
    with fastnc.Dataset(p) as d:
        a, b = d['empty'], d['scalar']
        assert list(d.variables) == ['empty', 'scalar', 'other']
        assert d['empty'] is a and d['scalar'] is b


def test_settings_reach_variables_not_read_yet(path):
    with fastnc.Dataset(path) as d:
        d.set_auto_maskandscale(False)
        assert not isinstance(d['a'][...], np.ma.MaskedArray)


def test_shape_follows_the_unlimited_dimension(path):
    """(Which a variable alone cannot know: the longest variable sets it.)"""
    with fastnc.Dataset(path) as d, netCDF4.Dataset(path) as r:
        assert d['short'].shape == r['short'].shape
        assert d['a'].shape == r['a'].shape


def test_one_read_scans_nothing(tmp_path, monkeypatch):
    """A big variable read alone: no other variable's metadata, readahead's
    notes included, sub-groups or not. (Records no axis falls short of: more
    records than the shortest axis takes the whole scan, to be sure.)"""
    monkeypatch.setattr(_readahead, 'BIG', 1)
    p = str(tmp_path / 'x.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('t', None)
        d.createDimension('y', 4)
        d.createVariable('a', 'f4', ('t', 'y'))[0:3] = np.ones((3, 4))
        g = d.createGroup('g')
        g.createVariable('b', 'f4', ('t', 'y'))[0:3] = np.ones((3, 4))
    with fastnc.Dataset(p) as d:
        d['a'][...]
        assert not d._backend._scanned and not d._backend._linked


def test_a_longer_variable_in_a_group_is_seen(tmp_path):
    p = str(tmp_path / 'x.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('t', None)
        d.createDimension('y', 9)
        d.createVariable('a', 'f4', ('t', 'y'))[0:2] = np.ones((2, 9))
        g = d.createGroup('g').createGroup('deep')
        g.createVariable('b', 'f4', ('t', 'y'))[0:5] = np.ones((5, 9))
    with netCDF4.Dataset(p) as r, fastnc.Dataset(p) as d:
        assert d['a'].shape == r['a'].shape == (5, 9)
        assert d['a'][...].shape == (5, 9)


def test_a_second_read_still_drives_readahead(path, monkeypatch):
    monkeypatch.setattr(_readahead, 'BIG', 1)
    with fastnc.Dataset(path) as d:
        for n in ('t', 'y', 'a'):
            d[n][...]
        assert d._readahead().last >= 0


def test_a_variable_with_a_class_attribute_takes_the_exact_path(tmp_path):
    p = str(tmp_path / 'x.h5')
    with h5py.File(p, 'w') as f:
        f['plain'] = np.arange(3.0)
        f['labelled'] = np.arange(3.0)
        f['labelled'].attrs['CLASS'] = 'NOT_A_SCALE'
        f['vlen_scale'] = np.arange(3.0)
        f['vlen_scale'].attrs['CLASS'] = 'DIMENSION_SCALE'
    with fastnc.Dataset(p) as d:
        assert d['labelled'][...].tolist() == [0, 1, 2]
    with fastnc.Dataset(p) as d:
        names = list(d.variables)
        assert names == ['labelled', 'plain', 'vlen_scale']


def test_many_attributes_are_found_in_dense_storage(tmp_path):
    p = str(tmp_path / 'x.h5')
    with h5py.File(p, 'w', libver='latest') as f:
        f['v'] = np.arange(4)
        for i in range(40):
            f['v'].attrs['a%d' % i] = i
        f['v'].attrs['CLASS'] = np.bytes_('DIMENSION_SCALE')
        f['w'] = np.arange(4)
    with fastnc.Dataset(p) as d:
        assert d['w'][...].tolist() == [0, 1, 2, 3]
        assert 'v' in d.dimensions or 'v' in d.variables
