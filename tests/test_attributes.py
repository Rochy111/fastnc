"""Attributes by attribute access (v._FillValue, g.title, hasattr, getattr
with a default), against netCDF4-python.

netCDF4-python gives every attribute ncattrs() lists as v.<name>, '_' names
too: a GOES ABI file's Rad._FillValue and Rad._Unsigned, which readers test
with hasattr(v, '_FillValue') and getattr(v, '_Unsigned', 'false'). Names it
does not list (ncdump -s's _ChunkSizes, _Storage ..., HDF5's dimension-scale
bookkeeping) are AttributeErrors, and so are Python's own special names
(__array_interface__ ...), even where the file has an attribute so called."""
import glob
import os
import pickle

import numpy as np
import pytest

import fastnc
from helpers import DATA

netCDF4 = pytest.importorskip('netCDF4')

FORMATS = ['NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC', 'NETCDF3_64BIT_DATA']
# not attributes in netCDF4-python (ncdump -s's virtual ones, netCDF-4's
# hidden HDF5 ones, never-written ones); netCDF-C's virtual globals
# _NCProperties, _IsNetcdf4 and _SuperblockVersion are left out: netCDF4-python
# gives them as attributes of the root group, fastnc does not (yet)
UNLISTED = ['_x', '_', '_ChunkSizes', '_Storage', '_DeflateLevel', '_Shuffle', '_Fletcher32',
            '_Endianness', '_NoFill', '_Filter', '_Netcdf4Dimid', '_Netcdf4Coordinates',
            '_nc3_strict', 'DIMENSION_LIST', 'REFERENCE_LIST', 'CLASS', 'NAME', 'absent']
SPECIAL = ['__array_interface__', '__array_struct__', '__array_priority__', '__setstate__',
           '__getstate__x__', '__x__', '__', '___']


def write(p, fmt):
    nc4 = fmt.startswith('NETCDF4')
    with netCDF4.Dataset(p, 'w', format=fmt) as d:
        d.title = 'scan'
        d._Gx = 'global, with an underscore'
        d.setncattr('__x__', 'a special name')
        d.createDimension('t', None)
        d.createDimension('x', 4)
        d.createVariable('x', 'f4', ('x',))[:] = np.arange(4)
        v = d.createVariable('Rad', 'i2', ('t', 'x'), fill_value=np.int16(4095),
                             **({'zlib': True, 'chunksizes': (1, 4), 'fletcher32': True} if nc4 else {}))
        v._Unsigned = 'true'
        v.scale_factor = np.float32(0.5)
        v._CoordinateAxisType = 'Time'
        v._valid = np.array([0, 4094], 'i2')
        v.setncattr('__array_interface__', 'not numpy')
        v.setncattr('__x__', 7)
        v.setncattr('_', 'one underscore')
        v[0] = [1, 2, 3, 4]
        c = d.createVariable('c', 'S1', ('x',))
        c._Encoding = 'ascii'
        if fmt == 'NETCDF4':
            g = d.createGroup('g')
            g._Inner = np.int64(3)
            w = g.createVariable('w', 'u2', ('x',), fill_value=np.uint16(9))
            w._ChunkSizes_x = 'not netCDF-C\'s'
            w._Storage = 'written, so an attribute'


@pytest.fixture(scope='module')
def files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('attributes')
    out = {}
    for fmt in FORMATS:
        out[fmt] = str(tmp / ('%s.nc' % fmt))
        write(out[fmt], fmt)
    return out


def pairs(r, f):
    """(netCDF4 object, fastnc object) for every group and variable."""
    out = []
    todo = [(r, f)]
    while todo:
        a, b = todo.pop()
        out.append((a, b))
        out += [(a.variables[k], b.variables[k]) for k in a.variables]
        todo += [(a.groups[k], b.groups[k]) for k in a.groups]
    return out


def outcome(o, name):
    try:
        r = getattr(o, name)
    except AttributeError:
        return ('AttributeError',)
    return (type(r).__name__, np.asarray(r).dtype.str, np.asarray(r).tolist())


@pytest.mark.parametrize('fmt', FORMATS)
def test_every_listed_attribute_as_netcdf4(files, fmt):
    with netCDF4.Dataset(files[fmt]) as r, fastnc.Dataset(files[fmt]) as f:
        for a, b in pairs(r, f):
            assert b.ncattrs() == a.ncattrs()
            for name in a.ncattrs():
                if name[:2] == '__' and name[-2:] == '__':
                    continue                            # (test_special_names)
                assert hasattr(b, name) and hasattr(a, name), name
                assert outcome(b, name) == outcome(a, name), name


@pytest.mark.parametrize('fmt', FORMATS)
def test_a_goes_reader_s_idioms(files, fmt):
    with netCDF4.Dataset(files[fmt]) as r, fastnc.Dataset(files[fmt]) as f:
        for d in (r, f):
            v = d['Rad']
            assert hasattr(v, '_FillValue') and v._FillValue == 4095
            assert type(v._FillValue) is np.int16
            assert getattr(v, '_Unsigned', 'false') == 'true'
            assert getattr(d['x'], '_Unsigned', 'false') == 'false'
            assert getattr(d['x'], '_FillValue', None) is None
            assert d['c']._Encoding == 'ascii'
            assert d._Gx == 'global, with an underscore'
        if fmt == 'NETCDF4':
            assert f['g']._Inner == r['g']._Inner and type(f['g']._Inner) is np.int64
            assert f['g/w']._Storage == r['g/w']._Storage == 'written, so an attribute'


@pytest.mark.parametrize('fmt', FORMATS)
def test_unlisted_names_are_not_attributes(files, fmt):
    with netCDF4.Dataset(files[fmt]) as r, fastnc.Dataset(files[fmt]) as f:
        for a, b in pairs(r, f):
            for name in UNLISTED:
                if name in a.ncattrs():
                    continue                            # (g/w's _Storage was written)
                assert not hasattr(a, name), name
                assert not hasattr(b, name), name
                with pytest.raises(AttributeError, match='Attribute not found'):
                    getattr(b, name)
                with pytest.raises(AttributeError):
                    b.getncattr(name)


@pytest.mark.parametrize('fmt', FORMATS)
def test_special_names(files, fmt):
    """Python's special names are never netCDF attributes, in either library,
    even those the file has: numpy, pickle and copy look them up."""
    with netCDF4.Dataset(files[fmt]) as r, fastnc.Dataset(files[fmt]) as f:
        for a, b in pairs(r, f):
            for name in SPECIAL:
                assert not hasattr(a, name), name
                assert not hasattr(b, name), name
        v, w = f['Rad'], r['Rad']
        assert v.getncattr('__array_interface__') == w.getncattr('__array_interface__') == 'not numpy'
        assert v.getncattr('__x__') == w.getncattr('__x__') == 7
        assert f.getncattr('__x__') == r.getncattr('__x__')
        assert v._ == w._ == 'one underscore'                # (one underscore: an attribute)
        a = np.asarray(v[:])
        assert a.shape == (1, 4)
        assert isinstance(v.__dict__, dict)


def test_half_made_and_unpickled(files):
    """No backend yet (being unpickled, or a failed open): an AttributeError
    for every name, not a recursion or a TypeError."""
    for cls in (fastnc.Variable, fastnc.Group, fastnc.Dataset):
        o = cls.__new__(cls)
        for name in ('_FillValue', 'title', '_bv', '_bg', '__setstate__'):
            assert not hasattr(o, name), (cls, name)
    with fastnc.Dataset(files['NETCDF4']) as f:
        v = pickle.loads(pickle.dumps(f['Rad']))
        assert v._FillValue == 4095 and v._Unsigned == 'true'
        g = pickle.loads(pickle.dumps(f['g']))
        assert g._Inner == 3


def test_corpus_attributes():
    """In every corpus file, v.<name> is v.getncattr(name) for every name
    ncattrs() lists, '_' ones too (no reference library needed)."""
    seen = 0
    for p in sorted(glob.glob(os.path.join(DATA, '*.nc'))):
        try:
            f = fastnc.Dataset(p)
        except (OSError, ValueError, NotImplementedError):
            continue
        with f:
            todo = [f]
            while todo:
                g = todo.pop()
                todo += list(g.groups.values())
                for o in [g] + list(g.variables.values()):
                    for name in o.ncattrs():
                        if name[:2] == '__' and name[-2:] == '__':
                            continue
                        want = o.getncattr(name)
                        got = getattr(o, name)
                        assert type(got) is type(want), (p, name)
                        assert np.array_equal(np.asarray(got), np.asarray(want)), (p, name)
                        seen += name.startswith('_')
    assert seen
