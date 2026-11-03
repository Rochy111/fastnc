"""A variable with fewer records than its unlimited dimension (records written
to other variables, the usual case in a record file). netCDF-C gives the rest as
fill values, and the shape is the dimension's, so every read must match
netCDF4-python's, whichever way it is made: by index, by load(), in a loop, by
readahead, through xarray, on the GPU."""
import numpy as np
import pytest

import fastnc
from fastnc import _chunks, _readahead
from helpers import digest, same_values

netCDF4 = pytest.importorskip('netCDF4')

KEYS = [Ellipsis, (), 0, 1, 2, 4, 6, -1, -3, slice(None), slice(1, 5), slice(0, 2), slice(3, 7),
        slice(None, None, 2), slice(1, None, 3), slice(None, None, -1), slice(5, 1, -1),
        (slice(1, 5), 1), (slice(None), slice(0, 1)), (4, slice(None)), (1, 1), (5, 1),
        [0, 3, 6], ([1, 5],), (slice(None, None, 2), [0, 1])]
NUMBERS = ('f4', 'f8', 'i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8')


def write(p):
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('t', None)
        d.createDimension('y', 2)
        d.createVariable('long', 'f4', ('t', 'y'))[:] = np.arange(10.0).reshape(5, 2)
        for dt in NUMBERS:
            one = np.ones((2, 2)).astype(dt)
            d.createVariable('d_' + dt, dt, ('t', 'y'))[0:2] = one
            d.createVariable('nofill_' + dt, dt, ('t', 'y'), fill_value=False)[0:2] = one
            d.createVariable('fill_' + dt, dt, ('t', 'y'), fill_value=7)[0:2] = one
        d.createVariable('char', 'S1', ('t', 'y'))[0:1] = np.array([[b'a', b'b']])
        d.createVariable('text', str, ('t',))[0] = 'hello'
        v = d.createVariable('scaled', 'i2', ('t', 'y'), fill_value=-5)
        v.set_auto_scale(False)
        v.scale_factor = 0.5
        v[0:2] = np.ones((2, 2), 'i2')
        v = d.createVariable('offset_range', 'f4', ('t', 'y'))
        v.valid_min = np.float32(0)
        v[0:2] = np.ones((2, 2))
        d.createVariable('compressed', 'f4', ('t', 'y'), zlib=True, shuffle=True,
                         chunksizes=(2, 2))[0:2] = np.ones((2, 2))
        d.createVariable('contiguous_ish', 'i4', ('t',))[0:1] = [3]
        d.createVariable('never_written', 'f4', ('t', 'y'))
        ct = d.createCompoundType(np.dtype([('a', 'f4'), ('b', 'i2')]), 'cmp')
        d.createVariable('compound', ct, ('t',))[0] = np.array((1.5, 2), dtype=ct.dtype)
        vt = d.createVLType('i4', 'vl')
        d.createVariable('vlen', vt, ('t',))[0] = np.array([1, 2, 3], 'i4')
        et = d.createEnumType('u1', 'en', {'a': 1, 'b': 2})
        d.createVariable('enum', et, ('t',))[0] = 1
        g = d.createGroup('g')
        g.createVariable('inner', 'f4', ('t', 'y'))[0:7] = np.ones((7, 2))      # the longest
        g.createVariable('inner_short', 'f4', ('t', 'y'))[0:3] = np.ones((3, 2))


@pytest.fixture(scope='module')
def path(tmp_path_factory):
    p = str(tmp_path_factory.mktemp('short') / 'x.nc')
    write(p)
    return p


def names(path):
    with netCDF4.Dataset(path) as d:
        return list(d.variables) + ['g/inner', 'g/inner_short']


NAMES = ['long'] + ['d_' + t for t in NUMBERS] + ['nofill_' + t for t in NUMBERS] + \
    ['fill_' + t for t in NUMBERS] + ['char', 'text', 'scaled', 'offset_range', 'compressed',
                                      'contiguous_ish', 'never_written', 'compound', 'vlen', 'enum',
                                      'g/inner', 'g/inner_short']


def outcome(fn):
    """What a call gives, as something to compare: values, or the error's kind."""
    try:
        r = fn()
    except (IndexError, ValueError, TypeError):
        return ('raised',)                          # (refused alike, in its own words)
    if isinstance(r, np.ndarray) and r.dtype == object:
        return ('objects', r.shape, repr(r.tolist()))
    if isinstance(r, (str, bytes)):
        return ('scalar', r)
    return digest(r)


def stepped(k):
    """Whether index k steps forward by more than one: a slice, or a list
    (netCDF4-python reads evenly spaced lists as stepped ones)."""
    return any(isinstance(x, list) or (isinstance(x, slice) and (x.step or 1) > 1)
               for x in (k if isinstance(k, tuple) else (k,)))


# netCDF-C before 4.10 drops stored values from a stepped read that runs past
# them (short[1::3] of [1, 2, fill ...] gives [fill, fill], against its own
# short[...]); there its whole-array answer, indexed, is the reference
OLD_STEPS = tuple(int(x) for x in netCDF4.__netcdf4libversion__.split('.')[:2]) < (4, 10)


def reference(b, k):
    return b[...][k] if OLD_STEPS and stepped(k) else b[k]


def same(got, want, what):
    if isinstance(want, tuple) or isinstance(got, tuple):
        assert got == want, what
    else:
        ok, msg = same_values(got, want)
        assert ok, (what, msg)


@pytest.mark.parametrize('masked', [True, False])
@pytest.mark.parametrize('name', NAMES)
def test_every_index(path, name, masked):
    with netCDF4.Dataset(path) as r, fastnc.Dataset(path) as d:
        a, b = d[name], r[name]
        a.set_auto_maskandscale(masked)
        b.set_auto_maskandscale(masked)
        assert a.shape == b.shape, name
        for k in KEYS:
            if OLD_STEPS and stepped(k):
                x, y = outcome(lambda: a[k]), outcome(lambda: reference(b, k))
                if isinstance(x, dict) and isinstance(y, dict):
                    # (an array cut from a whole one keeps its fill_value:
                    # nothing to compare; values and mask are)
                    x, y = [{q: v for q, v in c.items() if q != 'fill_value'} for c in (x, y)]
                same(x, y, (name, k))
                continue
            same(outcome(lambda: a[k]), outcome(lambda: b[k]), (name, k))


@pytest.mark.parametrize('name', NAMES)
def test_always_mask_off(path, name):
    with netCDF4.Dataset(path) as r, fastnc.Dataset(path) as d:
        d[name].set_always_mask(False)
        r[name].set_always_mask(False)
        same(outcome(lambda: d[name][...]), outcome(lambda: r[name][...]), name)


@pytest.mark.parametrize('kernels', [True, False])
@pytest.mark.parametrize('masked', [True, False])
def test_load(path, masked, kernels, monkeypatch):
    if kernels:
        fastnc.warm()
    else:
        monkeypatch.setattr(_chunks, 'ready', lambda: False)
    with netCDF4.Dataset(path) as r, fastnc.Dataset(path) as d:
        d.set_auto_maskandscale(masked)
        r.set_auto_maskandscale(masked)
        got = d.load(recursive=True)
        for n in NAMES:
            b = r[n]
            b.set_auto_maskandscale(masked)
            same(outcome(lambda: got[n]), outcome(lambda: b[...]), n)


def test_the_loop_and_readahead(path, monkeypatch):
    monkeypatch.setattr(_readahead, 'BIG', 1)
    with netCDF4.Dataset(path) as r, fastnc.Dataset(path) as d:
        for n in d.variables:
            same(outcome(lambda: d[n][...]), outcome(lambda: r[n][...]), n)


def test_asked_for_before_anything_else(path):
    """Each in a fresh file: a variable found alone must still see the others."""
    with netCDF4.Dataset(path) as r:
        for n in NAMES:
            with fastnc.Dataset(path) as d:
                same(outcome(lambda: d[n][...]), outcome(lambda: r[n][...]), n)


def test_two_unlimited_dimensions(tmp_path):
    """No comparing with netCDF4-python here: short along two unlimited
    dimensions, netCDF-C returns uninitialized memory, and values of other
    variables, at places that vary. What is stored stays where it is stored,
    and the rest is fill."""
    p = str(tmp_path / 'x.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('a', None)
        d.createDimension('b', None)
        d.createVariable('wide', 'f4', ('a', 'b'))[0:2, 0:6] = np.ones((2, 6))
        d.createVariable('tall', 'f4', ('a', 'b'))[0:5, 0:2] = 2 * np.ones((5, 2))
    fill = np.float32(netCDF4.default_fillvals['f4'])
    want = {'wide': np.full((5, 6), fill), 'tall': np.full((5, 6), fill)}
    want['wide'][:2, :6] = 1
    want['tall'][:5, :2] = 2
    with netCDF4.Dataset(p) as r, fastnc.Dataset(p) as d:
        d.set_auto_maskandscale(False)
        for n in ('wide', 'tall'):
            assert d[n].shape == r[n].shape == (5, 6)
            assert d[n][...].tobytes() == want[n].tobytes()
            for k in ((slice(1, 4), slice(1, 5)), (4, 5), (slice(None, None, 2), 0), (slice(None, None, -1),)):
                assert np.array_equal(d[n][k], want[n][k]), (n, k)


def test_nothing_longer_than_a_fixed_dimension_is_asked_about(tmp_path):
    """Fixed dimensions are not padded, however long the records are."""
    p = str(tmp_path / 'x.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('t', None)
        d.createDimension('y', 2)
        d.createVariable('rec', 'f4', ('t',))[:] = np.arange(9.0)
        d.createVariable('fixed', 'f4', ('y',))[:] = [1, 2]
    with netCDF4.Dataset(p) as r, fastnc.Dataset(p) as d:
        assert d['fixed'].shape == (2,) and not d['fixed']._short()
        same(outcome(lambda: d['fixed'][...]), outcome(lambda: r['fixed'][...]), 'fixed')


def test_full_variables_need_no_scan(tmp_path):
    """The usual file: nothing short, and one read reads one variable."""
    p = str(tmp_path / 'x.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('t', None)
        d.createDimension('y', 3)
        d.createVariable('a', 'f4', ('t', 'y'))[0:2] = np.ones((2, 3))
        d.createVariable('b', 'f4', ('t', 'y'))[0:2] = np.ones((2, 3))
    with fastnc.Dataset(p) as d:
        d['a'][...]
        assert not d._backend._scanned


def test_more_records_than_a_fixed_axis_needs_no_scan(tmp_path):
    """Time longer than lat: the usual climate file."""
    p = str(tmp_path / 'x.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('time', None)
        d.createDimension('lat', 3)
        d.createVariable('time', 'f8', ('time',))[:] = np.arange(20.0)
        d.createVariable('lat', 'f4', ('lat',))[:] = [1, 2, 3]
        d.createVariable('t', 'f4', ('time', 'lat'))[:] = np.ones((20, 3))
    with fastnc.Dataset(p) as d:
        assert d['t'][...].shape == (20, 3) and not d._backend._linked


def test_a_fixed_axis_on_an_unlimited_scale(tmp_path):
    """What netCDF-C does not write but does read: a dataset of fixed size
    attached to a dimension scale with an unlimited axis, here longer."""
    h5py = pytest.importorskip('h5py')
    p = str(tmp_path / 'x.h5')
    with h5py.File(p, 'w') as f:
        t = f.create_dataset('t', data=np.arange(6.0), maxshape=(None,), chunks=True)
        t.make_scale('t')
        v = f.create_dataset('v', data=np.ones(2, 'f4'))
        v.dims[0].attach_scale(t)
    with netCDF4.Dataset(p) as r, fastnc.Dataset(p) as d:
        assert d['v'].shape == r['v'].shape
        same(outcome(lambda: d['v'][...]), outcome(lambda: r['v'][...]), 'v')


def test_xarray_as_the_netcdf4_engine(path):
    xr = pytest.importorskip('xarray')
    from fastnc._xarray import FastncBackendEntrypoint
    with xr.open_dataset(path, engine=FastncBackendEntrypoint) as a, \
            xr.open_dataset(path, engine='netcdf4') as b:
        for n in NAMES:
            if '/' in n or n not in b or n == 'vlen' or b[n].dtype == object or b[n].dtype.names:
                continue
            x, y = a[n].load(), b[n].load()
            assert x.dtype == y.dtype and x.shape == y.shape, n
            assert x.values.tobytes() == y.values.tobytes(), n


@pytest.mark.parametrize('masked', [True, False])
def test_the_card_equals_the_cpu(path, masked, request):
    if not request.config.getoption('--gpu'):
        pytest.skip('needs --gpu')
    with fastnc.Dataset(path) as d:
        for n in NAMES:
            v = d[n]
            if not isinstance(v.dtype, np.dtype) or v.dtype.names or v.dtype.kind == 'S' or n == 'vlen':
                continue
            v.set_auto_maskandscale(masked)
            for k in (Ellipsis, 4, slice(1, 5)) + (((slice(1, 5), 1),) if v.ndim > 1 else ()):
                want = v[k]
                got = v.read(k, device='gpu')
                got = got.get() if hasattr(got, 'get') else got
                assert got.dtype == want.dtype and got.shape == want.shape, (n, k)
                assert np.ma.getdata(got).tobytes() == np.ma.getdata(want).tobytes(), (n, k)
                assert np.array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(want)), (n, k)
