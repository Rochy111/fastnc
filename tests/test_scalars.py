"""Scalar (0-d) variables indexed every way, against netCDF4-python and h5py.

netCDF4-python indexes a scalar as if it had shape (1,): v[:], v[0], v[0:1],
v[[0]] all give its value (a GOES ABI file's kappa0, planck_fk1 ... are read
with v[:]), and integers are not bounds-checked. h5py takes only () and
Ellipsis on a scalar dataset. fastnc must give what each gives, value, type
and mask, and refuse what each refuses, with the same kind of error."""
import glob
import os

import numpy as np
import pytest

import fastnc
from helpers import DATA, digest

netCDF4 = pytest.importorskip('netCDF4')

KEYS = [Ellipsis, (), (Ellipsis,), slice(None), (slice(None),), (slice(None), slice(None)),
        (slice(None), 0), (0, slice(None)), 0, -1, -2, 1, 5, True, False, np.True_, np.int64(0),
        np.uint8(3), np.array(0), np.array(True), 1.0, '0', 'a', None, (None,), (0,), (Ellipsis, 0),
        (0, Ellipsis), (Ellipsis, slice(None)), (slice(None), Ellipsis),
        [0], [1], [-1], [-2], [0, 0], [True], [False], [True, False], [], np.array([0]),
        np.array([], 'i8'), slice(0, 1), slice(None, 5), slice(-5, None), slice(1, None), slice(0, 0),
        slice(None, -1), slice(0, 1, 2), slice(None, 5, 3), slice(None, 1, 2), slice(None, None, -1),
        slice(0, 1, -1), slice(5, None, -1)]
# netCDF4-python 1.7.4 under numpy 2 fails on a strided slice without a stop
# on a scalar ('tuple index out of range', from looking up the scalar's first
# dimension); fastnc gives the value, as both do for the same slice with a stop
NC4_FAILS = [slice(None, None, 2), slice(None, None, -2), slice(0, None, 2), slice(-1, None, 3)]

NUMBERS = {'NETCDF4': ('f4', 'f8', 'i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8'),
           'NETCDF4_CLASSIC': ('f4', 'f8', 'i1', 'i2', 'i4'),
           'NETCDF3_CLASSIC': ('f4', 'f8', 'i1', 'i2', 'i4'),
           'NETCDF3_64BIT_DATA': ('f4', 'f8', 'i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8')}
MODES = ['masked', 'raw', 'always_mask_off', 'mask_only', 'scale_only']


def write(p, fmt):
    with netCDF4.Dataset(p, 'w', format=fmt) as d:
        for dt in NUMBERS[fmt]:
            d.createVariable('d_' + dt, dt)[...] = 3
            d.createVariable('fill_' + dt, dt, fill_value=7)[...] = 7          # the fill: masked
            d.createVariable('never_' + dt, dt)                              # the default fill
        v = d.createVariable('scaled', 'i2', fill_value=-5)
        v.set_auto_scale(False)
        v.scale_factor = np.float32(0.5)
        v.add_offset = np.float32(10)
        v[...] = 4
        v = d.createVariable('kappa0', 'f4')                                 # as in a GOES ABI file
        v.units = 'W-1 m2 sr um'
        v[...] = 0.0019
        v = d.createVariable('unsigned', 'i1')
        v._Unsigned = 'true'
        v.set_auto_scale(False)
        v[...] = -3
        v = d.createVariable('ranged', 'f4')
        v.valid_range = np.array([0, 1], 'f4')
        v[...] = 5
        v = d.createVariable('missing', 'f8')
        v.missing_value = 2.5
        v[...] = 2.5
        d.createVariable('char', 'S1')[...] = b'x'
        v = d.createVariable('encoded', 'S1')                                # chartostring's case
        v.set_auto_chartostring(False)
        v._Encoding = 'ascii'
        v[...] = b'y'
        if fmt == 'NETCDF4':
            d.createVariable('text', str)[...] = np.array('hello', object)
            ct = d.createCompoundType(np.dtype([('a', 'f4'), ('b', 'i2')]), 'cmp')
            d.createVariable('compound', ct)[...] = np.array((1.5, 2), dtype=ct.dtype)
            et = d.createEnumType('u1', 'en', {'a': 1, 'b': 2})
            d.createVariable('enum', et)[...] = 2
            g = d.createGroup('g')
            g.createVariable('inner', 'f4')[...] = 1.25


FORMATS = list(NUMBERS)


@pytest.fixture(scope='module')
def files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('scalars')
    out = {}
    for fmt in FORMATS:
        p = str(tmp / ('%s.nc' % fmt))
        write(p, fmt)
        out[fmt] = p
    return out


def names(fmt):
    out = [p + dt for dt in NUMBERS[fmt] for p in ('d_', 'fill_', 'never_')]
    out += ['scaled', 'kappa0', 'unsigned', 'ranged', 'missing', 'char', 'encoded']
    if fmt == 'NETCDF4':
        out += ['text', 'compound', 'enum', 'g/inner']
    return out


def outcome(fn):
    """What a call gives, as something to compare: its values (type, dtype,
    shape, mask, fill value), np.ma.masked, or the kind of error raised."""
    try:
        r = fn()
    except (IndexError, ValueError, TypeError) as e:
        return ('raised', type(e).__name__)
    if r is np.ma.masked:
        return ('masked',)
    if isinstance(r, np.ndarray) and r.dtype == object:
        return ('objects', type(r).__name__, r.shape, repr(r.tolist()))
    if isinstance(r, (str, bytes)):
        return ('scalar', type(r).__name__, r)
    return digest(r)


def set_mode(v, mode):
    if mode in ('masked', 'raw'):
        v.set_auto_maskandscale(mode == 'masked')
    elif mode == 'always_mask_off':
        v.set_always_mask(False)
    else:
        v.set_auto_mask(mode == 'mask_only')
        v.set_auto_scale(mode == 'scale_only')


@pytest.mark.parametrize('how', MODES)
@pytest.mark.parametrize('fmt', FORMATS)
def test_every_index_as_netcdf4(files, fmt, how):
    with netCDF4.Dataset(files[fmt]) as r, fastnc.Dataset(files[fmt]) as d:
        for name in names(fmt):
            a, b = d[name], r[name]
            assert a.shape == b.shape == (), name
            set_mode(a, how)
            set_mode(b, how)
            for k in KEYS:
                assert outcome(lambda: a[k]) == outcome(lambda: b[k]), (name, k)


@pytest.mark.parametrize('fmt', FORMATS)
def test_full_slice_reads_the_value(files, fmt):
    """v[:] of a scalar: what netCDF4-python gives for it, and fastnc's own v[...]."""
    with netCDF4.Dataset(files[fmt]) as r, fastnc.Dataset(files[fmt]) as d:
        for masked in (True, False):
            for name in names(fmt):
                a, b = d[name], r[name]
                a.set_auto_maskandscale(masked)
                b.set_auto_maskandscale(masked)
                assert outcome(lambda: a[:]) == outcome(lambda: b[:]) == outcome(lambda: a[...]), name
        d['kappa0'].set_auto_maskandscale(True)
        k = d['kappa0'][:]
        assert type(k) is np.ma.MaskedArray and k.shape == () and k.dtype == np.float32
        assert k == np.float32(0.0019)


@pytest.mark.parametrize('k', NC4_FAILS)
def test_strided_without_a_stop(files, k):
    """Where netCDF4-python fails on its own, fastnc gives what it gives
    for a slice of the one element with a stop (v[0:1:2], v[0:1]); where
    it does not fail, the same."""
    with netCDF4.Dataset(files['NETCDF4']) as r, fastnc.Dataset(files['NETCDF4']) as d:
        for name in names('NETCDF4'):
            want = outcome(lambda: r[name][0:1])
            assert outcome(lambda: r[name][k]) in (want, ('raised', 'IndexError')), (name, k)
            assert outcome(lambda: d[name][k]) == outcome(lambda: r[name][0:1:2]) == want, (name, k)


def test_scalar_dataset_as_h5py(files):
    h5py = pytest.importorskip('h5py')
    p = files['NETCDF4']
    with h5py.File(p, 'r') as r, fastnc.File(p, 'r') as f:
        for name in ['d_' + dt for dt in NUMBERS['NETCDF4']] + ['scaled', 'char', 'text', 'compound',
                                                              'enum', 'g/inner']:
            for k in KEYS:
                if isinstance(k, str):
                    continue                        # (field names: h5py's own rules, any rank)
                assert outcome(lambda: f[name][k]) == outcome(lambda: r[name][k]), (name, k)


def test_corpus_scalars():
    """The corpus files' scalar variable (no reference library needed): v[:],
    v[0], v[0:1] and v[[0]] give v[...]'s value, masked or raw."""
    paths = [p for p in sorted(glob.glob(os.path.join(DATA, '*.nc')))]
    seen = 0
    for p in paths:
        with fastnc.Dataset(p) as d:
            if 'scalar' not in d.variables:
                continue
            v = d['scalar']
            assert v.shape == ()
            seen += 1
            for masked in (True, False):
                v.set_auto_maskandscale(masked)
                whole = v[...]
                for k in (slice(None), 0, slice(0, 1), [0], -1):
                    got = v[k]
                    assert np.asarray(got).tolist() == np.asarray(whole).tolist(), (p, k)
                    assert np.asarray(got).dtype == np.asarray(whole).dtype, (p, k)
                    if masked:
                        assert type(got) is type(whole), (p, k)
                with pytest.raises(IndexError):
                    v[1:]
    assert seen


@pytest.mark.parametrize('masked', [True, False])
def test_the_card_equals_the_cpu(files, masked, request):
    if not request.config.getoption('--gpu'):
        pytest.skip('needs --gpu')
    with fastnc.Dataset(files['NETCDF4']) as d:
        for name in ['d_' + dt for dt in NUMBERS['NETCDF4']] + ['fill_f4', 'scaled', 'kappa0']:
            v = d[name]
            v.set_auto_maskandscale(masked)
            for k in (Ellipsis, slice(None), 0, slice(0, 1), [0]):
                want = v[k]
                got = v.read(k, device='gpu')
                got = got.get() if hasattr(got, 'get') else got
                assert np.asarray(got).dtype == np.asarray(want).dtype, (name, k)
                assert np.ma.getdata(got).tobytes() == np.ma.getdata(want).tobytes(), (name, k)
                assert np.array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(want)), (name, k)
