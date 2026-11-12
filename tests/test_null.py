"""HDF5 null dataspaces: datasets with a type and attributes but no values.
netCDF-C refuses such a file, so through fastnc.Dataset there is no reference
to match: a read gives None, masked or raw, on the CPU or the GPU, and one
such variable costs a load() of the others nothing. fastnc.File does as h5py."""
import numpy as np
import pytest

import fastnc
from fastnc import _gpu
from fastnc._h5py import Empty

h5py = pytest.importorskip('h5py')


@pytest.fixture(scope='module')
def odd(tmp_path_factory):
    """Types netCDF has no default fill value for (netCDF-C will not open
    this file), small and past the fast path's size."""
    p = str(tmp_path_factory.mktemp('odd') / 'odd.h5')
    x = np.arange(1, 40001, dtype='f2').reshape(200, 200)
    with h5py.File(p, 'w') as f:
        f['f2'] = x
        f['f2_small'] = x[:3, :4]
        f['f2_fill'] = x
        f['f2_fill'].attrs['_FillValue'] = x[0, 2]
        f.create_dataset('f2_chunked', data=x, chunks=(50, 50), compression='gzip', shuffle=True)
        f['f2_scaled'] = x
        f['f2_scaled'].attrs['scale_factor'] = np.float32(2)
        f['f2_range'] = x
        f['f2_range'].attrs['valid_max'] = np.float16(3)
        f.create_dataset('array', shape=(4,), dtype=np.dtype(('i4', (3, 2))))
        f['values'] = np.arange(5.0)
    return p, x


@pytest.mark.parametrize('name', ['f2', 'f2_small', 'f2_chunked'])
def test_no_default_fill_value_means_nothing_masked(odd, name):
    p, x = odd
    fastnc.warm()
    with fastnc.Dataset(p) as d:
        got = d[name][...]
    assert isinstance(got, np.ma.MaskedArray) and not np.ma.getmaskarray(got).any()
    assert got.dtype == np.dtype('f4')                  # (netCDF has no float16, as netCDF4 shows)
    assert np.array_equal(np.ma.getdata(got), x if name != 'f2_small' else x[:3, :4])


def test_an_explicit_fill_value_still_counts(odd):
    p, x = odd
    with fastnc.Dataset(p) as d:
        got = d['f2_fill'][...]
    assert np.ma.getmaskarray(got).sum() == 1 and got.mask[0, 2]
    assert got.fill_value == x[0, 2]


def test_a_range_still_counts(odd):
    p, x = odd
    with fastnc.Dataset(p) as d:
        got = d['f2_range'][...]
    assert np.array_equal(np.ma.getmaskarray(got), x > 3)


def test_scaling_still_counts(odd):
    p, x = odd
    with fastnc.Dataset(p) as d:
        got = d['f2_scaled'][...]
    assert np.array_equal(np.ma.getdata(got), x.astype('f4') * 2)


@pytest.mark.parametrize('always', [True, False])
def test_always_mask_on_odd_types(odd, always):
    p, x = odd
    with fastnc.Dataset(p) as d:
        d['f2'].set_always_mask(always)
        assert isinstance(d['f2'][...], np.ma.MaskedArray) == always


def test_odd_types_cost_load_nothing(odd):
    p, x = odd
    with fastnc.Dataset(p) as d:
        got = d.load()
    assert np.array_equal(np.ma.getdata(got['values']), np.arange(5.0))
    assert 'array' not in got                       # (an HDF5 array type: netCDF-c shows no such variable)
    assert np.ma.getdata(got['f2']).tobytes() == x.tobytes()
    with fastnc.File(p) as f:
        assert f['array'][()].shape == (4, 3, 2)    # (h5py's view reads it)


@pytest.mark.parametrize('masked', [True, False])
@pytest.mark.parametrize('name', ['f2', 'f2_fill', 'f2_scaled', 'f2_range'])
def test_the_card_equals_the_cpu(odd, name, masked, request):
    if not request.config.getoption('--gpu'):
        pytest.skip('needs --gpu')
    with fastnc.Dataset(odd[0]) as d:
        v = d[name]
        v.set_auto_maskandscale(masked)
        want, got = v[...], v.read(device='gpu').get()
    assert got.dtype == want.dtype
    assert np.ma.getdata(got).tobytes() == np.ma.getdata(want).tobytes()
    assert np.array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(want))


def test_every_variable_of_the_corpus_loads_masked():
    """One variable no default fill value exists for used to end load()."""
    import glob
    import os
    data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    for p in sorted(glob.glob(os.path.join(data, '*.nc')) + glob.glob(os.path.join(data, '*.h5'))):
        with fastnc.Dataset(p) as d:
            names = [(g.path.rstrip('/') + '/' + k).lstrip('/') for g in d._walk() for k in g.variables]
            assert set(d.load(recursive=True)) == set(names), p


@pytest.fixture(scope='module')
def path(tmp_path_factory):
    p = str(tmp_path_factory.mktemp('null') / 'null.h5')
    with h5py.File(p, 'w') as f:
        f.create_dataset('nothing', data=h5py.Empty('f4'))
        f['nothing'].attrs['_FillValue'] = np.float32(-1)
        f['nothing'].attrs['scale_factor'] = np.float32(2)
        f.create_dataset('nothing_i', data=h5py.Empty('i2'))
        f.create_dataset('values', data=np.arange(4.0))
        g = f.create_group('g')
        g.create_dataset('nothing', data=h5py.Empty('f8'))
        g.create_dataset('values', data=np.arange(3))
    return p


@pytest.mark.parametrize('masked', [True, False])
@pytest.mark.parametrize('key', [Ellipsis, ()])
@pytest.mark.parametrize('name', ['nothing', 'nothing_i'])
def test_read_gives_none(path, name, key, masked):
    with fastnc.Dataset(path) as d:
        d.set_auto_maskandscale(masked)
        assert d[name][key] is None
        assert d[name].read(key) is None


@pytest.mark.parametrize('masked', [True, False])
def test_other_variables_survive_load(path, masked):
    with fastnc.Dataset(path) as d:
        d.set_auto_maskandscale(masked)
        got = d.load(recursive=True)
    assert set(got) == {'nothing', 'nothing_i', 'values', 'g/nothing', 'g/values'}
    assert got['nothing'] is None and got['nothing_i'] is None and got['g/nothing'] is None
    assert np.array_equal(np.ma.getdata(got['values']), np.arange(4.0))
    assert np.array_equal(np.ma.getdata(got['g/values']), np.arange(3))


def test_always_mask_off(path):
    with fastnc.Dataset(path) as d:
        d['nothing'].set_always_mask(False)
        assert d['nothing'][...] is None


@pytest.mark.parametrize('masked', [True, False])
def test_a_bad_index_is_refused_alike(path, masked):
    with fastnc.Dataset(path) as d:
        d.set_auto_maskandscale(masked)
        with pytest.raises(IndexError):
            d['nothing'][0]


@pytest.mark.parametrize('masked', [True, False])
def test_nothing_goes_to_the_card(path, masked, monkeypatch):
    """(The buffer would go up unfilled: uninitialized memory as a value.)"""
    def never(*a, **k):
        raise AssertionError('a null dataset reached the GPU path')
    monkeypatch.setattr(_gpu, 'read', never)
    with fastnc.Dataset(path) as d:
        d.set_auto_maskandscale(masked)
        assert d['nothing'].read(device='gpu') is None
        with pytest.raises(IndexError):
            d['nothing'].read(0, device='gpu')
    with fastnc.Dataset(path, device='gpu') as d:
        d.set_auto_maskandscale(masked)
        assert d['nothing'][...] is None


def test_a_gpu_read_of_values_still_goes_to_the_card(path, monkeypatch):
    called = []
    monkeypatch.setattr(_gpu, 'read', lambda var, key, threads=None: called.append(var.name))
    with fastnc.Dataset(path) as d:
        d['values'].read(device='gpu')
    assert called == ['values']


def test_metadata_stays(path):
    with fastnc.Dataset(path) as d:
        v = d['nothing']
        assert v.ncattrs() == ['_FillValue', 'scale_factor']
        assert v.dtype == np.dtype('f4') and v.shape == ()


def test_xarray_opens_it(path):
    xr = pytest.importorskip('xarray')
    pytest.importorskip('fastnc._xarray')
    from fastnc._xarray import FastncBackendEntrypoint
    with xr.open_dataset(path, engine=FastncBackendEntrypoint) as ds:
        got = ds.load()
    assert np.array_equal(got['values'].values, np.arange(4.0))


@pytest.mark.parametrize('case', [lambda d: np.asarray(d), lambda d: np.array(d),
                                  lambda d: np.asarray(d, 'f8'), lambda d: list(d),
                                  lambda d: len(d), lambda d: d[0], lambda d: d[:],
                                  lambda d: d.read_direct(np.zeros(1, 'f4'))])
def test_file_refuses_what_h5py_refuses(path, case):
    with h5py.File(path) as a, fastnc.File(path) as b:
        with pytest.raises(Exception) as x:
            case(a['nothing'])
        with pytest.raises(type(x.value)):
            case(b['nothing'])


@pytest.mark.parametrize('key', [Ellipsis, ()])
def test_file_gives_empty_as_h5py(path, key):
    with h5py.File(path) as a, fastnc.File(path) as b:
        want, got = a['nothing'][key], b['nothing'][key]
        assert isinstance(want, h5py.Empty) and isinstance(got, Empty)
        assert got.dtype == want.dtype
        assert b['nothing'].shape is None and b['nothing'].size is None
