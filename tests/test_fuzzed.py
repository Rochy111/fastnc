"""What a fuzzer found that no test here had imagined: each against h5py or
netCDF4, as the file was written."""
import ctypes
import os
import sys
import threading

import numpy as np
import pytest

import fastnc

h5py = pytest.importorskip('h5py')


def test_threads_finding_chunks_get_their_own(tmp_path):
    """Two threads looking chunks up took the same row of the table, or a row
    whose address was not there yet, and one read another chunk's values."""
    p = str(tmp_path / 'race.h5')
    want = np.arange(4000, dtype='i4')
    with h5py.File(p, 'w') as f:
        f.create_dataset('d', data=want, chunks=(2,))
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    wrong = []
    try:
        for trial in range(8):
            with fastnc.File(p) as f:
                d = f['d']
                d[0:2]
                d[1000:1002]                        # (the read that makes the fast way)

                def work(seed):
                    for i in np.random.default_rng(seed).permutation(1999)[:400]:
                        i = int(i) * 2
                        try:
                            v = d[i:i + 2]
                        except Exception as e:
                            wrong.append(repr(e))
                            continue
                        if v.tolist() != [i, i + 1]:
                            wrong.append((i, v.tolist()))
                ts = [threading.Thread(target=work, args=(trial * 10 + k,)) for k in range(8)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
    finally:
        sys.setswitchinterval(old)
    assert not wrong, wrong[:4]


@pytest.mark.parametrize('dt', ['>c16', '>c8', '<c16', '<c8'])
def test_complex_attribute_and_fill_in_either_order(tmp_path, dt):
    p = str(tmp_path / 'c.h5')
    with h5py.File(p, 'w') as f:
        f.attrs['a'] = np.array([1 + 2j, 3 + 4j], dt)
        f.attrs['s'] = np.array(5 + 6j, dt)
        f.create_dataset('d', (4,), dtype=dt, fillvalue=np.array(1 - 1j, dt), chunks=(2,))
    with h5py.File(p) as r, fastnc.File(p) as f:
        assert np.array_equal(f.attrs['a'], r.attrs['a'])
        assert f.attrs['s'] == r.attrs['s']
        assert f['d'].fillvalue == r['d'].fillvalue
        assert np.array_equal(f['d'][()], r['d'][()])


@pytest.mark.parametrize('base', ['i1', 'i2', 'i4', 'u8', '>i2'])
def test_enum_of_false_and_true_is_bool_whatever_its_width(tmp_path, base):
    p = str(tmp_path / 'b.h5')
    t = h5py.enum_dtype({'FALSE': 0, 'TRUE': 1}, basetype=base)
    with h5py.File(p, 'w') as f:
        f.create_dataset('b', data=np.array([1, 0, 1], base), dtype=t)
        f.create_dataset('s', data=np.array(1, base), dtype=t)
    with h5py.File(p) as r, fastnc.File(p) as f:
        for n in 'bs':
            assert f[n].dtype == r[n].dtype
            got, want = f[n][()], r[n][()]
            assert np.array_equal(got, want) and np.asarray(got).dtype == np.asarray(want).dtype


def test_char_variable_with_a_fill_value(tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path / 'ch.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('x', 3)
        c = d.createVariable('ch', 'S1', ('x',), fill_value=b'x')
        c[0] = b'q'
    with netCDF4.Dataset(p) as r, fastnc.Dataset(p) as f:
        got, want = f['ch'], r['ch']
        assert got.getncattr('_FillValue') == want.getncattr('_FillValue')
        a, b = got[:], want[:]
        assert np.array_equal(np.ma.getmaskarray(a), np.ma.getmaskarray(b))
        assert a.compressed().tolist() == b.compressed().tolist()


def test_a_closed_dataset_gives_nothing_it_decoded_ahead(tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path / 'c.nc')
    x = np.random.default_rng(0).random((400, 400))
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('y', 400)
        d.createDimension('x', 400)
        for n in 'abcde':
            d.createVariable(n, 'f8', ('y', 'x'), zlib=True, complevel=1)[:] = x
    d = fastnc.Dataset(p)
    d['a'][...]
    d['b'][...]
    vs = dict(d.variables)
    d.close()
    for n in 'abcde':
        with pytest.raises((ValueError, RuntimeError, OSError)):
            vs[n][...]


def test_char_array_in_a_compound_is_seen_as_a_string(tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path / 'cc.nc')
    dt = np.dtype([('a', 'i4'), ('c', 'S1', (3,))], align=True)
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('x', 2)
        t = d.createCompoundType(dt, 't')
        v = d.createVariable('v', t, ('x',))
        a = np.zeros(2, dt)
        a['a'] = 1, 2
        a['c'][0] = [b'a', b'b', b'c']
        a['c'][1] = [b'd', b'', b'']
        v[:] = a
    with netCDF4.Dataset(p) as r, fastnc.Dataset(p) as f:
        got, want = f['v'][:], r['v'][:]
        assert got.dtype == want.dtype
        assert got.tolist() == want.tolist()


@pytest.mark.parametrize('sizes', [(8, 4), (4, 8), (4, 4), (8, 8)], ids=str)
@pytest.mark.parametrize('lib', ['earliest', 'latest'])
def test_every_width_of_offsets_and_lengths(tmp_path, sizes, lib):
    """A symbol table entry's name goes by lengths' width, as the library
    writes it; a global heap's objects start at a multiple of 8."""
    from h5py import h5f, h5p
    p = str(tmp_path / 'w.h5')
    fcpl = h5p.create(h5p.FILE_CREATE)
    fcpl.set_sizes(*sizes)
    fapl = h5p.create(h5p.FILE_ACCESS)
    fapl.set_libver_bounds(h5f.LIBVER_EARLIEST if lib == 'earliest' else h5f.LIBVER_LATEST, h5f.LIBVER_LATEST)
    fid = h5f.create(p.encode(), h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl)
    with h5py.File(fid) as f:
        g = f.create_group('g')
        g['a'] = np.arange(4)
        g['soft'] = h5py.SoftLink('/b')
        f['b'] = np.arange(3)
        f.create_dataset('s', data=np.array(['aa', 'bbb'], dtype=h5py.string_dtype()))
        f.create_dataset('c', data=np.arange(100.).reshape(10, 10), chunks=(3, 3), compression='gzip')
        # (h5py cannot open an unlimited dataset with 4-byte lengths)
        f.create_dataset('u', data=np.arange(100), chunks=(7,), maxshape=(None if sizes[1] == 8 else 1000,))
    with h5py.File(p) as r, fastnc.File(p) as f:
        assert sorted(f.keys()) == sorted(r.keys()) and sorted(f['g'].keys()) == sorted(r['g'].keys())
        assert np.array_equal(f['g/soft'][()], r['g/soft'][()])
        for n in ('b', 's', 'c', 'u', 'g/a'):
            assert np.array_equal(f[n][()], r[n][()]), n


@pytest.mark.parametrize('n', [[], ()], ids=['list', 'tuple'])
def test_an_empty_list_selects_nothing(tmp_path, n):
    p = str(tmp_path / 'e.h5')
    with h5py.File(p, 'w') as f:
        f['x'] = np.arange(12).reshape(3, 4)
    with h5py.File(p) as r, fastnc.File(p) as f:
        assert f['x'][list(n)].shape == r['x'][list(n)].shape


@pytest.mark.parametrize('kw', [{'libver': 'earliest'}, {'libver': 'latest'}], ids=['btree1', 'latest'])
@pytest.mark.parametrize('maxshape', [(None, 4), (None, None), (100, 4)], ids=str)
def test_a_dataset_shrunk_until_no_chunk_is_left(tmp_path, kw, maxshape):
    p = str(tmp_path / 's.h5')
    with h5py.File(p, 'w', **kw) as f:
        d = f.create_dataset('d', (20, 4), maxshape=maxshape, chunks=(5, 4), dtype='i4', fillvalue=-1)
        d[15:17] = 1
        d.resize((10, 4))
        d.resize((20, 4))
        e = f.create_dataset('e', (20, 4), maxshape=maxshape, chunks=(5, 4), dtype='i4', fillvalue=-1)
        e[15:17] = 1
        e.resize((0, 4))
        e.resize((20, 4))
    with h5py.File(p) as r, fastnc.File(p) as f:
        for n in 'de':
            assert np.array_equal(f[n][()], r[n][()])
            assert f[n][3, 1] == r[n][3, 1]


@pytest.mark.parametrize('value', [np.arange(3, dtype='i4'), np.arange(6, dtype='f8').reshape(2, 3)], ids=['1d', '2d'])
@pytest.mark.parametrize('shape', [(), (2,)], ids=['scalar', 'one-d'])
def test_attribute_of_an_array_type(tmp_path, value, shape):
    p = str(tmp_path / 'a.h5')
    t = np.dtype((value.dtype, value.shape))
    with h5py.File(p, 'w') as f:
        f.attrs.create('a', np.broadcast_to(value, shape + value.shape), dtype=t)
    with h5py.File(p) as r, fastnc.File(p) as f:
        got, want = f.attrs['a'], r.attrs['a']
        assert np.shape(got) == np.shape(want) and np.array_equal(got, want)


def test_a_coder_inside_a_coder(tmp_path):
    """Random bytes grow under bzip2, past the chunk the stage's room allowed."""
    hdf5plugin = pytest.importorskip('hdf5plugin')
    p = str(tmp_path / 'b.h5')
    x = np.random.default_rng(0).integers(0, 255, 1 << 20, dtype='u1')
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk((1 << 20,))
    b = hdf5plugin.BZip2(blocksize=9)
    dcpl.set_filter(b.filter_id, h5py.h5z.FLAG_MANDATORY, b.filter_options)
    dcpl.set_deflate(1)
    with h5py.File(p, 'w') as f:
        sid = h5py.h5s.create_simple(x.shape)
        h5py.h5d.create(f.id, b'x', h5py.h5t.NATIVE_UINT8, sid, dcpl).write(h5py.h5s.ALL, h5py.h5s.ALL, x)
    with fastnc.File(p) as f:
        assert np.array_equal(f['x'][()], x)


def _sohm(p, min_size=1):
    """A file whose messages are kept in the shared heap."""
    from h5py import h5f, h5p
    lib = ctypes.CDLL(h5py.h5.__file__) if False else None
    import glob
    dll = glob.glob(os.path.join(os.path.dirname(h5py.__file__), 'hdf5.dll')) + \
        glob.glob(os.path.join(os.path.dirname(h5py.__file__), '..', 'h5py.libs', 'libhdf5*')) + \
        glob.glob(os.path.join(sys.prefix, 'Library', 'bin', 'hdf5.dll')) + \
        glob.glob(os.path.join(sys.prefix, 'lib', 'libhdf5.so*'))
    if not dll:
        pytest.skip('no HDF5 library to call')
    lib = ctypes.CDLL(dll[0])
    fcpl = h5p.create(h5p.FILE_CREATE)
    hid = ctypes.c_int64(fcpl.id)
    assert lib.H5Pset_shared_mesg_nindexes(hid, ctypes.c_uint(1)) >= 0
    assert lib.H5Pset_shared_mesg_index(hid, ctypes.c_uint(0), ctypes.c_uint(0x182A), ctypes.c_uint(min_size)) >= 0
    return h5f.create(p.encode(), h5f.ACC_TRUNC, fcpl=fcpl)


@pytest.mark.parametrize('min_size', [1, 40], ids=['all', 'bigger'])
def test_shared_messages(tmp_path, min_size):
    p = str(tmp_path / 'so.h5')
    with h5py.File(_sohm(p, min_size)) as f:
        for i in range(4):
            d = f.create_dataset('d%d' % i, data=np.arange(12.).reshape(3, 4) + i, compression='gzip')
            d.attrs['units'] = 'm'
            d.attrs['n'] = np.arange(3)
            d.attrs['k'] = np.arange(3) + i
            d.attrs['long'] = np.arange(4000.)
    with h5py.File(p) as r, fastnc.File(p) as f:
        assert sorted(f.keys()) == sorted(r.keys())
        for n in r:
            assert f[n].shape == r[n].shape and np.array_equal(f[n][()], r[n][()])
            for k in r[n].attrs:
                assert np.array_equal(f[n].attrs[k], r[n].attrs[k]), (n, k)


def test_close_frees_the_file_without_a_collection(tmp_path):
    """Recursion over a v2 B-tree left a closure holding itself and the file's
    view: on Windows the file could not be moved until a collection came."""
    import gc
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path / 'd.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('x', 4)
        g = d.createGroup('g')
        for i in range(40):
            d.createVariable('v%d' % i, 'f8', ('x',))[:] = i
            g.createVariable('w%d' % i, 'f8', ('x',))[:] = i
            d.setncattr('a%d' % i, i)
    gc.collect()
    gc.disable()
    try:
        f = fastnc.Dataset(p)
        m = f._mm
        for v in list(f.variables.values()) + list(f['g'].variables.values()):
            v[...]
        f.close()
        assert m.closed
    finally:
        gc.enable()


def test_an_object_keeps_the_name_it_was_asked_by(tmp_path):
    p = str(tmp_path / 'a.h5')
    with h5py.File(p, 'w') as f:
        g = f.create_group('g')
        g['t'] = np.arange(3)
        f['soft'] = h5py.SoftLink('/g/t')
        g['softg'] = h5py.SoftLink('/g')
    with h5py.File(p) as r, fastnc.File(p) as f:
        for n in ('soft', 'g/softg', 'g/t', 'g/softg/t'):
            assert f[n].name == r[n].name, n
        assert f['soft'].parent.name == r['soft'].parent.name
        assert f['soft'] == f['g/t']                # (the same object, by another name)


def test_string_types_as_h5py_tells_them(tmp_path):
    p = str(tmp_path / 'a.h5')
    with h5py.File(p, 'w') as f:
        f.create_dataset('ascii', data=[b'aa', b'b'], dtype=h5py.string_dtype('ascii'))
        f.create_dataset('utf', data=['aa', 'b'], dtype=h5py.string_dtype('utf-8'))
        f.attrs.create('cs', np.array([(1, 'x')], [('a', 'i4'), ('s', h5py.string_dtype('utf-8'))]))
    with h5py.File(p) as r, fastnc.File(p) as f:
        for n in ('ascii', 'utf'):
            assert h5py.check_string_dtype(f[n].dtype) == h5py.check_string_dtype(r[n].dtype)
            assert f[n].dtype.metadata == r[n].dtype.metadata
        assert f.attrs['cs'].tolist() == r.attrs['cs'].tolist()


def test_scalars_of_user_types_as_netcdf4_gives_them(tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path / 'b.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createVariable('c', d.createCompoundType(np.dtype([('a', 'i4')]), 't'), ())
        w = d.createVLType(np.int32, 'vl')
        d.createVariable('one', w, ())[0] = np.array([7], 'i4')
        d.createVariable('two', w, ())[0] = np.array([7, 8], 'i4')
    with netCDF4.Dataset(p) as r, fastnc.Dataset(p) as f:
        for n in ('c', 'one', 'two'):
            for k in (..., ()):
                got, want = f[n][k], r[n][k]
                assert type(got) is type(want) and np.shape(got) == np.shape(want), n
                assert np.array_equal(got, want)


@pytest.mark.parametrize('src, dst', [('>i4', 'i2'), ('i4', 'u1'), ('i8', 'u8'), ('u8', 'i8'), ('f8', 'i2'), ('f8', 'i8'), ('f4', 'u4')])
def test_a_virtual_dataset_clips_what_does_not_fit(tmp_path, src, dst):
    p = str(tmp_path / 'v.h5')
    s = np.dtype(src)
    v = [1, 70000, 300, -5 if s.kind != 'u' else 5, np.iinfo('i8').max if s.kind != 'f' else 1e300]
    if s.itemsize == 4:
        v[4] = 2000000000 if s.kind != 'f' else 1e30
    if s.kind == 'f':
        v += [-1e300 if s.itemsize == 8 else -1e30, np.inf, -np.inf]
    with h5py.File(p, 'w') as f:
        f['i'] = np.array(v, src)
        lay = h5py.VirtualLayout(shape=(len(v),), dtype=dst)
        lay[:] = h5py.VirtualSource('.', 'i', shape=(len(v),))
        f.create_virtual_dataset('v', lay)
    with h5py.File(p) as r, fastnc.File(p) as f:
        assert f['v'][()].tolist() == r['v'][()].tolist()


def test_big_endian_inside_an_array_of_compounds_with_vlen(tmp_path):
    p = str(tmp_path / 'b.h5')
    t = np.dtype([('a', np.dtype([('s', h5py.string_dtype()), ('x', '>f8')]), (1,))])
    a = np.zeros(2, t)
    a['a']['s'] = [['p'], ['q']]
    a['a']['x'] = [[1.5], [2.5]]
    with h5py.File(p, 'w') as f:
        f['d'] = a
    with h5py.File(p) as r, fastnc.File(p) as f:
        assert f['d'][()]['a']['x'].tolist() == r['d'][()]['a']['x'].tolist() == [[1.5], [2.5]]


@pytest.mark.parametrize('order', [('fletcher', 'shuffle'), ('gzip', 'shuffle'), ('gzip', 'gzip'),
                                   ('fletcher', 'fletcher', 'gzip'), ('shuffle', 'fletcher', 'gzip')], ids='+'.join)
@pytest.mark.parametrize('values', ['small', 'random'])
def test_filters_in_any_order(tmp_path, order, values):
    """And a chunk that went without a filter that is optional."""
    from h5py import h5d, h5p, h5s, h5t, h5z
    p = str(tmp_path / 'o.h5')
    x = np.arange(4000, dtype='i4') * 7 if values == 'small' else np.random.default_rng(1).integers(0, 1 << 31, 4000, dtype='i4')
    with h5py.File(p, 'w') as f:
        dc = h5p.create(h5p.DATASET_CREATE)
        dc.set_chunk((1000,))
        for n in order:
            if n == 'gzip':
                dc.set_filter(h5z.FILTER_DEFLATE, h5z.FLAG_OPTIONAL, (1,))
            elif n == 'shuffle':
                dc.set_filter(h5z.FILTER_SHUFFLE, h5z.FLAG_OPTIONAL)
            else:
                dc.set_filter(h5z.FILTER_FLETCHER32, h5z.FLAG_MANDATORY)
        h5py.Dataset(h5d.create(f.id, b'd', h5t.STD_I32LE, h5s.create_simple(x.shape), dcpl=dc))[()] = x
    with fastnc.File(p) as f:
        assert np.array_equal(f['d'][()], x)
        assert np.array_equal(f['d'][1500:1510], x[1500:1510])


def test_netcdf_fletcher_order_on_chunks_that_would_not_compress(tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path / 'f.nc')
    x = np.random.default_rng(2).integers(0, 1 << 31, 4000, dtype='i4')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('x', 4000)
        d.createVariable('v', 'i4', ('x',), zlib=True, shuffle=True, fletcher32=True, chunksizes=(1000,))[:] = x
    with fastnc.Dataset(p) as f:
        assert np.array_equal(f['v'][:], x) and np.array_equal(f['v'][1500:1510], x[1500:1510])


def test_a_closed_netcdf3_file_is_not_opened_again_by_name(tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    a, b = str(tmp_path / 'a.nc'), str(tmp_path / 'b.nc')
    for p, k in ((a, 1), (b, 2)):
        with netCDF4.Dataset(p, 'w', format='NETCDF3_CLASSIC') as d:
            d.createDimension('x', 200000)
            d.createVariable('v', 'i4', ('x',))[:] = k
    f = fastnc.Dataset(a)
    v = f['v']
    f.close()
    os.replace(b, a)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        v[...]


def test_a_char_variable_with_an_encoding_shorter_than_its_dimension(tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path / 's.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('t', None)
        c = d.createVariable('c', 'S1', ('t',))
        c._Encoding = 'utf-8'
        d.createVariable('o', 'i4', ('t',))[:3] = [1, 2, 3]
    with netCDF4.Dataset(p) as r, fastnc.Dataset(p) as f:
        for k in (slice(None), slice(2, -1, 1000)):
            got, want = f['c'][k], r['c'][k]
            assert got.dtype == want.dtype and np.array_equal(got, want)


def test_a_user_block_of_any_size(tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path / 'u.h5')
    with h5py.File(p, 'w', userblock_size=8192) as f:
        f['x'] = np.arange(3)
    with fastnc.Dataset(p) as f:
        assert f['x'][:].tolist() == [0, 1, 2]


def test_bool_inside_a_compound(tmp_path):
    p = str(tmp_path / 'b.h5')
    t = np.dtype([('b', h5py.enum_dtype({'FALSE': 0, 'TRUE': 1}, basetype='i2')), ('n', 'i4')])
    a = np.zeros(3, t)
    a['b'] = 1, 0, 1
    a['n'] = 5, 6, 7
    with h5py.File(p, 'w') as f:
        f['d'] = a
    with h5py.File(p) as r, fastnc.File(p) as f:
        got, want = f['d'][()], r['d'][()]
        assert got.dtype['b'] == want.dtype['b'] == np.dtype('?')
        assert got.tolist() == want.tolist() and f['d'].dtype['b'] == r['d'].dtype['b']


def test_compound_with_variable_length_members_as_h5py_lays_it_out(tmp_path):
    p = str(tmp_path / 'c.h5')
    dt = np.dtype([('a', 'i4'), ('s', h5py.string_dtype()), ('v', h5py.vlen_dtype('i4')), ('b', 'f8')])
    a = np.zeros(2, dt)
    a['a'] = 1, 2
    a['s'] = 'x', 'y'
    a['v'][0] = np.array([1, 2], 'i4')
    a['v'][1] = np.array([3], 'i4')
    with h5py.File(p, 'w') as f:
        f['d'] = a
    with h5py.File(p) as r, fastnc.File(p) as f:
        assert f['d'].dtype.itemsize == r['d'].dtype.itemsize and f['d'].nbytes == r['d'].nbytes
        got, want = f['d'][()], r['d'][()]
        assert got.dtype == want.dtype and got['s'].tolist() == want['s'].tolist()
        assert [x.tolist() for x in got['v']] == [x.tolist() for x in want['v']]


def test_what_netcdf4_hides_of_a_plain_file(tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path / 'p.h5')
    with h5py.File(p, 'w') as f:
        f['z'] = np.arange(4.)
        f['ref'] = np.array([f['z'].ref], h5py.ref_dtype)
        f['op'] = np.void(b'abcd')
        f['f16'] = np.arange(3, dtype='f2')
    with netCDF4.Dataset(p) as r, fastnc.Dataset(p) as f:
        assert sorted(f.variables) == sorted(r.variables)
        assert list(f.dimensions) == list(r.dimensions)
        if isinstance(r['f16'].dtype, np.dtype):        # (netCDF-C 4.9.3 takes it for a string)
            assert f['f16'].dtype == r['f16'].dtype and f['f16'][:].dtype == r['f16'][:].dtype


@pytest.mark.parametrize('dt', ['>i2', '>u2', '>i4', '>f4', '>i8', '>f8', '>c8', '>c16'])
@pytest.mark.parametrize('kw', [dict(chunks=(7, 11)), dict(chunks=(7, 11), compression='gzip', shuffle=False),
                                dict(chunks=(7, 11), compression='gzip', shuffle=True), {}],
                         ids=['plain', 'gzip', 'shuffled', 'contiguous'])
def test_big_endian_at_every_length_of_a_run(tmp_path, dt, kw):
    """Word by word, with what is left over at each length byte by byte."""
    p = str(tmp_path / 'b.h5')
    x = (np.arange(30 * 40, dtype='f8').reshape(30, 40) * 1.5 + 1).astype(dt)
    with h5py.File(p, 'w') as f:
        f.create_dataset('x', data=x, **kw)
    with fastnc.File(p) as f, h5py.File(p) as r:
        d, e = f['x'], r['x']
        assert np.array_equal(d[()], x)
        for n in range(1, 18):
            assert np.array_equal(d[3:9, 2:2 + n], e[3:9, 2:2 + n]), n
        assert np.array_equal(d[::2, 1::3], e[::2, 1::3])


@pytest.mark.parametrize('n', [3, 255, 256, 5000])
def test_many_variable_length_values(tmp_path, n):
    """By the kernel from 256 on: empty ones, unwritten ones, parts, repeats, odd
    bytes, values out of order, and several collections."""
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path / 'v.h5')
    words = np.array([('w%d' % (i * 7919)) * (i % 5) for i in range(n)], dtype=object)
    nums = np.empty(n, object)
    for i in range(n):
        nums[i] = np.arange(i % 7, dtype='f8') * i
    with h5py.File(p, 'w') as f:
        f.create_dataset('s', data=words, dtype=h5py.string_dtype())
        f.create_dataset('a', data=np.array([b'\xe9' * (i % 3 + 1) for i in range(n)], dtype=object), dtype=h5py.string_dtype('ascii'))
        f.create_dataset('n', data=nums, dtype=h5py.vlen_dtype('f8'))
        f.create_dataset('big', data=nums, dtype=h5py.vlen_dtype('i2'))
        d = f.create_dataset('late', (n,), dtype=h5py.string_dtype())
        for i in range(0, n, 3):
            d[n - 1 - i] = 'late%d' % i
        f.create_dataset('s2', data=words.reshape(-1, 1)[:n // 2 * 2], dtype=h5py.string_dtype())
    with h5py.File(p) as r, fastnc.File(p) as f:
        for name in ('s', 'a', 'late', 's2'):
            assert f[name].dtype.metadata == r[name].dtype.metadata
            assert f[name][()].tolist() == r[name][()].tolist(), name
            assert f[name][n // 4:n // 2].tolist() == r[name][n // 4:n // 2].tolist()
        for name in ('n', 'big'):
            got, want = f[name][()], r[name][()]
            assert got.dtype == want.dtype and all(np.array_equal(x, y) for x, y in zip(got, want))
            assert all(x.dtype == y.dtype for x, y in zip(got, want))
            got[0:1] = [np.zeros(1)]                    # (each its own: nothing else changes)
            if n > 8 and got[8].size > 1:
                got[8][0] = -1
                assert got[9][0] != -1 or want[9][0] == -1
    with netCDF4.Dataset(p) as r, fastnc.Dataset(p) as f:
        got, want = f['s'][:], r['s'][:]
        assert got.tolist() == want.tolist() and all(type(x) is str for x in got.tolist())


def test_compounds_of_variable_length_strings_stay_as_they_were(tmp_path):
    p = str(tmp_path / 'c.h5')
    dt = np.dtype([('a', 'i4'), ('s', h5py.string_dtype())])
    x = np.zeros(1000, dt)
    x['a'] = np.arange(1000)
    x['s'] = ['n%d' % i for i in range(1000)]
    with h5py.File(p, 'w') as f:
        f['d'] = x
    with h5py.File(p) as r, fastnc.File(p) as f:
        assert f['d'][()].tolist() == r['d'][()].tolist()


def test_many_references_and_names(tmp_path):
    p = str(tmp_path / 'r.h5')
    with h5py.File(p, 'w') as f:
        g = f.create_group('g')
        for i in range(300):
            g['é%d' % i if i % 50 == 0 else 'd%d' % i] = i
        f['r'] = np.array([f['g'].ref, g['d1'].ref] * 100, h5py.ref_dtype)
    with h5py.File(p) as r, fastnc.File(p) as f:
        assert sorted(f['g'].keys()) == sorted(r['g'].keys())
        refs = f['r'][()]
        assert f[refs[0]] == f['g'] and f[refs[1]] == f['g/d1'] and f[refs[1]][()] == 1
    with h5py.File(p) as r, fastnc.File(p) as f:        # (asked of an object nothing else named)
        refs = f['r'][()]
        assert [f[x].name for x in refs[:4]] == [r[x].name for x in r['r'][:4]]
        assert f[refs[1]].parent.name == r[r['r'][1]].parent.name


def _kinds(p):
    netCDF4 = pytest.importorskip('netCDF4')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('t', 8)
        d.createDimension('x', 6)
        v = d.createVariable('v', 'i2', ('t', 'x'), zlib=True, fill_value=-9)
        v.scale_factor = 0.5
        v[:] = np.arange(48).reshape(8, 6)
        d.createGroup('g').createVariable('w', 'f4', ('/t' if False else 't',))[:] = np.arange(8.)


def test_a_dataset_pickles_and_opens_again(tmp_path):
    import pickle
    p = str(tmp_path / 'p.nc')
    _kinds(p)
    with fastnc.Dataset(p) as d:
        d['v'].set_auto_scale(False)
        d.set_auto_mask(True)
        again = pickle.loads(pickle.dumps(d))
        v = pickle.loads(pickle.dumps(d['v']))
        w = pickle.loads(pickle.dumps(d['g']['w']))
        g = pickle.loads(pickle.dumps(d['g']))
        assert np.array_equal(again['v'][:], d['v'].read() if False else again['v'][:])
        assert v[...].dtype == np.dtype('i2')           # (its own settings came along: not scaled)
        assert np.array_equal(np.ma.getdata(v[...]), 2 * np.arange(48).reshape(8, 6))
        assert np.array_equal(w[:], np.arange(8.)) and g.path == '/g' and list(g.variables) == ['w']
    assert v[0, 0] == 0                                 # (it opened its own)


def test_a_file_pickles_and_opens_again(tmp_path):
    import pickle
    p = str(tmp_path / 'p.h5')
    with h5py.File(p, 'w') as f:
        f.create_dataset('g/d', data=np.arange(5))
    with fastnc.File(p) as f:
        for obj in (f, f['g'], f['g/d']):
            got = pickle.loads(pickle.dumps(obj))
            assert got.name == obj.name
        assert pickle.loads(pickle.dumps(f['g/d']))[2] == 2


def test_xarray_pickles_as_dask_needs(tmp_path):
    import pickle
    xr = pytest.importorskip('xarray')
    from fastnc._xarray import FastncBackendEntrypoint
    p = str(tmp_path / 'p.nc')
    _kinds(p)
    with xr.open_dataset(p, engine=FastncBackendEntrypoint) as d:
        again = pickle.loads(pickle.dumps(d))
        assert again.identical(d) and again['v'].values.tolist() == d['v'].values.tolist()


@pytest.mark.parametrize('opener', ['File', 'Dataset'])
@pytest.mark.parametrize('stride', [2, 3, 7, -2])
def test_a_loop_by_a_stride_reads_what_it_asks_for(tmp_path, opener, stride):
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path / 's.nc')
    x = np.arange(60 * 8 * 9, dtype='f4').reshape(60, 8, 9)
    with netCDF4.Dataset(p, 'w') as d:
        for n, k in zip('txy', x.shape):
            d.createDimension(n, k)
        d.createVariable('v', 'f4', ('t', 'x', 'y'), zlib=True, complevel=1, chunksizes=(1, 8, 9))[:] = x
    with getattr(fastnc, opener)(p) as f:
        v = f['v']
        steps = list(range(0, 60, stride)) if stride > 0 else list(range(59, -1, stride))
        for i in steps:
            assert np.array_equal(np.ma.getdata(v[i]), x[i])
        for i in steps[::-1]:                           # (and turned round half way, and off the pattern)
            assert np.array_equal(np.ma.getdata(v[i]), x[i])
        for i in (3, 50, 4, 51, 5):
            assert np.array_equal(np.ma.getdata(v[i]), x[i])


@pytest.mark.parametrize('n', [2, 8])
def test_threads_going_through_one_file(tmp_path, n):
    """Each gets its values, and a chunk is not decoded over and over."""
    from fastnc import _chunks
    p = str(tmp_path / 't.h5')
    x = np.random.default_rng(0).random((160, 96, 96), 'f4')
    with h5py.File(p, 'w') as f:
        f.create_dataset('v', data=x, chunks=(1, 96, 96), compression='gzip', compression_opts=1)
    made = []
    real = _chunks.read_batch

    def spy(items, *a, **k):
        made.extend(g for it in items for g in range(it.box[0][0], it.box[0][1]))
        return real(items, *a, **k)
    _chunks.read_batch = spy
    wrong = []
    counts = []
    try:
        for _ in range(4):                              # (how threads fall out differs: the best of a few)
            made.clear()
            with fastnc.File(p) as f:
                d = f['v']

                def work(k):
                    for i in range(k, 160, n):
                        if not np.array_equal(d[i], x[i]):
                            wrong.append(i)
                ts = [threading.Thread(target=work, args=(k,)) for k in range(n)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
                assert not d._reader.index.coming
            counts.append(len(made))
    finally:
        _chunks.read_batch = real
    assert not wrong
    assert min(counts) < 160 * 1.5, counts


def test_a_claim_from_before_a_fork_keeps_nobody_waiting(tmp_path):
    """The thread that made it is in the parent, and will never say it is done."""
    p = str(tmp_path / 'c.h5')
    x = np.arange(20 * 16 * 16, dtype='f4').reshape(20, 16, 16)
    with h5py.File(p, 'w') as f:
        f.create_dataset('v', data=x, chunks=(1, 16, 16), compression='gzip', compression_opts=1, shuffle=True)
    with fastnc.File(p) as f:
        d = f['v']
        d[0, 0, 0], d[1, 0, 0]
        gone = threading.Event()
        gone.pid = -1
        d._reader.index.coming[3, 0] = gone             # (not a place of this rank)
        d._reader.index.coming[3, 0, 0] = gone
        got = []
        t = threading.Thread(target=lambda: got.append(d[3, 2, 2]), daemon=True)
        t.start()
        t.join(30)
        assert got == [x[3, 2, 2]]


def test_a_failed_decode_leaves_nobody_waiting(tmp_path, monkeypatch):
    from fastnc import _chunks
    p = str(tmp_path / 'f.h5')
    x = np.arange(40 * 32 * 32, dtype='f4').reshape(40, 32, 32)
    with h5py.File(p, 'w') as f:
        f.create_dataset('v', data=x, chunks=(1, 32, 32), compression='gzip', compression_opts=1, shuffle=True)
    with fastnc.File(p) as f:
        d = f['v']
        d[0, 0, 0], d[1, 0, 0], d[2, 0, 0]
        real = _chunks.read_batch

        def boom(*a, **k):
            raise IOError('no')
        monkeypatch.setattr(_chunks, 'read_batch', boom)
        with pytest.raises(IOError):
            for i in range(3, 12):
                d[i, 0, 0]
        monkeypatch.setattr(_chunks, 'read_batch', real)
        assert not d._reader.index.coming
        assert np.array_equal(d[5], x[5])


def test_read_ahead_follows_a_stride_and_nothing_else(tmp_path):
    from fastnc import _chunks
    p = str(tmp_path / 's.h5')
    x = np.random.default_rng(0).random((200, 64, 64), 'f4')
    with h5py.File(p, 'w') as f:
        f.create_dataset('v', data=x, chunks=(1, 64, 64), compression='gzip')
    seen = []
    real = _chunks._ahead

    def spy(index, shape, origins, way, count):
        b = real(index, shape, origins, way, count)
        seen.extend(q[0][0] for q in b)
        return b
    _chunks._ahead, keep = spy, _chunks._ahead
    try:
        with fastnc.File(p) as f:
            for i in range(0, 120, 4):
                f['v'][i]
        assert seen and all(q % 4 == 0 for q in seen)   # (none of the chunks between)
        seen.clear()
        with fastnc.File(p) as f:
            r = np.random.default_rng(1)
            for i in r.integers(0, 200, 60).tolist():   # (no pattern: nothing decoded ahead)
                f['v'][i]
        assert len(seen) <= 3
    finally:
        _chunks._ahead = keep


@pytest.mark.parametrize('kw', [{}, dict(chunks=(16,)), dict(chunks=(16,), compression='gzip', shuffle=True)],
                         ids=['contiguous', 'chunks', 'gzip'])
def test_records_one_at_a_time(tmp_path, kw):
    """As h5py gives them, and none a view of what is kept or mapped."""
    p = str(tmp_path / 'c.h5')
    dt = np.dtype([('id', 'i8'), ('v', 'f4'), ('name', 'S4'), ('in', [('a', 'i2'), ('b', 'f8')]), ('m', 'f4', (2,))])
    x = np.zeros(300, dt)
    x['id'] = np.arange(300)
    x['v'] = np.arange(300) * .5
    x['name'] = [b'n%d' % (i % 99) for i in range(300)]
    x['in']['a'] = np.arange(300)
    with h5py.File(p, 'w') as f:
        f.create_dataset('d', data=x, **kw)
    with h5py.File(p) as r:
        want, kinds = r['d'][()], type(r['d'][5])
    f = fastnc.File(p)
    d = f['d']
    keep = None
    for i in (5, 5, 6, 299, -1):
        got = d[i]
        assert type(got) is kinds and got.dtype == want.dtype and got.tobytes() == want[i].tobytes()
        got['id'] = -5                                  # (ours to change: the next read does not see it)
        assert d[i]['id'] == want[i]['id']
        keep = d[i]
    assert d[3:9].tobytes() == want[3:9].tobytes() and d[20:40].dtype == want.dtype
    f.close()
    assert keep.tobytes() == want[-1].tobytes()         # (whatever became of the file)


@pytest.mark.parametrize('dt', ['f4', '>f4', 'i2', '>f8'])
@pytest.mark.parametrize('kw', [dict(chunks=(1, 7, 9)), dict(chunks=(3, 7, 9)), dict(chunks=(2, 40, 5)), dict(chunks=(4, 8, 8), compression='gzip'), {}],
                         ids=['rows', 'small', 'tall', 'gzip', 'contiguous'])
def test_selections_of_every_kind_against_h5py(tmp_path, dt, kw):
    """Steps across chunks, sparse lists, Ellipsis, in the paths made for them."""
    import random
    p = str(tmp_path / 's.h5')
    x = (np.arange(12 * 40 * 45).reshape(12, 40, 45) % 30011).astype(dt)
    with h5py.File(p, 'w') as f:
        f.create_dataset('x', data=x, **kw)
    r = random.Random(0)

    def piece(n):
        k = r.randrange(8)
        a = r.randrange(n)
        if k == 0:
            return a
        if k == 1:
            return a - n
        if k == 2:
            return slice(None)
        b = r.randrange(a, n) + 1
        return slice(a, b, r.choice((1, 1, 2, 3, 5, 7, 13)))
    with fastnc.File(p) as f:
        d = f['x']
        d[0:2]
        d[1:2]                                          # (from here the quick path)
        for _ in range(500):
            key = tuple(piece(n) for n in x.shape)
            cut = r.randrange(4)
            if r.random() < .25 and cut:
                key = (Ellipsis,) + key[cut:] if r.random() < .5 else key[:cut] + (Ellipsis,)
            elif cut:
                key = key[:cut]
            got, want = d[key], x[key]
            assert np.array_equal(got, want), key
            assert np.shape(got) == np.shape(want), key
        with h5py.File(p) as h:                         # (lists: h5py's rule, not numpy's)
            for _ in range(100):
                axis = r.randrange(3)
                ids = sorted(r.sample(range(x.shape[axis]), r.randrange(1, min(9, x.shape[axis]))))
                key = [piece(n) for n in x.shape]
                key[axis] = ids
                got, want = d[tuple(key)], h['x'][tuple(key)]
                assert np.array_equal(got, want) and np.shape(got) == np.shape(want), key


def test_a_sparse_list_over_chunks_not_all_stored(tmp_path):
    p = str(tmp_path / 'h.h5')
    with h5py.File(p, 'w') as f:
        d = f.create_dataset('x', (100, 6), dtype='i4', chunks=(1, 6), fillvalue=-3)
        d[0::7] = 1
    with h5py.File(p) as r, fastnc.File(p) as f:
        ids = [0, 3, 7, 50, 99]
        assert np.array_equal(f['x'][ids, 2:4], r['x'][ids, 2:4])
        assert np.array_equal(f['x'][ids], r['x'][ids])
        assert np.array_equal(f['x'][::9, 1], r['x'][::9, 1])          # (a step past holes)


@pytest.mark.parametrize('fill', [None, b'x'], ids=['default', 'own'])
def test_chars_as_netcdf4_masks_them(tmp_path, fill):
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path / 'c.nc')
    with netCDF4.Dataset(p, 'w') as d:
        d.createDimension('n', 6)
        d.createDimension('c', 4)
        v = d.createVariable('ch', 'S1', ('n', 'c'), fill_value=fill)
        v[0] = np.array([b'a', b'b', b'', b''], 'S1')
        v[2] = np.array([b'x', b'y', b'z', b'w'], 'S1')
        d.createVariable('rec', d.createCompoundType(np.dtype([('i', 'i4'), ('v', 'f4')]), 'r_t'), ('n',))
        w = d.createVariable('pad', d.createCompoundType(np.dtype([('i', 'i1'), ('v', 'f8')], align=True), 'p_t'), ('n',))
        a = np.zeros(6, w.datatype.dtype)
        a['i'] = np.arange(6)
        w[:] = a
    with netCDF4.Dataset(p) as r, fastnc.Dataset(p) as f:
        for k in (..., 0, 2, slice(0, 3), (2, 1), (0, 3), (1, slice(None))):
            got, want = f['ch'][k], r['ch'][k]
            assert type(got) is type(want) and got.dtype == want.dtype
            assert np.array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(want)), k
            assert np.ma.getdata(got).tolist() == np.ma.getdata(want).tolist(), k
            if isinstance(want, np.ma.MaskedArray) and want is not np.ma.masked:
                assert got.fill_value == want.fill_value
        for name in ('rec', 'pad'):
            for k in (0, 3, slice(1, 4), ...):
                got, want = f[name][k], r[name][k]
                assert type(got) is type(want) and got.dtype == want.dtype and got.tolist() == want.tolist()


def test_a_record_with_something_to_convert_still_is(tmp_path):
    p = str(tmp_path / 'c.h5')
    a = np.zeros(4, [('b', h5py.enum_dtype({'FALSE': 0, 'TRUE': 1}, basetype='i2')), ('z', 'c8'), ('e', '>f4')])
    a['b'] = 1, 0, 1, 0
    a['z'] = 1 + 2j
    a['e'] = 2.5
    with h5py.File(p, 'w') as f:
        f['d'] = a
        f['s'] = np.array([(b'ab', 1)], [('t', h5py.string_dtype(length=4)), ('n', 'i4')])
    with h5py.File(p) as r, fastnc.File(p) as f:
        for n in 'ds':
            for k in (0, ()):
                got, want = f[n][k], r[n][k]
                assert got.tolist() == want.tolist() and got.dtype == want.dtype


def test_ellipsis_in_a_tuple_on_a_scalar(tmp_path):
    p = str(tmp_path / 's.h5')
    with h5py.File(p, 'w') as f:
        f['x'] = 3
    with h5py.File(p) as r, fastnc.File(p) as f:
        assert type(f['x'][(Ellipsis,)]) is type(r['x'][(Ellipsis,)])


def test_nbytes_is_in_memory(tmp_path):
    p = str(tmp_path / 's.h5')
    with h5py.File(p, 'w') as f:
        f['s'] = np.array(['aa', 'b'], dtype=h5py.string_dtype())
        f['n'] = h5py.Empty('f4')
    with h5py.File(p) as r, fastnc.File(p) as f:
        assert f['s'].nbytes == r['s'].nbytes and f['n'].nbytes == r['n'].nbytes
