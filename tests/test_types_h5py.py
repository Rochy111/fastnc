"""Types and layouts that only odd files have, as the HDF Group's own test files
showed them wrong: what h5py gives is the answer."""
import numpy as np
import pytest

import fastnc

h5py = pytest.importorskip('h5py')


def equal(a, b):
    if a.dtype.names:
        return all(equal(a[n], b[n]) for n in a.dtype.names)
    if a.dtype.kind == 'O':
        return all(equal(np.asarray(x), np.asarray(y)) if isinstance(y, np.ndarray) else x == y
                   for x, y in zip(a.ravel().tolist(), b.ravel().tolist()))
    return a.shape == b.shape and np.array_equal(a, b)


@pytest.fixture(scope='module')
def path(tmp_path_factory):
    p = str(tmp_path_factory.mktemp('types') / 'types.h5')
    vstr = h5py.string_dtype()
    with h5py.File(p, 'w') as h:
        # strings padded with spaces, and ended by a NUL with bytes after it
        for name, pad in (('spaces', h5py.h5t.STR_SPACEPAD), ('nul', h5py.h5t.STR_NULLTERM)):
            t = h5py.h5t.C_S1.copy()
            t.set_size(8)
            t.set_strpad(pad)
            d = h5py.h5d.create(h.id, name.encode(), t, h5py.h5s.create_simple((3,)))
            raw = b'ab      ' + b'cd e    ' + b'fghijklm' if name == 'spaces' else \
                b'ab\0junk\0' + b'\0xxxxxxx' + b'fghijklm'
            d.write(h5py.h5s.ALL, h5py.h5s.ALL, np.frombuffer(raw, 'S8'), mtype=t)
        d = np.zeros(4, [('name', 'S8'), ('x', 'i4')])
        d['name'] = [b'a', b'bb', b'ccc', b'dddd']
        d['x'] = np.arange(4)
        h['plain'] = d
        # variable-length values inside compounds and array types
        t = np.dtype([('s', vstr), ('v', h5py.vlen_dtype('i4')), ('n', 'i4')])
        c = h.create_dataset('compound', (3,), t)
        for i in range(3):
            c[i] = ('s%d' % i, np.arange(i + 1, dtype='i4'), i)
        a = h.create_dataset('arrayvlen', (2,), np.dtype((h5py.vlen_dtype('i4'), (3,))))
        for i in range(2):
            a[i] = [np.arange(k + i, dtype='i4') for k in range(3)]
        n = h.create_dataset('nested', (2,), np.dtype([('r', np.dtype([('s', vstr), ('q', 'f8')]))]))
        n[0] = (('x', 1.5),)
        n[1] = (('yy', 2.5),)
        s = h.create_dataset('strarray', (2,), np.dtype([('m', vstr, (2,)), ('k', 'i2')]))
        s[0] = (['a', 'bb'], 1)
        s[1] = (['ccc', ''], 2)
        # array types: nested, big-endian, past the size read straight
        h.create_dataset('array', (3,), np.dtype(('i4', (4,))))[...] = np.arange(36).reshape(3, 4, 3)[:, :, 0]
        t = h5py.h5t.array_create(h5py.h5t.array_create(h5py.h5t.NATIVE_INT32, (2, 3)), (4,))
        d = h5py.h5d.create(h.id, b'array2', t, h5py.h5s.create_simple((2,)))
        d.write(h5py.h5s.ALL, h5py.h5s.ALL, np.arange(48, dtype='i4').reshape(2, 4, 2, 3), mtype=t)
        h.create_dataset('arraybig', (3,), np.dtype(('>f8', (2,))))[...] = np.arange(6.).reshape(3, 2)
        h.create_dataset('arraylong', (2000,), np.dtype(('i4', (1000,))))[...] = np.arange(2000000).reshape(2000, 1000)
    return p


@pytest.mark.parametrize('name', ['spaces', 'nul', 'plain', 'compound', 'arrayvlen', 'nested', 'strarray',
                                  'array', 'array2', 'arraybig', 'arraylong'])
def test_as_h5py(path, name):
    with fastnc.File(path) as f, h5py.File(path, 'r') as h:
        x, y = f[name], h[name]
        assert equal(x[()], y[()])
        n = x.shape[0]
        for k in (0, n - 1, slice(0, n, 2)):
            assert equal(np.asarray(x[k]), np.asarray(y[k])), k


def test_lone_chunk_is_where_it_is(tmp_path):
    """One chunk stored, not the first: reads before it are fill values."""
    p = str(tmp_path / 'lone.h5')
    with h5py.File(p, 'w') as h:
        d = h.create_dataset('x', (1 << 24,), 'i1', chunks=(1 << 16,), fillvalue=-1)
        d[1 << 20:(1 << 20) + (1 << 16)] = 31
    with fastnc.File(p) as f, h5py.File(p, 'r') as h:
        x, y = f['x'], h['x']
        for k in (0, slice(0, 8), 1 << 20, slice((1 << 20) - 4, (1 << 20) + 4), (1 << 24) - 1):
            assert np.array_equal(x[k], y[k]), k


@pytest.mark.parametrize('dt', [np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), np.dtype('c16'),
                                np.dtype([('a', 'i4'), ('b', 'f8'), ('c', 'u1')]), np.dtype('>i4')])
def test_contiguous_dataset_ending_the_file(tmp_path, monkeypatch, dt):
    """A contiguous dataset read as row blocks whose last block runs past the
    end of the file (the dataset ends it, its rows not a multiple of the
    block): compounds and h5py's complex numbers failed to reshape, read as a
    process reads before its kernels are loaded."""
    from fastnc import _chunks, _h5read
    monkeypatch.setattr(_chunks, 'ready', lambda: False)
    monkeypatch.setattr(_h5read, 'VIRTUAL_MIN', 1 << 12)
    monkeypatch.setattr(_h5read, 'VIRTUAL_CHUNK', 1 << 16)  # (smaller blocks: a smaller file shows it)
    p = str(tmp_path / 'end.h5')
    raw = np.random.default_rng(5).integers(0, 255, 3001 * 300 * dt.itemsize, dtype=np.uint8)
    with h5py.File(p, 'w') as h:
        h['first'] = np.arange(3)
        h['v'] = raw.view(dt).reshape(3001, 300)
    with fastnc.File(p) as f, h5py.File(p, 'r') as h:
        for k in ((), (slice(None, None, 3),), (slice(-5, None),), (slice(1, None, 2), slice(None, None, 5))):
            a, b = h['v'][k], f['v'][k]
            assert a.tobytes() == np.ascontiguousarray(b).astype(a.dtype).tobytes(), k


def test_old_format_array_members_through_dataset(tmp_path):
    """A compound's array members, read by the netCDF4 API."""
    p = str(tmp_path / 'c.h5')
    t = np.dtype([('a', 'i4', (2, 3)), ('b', 'f8')])
    with h5py.File(p, 'w', libver='earliest') as h:
        d = h.create_dataset('x', (2,), t)
        d[...] = np.array([(np.arange(6).reshape(2, 3), 1.5), (np.arange(6, 12).reshape(2, 3), 2.5)], t)
    with fastnc.Dataset(p) as a, h5py.File(p, 'r') as h:
        got, want = a.variables['x'][:], h['x'][()]
        assert np.array_equal(got['a'], want['a']) and np.array_equal(got['b'], want['b'])


@pytest.mark.parametrize('libver', ['earliest', 'latest'])
def test_region_references(tmp_path, libver):
    p = str(tmp_path / 'r.h5')
    a = np.arange(120, dtype='i4').reshape(10, 12)
    with h5py.File(p, 'w', libver=libver) as h:
        d = h.create_dataset('x', data=a)
        s = h.create_dataset('s', (9,), dtype=h5py.regionref_dtype)
        s[0] = d.regionref[...]
        s[1] = d.regionref[2:5, 3:7]
        s[2] = d.regionref[0:10:2, 1]
        s[3] = d.regionref[1:9:3, 2:10:2]
        s[4] = d.regionref[3, 4]
        mask = np.zeros(a.shape, bool)
        mask[[1, 1, 4, 8], [2, 9, 4, 0]] = True
        s[5] = d.regionref[mask]
        s[6] = d.regionref[0:0]
        sid = d.id.get_space()
        sid.select_elements([(7, 1), (2, 3), (5, 5)])
        s[7] = h5py.h5r.create(d.id, b'.', h5py.h5r.DATASET_REGION, sid)
        sid.select_hyperslab((0, 0), (2, 2))
        sid.select_hyperslab((5, 5), (2, 2), op=h5py.h5s.SELECT_OR)
        s[8] = h5py.h5r.create(d.id, b'.', h5py.h5r.DATASET_REGION, sid)
    with fastnc.File(p) as f, h5py.File(p, 'r') as h:
        got, want = f['s'][()], h['s'][()]
        assert h['s'].dtype.metadata['ref'] is h5py.RegionReference
        for i, (r, w) in enumerate(zip(got, want)):
            assert isinstance(r, fastnc._h5py.RegionReference)
            assert bool(r) == bool(w)
            x, y = f['x'][r], h['x'][w]
            assert x.shape == y.shape and np.array_equal(x, y), i
            assert f[r].name == '/x' or f[r]._address == f['x']._address


def test_region_of_another_dataset(tmp_path):
    p = str(tmp_path / 'o.h5')
    with h5py.File(p, 'w') as h:
        d = h.create_dataset('x', data=np.arange(5))
        h.create_dataset('y', data=np.arange(5))
        h.create_dataset('s', (1,), dtype=h5py.regionref_dtype)[0] = d.regionref[1:3]
    with fastnc.File(p) as f:
        with pytest.raises(ValueError):
            f['y'][f['s'][0]]


def test_external_links(tmp_path):
    """Followed as h5py follows them, relative to the file; a missing one a KeyError."""
    other = str(tmp_path / 'other.h5')
    p = str(tmp_path / 'main.h5')
    with h5py.File(other, 'w') as h:
        h['a/b'] = np.arange(4)
    with h5py.File(p, 'w') as h:
        h['ok'] = h5py.ExternalLink('other.h5', '/a/b')
        h['grp'] = h5py.ExternalLink('other.h5', '/a')
        h['gone'] = h5py.ExternalLink('nothing.h5', '/x')
    with fastnc.File(p) as f, h5py.File(p, 'r') as h:
        assert np.array_equal(f['ok'][()], h['ok'][()])
        assert np.array_equal(f['grp/b'][()], h['grp/b'][()])
        assert f['ok'].name == h['ok'].name
        assert 'gone' not in f
        with pytest.raises(KeyError):
            f['gone']


def test_soft_links_to_soft_links(tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    p = str(tmp_path / 's.nc')
    with h5py.File(p, 'w') as h:
        h['x'] = np.arange(3)
        h['one'] = h5py.SoftLink('/x')
        h['two'] = h5py.SoftLink('/one')
    with fastnc.Dataset(p) as a, netCDF4.Dataset(p) as n:
        assert sorted(a.variables) == sorted(n.variables)
        assert np.array_equal(a.variables['two'][:], n.variables['two'][:])


def test_more_filters_than_the_kernels_hold(tmp_path):
    p = str(tmp_path / 'f.h5')
    a = np.arange(4000, dtype='i4').reshape(40, 100)
    with h5py.File(p, 'w') as h:
        pl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        pl.set_chunk((10, 25))
        pl.set_shuffle()
        for _ in range(9):
            pl.set_deflate(1)
        d = h5py.h5d.create(h.id, b'x', h5py.h5t.NATIVE_INT32, h5py.h5s.create_simple((40, 100)), dcpl=pl)
        d.write(h5py.h5s.ALL, h5py.h5s.ALL, a)
    with fastnc.File(p) as f:
        assert f['x']._ds.filters.__len__() == 10
        assert np.array_equal(f['x'][()], a)
        assert np.array_equal(f['x'][3:17, 20:60], a[3:17, 20:60])
    with fastnc.Dataset(p) as d:
        assert np.array_equal(d.variables['x'][...], a)


def test_big_endian_enums(tmp_path):
    p = str(tmp_path / 'e.h5')
    with h5py.File(p, 'w') as h:
        t = h5py.h5t.enum_create(h5py.h5t.STD_I32BE)
        for i, k in enumerate((b'a', b'b', b'c')):
            t.enum_insert(k, i)
        d = h5py.h5d.create(h.id, b'x', t, h5py.h5s.create_simple((6,)))
        d.write(h5py.h5s.ALL, h5py.h5s.ALL, np.array([0, 1, 2, 2, 1, 0], '>i4'), mtype=t)
    with fastnc.File(p) as f, h5py.File(p, 'r') as h:
        assert np.array_equal(f['x'][()], h['x'][()]) and f['x'][2] == h['x'][2] == 2


def test_shuffled_by_another_size(tmp_path):
    """The filter's own element size, not the type's (PyTables, variable-length records)."""
    p = str(tmp_path / 'sh.h5')
    a = np.arange(600, dtype='i8').reshape(100, 6)
    with h5py.File(p, 'w') as h:
        pl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        pl.set_chunk((10,))
        pl.set_filter(h5py.h5z.FILTER_SHUFFLE, 0, (8,))
        pl.set_deflate(1)
        t = h5py.h5t.array_create(h5py.h5t.NATIVE_INT64, (2,))
        d = h5py.h5d.create(h.id, b'x', h5py.h5t.py_create(np.dtype([('a', 'i8'), ('b', 'i8')])), h5py.h5s.create_simple((50,)), dcpl=pl)
        v = np.zeros(50, [('a', 'i8'), ('b', 'i8')])
        v['a'], v['b'] = np.arange(50), np.arange(50) * 3
        d.write(h5py.h5s.ALL, h5py.h5s.ALL, v)
        del t, a
    with fastnc.File(p) as f, h5py.File(p, 'r') as h:
        for k in ((), 7, slice(3, 40)):
            assert equal(np.asarray(f['x'][k]), np.asarray(h['x'][k])), k


def test_odd_width_integers(tmp_path):
    p = str(tmp_path / 'i.h5')
    with h5py.File(p, 'w') as h:
        for name, size, sign, order in (('s3', 3, h5py.h5t.SGN_2, h5py.h5t.ORDER_LE), ('u5', 5, h5py.h5t.SGN_NONE, h5py.h5t.ORDER_LE),
                                        ('b6', 6, h5py.h5t.SGN_2, h5py.h5t.ORDER_BE)):
            t = h5py.h5t.STD_I64LE.copy()
            t.set_order(order)
            t.set_sign(sign)
            t.set_precision(size * 8)
            t.set_size(size)
            d = h5py.h5d.create(h.id, name.encode(), t, h5py.h5s.create_simple((4,)))
            v = np.array([0, 1, 200, -5 if sign == h5py.h5t.SGN_2 else 77], 'i8')
            d.write(h5py.h5s.ALL, h5py.h5s.ALL, v, mtype=h5py.h5t.NATIVE_INT64)
    with fastnc.File(p) as f:                       # (h5py: "data type '<i3' not understood")
        assert f['s3'][()].tolist() == [0, 1, 200, -5] and f['s3'].dtype == np.dtype('i4')
        assert f['u5'][()].tolist() == [0, 1, 200, 77] and f['u5'].dtype == np.dtype('u8')
        assert f['b6'][()].tolist() == [0, 1, 200, -5] and f['b6'].dtype == np.dtype('i8')


def test_nc3_name_with_a_nul(tmp_path):
    p = str(tmp_path / 'n.nc')
    netCDF4 = pytest.importorskip('netCDF4')
    with netCDF4.Dataset(p, 'w', format='NETCDF3_CLASSIC') as d:
        d.createDimension('ii', 3)
        d.createVariable('yy', 'i2', ('ii',))[:] = [1, 2, 3]
    raw = bytearray(open(p, 'rb').read())
    at = raw.index(b'yy\0\0')
    raw[at - 1] = 3                                 # (counted as three bytes: the third a NUL)
    open(p, 'wb').write(raw)
    with fastnc.Dataset(p) as d, netCDF4.Dataset(p) as n:
        assert list(d.variables) == list(n.variables) == ['yy']


def test_hdf4_szip():
    """A real stream, from the HDF Group's sds_compressed.hdf: 16 by 5 big-endian
    int32, 1..20 along each row; blocks of two samples, flag and length in front."""
    from fastnc import _hdf4
    src = np.frombuffer(bytes.fromhex(
        '0000000073020408102040810204081020408102040810204081020408102040810204081020408102040810204081'
        '294991c9544a8624cd590e44ad6d0635669c41cd6d922ce1a7192ccf1922c9b63064b45718d19371656388ae45b98f'
        '22cb17761845ba27861a45de120b0e22788494c3c8483094e1c0'), np.uint8)
    out = np.zeros(320, np.uint8)
    assert _hdf4._szip_decode((80, 5, 65668, 32, 2), src, 0, src.size, out) >= 0
    want = np.array([[i + j + 1 for j in range(5)] for i in range(16)], '>i4')
    assert np.array_equal(out.view('>i4').reshape(16, 5), want)


def test_group_inside_itself(tmp_path):
    """netCDF-C overflows its stack on such a file."""
    p = str(tmp_path / 'loop.h5')
    with h5py.File(p, 'w') as h:
        g = h.create_group('a')
        g['x'] = np.arange(3)
        s = g.create_group('b')
        s['up'] = g
        s['root'] = h['/']
    with fastnc.Dataset(p) as d:
        assert list(d.groups) == ['a'] and list(d['a'].groups) == ['b']
        assert not d['a/b'].groups
        assert d['a'].variables['x'][1] == 1


def test_shared_datatype_of_the_old_format(tmp_path):
    p = str(tmp_path / 'old.h5')
    with h5py.File(p, 'w', libver='earliest') as h:
        h['t'] = np.dtype([('a', '>i4'), ('b', '>f4')])
        d = h.create_dataset('x', (3,), h['t'])
        d[...] = np.array([(1, 1.5), (2, 2.5), (3, 3.5)], d.dtype)
    with fastnc.File(p) as f, h5py.File(p, 'r') as h:
        assert equal(f['x'][()], h['x'][()])
