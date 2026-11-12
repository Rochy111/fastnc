"""fastnc.File's h5py conveniences against h5py itself on the test files:
asstr(), astype(), fields(), len(), iter_chunks(), .ref and .dims."""
import os
import warnings

import numpy as np
import pytest

import fastnc
from fastnc import _hdf5

h5py =pytest.importorskip('h5py')
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FILES = ['types.h5', 'layouts.h5', 'nc4.nc', 'groups_new.h5', 'filters.h5']


def pairs():
    for fn in FILES:
        p = os.path.join(DATA, fn)
        with h5py.File(p, 'r') as h:
            names = []
            h.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        yield from ((p, n) for n in names)


PAIRS = list(pairs())


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == 'O' or b.dtype.kind == 'O':
        return a.shape == b.shape and a.tolist() == b.tolist()
    return a.shape == b.shape and a.dtype == b.dtype.newbyteorder('=') and \
        a.tobytes() == b.astype(a.dtype).tobytes()


def outcome(fn):
    try:
        return 'ok', fn()
    except Exception as e:
        return 'raises', type(e)


@pytest.mark.parametrize('path,name', PAIRS, ids=['%s:%s' % (os.path.basename(p), n) for p, n in PAIRS])
def test_conveniences_as_h5py(path, name):
    with warnings.catch_warnings(), h5py.File(path, 'r') as hf, fastnc.File(path) as ff:
        warnings.simplefilter('ignore')
        h, f = hf[name], ff[name]
        if outcome(lambda: h[()])[0] == 'raises':
            pytest.skip('h5py cannot read it')                  # (external files, filters it lacks)
        if h.shape and h.size:
            assert f.len() == h.len()
            if h.dtype.kind == 'S' or h5py.check_string_dtype(h.dtype):
                assert same(f.asstr()[()], h.asstr()[()])
                assert same(f.asstr()[0], h.asstr()[0])
            else:
                with pytest.raises(TypeError):
                    f.asstr()
            if h.dtype.names:
                first = h.dtype.names[0]
                assert same(f.fields(first)[()], h.fields(first)[()])
                two = list(h.dtype.names[:2])
                assert same(f.fields(two)[()], h.fields(two)[()])
            if h.dtype.kind in 'iuf' and h.dtype.names is None and h.dtype.itemsize > 2:
                assert same(f.astype('f8')[()], h.astype('f8')[()])
            if h.chunks:
                sel = (slice(1, min(7, h.shape[0])),) + tuple(slice(0, n) for n in h.shape[1:])
                assert list(f.iter_chunks()) == list(h.iter_chunks())
                assert list(f.iter_chunks(sel)) == list(h.iter_chunks(sel))
        assert ff[f.ref] == f and ff[f.ref].name == hf[h.ref].name
        assert ff[f.parent.ref] == f.parent
        assert len(f.dims) == len(h.dims)
        for fd, hd in zip(f.dims, h.dims):
            assert fd.label == hd.label
            assert fd.keys() == hd.keys() and len(fd) == len(hd)
            assert [s.name for s in fd.values()] == [s.name for s in hd.values()]
            for k in hd.keys():
                assert fd[k].name == hd[k].name
            if len(hd):
                assert fd[0].name == hd[0].name


def test_links_and_file_properties(tmp_path):
    """get(getlink=True), visit_links / visititems_links, require_group,
    is_scale, userblock_size, driver, as h5py."""
    other = str(tmp_path / 'other.h5')
    with h5py.File(other, 'w') as h:
        h['far/x'] = np.arange(3)
    p = str(tmp_path / 'links.h5')
    with h5py.File(p, 'w', userblock_size=1024) as h:
        h['g/a'] = np.arange(5)
        h['g/sub/b'] = np.ones(2)
        h['z'] = np.zeros(3)
        h['soft'] = h5py.SoftLink('/g/sub/b')
        h['dangling'] = h5py.SoftLink('/nowhere')
        h['ext'] = h5py.ExternalLink('other.h5', '/far/x')
        h['g/a'].make_scale('a')
        h['z'].make_scale()
    with h5py.File(p, 'r') as h, fastnc.File(p) as f:
        assert f.userblock_size == h.userblock_size == 1024
        assert f.driver == h.driver and f.swmr_mode is False
        f.flush()
        for name in ('g', 'g/a', 'soft', 'dangling', 'ext', '/g/sub', 'nope', 'g/nope'):
            a, b = h.get(name, getlink=True), f.get(name, getlink=True)
            assert type(a).__name__ == type(b).__name__, name
            for att in ('path', 'filename'):
                assert getattr(a, att, None) == getattr(b, att, None), (name, att)
            ca, cb = h.get(name, getclass=True, getlink=True), f.get(name, getclass=True, getlink=True)
            assert getattr(ca, '__name__', ca) == getattr(cb, '__name__', cb), name
        got, want = [], []
        f.visititems_links(lambda n, link: got.append((n, type(link).__name__, getattr(link, 'path', None))))
        h.visititems_links(lambda n, link: want.append((n, type(link).__name__, getattr(link, 'path', None))))
        assert got == want
        names = []
        assert f['g'].visit_links(lambda n: names.append(n)) is None
        hn = []
        h['g'].visit_links(lambda n: hn.append(n))
        assert names == hn
        assert f.visit_links(lambda n: n if n == 'g/sub' else None) == 'g/sub'     # (stops)
        assert f.require_group('g') == f['g']
        with pytest.raises(TypeError):
            f.require_group('z')
        with pytest.raises(ValueError):
            f.require_group('new')
        for n in ('g/a', 'z', 'g/sub/b'):
            assert f[n].is_scale == h[n].is_scale, n
        f['z'].refresh()
        f['z'].flush()


def test_float_index_refused_as_h5py():
    p = os.path.join(DATA, 'layouts.h5')
    with h5py.File(p, 'r') as h, fastnc.File(p) as f:
        name = next(n for n in h if isinstance(h[n], h5py.Dataset) and h[n].shape)
        for key in (0.0, np.float32(1), (0, 1.5)):
            with pytest.raises(TypeError):
                h[name][key]
            with pytest.raises(TypeError):
                f[name][key]


def test_astype_saturates_as_hdf5(tmp_path):
    """To a narrower integer type HDF5 truncates floats and saturates what
    is out of range; numpy would wrap."""
    p = str(tmp_path / 'conv.h5')
    f8 = np.array([np.inf, -np.inf, 1e10, -1e10, 2.7, -2.7, 0.5, -0.5, 1.5, 32767.9, -32768.9,
                   65535.5, 1e300, -2.0 ** 63], 'f8')     # (not 2^63 itself: HDF5's own answer
                                                            # there for float32 is its hardware's)
    i8 = np.array([-2 ** 63, -70000, -1, 0, 255, 256, 40000, 70000, 2 ** 62, 2 ** 63 - 1], 'i8')
    u8 = np.array([0, 255, 256, 40000, 70000, 2 ** 63, 2 ** 64 - 1], 'u8')
    with h5py.File(p, 'w') as h:
        h['f8'], h['f4'], h['i8'], h['u8'], h['i2'] = f8, f8.astype('f4'), i8, u8, i8.astype('i2')
    with h5py.File(p, 'r') as h, fastnc.File(p) as f:
        for src in h:
            for t in ('i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'f4', 'f8'):
                want = h[src].astype(t)[()]
                got = f[src].astype(t)[()]
                assert got.dtype == want.dtype and got.tobytes() == want.tobytes(), (src, t)
                assert f[src].astype(t)[2:5].tobytes() == want[2:5].tobytes(), (src, t)


@pytest.mark.parametrize('fn', FILES + ['complex/h5py.h5'])
def test_attribute_ids_as_h5py(fn, tmp_path):
    p = os.path.join(DATA, fn)
    with h5py.File(p, 'r') as h, fastnc.File(p) as f:
        objs = ['']
        h.visit(objs.append)
        for n in objs:
            a, b = (h[n], f[n]) if n else (h, f)
            if not hasattr(a, 'attrs'):
                continue
            for k in a.attrs:
                x, y = a.attrs.get_id(k), b.attrs.get_id(k)
                assert (x.name, x.shape, x.dtype) == (y.name, y.shape, y.dtype), (n, k)
                assert x.dtype.metadata == y.dtype.metadata, (n, k)
                assert x.get_storage_size() == y.get_storage_size(), (n, k)
    q = str(tmp_path / 'empty.h5')
    with h5py.File(q, 'w') as h:
        h.attrs['e'] = h5py.Empty('f4')
    with fastnc.File(q) as f:
        i = f.attrs.get_id('e')
        assert i.shape is None and i.dtype == np.float32 and i.get_name() == b'e'
        with pytest.raises(RuntimeError):
            i.get_storage_size()                    # (as h5py: nothing stored)


@pytest.mark.parametrize('fn', FILES + ['complex/anonymous.h5'])
def test_repr_as_h5py(fn):
    p = os.path.join(DATA, fn)
    with h5py.File(p, 'r') as h, fastnc.File(p) as f:
        objs = [('', h, f)]
        h.visititems(lambda n, o: objs.append((n, o, f[n])))
        for n, a, b in objs:
            assert repr(b) == repr(a), n


def test_fillvalue_as_h5py(tmp_path):
    """Strings with a fill value (kept in the global heap) and fill values
    HDF5 calls undefined (h5py raises; written by H5Pset_fill_value(NULL),
    as in GPM IMERG files: here one flag patched in each message version)."""
    p = str(tmp_path / 'fill.h5')
    with h5py.File(p, 'w') as h:
        h.create_dataset('s', (6,), h5py.string_dtype(), chunks=(2,), fillvalue=b'UN')
        h['s'][0] = 'x'
        h.create_dataset('utf', (3,), h5py.string_dtype('utf-8'), fillvalue='é'.encode())
        h.create_dataset('none', (3,), h5py.string_dtype())
        h.create_dataset('vlen', (4,), h5py.vlen_dtype('i4'))
        h.create_dataset('fixed', (4,), 'S3', fillvalue=b'zz')
        h.create_dataset('i', (4,), 'i4')
    with h5py.File(p) as h, fastnc.File(p) as f:
        for k in h:
            assert repr(f[k].fillvalue) == repr(h[k].fillvalue), k
        assert f['s'][()].tolist() == [b'x'] + [b'UN'] * 5
    for libver in ('earliest', 'latest'):
        q = str(tmp_path / ('undefined_%s.h5' % libver))
        with h5py.File(q, 'w', libver=libver) as h:
            h.create_dataset('x', data=np.arange(4.0))
        with fastnc.File(q) as f:
            ds = f['x']._ds
            at = ds.header.first(5)[2]
            version = ds.file.b[at]
        with open(q, 'rb') as fh:
            b = bytearray(fh.read())
        if version < 3:
            b[at + 3] = 0
        else:
            b[at + 1] = (b[at + 1] & ~32) | 16
            o = b.rindex(b'OHDR', 0, at)                # (a version 2 header: its checksum again)
            flags = b[o + 5]
            q0 = o + 6 + (16 if flags & 32 else 0) + (4 if flags & 16 else 0)
            w = 1 << (flags & 3)
            end = q0 + w + int.from_bytes(b[q0:q0 + w], 'little')
            b[end:end + 4] = _hdf5.lookup3(bytes(b[o:end])).to_bytes(4, 'little')
        with open(q, 'wb') as fh:
            fh.write(b)
        with h5py.File(q) as h, fastnc.File(q) as f:
            assert h['x'].id.get_create_plist().fill_value_defined() == 0
            with pytest.raises(RuntimeError):
                h['x'].fillvalue
            with pytest.raises(RuntimeError):
                f['x'].fillvalue
            assert f['x'][()].tolist() == h['x'][()].tolist() == [0.0, 1.0, 2.0, 3.0]
