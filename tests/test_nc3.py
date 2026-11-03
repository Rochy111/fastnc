"""fastnc's netCDF3 reader (fastnc/_nc3.py) against netCDF4-python's answers
stored in tests/data/nc3_expected.npz (made by
tools/check/nc3_make_expected.py): metadata, attribute values with their
types, and every variable's values, whole and for random keys, on the
committed nc3 files and on edge-case files (stored there too), under the
default settings and with every read path forced on tiny requests.
"""
import hashlib
import json
import os
import struct
import subprocess
import sys
import threading

import numpy as np
import pytest

import fastnc
from fastnc import _nc3, _pool

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
ROOT = os.path.dirname(HERE)
EXP = np.load(os.path.join(DATA, 'nc3_expected.npz'), allow_pickle=False)
INDEX = json.loads(EXP['index'].tobytes().decode('utf-8'))

TINY = dict(PIECE=4096, RUN_MIN=256, PAR_MIN=8192, MAP_MAX=64, SWAP_NUMPY=512,
            GATHER_PAR=1024, GATHER_BLOCK=300)
SETTINGS = {
    'defaults': {},
    'kernels': dict(KERNELS=True),
    'tiny-numpy': dict(TINY, KERNELS=False),
    'tiny-kernels': dict(TINY, KERNELS=True),
    'tiny-kernels-handles': dict(TINY, KERNELS=True, _IO=_nc3._Handles),
    'tiny-1-thread': dict(TINY, KERNELS=True, threads=1),
}


@pytest.fixture(params=list(SETTINGS))
def settings(request, monkeypatch):
    st = dict(SETTINGS[request.param])
    threads = st.pop('threads', None)
    for k, v in st.items():
        monkeypatch.setattr(_nc3, k, v)
    if threads:
        _pool.set_threads(threads)
    yield request.param
    if threads:
        _pool.set_threads(None)


@pytest.fixture(scope='session')
def edge_dir(tmp_path_factory):
    """The stored edge-case files written out, plus 'streaming' copies of a
    few (record count all ones: computed from the file's size)."""
    d = tmp_path_factory.mktemp('nc3_edge')
    for name, e in INDEX['edge'].items():
        (d / name).write_bytes(EXP[e['file']].tobytes())
    for tag in ('c1', 'c2', 'c5'):
        for base in ('records', 'lone_i1_3', 'lone_i2_3', 'norecs'):
            raw = bytearray((d / ('%s_%s.nc' % (tag, base))).read_bytes())
            if raw[3] == 5:
                raw[4:12] = b'\xff' * 8
            else:
                raw[4:8] = b'\xff' * 4
            (d / ('%s_%s_stream.nc' % (tag, base))).write_bytes(bytes(raw))
    return d


def expected_attr(enc):
    t = enc['t']
    if t == 'str':
        return enc['v']
    if t == 'bytes':
        return bytes.fromhex(enc['v'])
    if t == 'scalar':
        return np.frombuffer(bytes.fromhex(enc['hex']), enc['dtype'])[0]
    return EXP[enc['key']]


def same_value(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, np.generic):
        return a.dtype == b.dtype and a.tobytes() == b.tobytes()
    return a == b


def check_attrs(got, want):
    assert list(got) == [k for k, _ in want]
    for k, enc in want:
        w = expected_attr(enc)
        assert same_value(got[k], w), (k, got[k], w)


def check_meta(f, meta):
    g = f.root
    assert f.data_model == meta['data_model']
    assert g.name == '/' and g.groups == {}
    assert list(g.dimensions.items()) == [(k, (n, u)) for k, n, u in meta['dimensions']]
    check_attrs(g.attrs, meta['attrs'])
    assert list(g.variables) == [m['name'] for m in meta['variables']]
    for m in meta['variables']:
        v = g.variables[m['name']]
        assert v.name == m['name']
        assert v.dimensions == tuple(m['dimensions'])
        assert v.shape == tuple(m['shape'])
        assert v.dtype == np.dtype(m['dtype'])
        assert v.endian == m['endian'] == 'native'
        assert v.filters is None and m['filters'] is None
        assert v.chunks is None and m['chunking'] is None
        check_attrs(v.attrs, m['attrs'])


def random_key(rng, shape):
    key = []
    for n in shape:
        r = rng.random()
        if n == 0:
            key.append(slice(0, 0, 1))
        elif r < 0.25:
            key.append(int(rng.integers(0, n)))
        elif r < 0.4:
            key.append(slice(0, n, 1))
        else:
            a = int(rng.integers(0, n))
            b = int(rng.integers(a, n + 1))
            key.append(slice(a, b, int(rng.choice([1, 1, 2, 3, 7, max(1, n // 3)]))))
    return tuple(key)


def check_values(f, meta, nkeys=25, seed=0):
    rng = np.random.default_rng(seed)
    for m in meta['variables']:
        v = f.root.variables[m['name']]
        whole = v.read(tuple(slice(0, n, 1) for n in v.shape))
        if 'value' in m:
            want = EXP[m['value']]
            assert whole.dtype == want.dtype and whole.shape == want.shape, m['name']
            assert whole.tobytes() == want.tobytes(), m['name']
        else:
            assert whole.dtype.str == m['value_dtype']
            assert hashlib.sha256(whole.tobytes()).hexdigest() == m['sha256'], m['name']
        for _ in range(nkeys):
            key = random_key(rng, v.shape)
            got = v.read(key)
            want = whole[key + (Ellipsis,)]          # (an array even for all ints)
            assert got.dtype == want.dtype and got.shape == want.shape, (m['name'], key)
            assert got.tobytes() == np.ascontiguousarray(want).tobytes(), (m['name'], key)


@pytest.mark.parametrize('name', sorted(INDEX['committed']))
def test_committed(name, settings):
    with _nc3.NC3File(os.path.join(DATA, name)) as f:
        meta = INDEX['committed'][name]
        check_meta(f, meta)
        check_values(f, meta)


@pytest.mark.parametrize('name', sorted(INDEX['edge']))
def test_edge(name, settings, edge_dir):
    with _nc3.NC3File(str(edge_dir / name)) as f:
        meta = INDEX['edge'][name]['meta']
        check_meta(f, meta)
        check_values(f, meta)


@pytest.mark.parametrize('tag', ['c1', 'c2', 'c5'])
@pytest.mark.parametrize('base', ['records', 'lone_i1_3', 'lone_i2_3', 'norecs'])
def test_streaming(tag, base, edge_dir):
    """A record count of all ones ('streaming') is worked out from the
    file's size; netCDF4-python takes it literally (and fails)."""
    name = '%s_%s.nc' % (tag, base)
    with _nc3.NC3File(str(edge_dir / ('%s_%s_stream.nc' % (tag, base)))) as f:
        assert f.streaming
        meta = INDEX['edge'][name]['meta']
        check_meta(f, meta)
        check_values(f, meta)


def external_files():
    data = os.environ.get('FASTNC_DATA') or (
        os.path.dirname(os.path.dirname(os.path.dirname(ROOT))) if '.claude' in ROOT else ROOT)
    return [(rel, os.path.join(data, rel)) for rel in sorted(INDEX['external'])]


@pytest.mark.parametrize('rel,path', external_files())
@pytest.mark.parametrize('kernels', [False, True])
def test_external(rel, path, kernels, monkeypatch):
    """The real / synthetic files that are not committed, where present:
    metadata and a digest of every variable's values."""
    if not os.path.exists(path):
        pytest.skip('%s not here' % rel)
    monkeypatch.setattr(_nc3, 'KERNELS', kernels)
    with _nc3.NC3File(path) as f:
        meta = INDEX['external'][rel]
        check_meta(f, meta)
        check_values(f, meta, nkeys=5)


def test_truncated_reads_zeros(edge_dir):
    """Bytes past the end of a truncated file read as zeros."""
    with _nc3.NC3File(str(edge_dir / 'c1_trunc13.nc')) as f, \
            _nc3.NC3File(str(edge_dir / 'c1_records.nc')) as g:
        assert f.numrecs == g.numrecs == 37
        a = f.root.variables['r_int'].read(())
        b = g.root.variables['r_int'].read(())
        assert (a[:-2] == b[:-2]).all()
        assert (a[-1] == 0).all()


def test_out(edge_dir):
    with _nc3.NC3File(str(edge_dir / 'c2_records.nc')) as f:
        v = f.root.variables['r_double']
        want = v.read(())
        out = np.empty(v.shape, v.dtype)
        assert v.read((), out=out) is out and out.tobytes() == want.tobytes()
        out = np.empty(v.shape, v.dtype, order='F')         # not C-contiguous: copied in
        assert v.read((), out=out) is out and np.array_equal(out, want, equal_nan=True)
        big = np.zeros((v.shape[0] * 2,) + v.shape[1:], v.dtype)
        out = big[::2]
        assert v.read((), out=out) is out
        assert np.array_equal(out, want, equal_nan=True) and not big[1::2].any()
        out = np.empty(v.shape, '>f8')                      # other byte order: converted
        assert v.read((), out=out) is out and np.array_equal(out, want, equal_nan=True)


def test_keys(edge_dir):
    with _nc3.NC3File(str(edge_dir / 'c1_records.nc')) as f:
        v = f.root.variables['r_double']
        whole = v.read(())
        assert v.read((slice(None), 1)).tobytes() == whole[:, 1].tobytes()  # short key, None
        assert v.read((-1, 0, 2)).tobytes() == whole[-1, 0, 2].tobytes()   # negative int
        assert v.read((slice(5, 3, 1),)).shape == (0, 2, 3)
        assert v.read((slice(0, 1000, 5),)).shape == whole[0:1000:5].shape
        with pytest.raises(IndexError):
            v.read((37,))
        with pytest.raises(IndexError):
            v.read((0, 0, 0, 0))
        with pytest.raises(IndexError):
            v.read((slice(None, None, -1),))
        s = f.root.variables['fixed_after']
        assert s.read(3).shape == ()


def test_not_netcdf3(tmp_path):
    with pytest.raises(_nc3.FormatError):
        _nc3.NC3File(os.path.join(DATA, 'nc4.nc'))
    p = tmp_path / 'junk.nc'
    p.write_bytes(b'CDF\x03' + bytes(100))
    with pytest.raises(_nc3.FormatError):
        _nc3.NC3File(str(p))
    raw = open(os.path.join(DATA, 'nc3_classic.nc'), 'rb').read()
    p.write_bytes(raw[:200])                                # header cut short
    with pytest.raises(_nc3.FormatError):
        _nc3.NC3File(str(p))


def test_close_twice():
    f = _nc3.NC3File(os.path.join(DATA, 'nc3_64bit.nc'))
    for v in f.root.variables.values():
        v.read(())
    f.close()
    f.close()


def test_threads_reading_at_once(edge_dir, monkeypatch):
    """Several user threads reading one file at once (each read also on the
    pool, tiny pieces)."""
    for k, v in TINY.items():
        monkeypatch.setattr(_nc3, k, v)
    with _nc3.NC3File(str(edge_dir / 'c5_records.nc')) as f:
        vs = list(f.root.variables.values())
        want = {v.name: v.read(()).tobytes() for v in vs}
        bad = []

        def run():
            for _ in range(20):
                for v in vs:
                    if v.read(()).tobytes() != want[v.name]:
                        bad.append(v.name)
        ts = [threading.Thread(target=run) for _ in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert not bad


def test_no_numba_for_metadata_and_small_reads():
    code = ('import sys; sys.path.insert(0, %r); from fastnc import _nc3; '
            'f = _nc3.NC3File(%r); f.root.attrs; '
            '[(v.attrs, v.read(())) for v in f.root.variables.values()]; f.close(); '
            'print("numba" in sys.modules)' % (ROOT, os.path.join(DATA, 'nc3_cdf5.nc')))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == 'False'


def explicit_key(rng, shape):
    """Slices as Dataset hands them to a backend: every part a number."""
    key = []
    for n in shape:
        if n == 0:
            key.append(slice(0, 0, 1))
            continue
        a = int(rng.integers(0, n))
        b = int(rng.integers(a + 1, n + 1))
        key.append(slice(a, b, int(rng.choice([1, 1, 2, 3, max(1, n // 3)]))))
    return tuple(key)


@pytest.mark.parametrize('name', sorted(INDEX['edge']))
def test_slices_as_dataset_gives_them(name, settings, edge_dir):
    """Keys the closure takes, small and big, on every record layout; each
    read twice over, the second through what the first left."""
    rng = np.random.default_rng(9)
    with _nc3.NC3File(str(edge_dir / name)) as f:
        for v in f.root.variables.values():
            whole = v.read(())
            for _ in range(25):
                key = explicit_key(rng, v.shape)
                want = whole[key + (Ellipsis,)]
                for _ in range(2):
                    got = v.read(key)
                    assert got.dtype == want.dtype and got.shape == want.shape, (v.name, key)
                    assert got.tobytes() == np.ascontiguousarray(want).tobytes(), (v.name, key)


def test_slices_past_the_end_of_a_truncated_file(edge_dir):
    with _nc3.NC3File(str(edge_dir / 'c1_trunc13.nc')) as f:
        v = f.root.variables['r_int']
        whole = v.read(())
        f._source()                                 # (the mapping there: the closure has it to use)
        n = v.shape[0]
        for a, b in ((n - 1, n), (n - 3, n), (0, n)):
            key = (slice(a, b, 1),) + tuple(slice(0, m, 1) for m in v.shape[1:])
            assert v.read(key).tobytes() == whole[a:b].tobytes()


def test_a_strided_read_spans_once(edge_dir, monkeypatch):
    """The first selection of a form reads the bytes among its elements; the
    next goes through the mapping. Both give the values, past a block too."""
    monkeypatch.setattr(_nc3, 'RUN_MIN', 16)
    monkeypatch.setattr(_nc3, 'MAP_MAX', 8)
    monkeypatch.setattr(_nc3, 'PIECE', 256)
    calls = []
    span = _nc3._span
    monkeypatch.setattr(_nc3, '_span', lambda *a: (calls.append(a[4:6]), span(*a))[1])
    with _nc3.NC3File(str(edge_dir / 'c1_records.nc')) as f:
        v = f.root.variables['r_double']
        whole = v.read(())
        for j, k in ((0, 0), (1, 2), (0, 1)):
            key = (slice(0, v.shape[0], 1), slice(j, j + 1, 1), slice(k, k + 1, 1))
            assert v.read(key).tobytes() == whole[:, j:j + 1, k:k + 1].tobytes()
        assert len(calls) == 1
        assert f._src is not None                   # (the mapping, from the second on)
        key = (slice(0, v.shape[0], 2), slice(0, 2, 1), slice(1, 2, 1))
        assert v.read(key).tobytes() == whole[::2, 0:2, 1:2].tobytes()
        assert len(calls) == 2


def test_no_mapping_for_a_first_strided_read(edge_dir, monkeypatch):
    monkeypatch.setattr(_nc3, 'RUN_MIN', 16)
    monkeypatch.setattr(_nc3, 'MAP_MAX', 8)
    with _nc3.NC3File(str(edge_dir / 'c5_records.nc')) as f:
        v = f.root.variables['r_double']
        v.read((slice(0, v.shape[0], 1), slice(0, 1, 1), slice(0, 1, 1)))
        assert f._src is None and f._mm is None


def test_variables_made_when_asked(edge_dir):
    with _nc3.NC3File(str(edge_dir / 'c1_records.nc')) as f:
        g = f.root
        names = list(g._rows)
        v = g.peek(names[1])
        assert list(g._made) == [names[1]] and g.peek(names[1]) is v
        with pytest.raises(KeyError):
            g.peek('no such')
        assert not g.holds_no_data()
        assert list(g.variables) == names and g.variables[names[1]] is v


def test_dataset_makes_one_variable(edge_dir):
    with fastnc.Dataset(str(edge_dir / 'c1_records.nc')) as d:
        g = d._backend.root
        d.variables['r_int'][0]
        assert list(g._made) == ['r_int']
        assert list(d.variables) == list(g._rows)


def classic(title, units='K', begin=512):
    """A netCDF-3 classic file of one variable, by hand: its global attribute
    decides where the variable list starts. Values at `begin`: netCDF-C pads a
    header to its chunk size, so that files of a run keep them in one place."""
    def name(s):
        return struct.pack('>I', len(s)) + s + bytes(-len(s) & 3)
    t = title.encode()
    gatt = struct.pack('>II', 12, 1) + name(b'title') + struct.pack('>II', 2, len(t)) + t + bytes(-len(t) & 3)
    u = units.encode()
    att = struct.pack('>II', 12, 1) + name(b'units') + struct.pack('>II', 2, len(u)) + u + bytes(-len(u) & 3)
    head = b'CDF\x01' + struct.pack('>I', 0) + struct.pack('>II', 10, 1) + name(b'x') + struct.pack('>I', 4) + gatt
    tail = struct.pack('>II', 11, 1) + name(b'v') + struct.pack('>II', 1, 0) + att + struct.pack('>II', 5, 16)
    header = head + tail + struct.pack('>I', begin)
    return header + bytes(begin - len(header)) + struct.pack('>4f', 1, 2, 3, 4)


def test_a_header_like_the_last(tmp_path, monkeypatch):
    """A variable list as the last file's, moved by its global attributes: its
    attributes and values are found where this file has them."""
    paths = []
    for i, (title, units) in enumerate((('a', 'K'), ('a longer title', 'K'), ('a longer title', 'm/s'))):
        p = tmp_path / ('f%d.nc' % i)
        p.write_bytes(classic(title, units))
        paths.append(str(p))
    monkeypatch.setattr(_nc3._Last, 'list', None)
    with _nc3.NC3File(paths[0]) as f:
        first = _nc3._Last.list
        assert f.root.variables['v'].attrs['units'] == 'K'
    with _nc3.NC3File(paths[1]) as f:
        assert _nc3._Last.list is first             # (nothing walked)
        assert f.root.attrs['title'] == 'a longer title'
        v = f.root.variables['v']
        assert v.attrs['units'] == 'K'
        assert v.read(()).tolist() == [1, 2, 3, 4]
        again = _nc3._parse(f._header, f._size)
        monkeypatch.setattr(_nc3._Last, 'list', None)
        assert again == _nc3._parse(f._header, f._size)  # (as a walk finds it)
    with _nc3.NC3File(paths[0]):
        first = _nc3._Last.list
    with _nc3.NC3File(paths[2]) as f:
        assert _nc3._Last.list is not first         # (other bytes: read)
        assert f.root.variables['v'].attrs['units'] == 'm/s'


def test_a_header_with_values_elsewhere_is_read(tmp_path, monkeypatch):
    """`begin` is in the list: values elsewhere, other bytes, the file's own place."""
    a, b = tmp_path / 'a.nc', tmp_path / 'b.nc'
    a.write_bytes(classic('t', begin=512))
    b.write_bytes(classic('t', begin=1024))
    monkeypatch.setattr(_nc3._Last, 'list', None)
    with _nc3.NC3File(str(a)) as f:
        assert f.root.variables['v'].read(()).tolist() == [1, 2, 3, 4]
    with _nc3.NC3File(str(b)) as f:
        v = f.root.variables['v']
        assert v.begin == 1024 and v.read(()).tolist() == [1, 2, 3, 4]


def test_a_closed_file_raises(edge_dir, monkeypatch):
    """Every way of reading, and no handle made anew behind the caller's back."""
    f = _nc3.NC3File(str(edge_dir / 'c1_records.nc'))
    v = f.root.variables['r_double']
    small = (slice(0, 1, 1), slice(0, 1, 1), slice(0, 1, 1))
    big = tuple(slice(0, n, 1) for n in v.shape)
    v.read(small)
    v.read(big)
    f.close()
    monkeypatch.setattr(_nc3, 'MAP_MAX', 8)
    for key in (small, big, ()):
        with pytest.raises(ValueError):
            v.read(key)
    assert f._rd is None


def test_a_header_with_other_dimensions_is_read(tmp_path, monkeypatch):
    """The same bytes of variables mean nothing with another count of dimensions."""
    a = bytearray(classic('a'))
    p = tmp_path / 'a.nc'
    p.write_bytes(bytes(a))
    monkeypatch.setattr(_nc3._Last, 'list', None)
    with _nc3.NC3File(str(p)):
        pass
    ver, ndim, raw, start, rows = _nc3._Last.list
    monkeypatch.setattr(_nc3._Last, 'list', (ver, ndim + 1, raw, start, []))
    with _nc3.NC3File(str(p)) as f:
        assert list(f.root.variables) == ['v']


def test_attributes_remembered():
    with fastnc.Dataset(os.path.join(DATA, 'nc3_classic.nc')) as d:
        for v in d.variables.values():
            if len(v.shape):
                a = v[0]
                assert np.array_equal(np.ma.getmaskarray(v[0]), np.ma.getmaskarray(a))
                assert v._bv.attrs is v._bv.attrs and isinstance(v._bv.attrs, dict)
                assert isinstance(v._bv.attrs, _nc3.Attrs)


def test_swap_kernel():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, 1000, dtype=np.uint8)
    for es in (1, 2, 4, 8):
        for lo in (0, 1, 3, 8):
            for n in (0, 1, 7, 13, 100):
                hi = lo + n * es
                b = a.copy()
                _nc3._swap(b, lo, hi, es)
                want = a.copy()
                if es > 1:
                    want[lo:hi] = a[lo:hi].reshape(-1, es)[:, ::-1].reshape(-1)
                assert (b == want).all(), (es, lo, n)
