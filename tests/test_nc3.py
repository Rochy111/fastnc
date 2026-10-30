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
import subprocess
import sys
import threading

import numpy as np
import pytest

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
