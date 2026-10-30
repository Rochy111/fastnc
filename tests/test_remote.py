"""Remote reading (fastnc/_remote.py), offline: tests/data served over HTTP
from a thread (a small handler with Range support), and file-like objects;
everything read must equal the local read of the same file.

Tiny blocks force many requests, so that every structure the parser reads
must have been asked for; new buffers are filled with a poison byte instead
of zeros, so that bytes read before they were fetched show up as wrong
metadata or values rather than as plausible zeros.
"""
import http.server
import io
import os
import re
import threading

import numpy as np
import pytest

import fastnc
from fastnc import _remote

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
HDF5 = ['filters.h5', 'groups_new.h5', 'groups_old.h5', 'idx_implicit.h5', 'idx_latest.h5',
        'idx_v1btree.h5', 'layouts.h5', 'types.h5', 'nc4.nc', 'nc4_classic.nc']
NETCDF4 = ['nc4.nc', 'nc4_classic.nc']
TINY = {'block_size': 512, 'initial': 1024, 'gap': 0, 'part': 4096, 'tail': 512}
EXACT = {'block_size': 8, 'initial': 64, 'gap': 0, 'part': 4096, 'tail': 0}   # read-ahead: 8 bytes


class Handler(http.server.BaseHTTPRequestHandler):
    """GET with an optional 'Range: bytes=a-b' (206 + Content-Range) over DATA;
    `fail[path]` answers that many requests 503 first."""
    protocol_version = 'HTTP/1.1'
    log = []
    fail = {}

    def do_GET(self):
        name = self.path.split('?')[0].lstrip('/')
        Handler.log.append((name, self.headers.get('Range')))
        if Handler.fail.get(name, 0) > 0:
            Handler.fail[name] -= 1
            return self._send(503, b'busy')
        path = os.path.join(DATA, name)
        if not os.path.isfile(path):
            return self._send(404, b'<Error>NoSuchKey</Error>')
        with open(path, 'rb') as f:
            data = f.read()
        rng = self.headers.get('Range')
        m = re.match(r'bytes=(\d*)-(\d*)$', rng or '')
        if not m or not (m.group(1) or m.group(2)):
            return self._send(200, data)
        if not m.group(1):                          # the last n bytes
            a, b = max(0, len(data) - int(m.group(2))), len(data) - 1
        else:
            a = int(m.group(1))
            b = min(int(m.group(2)) if m.group(2) else len(data) - 1, len(data) - 1)
        if a >= len(data):
            return self._send(416, b'')
        self._send(206, data[a:b + 1], {'Content-Range': 'bytes %d-%d/%d' % (a, b, len(data))})

    def _send(self, status, body, headers=()):
        self.send_response(status)
        self.send_header('Content-Length', str(len(body)))
        for k, v in dict(headers).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(scope='module')
def server():
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    _remote._POISON = 0xA5
    yield 'http://127.0.0.1:%d/' % srv.server_address[1]
    _remote._POISON = None
    srv.shutdown()


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        if a.dtype.kind == 'O':
            return all(same(x, y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist()))
        if np.ma.isMaskedArray(a) or np.ma.isMaskedArray(b):
            return (np.array_equal(np.ma.getmaskarray(a), np.ma.getmaskarray(b))
                    and np.ma.getdata(a).tobytes() == np.ma.getdata(b).tobytes())
        return a.tobytes() == b.tobytes()
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and a != a:
        return b != b
    return type(a) is type(b) and a == b


def h5_tree(f):
    """Everything the h5py front end shows: names, attributes, shapes,
    dtypes and values (whole and a window) of every object."""
    out = {}

    def visit(name, obj):
        rec = {'attrs': {k: obj.attrs[k] for k in obj.attrs}}
        if isinstance(obj, fastnc._h5py.Dataset):
            rec.update(shape=obj.shape, dtype=obj.dtype, chunks=obj.chunks)
            try:
                rec['all'] = obj[()]
            except NotImplementedError as e:
                rec['all'] = 'unsupported: %s' % e
            if obj.shape and len(obj.shape) >= 1 and obj.shape[0] > 2:
                try:
                    rec['part'] = obj[1:obj.shape[0] - 1]
                except NotImplementedError as e:
                    rec['part'] = 'unsupported: %s' % e
        out[name] = rec
    out['/'] = {'attrs': {k: f.attrs[k] for k in f.attrs}}
    f.visititems(visit)
    return out


def compare(a, b, where=''):
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), where
        for k in a:
            compare(a[k], b[k], '%s/%s' % (where, k))
        return
    if hasattr(a, 'dtype') and isinstance(a, fastnc._h5py.Empty):
        assert isinstance(b, fastnc._h5py.Empty), where
        return
    assert same(a, b), where


@pytest.mark.parametrize('name', HDF5)
@pytest.mark.parametrize('opts', ['TINY', 'EXACT'])
def test_h5py_http(server, name, opts):
    with fastnc.File(os.path.join(DATA, name)) as f:
        want = h5_tree(f)
    with fastnc.File(server + name, storage_options=globals()[opts]) as f:
        got = h5_tree(f)
        assert f._source.requests > 1
    compare(want, got)


@pytest.mark.parametrize('name', HDF5)
def test_h5py_default_blocks(server, name):
    with fastnc.File(os.path.join(DATA, name)) as f:
        want = h5_tree(f)
    with fastnc.File(server + name) as f:
        got = h5_tree(f)
        assert f._source.requests == 2           # the whole small file at open (head and tail)
    compare(want, got)


def values(v):
    try:
        return v[...]
    except (NotImplementedError, ImportError) as e:     # a filter / module not here yet
        return 'unsupported: %s' % type(e).__name__


def nc_tree(ds):
    out = {}
    for g in ds._walk():
        rec = {'dims': {k: (len(d), d.isunlimited()) for k, d in g.dimensions.items()},
               'attrs': {k: g.getncattr(k) for k in g.ncattrs()}}
        for k, v in g.variables.items():
            r = {'dims': v.dimensions, 'shape': v.shape, 'dtype': str(v.dtype),
                 'attrs': {a: v.getncattr(a) for a in v.ncattrs()}, 'masked': values(v)}
            v.set_auto_maskandscale(False)
            r['raw'] = values(v)
            rec[k] = r
        out[g.path] = rec
    return out


def readable(ds):
    """Paths of the variables this checkout can read (raw)."""
    out = []
    for g in ds._walk():
        for k, v in g.variables.items():
            v.set_auto_maskandscale(False)
            if not isinstance(values(v), str):
                out.append(('%s/%s' % (g.path.rstrip('/'), k)).lstrip('/'))
    return out


@pytest.mark.parametrize('name', NETCDF4)
@pytest.mark.parametrize('opts', ['TINY', 'EXACT'])
def test_netcdf4_http(server, name, opts):
    with fastnc.Dataset(os.path.join(DATA, name)) as ds:
        want = nc_tree(ds)
        model = ds.data_model
    with fastnc.Dataset(server + name, storage_options=globals()[opts]) as ds:
        got = nc_tree(ds)
        assert ds.data_model == model
        assert ds.filepath() == server + name
    compare(want, got)


def test_load_many(server):
    with fastnc.Dataset(os.path.join(DATA, 'nc4.nc')) as ds:
        names = readable(ds)
        ds.set_auto_maskandscale(False)
        want = ds.load(names)
    with fastnc.Dataset(server + 'nc4.nc', storage_options=TINY) as ds:
        ds.set_auto_maskandscale(False)
        got = ds.load(names)
    assert len(names) > 3
    compare(want, got)


def test_mode_bytes_suffix(server):
    with fastnc.Dataset(server + 'nc4.nc#mode=bytes') as ds:
        assert ds.filepath() == server + 'nc4.nc'
        assert len(ds.variables)


class ReadOnly:
    """The least a file-like object offers: seek, tell and read."""

    def __init__(self, data):
        self.f = io.BytesIO(data)
        self.reads = 0

    def seek(self, a, whence=0):
        return self.f.seek(a, whence)

    def tell(self):
        return self.f.tell()

    def read(self, n=-1):
        self.reads += 1
        return self.f.read(n)


@pytest.mark.parametrize('name', ['idx_v1btree.h5', 'groups_new.h5', 'types.h5', 'filters.h5'])
def test_file_objects(name):
    _remote._POISON = 0xA5
    try:
        path = os.path.join(DATA, name)
        with fastnc.File(path) as f:
            want = h5_tree(f)
        data = open(path, 'rb').read()
        for fo in (io.BytesIO(data), ReadOnly(data)):
            with fastnc.File(fo, storage_options=TINY) as f:
                compare(want, h5_tree(f))
        with open(path, 'rb') as fo:
            with fastnc.File(fo) as f:
                compare(want, h5_tree(f))
    finally:
        _remote._POISON = None


def test_dataset_file_object():
    path = os.path.join(DATA, 'nc4.nc')
    with fastnc.Dataset(path) as ds:
        want = nc_tree(ds)
    with fastnc.Dataset(io.BytesIO(open(path, 'rb').read()), storage_options=TINY) as ds:
        compare(want, nc_tree(ds))


def test_coalescing(server):
    """One read of many chunks: ranges merged across small gaps, nothing
    fetched twice, a second read of the same data fetches nothing."""
    with fastnc.File(server + 'idx_v1btree.h5', storage_options={'initial': 4096, 'block_size': 4096}) as f:
        s = f._source
        names = [k for k in f.keys() if isinstance(f[k], fastnc._h5py.Dataset) and f[k].chunks]
        d = f[names[0]]
        d._reader.index                             # the chunk index (B-tree) first
        r0, b0 = s.requests, s.fetched
        a = d[()]
        r1, b1 = s.requests, s.fetched
        assert r1 - r0 == 1 and b1 - b0 <= s.size
        assert same(d[()], a)
        assert (s.requests, s.fetched) == (r1, b1)


def test_contiguous_runs():
    from fastnc._h5read import contiguous_runs
    st, sz = contiguous_runs(100, (10, 20), 4, [(2, 5, 1), (3, 9, 2)])
    assert st.tolist() == [100 + (r * 20 + 3) * 4 for r in (2, 3, 4)]
    assert sz.tolist() == [(7 - 3 + 1) * 4] * 3
    st, sz = contiguous_runs(0, (10, 20), 4, [(2, 5, 1), (3, 9, 2)], most=2)
    assert st.tolist() == [(2 * 20 + 3) * 4] and sz.tolist() == [((4 * 20 + 7) - (2 * 20 + 3) + 1) * 4]


def test_errors(server):
    with pytest.raises(OSError, match='404'):
        fastnc.File(server + 'missing.h5')
    Handler.fail['layouts.h5'] = 2                  # two 503s, then fine: retried
    with fastnc.File(server + 'layouts.h5', storage_options={'retries': 3}) as f:
        assert len(f.keys())
    Handler.fail['layouts.h5'] = 10
    with pytest.raises(OSError, match='503'):
        fastnc.File(server + 'layouts.h5', storage_options={'retries': 1})
    Handler.fail.clear()
    with pytest.raises(OSError):
        fastnc.File('http://127.0.0.1:1/nothing.h5', storage_options={'retries': 0, 'timeout': 2})


def test_s3_urls():
    t = _remote._http_for('s3://noaa-goes19/ABI-L1b-RadF/a b.nc', {})
    assert t.url == 'https://noaa-goes19.s3.amazonaws.com/ABI-L1b-RadF/a%20b.nc'
    t = _remote._http_for('s3://my-bucket/k.nc', {'region': 'eu-west-1'})
    assert t.url == 'https://my-bucket.s3.eu-west-1.amazonaws.com/k.nc'
    t = _remote._http_for('s3://my.bucket/k.nc', {})
    assert t.url == 'https://s3.amazonaws.com/my.bucket/k.nc'
    t = _remote._http_for('s3://b/k.nc', {'endpoint_url': 'http://localhost:9000/'})
    assert t.url == 'http://localhost:9000/b/k.nc'


def test_source_ranges():
    data = bytes(range(256)) * 64

    class Mem:
        parallel = 1
        calls = []

        def open(self, initial, tail):
            return len(data), data[:initial], len(data), b''

        def read_into(self, a, out):
            Mem.calls.append((a, a + len(out)))
            out[:] = data[a:a + len(out)]

        def close(self):
            pass
    s = _remote.Source(Mem(), 'mem', block_size=256, initial=512, gap=100, part=1 << 20)
    s.fetch(np.array([1000, 1150, 3000, 600]), np.array([100, 50, 10, 10]))
    assert Mem.calls == [(600, 610), (1000, 1200), (3000, 3010)]
    assert s.buf[1000:1200].tobytes() == data[1000:1200]
    s.need(5000, 10)                                # whole blocks and one more
    assert Mem.calls[-1] == (4864, 5376)
    s.fetch(np.array([590, 1190]), np.array([30, 20]))  # only the missing parts
    assert Mem.calls[-2:] == [(590, 620), (1200, 1210)]   # small present holes refetched
    assert s.has(0, 512) and not s.has(0, 513)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
