"""HDF4 through fastnc.Dataset: metadata and raw values against digests
taken from netCDF4-python and pyhdf (tests/data/hdf4/expected.json, made by
tools/check/hdf4_make_expected.py; neither library is needed here), the
reads before and after numba's kernels load, read_many, the HDF4 coders
on edge cases, and corrupted files (errors, never a crash)."""
import hashlib
import io
import json
import os
import struct
import threading

import numpy as np
import pytest

import fastnc
from fastnc import _chunks, _hdf4, _hdf4_codecs, _hold, _nc3, _pio, _pool

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'hdf4')
with open(os.path.join(DATA, 'expected.json')) as _f:
    EXPECTED = json.load(_f)
FILES = sorted(EXPECTED['files'])


def digest(a):
    a = np.asarray(a)
    if a.dtype.kind not in 'SU':
        a = a.astype(a.dtype.newbyteorder('='))
    return '%s %s %s' % (a.dtype.str, list(a.shape),
                         hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:24])


def attr_repr(v):
    if isinstance(v, (str, bytes)):
        return [type(v).__name__, v if isinstance(v, str) else v.hex()]
    a = np.asarray(v)
    return [type(v).__name__, a.dtype.str, list(a.shape), a.tobytes().hex()]


def check_file(fn, values=True):
    e = EXPECTED['files'][fn]
    with fastnc.Dataset(os.path.join(DATA, fn)) as d:
        d.set_auto_maskandscale(False)
        assert (d.data_model, d.file_format, d.disk_format) == \
            (e['data_model'], e['file_format'], e['disk_format'])
        assert [[k, len(v), v.isunlimited()] for k, v in d.dimensions.items()] == e['dimensions']
        assert {k: attr_repr(d.getncattr(k)) for k in d.ncattrs()} == e['attrs']
        assert list(d.ncattrs()) == list(e['attrs'])
        assert list(d.variables) == list(e['variables'])
        for k, ve in e['variables'].items():
            v = d.variables[k]
            assert list(v.dimensions) == ve['dimensions'], k
            assert list(v.shape) == ve['shape'], k
            assert v.dtype.str == ve['dtype'], k
            assert v.chunking() == ve['chunking'], k
            assert v.endian() == ve['endian'], k
            assert v.filters() == ve['filters'], k
            assert list(v.ncattrs()) == list(ve['attrs']), k
            assert {n: attr_repr(v.getncattr(n)) for n in v.ncattrs()} == ve['attrs'], k
            if values:
                for sel, want in ve['values'].items():
                    assert digest(v[eval(sel)]) == want, (k, sel)


@pytest.mark.parametrize('fn', FILES)
def test_against_references(fn):
    _chunks.warm()
    check_file(fn)


@pytest.mark.parametrize('fn', FILES)
def test_before_kernels_load(fn, monkeypatch):
    """Reads before numba is up go through zlib and numpy: same values."""
    monkeypatch.setattr(_chunks, '_READY', __import__('threading').Event())
    monkeypatch.setattr(_chunks, 'IMAGE', False)
    monkeypatch.setattr(_chunks, 'LOAD_AFTER', float('inf'))
    e = EXPECTED['files'][fn]
    with fastnc.Dataset(os.path.join(DATA, fn)) as d:
        d.set_auto_maskandscale(False)
        for k, ve in e['variables'].items():
            if d.variables[k]._bv.compression in ('rle', 'nbit', 'skphuff'):
                continue                            # (numba's own coders: no zlib path)
            assert digest(d.variables[k][...]) == ve['values']['...'], k


@pytest.mark.parametrize('fn', FILES)
def test_read_many(fn):
    _chunks.warm()
    e = EXPECTED['files'][fn]
    with fastnc.Dataset(os.path.join(DATA, fn)) as d:
        names = list(d.variables)
        got = _hdf4.read_many([d.variables[k]._bv for k in names])
        for k, a in zip(names, got):
            assert digest(a) == e['variables'][k]['values']['...'], k


def test_out_argument_and_threads():
    _chunks.warm()
    with fastnc.Dataset(os.path.join(DATA, 'layout.hdf')) as d:
        bv = d.variables['contig_big']._bv
        full = bv.read((Ellipsis,))
        out = np.empty(full.shape, full.dtype)
        r = bv.read((Ellipsis,), out=out, threads=1)
        assert r is out and np.array_equal(out, full)
        part = np.empty((3, full.shape[1]), full.dtype)
        bv.read((slice(10, 13),), out=part)
        assert np.array_equal(part, full[10:13])
        assert np.array_equal(bv.read(([5, 2, 900], slice(None, None, 7))), full[[5, 2, 900]][:, ::7])


def test_storage_kinds():
    """The test files cover what they are meant to."""
    with fastnc.Dataset(os.path.join(DATA, 'comp.hdf')) as d:
        kinds = {(v._bv.elem.kind, v._bv.compression) for v in d.variables.values()}
    assert {('comp', 'deflate'), ('comp', 'rle'), ('comp', 'skphuff'), ('comp', 'nbit'),
            ('chunked', None), ('chunked', 'deflate'), ('chunked', 'rle'), ('chunked', 'skphuff'),
            ('chunked', 'nbit')} <= kinds
    with fastnc.Dataset(os.path.join(DATA, 'layout.hdf')) as d:
        kinds = {k: v._bv.elem.kind for k, v in d.variables.items()}
    assert kinds == {'contig_big': 'plain', 'linked': 'linked', 'external': 'ext', 'vector': 'plain'}
    with fastnc.Dataset(os.path.join(DATA, 'types.hdf')) as d:
        assert d.variables['lfloat32'].endian() == 'little'
        assert d.variables['float32'].endian() == 'big'


# ---------------------------------------------------------------- coders

def _guarded(n):
    """An output of n bytes inside guard bytes, to see nothing is written outside."""
    buf = np.full(n + 32, 0xA5, np.uint8)
    return buf, 16, 16 + n


def _rle(data):
    """A simple RLE encoder (runs of >= 3 as runs, the rest as mixes)."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        j = i
        while j < n and data[j] == data[i] and j - i < 130:
            j += 1
        if j - i >= 3:
            out += bytes([0x80 | (j - i - 3), data[i]])
            i = j
        else:
            k = i
            while k < n and k - i < 128 and not (k + 2 < n and data[k] == data[k + 1] == data[k + 2]):
                k += 1
            out += bytes([k - i - 1]) + bytes(data[i:k])
            i = k
    return bytes(out)


@pytest.mark.parametrize('seed', range(5))
def test_rle(seed):
    rng = np.random.default_rng(seed)
    data = bytes(np.repeat(rng.integers(0, 4, 300, dtype=np.uint8), rng.integers(1, 9, 300)))
    enc = np.frombuffer(_rle(data), np.uint8)
    cd = np.zeros(3, np.int64)
    ws = np.zeros(1, np.uint8)
    buf, o, oend = _guarded(len(data))
    assert _hdf4_codecs.rle_decode(enc, 0, enc.size, buf, o, oend, cd, ws) == oend
    assert buf[o:oend].tobytes() == data
    assert (buf[:o] == 0xA5).all() and (buf[oend:] == 0xA5).all()
    # only the head asked for; the stream cut short
    buf, o, oend = _guarded(37)
    assert _hdf4_codecs.rle_decode(enc, 0, enc.size, buf, o, oend, cd, ws) == oend
    assert buf[o:oend].tobytes() == data[:37] and (buf[oend:] == 0xA5).all()
    buf, o, oend = _guarded(len(data))
    assert _hdf4_codecs.rle_decode(enc, 0, enc.size // 2, buf, o, oend, cd, ws) == -3
    assert (buf[oend:] == 0xA5).all() and (buf[:o] == 0xA5).all()


def test_nbit_and_skphuff_bounds():
    rng = np.random.default_rng(1)
    src = rng.integers(0, 256, 200, dtype=np.uint8)
    ws = np.zeros(1, np.uint8)
    for cd in ([2, 64, 5, 2, 10, 6, 1, 0], [4, 64, 5, 4, 31, 32, 0, 1], [1, 64, 5, 1, 5, 4, 0, 0]):
        cd = np.array(cd, np.int64)
        buf, o, oend = _guarded(64)
        r = _hdf4_codecs.nbit_decode(src, 0, 3, buf, o, oend, cd, ws)    # input runs out: zeros
        assert r == oend
        assert (buf[:o] == 0xA5).all() and (buf[oend:] == 0xA5).all()
    for bad in ([2, 64, 5, 2, 16, 6, 1, 0], [2, 64, 5, 2, 3, 6, 1, 0], [2, 64, 5, 0, 3, 1, 1, 0]):
        buf, o, oend = _guarded(64)
        assert _hdf4_codecs.nbit_decode(src, 0, 200, buf, o, oend, np.array(bad, np.int64), ws) == -1
        assert (buf == 0xA5).all()
    for skip in (1, 2, 4, 8):
        cd = np.array([skip, 64, 1, skip], np.int64)
        buf, o, oend = _guarded(500)
        r = _hdf4_codecs.skphuff_decode(src, 0, 20, buf, o, oend, cd, ws)
        assert r in (oend, -3)
        assert (buf[:o] == 0xA5).all() and (buf[oend:] == 0xA5).all()
    assert _hdf4_codecs.skphuff_decode(src, 0, 20, buf, 0, 10, np.array([1, 64, 1, 0], np.int64), ws) == -1


# ---------------------------------------------------------------- corrupt files

ALLOWED = (ValueError, OSError, NotImplementedError, KeyError, MemoryError)


def _try(path):
    try:
        d = fastnc.Dataset(path)
    except ALLOWED:
        return
    try:
        d.set_auto_maskandscale(False)
        list(d.ncattrs())
        for k in d.ncattrs():
            d.getncattr(k)
        for v in d.variables.values():
            try:
                v.ncattrs()
                if v.size > 1 << 24:
                    continue
                v[...]
                if v.ndim:
                    v[::2]
            except ALLOWED:
                pass
    finally:
        d.close()


@pytest.mark.parametrize('fn', ['types.hdf', 'comp.hdf', 'many.hdf'])
def test_fuzz(fn, tmp_path):
    _chunks.warm()
    raw = open(os.path.join(DATA, fn), 'rb').read()
    rng = np.random.default_rng(len(raw))
    # the DD list and metadata sit at the front and the end: aim there
    n = len(raw)
    for it in range(60):
        path = str(tmp_path / ('fuzz%d.hdf' % it))   # (a mapping still open holds a file on Windows)
        b = bytearray(raw)
        for _ in range(int(rng.integers(1, 12))):
            where = int(rng.integers(0, n)) if it % 3 == 0 else \
                int(rng.integers(0, min(n, 4096))) if it % 3 == 1 else int(rng.integers(max(0, n - 8192), n))
            b[where] = int(rng.integers(0, 256))
        with open(path, 'wb') as f:
            f.write(b)
        _try(path)
    for cut in (10, 100, 1000, n // 3, n // 2, n - 100):
        path = str(tmp_path / ('cut%d.hdf' % cut))
        with open(path, 'wb') as f:
            f.write(raw[:cut])
        _try(path)


def test_not_hdf4(tmp_path):
    p = str(tmp_path / 'x.hdf')
    with open(p, 'wb') as f:
        f.write(b'\x0e\x03\x13\x01' + struct.pack('>HI', 1000, 0))
    with pytest.raises(ValueError):
        fastnc.Dataset(p)


# ---------------------------------------------------------------- every path, one answer

TINY = dict(PIECE=4096, RUN_MIN=256, PAR_MIN=8192, MAP_MAX=64, SWAP_NUMPY=512, GATHER_PAR=1024, GATHER_BLOCK=300)


def explicit_key(rng, shape):
    """Slices as Dataset hands them to a backend: every part a number."""
    key = []
    for n in shape:
        a = int(rng.integers(0, n))
        b = int(rng.integers(a + 1, n + 1))
        key.append(slice(a, b, int(rng.choice([1, 1, 1, 2, 3]))))
    return tuple(key)


def check_keys(d, e, seed):
    rng = np.random.default_rng(seed)
    for k, ve in e['variables'].items():
        v = d.variables[k]
        whole = v[...]
        assert digest(whole) == ve['values']['...'], k
        if not v.shape or not whole.size:
            continue
        for _ in range(30):
            key = explicit_key(rng, v.shape)
            got = v[key]
            want = whole[key]
            assert got.dtype == want.dtype and got.shape == want.shape, (k, key)
            assert got.tobytes() == np.ascontiguousarray(want).tobytes(), (k, key)


@pytest.mark.parametrize('fn', FILES)
@pytest.mark.parametrize('how', ['small reads', 'no small reads', 'tiny', 'tiny numpy'])
def test_every_path(fn, how, monkeypatch):
    """The closures, the general path, and netCDF-3's reader with every way of
    its forced (big-endian, little-endian, in memory, mapped)."""
    _chunks.warm()
    if how.startswith('tiny'):
        for k, x in TINY.items():
            monkeypatch.setattr(_nc3, k, x)
        monkeypatch.setattr(_nc3, 'KERNELS', how == 'tiny')
    with fastnc.Dataset(os.path.join(DATA, fn)) as d:
        d.set_auto_maskandscale(False)
        if how == 'no small reads':
            for v in d.variables.values():
                v._bv._quick = False
        check_keys(d, EXPECTED['files'][fn], 5)


def test_dds_found_as_a_walk_finds_them():
    """Elements in the dict and in the arrays, as a plain walk of the DDs has them."""
    for fn in FILES:
        with open(os.path.join(DATA, fn), 'rb') as f:
            raw = f.read()
        want = {}
        pos = 4
        while pos > 0:
            n, nxt = struct.unpack_from('>Hi', raw, pos)
            for i in range(n):
                t, r, o, ln = struct.unpack_from('>HHii', raw, pos + 6 + 12 * i)
                if t not in (0, 1):
                    want[t, r] = (o, ln)
            pos = nxt
        h = _hdf4.H4File(raw)
        for (t, r), place in want.items():
            assert h._dd(t, r) == place, (fn, t, r)
        assert h._dd(_hdf4.T_CHUNK, 65535) is None and h._dd(_hdf4.T_VG, 65535) is None
        assert h.vgroups() == sorted(r for t, r in want if t == _hdf4.T_VG)
        h.b.release()


def test_a_later_dd_stands():
    """Two DDs of one chunk element: the last, in the arrays as in a dict."""
    b = bytearray(b'\x0e\x03\x13\x01')
    b += struct.pack('>Hi', 3, 0)
    b += struct.pack('>HHii', _hdf4.T_CHUNK, 7, 100, 1)
    b += struct.pack('>HHii', _hdf4.T_CHUNK, 7, 200, 2)
    b += struct.pack('>HHii', _hdf4.T_CHUNK, 8, 300, 3)
    h = _hdf4.H4File(bytes(b) + bytes(400))
    assert h._dd(_hdf4.T_CHUNK, 7) == (200, 2) and h._dd(_hdf4.T_CHUNK, 8) == (300, 3)
    h.b.release()


def test_chunk_table_both_ways(monkeypatch):
    """The table read at once says what going through its rows says."""
    _chunks.warm()
    with fastnc.Dataset(os.path.join(DATA, 'comp.hdf')) as d:
        fast = {k: v._bv.layout() for k, v in d.variables.items() if v._bv.elem.kind == 'chunked'}
        fast = {k: lay for k, lay in fast.items() if lay.kind == 'file'}
        assert {'chunk_none', 'chunk_deflate', 'chunk_rows', 'chunk_3d'} <= set(fast)
    monkeypatch.setattr(_hdf4, '_plain_table', lambda *a: None)
    with fastnc.Dataset(os.path.join(DATA, 'comp.hdf')) as d:
        for k, a in fast.items():
            b = d.variables[k]._bv.layout()
            for name in ('offsets', 'addr', 'size', 'mask'):
                assert np.array_equal(getattr(a.index, name), getattr(b.index, name)), (k, name)
            assert a.index.complete == b.index.complete, k
        check_keys(d, dict(variables={k: EXPECTED['files']['comp.hdf']['variables'][k] for k in fast}), 2)


def test_masked_reads_prove_or_pass(monkeypatch):
    """A read that proves nothing is masked gives what the pass over it gives."""
    _chunks.warm()
    from fastnc import _maskscale as ms
    with fastnc.Dataset(os.path.join(DATA, 'layout.hdf')) as d:
        v = d.variables['contig_big']
        bv = v._bv
        raw = bv.read(())
        slow = ms.apply(raw.copy(), bv.attrs, bv.dtype, var_shape=tuple(bv.shape), key=(slice(None),) * 2)
        monkeypatch.setattr(_nc3, 'PAR_MIN', 1 << 12)
        p = v._probe()
        got = v[...]
        assert type(got) is type(slow) and got.dtype == slow.dtype
        assert np.ma.getdata(got).tobytes() == np.ma.getdata(slow).tobytes()
        assert np.array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(slow))
        if p is not None:                           # (nothing proved while part of it is unread)
            part = np.empty((bv.shape[0] + 1,) + tuple(bv.shape[1:]), bv.dtype)
            bv._read_into([(0, bv.shape[0] + 1, 1), (0, bv.shape[1], 1)], part, None, p)
            assert not p.proven


def test_regions_kept_and_let_go(monkeypatch):
    """Columns of a file of row chunks: the same values kept as decoded, from
    the third read on out of one region; it goes with the file."""
    _chunks.warm()
    monkeypatch.setattr(_chunks, 'CACHE_CHUNKS', 1)
    with fastnc.Dataset(os.path.join(DATA, 'comp.hdf')) as d:
        d.set_auto_maskandscale(False)
        v = d.variables['chunk_rows']
        whole = v[...]
        for x in (5, 6, 7, 40):
            assert v[:, x].tobytes() == whole[:, x].tobytes()
        index = v._bv.layout().index
        assert index in _hold._held
    assert index not in _hold._held


@pytest.mark.parametrize('name', ['contig_big', 'vector'])
def test_close_unmaps(name, tmp_path):
    p = str(tmp_path / 'x.hdf')
    with open(os.path.join(DATA, 'layout.hdf'), 'rb') as f, open(p, 'wb') as g:
        g.write(f.read())
    d = fastnc.Dataset(p)
    v = d.variables[name]
    v[...]
    v[3]
    mm = d._mm
    d.close()
    assert mm.closed
    with pytest.raises(ValueError):
        v[3]
    with pytest.raises(ValueError):
        v[...]
    os.remove(p)                                    # (Windows: a mapped file cannot go)


@pytest.mark.parametrize('stage_min', [0, None])
def test_close_lets_go_of_the_chunk_reader(stage_min, tmp_path, monkeypatch):
    """Chunks staged by positional reads leave a reader on the file: it closes."""
    _chunks.warm()
    if stage_min is not None:
        monkeypatch.setattr(_pio, 'STAGE_MIN', stage_min)
    p = str(tmp_path / 'x.hdf')
    with open(os.path.join(DATA, 'comp.hdf'), 'rb') as f, open(p, 'wb') as g:
        g.write(f.read())
    d = fastnc.Dataset(p)
    d.set_auto_maskandscale(False)
    v = d.variables['chunk_none']
    want = EXPECTED['files']['comp.hdf']['variables']['chunk_none']['values']['...']
    v._bv._quick = False
    assert digest(v[...]) == want
    h = d._backend.f
    staged = h.__dict__.get('_pio') is not None
    assert staged == (stage_min is not None)
    d.close()
    assert '_pio' not in h.__dict__
    os.remove(p)


def test_close_twice_and_a_file_object():
    with open(os.path.join(DATA, 'comp.hdf'), 'rb') as f:
        raw = f.read()
    d = fastnc.Dataset(io.BytesIO(raw))             # (bytes in memory: no file to read from)
    d.set_auto_maskandscale(False)
    check_keys(d, dict(variables={k: x for k, x in EXPECTED['files']['comp.hdf']['variables'].items()
                                  if k in ('deflate1', 'chunk_deflate', 'chunk_missing')}), 1)
    assert not isinstance(d.variables['deflate1']._bv.nc.f.io, _hdf4._File)
    d.close()
    d.close()
    d = fastnc.Dataset(os.path.join(DATA, 'comp.hdf'))
    d._backend.close()
    d._backend.close()
    d.close()


def test_swapped_on_threads(monkeypatch):
    monkeypatch.setattr(_hdf4, 'SWAP_BLOCK', 64)
    monkeypatch.setattr(_pool, 'memory_threads', lambda n, most, speed=64: 4)
    monkeypatch.setattr(_nc3, 'KERNELS', True)
    rng = np.random.default_rng(4)
    for t in ('i2', 'f4', 'f8'):
        a = rng.integers(0, 255, 9600, np.uint8).view(t).reshape(-1, 3)
        b = a.copy()
        _hdf4._swapped(b)
        assert b.tobytes() == a.byteswap().tobytes(), t


def test_small_reads_and_threads_with_kept_chunks():
    """User threads over one file: chunks kept, decoded, read and let go."""
    _chunks.warm()
    bad = []
    with fastnc.Dataset(os.path.join(DATA, 'comp.hdf')) as d:
        d.set_auto_maskandscale(False)
        names = ['chunk_rows', 'chunk_deflate', 'chunk_none', 'chunk_3d']
        whole = {k: d.variables[k][...] for k in names}

        def run(seed):
            rng = np.random.default_rng(seed)
            for _ in range(200):
                k = names[int(rng.integers(0, len(names)))]
                key = explicit_key(rng, whole[k].shape)
                if d.variables[k][key].tobytes() != np.ascontiguousarray(whole[k][key]).tobytes():
                    bad.append((k, key))
        ts = [threading.Thread(target=run, args=(i,)) for i in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    assert not bad


def test_attributes_remembered():
    with fastnc.Dataset(os.path.join(DATA, 'types.hdf')) as d:
        v = d.variables['float32']
        v[0]
        v[1]
        assert isinstance(v._bv.attrs, _nc3.Attrs) and isinstance(v._bv.attrs, dict)
        assert v._bv.attrs is v._bv.attrs
