"""HDF4 through fastnc.Dataset: metadata and raw values against digests
taken from netCDF4-python and pyhdf (tests/data/hdf4/expected.json, made by
tools/check/hdf4_make_expected.py; neither library is needed here), the
reads before and after numba's kernels load, read_many, the HDF4 coders
on edge cases, and corrupted files (errors, never a crash)."""
import hashlib
import json
import os
import struct

import numpy as np
import pytest

import fastnc
from fastnc import _chunks, _hdf4, _hdf4_codecs

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
