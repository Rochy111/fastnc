"""HDF5's own filters -- szip, scale-offset, n-bit, fletcher32, shuffle --
and bzip2, against HDF5's decoding of the same chunks (digests made by
tools/check/filters_hdf_make.py with HDF5 1.14.6 and 2.0 through h5py),
against libaec's SZ layer for streams HDF5 cannot make, and fuzzed.
Needs neither h5py nor HDF5.
"""
import hashlib
import json
import os

import numpy as np
import pytest

from fastnc import _bzip2, _fletcher32, _hdf5, _inflate, _nbit, _scaleoffset, _shuffle, _szip

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FH = os.path.join(DATA, 'filters_hdf')
with open(os.path.join(FH, 'expected.json')) as _fh:
    EXPECTED = json.load(_fh)
MODULES = {2: _shuffle, 3: _fletcher32, 4: _szip, 5: _nbit, 6: _scaleoffset, 307: _bzip2}


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


def walk(f, g=None, path='', out=None):
    out = {} if out is None else out
    g = f.root if g is None else g
    for name, (kind, target) in g.links.items():
        if kind != 'hard':
            continue
        p = path + '/' + name
        h = f.object_header(target)
        if h.first(8) is not None:
            out[p] = _hdf5.Dataset(f, target, name, h)
        elif h.first(3) is None:
            walk(f, _hdf5.Group(f, target, name, h), p, out)
    return out


def cd_array(ds, filt):
    nbytes = int(np.prod(ds.chunks)) * ds.type.size
    return np.array([ds.type.size, nbytes, len(filt.cd)] + list(filt.cd), np.int64)


def decode_chunk(src, ds, addr, size, mask):
    """The chunk through the reverse of its pipeline: a uint8 array, or the
    first error code."""
    nbytes = int(np.prod(ds.chunks)) * ds.type.size
    buf, p, pend = src, addr, addr + size
    for i in range(len(ds.filters) - 1, -1, -1):
        filt = ds.filters[i]
        if mask & (1 << i):
            continue
        room = max(nbytes, pend - p)
        out = np.zeros(room, np.uint8)
        if filt.id == 1:
            tab, iws = _inflate.workspace()
            end = _inflate.zlib_inflate(buf, p, pend, out, 0, room, room, tab, iws)
        else:
            mod = MODULES[filt.id]
            ws = np.zeros(max(1, mod.workspace_size(nbytes)), np.uint8)
            end = mod.decode(buf, p, pend, out, 0, room, cd_array(ds, filt), ws)
        if end < 0:
            return end
        buf, p, pend = out, 0, end
    return np.array(buf[p:pend])


def _load(key):
    with open(os.path.join(DATA, key), 'rb') as fh:
        b = np.frombuffer(fh.read(), np.uint8)
    f = _hdf5.H5File(b)
    return b, walk(f)


@pytest.mark.parametrize('key', sorted(EXPECTED))
def test_chunks_equal_hdf5(key):
    """Every chunk of every dataset, decoded whole, equals HDF5's decoding."""
    b, dsets = _load(key)
    n = 0
    bad = []
    for name, rec in EXPECTED[key].items():
        ds = dsets[name]
        for offs, addr, size, mask in ds.chunk_table():
            want = rec.get(','.join(map(str, offs)))
            if want is None:
                continue
            got = decode_chunk(b, ds, addr, size, mask)
            n += 1
            if isinstance(got, (int, np.integer)) or got.size != want['n'] or sha(got) != want['sha']:
                bad.append((name, offs, got if isinstance(got, (int, np.integer)) else got.size))
    assert not bad, bad[:5]
    assert n > 0


def test_every_chunk_has_a_reference():
    """Each chunk of the edge-case files was decoded by some HDF5 (1.14 or 2.0)."""
    for key, recs in EXPECTED.items():
        b, dsets = _load(key)
        for name, rec in recs.items():
            assert len(rec) == len(dsets[name].chunk_table()), (key, name)


# ------------------------------------------------------------------ szip

def _szlib_cases():
    z = np.load(os.path.join(FH, 'szlib_cases.npz'), allow_pickle=True)
    return z['params'], z['streams'], z['decoded_sha']


def szip_cd(mask, ppb, bpp, pps, nbytes, esize=1):
    return np.array([esize, nbytes, 4, mask, ppb, bpp, pps], np.int64)


def test_szip_against_libaec():
    """Streams libaec's SZ layer made for every bits-per-pixel 1-32 and 64,
    option masks with and without RAW / NN / MSB / CHIP, pixels per block
    8-64 and 10, scanlines shorter than a block or padded."""
    params, streams, shas = _szlib_cases()
    bad = []
    for (bpp, mask, ppb, pps, n), st, want in zip(params, streams, shas):
        src = np.concatenate([np.frombuffer(int(n).to_bytes(4, 'little'), np.uint8), st])
        dst = np.zeros(n + 16, np.uint8)
        ws = np.zeros(_szip.workspace_size(n), np.uint8)
        end = _szip.decode(src, 0, src.size, dst, 0, n, szip_cd(mask, ppb, bpp, pps, n), ws)
        if end != n or sha(dst[:n]) != want or dst[n:].any():
            bad.append((bpp, mask, ppb, pps, n, end))
    assert not bad, bad[:5]


def test_szip_size_header():
    params, streams, shas = _szlib_cases()
    bpp, mask, ppb, pps, n = params[0]
    st = streams[0]
    src = np.concatenate([np.frombuffer(int(n).to_bytes(4, 'little'), np.uint8), st])
    ws = np.zeros(n, np.uint8)
    cd = szip_cd(mask, ppb, bpp, pps, n)
    assert _szip.decode(src, 0, src.size, np.zeros(n, np.uint8), 0, n - 1, cd, ws) == _szip.ERR_SPACE
    assert _szip.decode(src, 0, 3, np.zeros(n, np.uint8), 0, n, cd, ws) == _szip.ERR_INPUT
    for bad in (szip_cd(mask, 7, bpp, pps, n), szip_cd(mask, 66, bpp, pps, n),
                szip_cd(mask, ppb, 40, pps, n), szip_cd(mask, ppb, bpp, 0, n)):
        assert _szip.decode(src, 0, src.size, np.zeros(n, np.uint8), 0, n, bad, ws) == _szip.ERR_DATA


def _bits_to_bytes(bits):
    bits = bits + '0' * (-len(bits) % 8)
    return np.array([int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)], np.uint8)


def test_szip_second_extension_long_codes():
    """Second-extension codes of 57-90 zeros (longer than one 64-bit load
    holds), and 91 (libaec's table ends at 90: an error)."""
    se = []
    for s in range(14):
        for d1 in range(s + 1):
            se.append((s - d1, d1))
    for codes, ok in (([60, 0, 3, 90], True), ([0, 1, 2, 57], True), ([91, 0, 0, 0], False)):
        bits = '000' + '1' + ''.join('0' * m + '1' for m in codes)     # id 0, selector 1: SE
        src = np.concatenate([np.frombuffer((8).to_bytes(4, 'little'), np.uint8), _bits_to_bytes(bits)])
        dst = np.zeros(8, np.uint8)
        cd = szip_cd(128 | 4 | 8, 8, 8, 8, 8)                          # EC, 8 bits, J 8, 8 a line
        end = _szip.decode(src, 0, src.size, dst, 0, 8, cd, np.zeros(8, np.uint8))
        if ok:
            assert end == 8
            assert dst.tolist() == [v for m in codes for v in se[m]]
        else:
            assert end == _szip.ERR_DATA


# ------------------------------------------------------------------ fletcher32

def fletcher_ref(b):
    """H5_checksum_fletcher32, transcribed."""
    s1 = s2 = 0
    n = len(b) // 2
    i = 0
    while n:
        t = min(n, 360)
        n -= t
        for _ in range(t):
            s1 = (s1 + ((b[i] << 8) | b[i + 1])) & 0xFFFFFFFF
            s2 = (s2 + s1) & 0xFFFFFFFF
            i += 2
        s1 = (s1 & 0xFFFF) + (s1 >> 16)
        s2 = (s2 & 0xFFFF) + (s2 >> 16)
    if len(b) % 2:
        s1 += b[i] << 8
        s2 += s1
        s1 = (s1 & 0xFFFF) + (s1 >> 16)
        s2 = (s2 & 0xFFFF) + (s2 >> 16)
    s1 = (s1 & 0xFFFF) + (s1 >> 16)
    s2 = (s2 & 0xFFFF) + (s2 >> 16)
    return ((s2 << 16) | s1) & 0xFFFFFFFF


def test_fletcher32_sums():
    r = np.random.default_rng(5)
    for n in list(range(0, 40)) + [718, 719, 720, 721, 722, 1439, 1440, 1441, 5000, 20001]:
        for kind in range(3):
            a = (r.integers(0, 256, n, dtype=np.uint8) if kind == 0 else
                 np.full(n, 255, np.uint8) if kind == 1 else np.zeros(n, np.uint8))
            assert _fletcher32.fletcher32(a, 0, n) == fletcher_ref(a.tolist()), (n, kind)


def test_fletcher32_verify_strip():
    r = np.random.default_rng(6)
    data = r.integers(0, 256, 1001, dtype=np.uint8)
    f = fletcher_ref(data.tolist())
    swapped = ((f & 0x00FF00FF) << 8) | ((f >> 8) & 0x00FF00FF)
    for c in (f, swapped):                      # (the old byte order is accepted, as by HDF5)
        src = np.concatenate([np.zeros(3, np.uint8), data,
                              np.frombuffer(c.to_bytes(4, 'little'), np.uint8), np.ones(5, np.uint8)])
        dst = np.full(1100, 7, np.uint8)
        end = _fletcher32.decode(src, 3, 3 + 1005, dst, 10, 1011, None, None)
        assert end == 1011 and (dst[10:1011] == data).all() and (dst[:10] == 7).all() and (dst[1011:] == 7).all()
    src = np.concatenate([data, np.frombuffer((f ^ 1).to_bytes(4, 'little'), np.uint8)])
    dst = np.zeros(1001, np.uint8)
    assert _fletcher32.decode(src, 0, src.size, dst, 0, 1001, None, None) == _fletcher32.ERR_DATA
    assert _fletcher32.strip(src, 0, src.size, dst, 0, 1001, None, None) == 1001
    assert (dst == data).all()
    assert _fletcher32.decode(src, 0, 3, dst, 0, 1001, None, None) == _fletcher32.ERR_INPUT
    assert _fletcher32.decode(src, 0, src.size, dst, 0, 1000, None, None) == _fletcher32.ERR_SPACE


# ------------------------------------------------------------------ shuffle

def unshuffle_ref(b, es):
    """H5Zshuffle.c's reverse pass."""
    n = len(b)
    m = n // es if es else 0
    if es <= 1 or m <= 1:
        return bytes(b)
    out = bytearray(n)
    for i in range(es):
        for j in range(m):
            out[j * es + i] = b[i * m + j]
    out[m * es:] = b[m * es:]
    return bytes(out)


def test_unshuffle():
    r = np.random.default_rng(8)
    for es in (1, 2, 3, 4, 5, 7, 8, 12, 16):
        for n in (0, 1, es - 1, es, es + 1, 2 * es, 2 * es + 1, 7 * es + 3, 64 * es, 1001):
            if n < 0:
                continue
            a = r.integers(0, 256, n, dtype=np.uint8)
            src = np.concatenate([np.full(5, 9, np.uint8), a, np.full(5, 9, np.uint8)])
            dst = np.full(n + 20, 3, np.uint8)
            cd = np.array([es, n, 1, es], np.int64)
            end = _shuffle.decode(src, 5, 5 + n, dst, 10, 10 + n, cd, None)
            assert end == 10 + n
            assert dst[10:10 + n].tobytes() == unshuffle_ref(a.tobytes(), es), (es, n)
            assert (dst[:10] == 3).all() and (dst[10 + n:] == 3).all()


# ------------------------------------------------------------------ scale-offset, n-bit edge cases

def test_scaleoffset_header_checks():
    cd = np.array([2, 40, 20, 2, 0, 10, 0, 4, 1, 0, 0] + [0] * 12, np.int64)
    hdr = np.zeros(21, np.uint8)
    hdr[0] = 33                                 # minbits > 32
    dst = np.zeros(40, np.uint8)
    assert _scaleoffset.decode(hdr, 0, 21, dst, 0, 40, cd, None) == _scaleoffset.ERR_DATA
    hdr[0] = 5                                  # 10 x 5 bits: 7 bytes needed, none there
    assert _scaleoffset.decode(hdr, 0, 21, dst, 0, 40, cd, None) == _scaleoffset.ERR_INPUT
    assert _scaleoffset.decode(hdr, 0, 20, dst, 0, 40, cd, None) == _scaleoffset.ERR_INPUT
    assert _scaleoffset.decode(hdr, 0, 21, dst, 0, 39, cd, None) == _scaleoffset.ERR_SPACE
    hdr[0] = 32                                 # full precision, big-endian: swapped
    cd[9] = 1
    src = np.concatenate([hdr, np.arange(40, dtype=np.uint8)])
    assert _scaleoffset.decode(src, 0, src.size, dst, 0, 40, cd, None) == 40
    assert (dst == np.arange(40, dtype=np.uint8).reshape(10, 4)[:, ::-1].ravel()).all()


def test_nbit_bad_parameters():
    dst = np.zeros(400, np.uint8)
    src = np.zeros(400, np.uint8)
    good = [8, 0, 100, 1, 4, 0, 12, 2]
    cd = np.array([4, 400, len(good)] + good, np.int64)
    assert _nbit.decode(src, 0, 400, dst, 0, 400, cd, None) == 400
    for i, v in ((3, 30), (6, 33), (7, 21), (3, 7), (0, 9)):    # precision, offset, class, count
        g = list(good)
        g[i] = v
        cd = np.array([4, 400, len(g)] + g, np.int64)
        assert _nbit.decode(src, 0, 400, dst, 0, 400, cd, None) == _nbit.ERR_DATA, (i, v)
    cd = np.array([4, 400, len(good)] + good, np.int64)
    assert _nbit.decode(src, 0, 149, dst, 0, 400, cd, None) == _nbit.ERR_INPUT
    assert _nbit.decode(src, 0, 400, dst, 0, 399, cd, None) == _nbit.ERR_SPACE
    # a compound whose members overflow it
    comp = [3, 4, 2, 0, 1, 2, 0, 8, 0, 1, 1, 4, 0, 8, 0]
    cd = np.array([4, 400, len(comp) + 3, len(comp) + 3, 0, 10] + comp, np.int64)
    assert _nbit.decode(src, 0, 400, dst, 0, 400, cd, None) == _nbit.ERR_DATA


def test_bzip2_errors():
    import bz2
    raw = np.arange(5000, dtype='<i4').tobytes()
    z = bz2.compress(raw)
    assert _bzip2.decode_py(z, len(raw), (9,)) == raw
    assert _bzip2.decode_py(z + b'junk', len(raw), (9,)) == raw
    with pytest.raises(ValueError):
        _bzip2.decode_py(z[:-10], len(raw), (9,))
    with pytest.raises(ValueError):
        _bzip2.decode_py(z, len(raw) - 1, (9,))
    with pytest.raises(ValueError):
        _bzip2.decode_py(b'BZh9' + b'\0' * 50, len(raw), (9,))
    ws = np.zeros(_bzip2.workspace_size(0), np.uint8)
    dst = np.zeros(len(raw), np.uint8)
    src = np.frombuffer(z + b'junk', np.uint8)
    assert _bzip2.decode(src, 0, src.size, dst, 0, len(raw), None, ws) == len(raw)
    assert dst.tobytes() == raw
    assert _bzip2.decode(src, 0, len(z) - 10, dst, 0, len(raw), None, ws) == _bzip2.ERR_INPUT
    assert _bzip2.decode(src, 0, src.size, dst, 0, len(raw) - 1, None, ws) == _bzip2.ERR_SPACE
    bad = np.frombuffer(z, np.uint8).copy()
    bad[len(z) // 2] ^= 4                       # (caught by the block CRC at worst)
    assert _bzip2.decode(bad, 0, bad.size, dst, 0, len(raw), None, ws) < 0


def test_bzip2_numba_equals_libbzip2():
    """Levels 1-9, several blocks, runs longer than the RLE1 limit, every
    byte value, empty input."""
    import bz2
    r = np.random.default_rng(3)
    side = 600
    x = np.cumsum(np.cumsum(r.standard_normal((side, side)), 0), 1).astype('f4').tobytes()
    datas = [b'', b'a', bytes(range(256)) * 9, bytes(300000), x, x[:5000] * 7,
             np.repeat(r.integers(0, 256, 3000, dtype=np.uint8), r.integers(1, 600, 3000)).tobytes()]
    ws = np.zeros(_bzip2.workspace_size(0), np.uint8)
    for raw in datas:
        for lev in (1, 5, 9):
            z = bz2.compress(raw, lev)
            src = np.frombuffer(z, np.uint8)
            dst = np.zeros(len(raw) + 8, np.uint8)
            assert _bzip2.decode(src, 0, src.size, dst, 0, len(raw), None, ws) == len(raw)
            assert dst[:len(raw)].tobytes() == raw == _bzip2.decode_py(z, len(raw), (lev,))


# ------------------------------------------------------------------ fuzz

def _fuzz_inputs():
    """(module, cd, chunk bytes, output size) from the committed files."""
    out = []
    for key, want_ids in (('filters_hdf/szip.h5', {4}), ('filters_hdf/szip_nbit.h5', {4}),
                          ('filters_hdf/scaleoffset.h5', {6}), ('filters_hdf/nbit.h5', {5}),
                          ('filters_hdf/fletcher.h5', {3}), ('filters_hdf/bzip2.h5', {307})):
        b, dsets = _load(key)
        names = sorted(dsets)[::max(1, len(dsets) // 40)]
        for name in names:
            ds = dsets[name]
            if len(ds.filters) != 1 or ds.filters[0].id not in want_ids:
                continue
            offs, addr, size, mask = ds.chunk_table()[0]
            if mask:
                continue
            cd = cd_array(ds, ds.filters[0])
            out.append((MODULES[ds.filters[0].id], cd, np.array(b[addr:addr + size]), int(cd[1])))
    return out


def test_fuzz_no_crash_no_stray_writes():
    """Damaged chunks (bits flipped, bytes replaced, cut short, random
    bytes) return an error or some output, never write outside the range
    given, and never crash."""
    r = np.random.default_rng(1234)
    cases = _fuzz_inputs()
    assert len(cases) > 40
    for mod, cd, chunk, nbytes in cases:
        ws = np.zeros(max(1, mod.workspace_size(nbytes)), np.uint8)
        for trial in range(25):
            c = chunk.copy()
            kind = trial % 5
            if kind == 0 and c.size:
                for _ in range(1 + trial // 5):
                    i = r.integers(0, c.size)
                    c[i] ^= np.uint8(1 << r.integers(0, 8))
            elif kind == 1 and c.size:
                c[r.integers(0, c.size, 4)] = r.integers(0, 256, 4)
            elif kind == 2:
                c = c[:r.integers(0, c.size + 1)]
            elif kind == 3:
                c = r.integers(0, 256, c.size, dtype=np.uint8)
            elif kind == 4 and c.size > 4:
                c[4:] = r.integers(0, 256, c.size - 4, dtype=np.uint8)
            pad = 64
            src = np.concatenate([np.full(pad, 0xA5, np.uint8), c, np.full(pad, 0xA5, np.uint8)])
            dst = np.full(nbytes + 2 * pad, 0x5A, np.uint8)
            end = mod.decode(src, pad, pad + c.size, dst, pad, pad + nbytes, cd, ws)
            assert end < 0 or pad <= end <= pad + nbytes, (mod.__name__, end)
            assert (dst[:pad] == 0x5A).all() and (dst[pad + nbytes:] == 0x5A).all(), mod.__name__


def test_fuzz_cd_values():
    """Random filter parameters: an error or an output, nothing else."""
    r = np.random.default_rng(99)
    src = r.integers(0, 256, 3000, dtype=np.uint8)
    src[:4] = np.frombuffer((1000).to_bytes(4, 'little'), np.uint8)
    for mod in (_szip, _nbit, _scaleoffset, _shuffle, _bzip2):
        for _ in range(300):
            n = int(r.integers(0, 30))
            vals = r.integers(0, 70, n)
            if r.random() < 0.3:
                vals = r.integers(0, 2 ** 32, n)
            cd = np.array([int(r.integers(1, 9)), 1000, n] + list(vals), np.int64)
            if mod is _nbit and n:
                cd[3] = n
            dst = np.full(1200, 0x5A, np.uint8)
            ws = np.zeros(max(2000, mod.workspace_size(1000)), np.uint8)
            end = mod.decode(src, 0, src.size, dst, 100, 1100, cd, ws)
            assert end < 0 or 100 <= end <= 1100
            assert (dst[:100] == 0x5A).all() and (dst[1100:] == 0x5A).all()
