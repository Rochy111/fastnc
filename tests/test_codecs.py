"""The codec kernels (lzf, lz4, bitshuffle, zstd, blosc, blosc2) byte for
byte against the reference filters, and their behaviour on damaged input.

The chunks are those of tests/data/filters.h5 and tests/data/codecs/*.h5
(written by tools/check/codecs_make.py, which also recorded what
hdf5plugin decodes each into: tests/data/codecs/expected.json), read with
fastnc's own HDF5 parser -- no h5py or hdf5plugin needed here.

Damaged input (truncated, bytes changed, too little room for the output)
must give an error code or some output, never a crash, a write outside
dst[o:oend] or the workspace, or a read outside src[p:pend] (checked by
decoding with two different paddings around the input and comparing).
"""
import hashlib
import json
import os

import numpy as np
import pytest

from fastnc import _hdf5

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
CODECS = os.path.join(DATA, 'codecs')
MODULES = {32000: 'lzf', 32001: 'blosc', 32004: 'lz4', 32008: 'bitshuffle', 32015: 'zstd',
           32026: 'blosc2'}

with open(os.path.join(CODECS, 'expected.json')) as fh:
    EXPECTED = json.load(fh)


def module(fid):
    import importlib
    return importlib.import_module('fastnc._' + MODULES[fid])


def _walk(f, g, path, out):
    for name, (kind, target) in g.links.items():
        if kind != 'hard':
            continue
        p = path.rstrip('/') + '/' + name
        h = f.object_header(target)
        if h.first(8) is not None:
            out[p] = _hdf5.Dataset(f, target, name, h)
        elif h.first(3) is None:
            _walk(f, _hdf5.Group(f, target, name, h), p, out)


_FILES = {}


def datasets(fname):
    """{path: (file bytes, Dataset)} of one corpus file."""
    if fname not in _FILES:
        with open(os.path.join(CODECS, fname), 'rb') as fh:
            buf = fh.read()
        f = _hdf5.H5File(buf)
        out = {}
        _walk(f, f.root, '/', out)
        _FILES[fname] = (np.frombuffer(buf, np.uint8), out)
    return _FILES[fname]


def cd_array(ds):
    flt = ds.filters[-1]
    nb = ds.type.size * int(np.prod(ds.chunks))
    return np.array([ds.type.size, nb, len(flt.cd)] + list(flt.cd), np.int64)


def chunks_of(fname):
    """[(name, chunk offsets, mask, src, p, pend, module, cd, expected (n, sha1))]."""
    buf, sets = datasets(fname)
    out = []
    for name, want in sorted(EXPECTED[fname].items()):
        ds = sets[name]
        fid = ds.filters[-1].id
        table = {','.join(map(str, c[0])): c for c in ds.chunk_table()}
        for key, (mask, n, sha) in sorted(want.items()):
            offs, addr, size, m = table[key]
            assert m == mask
            out.append((name, key, mask, buf, addr, addr + size, fid, cd_array(ds), (n, sha)))
    return out


FILES = sorted(EXPECTED)


@pytest.mark.parametrize('fname', FILES)
def test_reference(fname):
    """Every chunk decodes to exactly what the reference filter gives."""
    bad = []
    n = 0
    for name, key, mask, buf, p, pend, fid, cd, (nb, sha) in chunks_of(fname):
        if mask & 1:
            continue                            # filter skipped for this chunk
        m = module(fid)
        ws = np.zeros(m.workspace_size(int(cd[1])), np.uint8)
        dst = np.full(int(cd[1]) + 16, 0xA5, np.uint8)
        r = m.decode(buf, p, pend, dst, 8, 8 + int(cd[1]), cd, ws)
        got = dst[8:8 + nb].tobytes()
        if r != 8 + nb or hashlib.sha1(got).hexdigest() != sha or (dst[:8] != 0xA5).any() \
                or (dst[8 + int(cd[1]):] != 0xA5).any():
            bad.append((name, key, r))
        n += 1
    assert n > 0
    assert not bad, bad[:10]


def _samples(per_codec=6):
    """A few chunks of every codec for the damage tests: the smallest
    compressed ones of different datasets, and one big one."""
    pick = {}
    for fname in FILES:
        for c in chunks_of(fname):
            if c[2] & 1:
                continue
            pick.setdefault(c[6], []).append(c)
    out = []
    for fid, cs in sorted(pick.items()):
        cs = sorted(cs, key=lambda c: c[5] - c[4])
        step = max(1, len(cs) // per_codec)
        out += cs[::step][:per_codec] + [cs[-1]]
    return out


def _run(m, src, p, pend, cd, nb, room, rng):
    """decode() with the input between two random paddings (twice, different
    paddings), the output between guards, the workspace inside a larger
    array: (result, output) after checking nothing outside was touched or
    read."""
    results = []
    for k in range(2):
        pad = rng.integers(0, 256, 64, dtype=np.uint8)
        s = np.concatenate([pad, src[p:pend], pad[::-1]])
        dst = np.full(room + 32, 0x5A, np.uint8)
        wsz = m.workspace_size(nb)
        wsb = np.full(wsz + 32, 0x3C, np.uint8)
        ws = wsb[16:16 + wsz]
        r = m.decode(s, 64, 64 + pend - p, dst, 16, 16 + room, cd, ws)
        assert (dst[:16] == 0x5A).all() and (dst[16 + room:] == 0x5A).all(), 'wrote outside dst'
        assert (wsb[:16] == 0x3C).all() and (wsb[16 + wsz:] == 0x3C).all(), 'wrote outside ws'
        assert r < 0 or 16 <= r <= 16 + room
        results.append((r, dst[16:r].tobytes() if r > 0 else b''))
    assert results[0] == results[1], 'read outside src[p:pend]'
    return results[0]


@pytest.mark.parametrize('trial', range(4))
def test_damaged(trial):
    """Truncated and changed input, and too little room: an error or some
    output, never a write outside the given ranges or a read past the
    input."""
    rng = np.random.default_rng(trial)
    for name, key, mask, buf, p, pend, fid, cd, (nb, sha) in _samples():
        m = module(fid)
        src = buf
        n = pend - p
        # the intact chunk still decodes, through the padded harness
        r, out = _run(m, src, p, pend, cd, nb, nb, rng)
        assert r == 16 + nb and hashlib.sha1(out).hexdigest() == sha, (name, key)
        # truncated
        cuts = {0, 1, 2, 3, 5, 8, 12, 16, 31, 32, 40, n // 2, n - 4, n - 1}
        cuts |= set(int(x) for x in rng.integers(0, n, 8))
        for c in sorted(x for x in cuts if 0 <= x < n):
            _run(m, src, p, p + c, cd, nb, nb, rng)
        # bytes changed
        for _ in range(40):
            bad = src[p:pend].copy()
            for _ in range(int(rng.integers(1, 4))):
                i = int(rng.integers(0, n)) if rng.random() < 0.5 else int(rng.integers(0, min(n, 128)))
                bad[i] = rng.integers(0, 256)
            _run(m, bad, 0, n, cd, nb, nb, rng)
        # too little room
        for room in (0, 1, nb // 2, nb - 1):
            if room >= 0:
                r, out = _run(m, src, p, pend, cd, nb, room, rng)
                assert r < 0 or r - 16 <= room


def test_zstd_frames():
    """Frame-level features ZSTD_compress through HDF5 does not produce:
    several frames, skippable frames, a checksum flag, an empty input."""
    from fastnc import _zstd
    buf, sets = datasets('zstd.h5')
    a = sets['/zstd/size300']
    b = sets['/zstd/smooth_l3']
    (_, aa, an, _), = [c for c in a.chunk_table()]
    ba = b.chunk_table()[0]
    fa = buf[aa:aa + an]
    fb = buf[ba[1]:ba[1] + ba[2]]
    ws = np.zeros(_zstd.WS_SIZE, np.uint8)

    def dec(src, room):
        dst = np.zeros(room, np.uint8)
        r = _zstd.zstd_decompress(src, 0, len(src), dst, 0, room, ws)
        return r, dst[:max(r, 0)].tobytes()

    ra, da = dec(fa, 1 << 20)
    rb, db = dec(fb, 1 << 20)
    assert ra == 300 and rb > 0
    skip = np.frombuffer(b'\x5a\x2a\x4d\x18\x05\x00\x00\x00hello', np.uint8)
    two = np.concatenate([fa, skip, fb, skip])
    assert dec(two, 1 << 20) == (ra + rb, da + db)
    assert dec(np.zeros(0, np.uint8), 10) == (0, b'')
    assert dec(skip, 10) == (0, b'')
    assert dec(two, ra + rb - 1)[0] < 0
    # the checksum flag: four more bytes after the last block, not checked
    c = fa.copy()
    c[4] |= 4
    c = np.concatenate([c, np.zeros(4, np.uint8)])
    assert dec(c, 1 << 20) == (ra, da)
    assert dec(c[:-1], 1 << 20)[0] < 0
    # a dictionary id: not supported, an error
    d = fa.copy()
    d[4] |= 1
    d = np.concatenate([d[:6], np.ones(1, np.uint8), d[6:]])
    assert dec(d, 1 << 20)[0] < 0


def test_bitshuffle_transpose():
    """The bit transpose against a plain numpy one, for element sizes 1 to
    17 and element counts with leftovers."""
    from fastnc import _bitshuffle
    rng = np.random.default_rng(5)
    y = np.zeros(64, np.uint64)
    for es in list(range(1, 18)) + [24, 32]:
        for n in (8, 64, 72, 200, 1024):
            x = rng.integers(0, 256, (n, es), dtype=np.uint8)
            bits = np.unpackbits(x, axis=1, bitorder='little').reshape(n, es, 8)
            shuffled = np.packbits(bits.transpose(1, 2, 0).reshape(es * 8, n), axis=1,
                                   bitorder='little').ravel()
            out = np.zeros(n * es + 16, np.uint8)
            _bitshuffle.bitunshuffle(shuffled, 0, out, 0, n, es, n * es, y)
            assert out[:n * es].tobytes() == x.tobytes(), (es, n)
            out2 = np.zeros(n * es + 16, np.uint8)
            _bitshuffle.bitunshuffle(shuffled, 0, out2, 0, n, es, len(out2), y)
            assert out2[:n * es].tobytes() == x.tobytes(), (es, n)


def test_unshuffle():
    """blosc's byte unshuffle against numpy, with leftover bytes."""
    from fastnc import _blosc
    rng = np.random.default_rng(6)
    for ts in (1, 2, 3, 4, 5, 8, 12, 16, 24):
        for n in (0, 7, 64, 100, 1000, 4099):
            src = rng.integers(0, 256, n, dtype=np.uint8)
            ne = n // ts
            want = np.concatenate([src[:ne * ts].reshape(ts, ne).T.ravel(), src[ne * ts:]])
            out = np.zeros(n, np.uint8)
            _blosc.unshuffle(src, 0, out, 0, n, ts)
            assert out.tobytes() == want.tobytes(), (ts, n)
