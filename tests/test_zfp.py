"""The zfp filter (32013), against what the real decoder gives: two chunks and the
file of the HDF Group's example (H5Z-ZFP 1.0.1, zfp 0.5.5), and every type, dimension
and mode as made and decoded by H5Z-ZFP 1.1.1 with zfp 1.0.1 (digests in tests/data/zfp,
by tools/check/zfp_make.py). Needs neither h5py nor the plugin.
"""
import hashlib
import json
import os
import shutil

import numpy as np
import pytest

import fastnc
from fastnc import _codecs, _staged, _zfp

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'zfp')
with open(os.path.join(DATA, 'expected.json')) as _fp:
    EXPECTED = json.load(_fp)
STREAMS = {s['name']: s for s in EXPECTED['streams']}

# h5ex_d_zfp.h5's DS1: 32 x 64 float32 in chunks of 4 x 8, tolerance 0, whose values are
# (row - 1) * column; a long header, of six cd values
DS1_CD = [5570817, 91252346, 805306486, 4293918720, 3767009280, 493487]
DS1_CHUNKS = {
    (4, 8): '0d2d161db74c00000000000000000000000000d0d042c174cb0400000000000000000000000000',
    (28, 56): '15ade81b38e3960900000000000000000000a0a2417c1323dd32010000000000000000000000',
}
GUARD = 64


def sha(a):
    return hashlib.sha256(bytes(a)).hexdigest()[:32]


def ds1(rows=range(32), cols=range(64)):
    return np.array([[(i - 1) * j for j in cols] for i in rows], 'f4')


def decode(src, cd, esize, nbytes, room=None):
    """The kernel with guard bytes around its output: (what it returned, the output)."""
    room = nbytes if room is None else room
    src = np.array(src, np.uint8)
    dst = np.full(room + 2 * GUARD, 0xA5, np.uint8)
    ws = np.empty(_zfp.workspace_size(nbytes), np.uint8)
    c = np.array([esize, nbytes, len(cd)] + list(cd), np.int64)
    r = _zfp.decode(src, 0, src.size, dst, GUARD, GUARD + room, c, ws)
    assert (dst[:GUARD] == 0xA5).all() and (dst[GUARD + room:] == 0xA5).all(), 'wrote outside its range'
    assert r == GUARD + nbytes or r in (-1, -2, -3)
    return r, dst[GUARD:GUARD + nbytes]


def stream(s):
    esize = np.dtype(s['dtype']).itemsize
    return np.frombuffer(bytes.fromhex(s['hex']), np.uint8), s['cd'], esize, int(np.prod(s['chunk'])) * esize


@pytest.mark.parametrize('at', sorted(DS1_CHUNKS))
def test_real_chunks(at):
    src = np.frombuffer(bytes.fromhex(DS1_CHUNKS[at]), np.uint8)
    r, out = decode(src, DS1_CD, 4, 128)
    assert r == GUARD + 128
    want = ds1(range(at[0], at[0] + 4), range(at[1], at[1] + 8))
    assert np.array_equal(np.frombuffer(out.tobytes(), 'f4').reshape(4, 8), want)


def test_real_file():
    with fastnc.File(os.path.join(DATA, 'h5ex_d_zfp.h5')) as f:
        d = f['DS1']
        assert (d.shape, d.dtype, d.chunks) == ((32, 64), np.dtype('f4'), (4, 8))
        assert (d.compression, d.compression_opts) == ('unknown', None)     # (as h5py says)
        assert np.array_equal(d[()], ds1())
        assert d[()].max() == 1890.0                                        # the example's own printout
        assert np.array_equal(d[3:9, 60:], ds1(range(3, 9), range(60, 64)))
        assert d[31, 63] == 1890.0


def test_every_type_dimension_and_mode_was_made():
    have = {(s['name'].split('-')[0], s['dtype'], s['name'].split('-')[2]) for s in EXPECTED['streams']}
    for d in ('1d', '2d', '3d', '4d'):
        for t in ('f4', 'f8', 'i4', 'i8', 'u4', 'u8'):
            for m in ('rate', 'precision', 'accuracy', 'reversible', 'expert'):
                assert (d, t, m) in have
    assert 'H5Z-ZFP-1.1.1' in EXPECTED['source'] and 'ZFP-1.0.1' in EXPECTED['source']
    assert {len(s['cd']) for s in EXPECTED['streams']} == {4, 6}            # short and long headers


@pytest.mark.parametrize('name', sorted(STREAMS))
def test_as_the_reference_decodes(name):
    src, cd, esize, nbytes = stream(STREAMS[name])
    r, out = decode(src, cd, esize, nbytes)
    assert r == GUARD + nbytes
    assert sha(out) == STREAMS[name]['sha']


@pytest.mark.parametrize('name', sorted(EXPECTED['files']))
def test_through_the_file(name):
    want = EXPECTED['files'][name]
    with fastnc.File(os.path.join(DATA, 'made.h5')) as f:
        a = f[name][()]
    assert list(a.shape) == want['shape'] and a.dtype == np.dtype(want['dtype'])
    assert sha(a.tobytes()) == want['sha']


def test_stream_ending_inside_a_word():
    """H5Z-ZFP stores bytes, not words: the last load must not read past the input,
    and a stream that just fits is whole."""
    src = np.frombuffer(bytes.fromhex(DS1_CHUNKS[(4, 8)]), np.uint8)
    assert src.size % 8 != 0
    assert decode(src, DS1_CD, 4, 128)[0] == GUARD + 128


def test_all_zero_blocks():
    """A block of zeros is one bit in a variable-rate mode, and padded to the full
    rate in a fixed one: six blocks of 9 x 6, of 152 bits each."""
    r, out = decode(np.zeros(1, np.uint8), STREAMS['2d-f4-precision']['cd'], 4, 216)
    assert r == GUARD + 216 and not out.any()
    cd = STREAMS['2d-f4-rate']['cd']
    r, out = decode(np.zeros(114, np.uint8), cd, 4, 216)
    assert r == GUARD + 216 and not out.any()
    assert decode(np.zeros(113, np.uint8), cd, 4, 216)[0] == -3


def test_driver_and_chunks_without_the_filter():
    """The batch kernel, several chunks: one stored as it is (HDF5's filter mask)."""
    src0 = np.frombuffer(bytes.fromhex(DS1_CHUNKS[(4, 8)]), np.uint8)
    plain = ds1(range(8, 12), range(16, 24)).tobytes()
    src1 = np.frombuffer(plain, np.uint8)
    src = np.concatenate([src0, src1, src0])
    ioff = np.array([0, src0.size, src0.size + src1.size], np.int64)
    ilen = np.array([src0.size, src1.size, src0.size], np.int64)
    ooff = np.arange(3, dtype=np.int64) * 192
    olen = np.full(3, 192, np.int64)
    dst = np.zeros(3 * 192, np.uint8)
    cd = np.array([4, 128, len(DS1_CD)] + DS1_CD, np.int64)
    status = np.zeros(3, np.int64)
    outlen = np.zeros(3, np.int64)
    _codecs.run_zfp(src, ioff, ilen, dst, ooff, olen, np.array([0, 1, 0], np.uint8), cd,
                    np.empty(_zfp.workspace_size(128), np.uint8), np.zeros(1, np.int64), status, outlen)
    assert not status.any() and outlen.tolist() == [128, 128, 128]
    a = np.frombuffer(dst.tobytes(), 'f4')
    want = ds1(range(4, 8), range(8, 16)).ravel()
    assert np.array_equal(a[:32], want) and np.array_equal(a[48:80], np.frombuffer(plain, 'f4'))
    assert np.array_equal(a[96:128], want)


def test_registered():
    driver, mod = _codecs.DRIVERS[32013]                # (a stand-in or numba's, by what has run)
    assert driver.__name__ == 'run_zfp' and mod is _zfp

    class F:
        id = 32013
    assert _staged.handles([F()])


@pytest.mark.parametrize('name', ['2d-f4-rate', '3d-f8-accuracy', '1d-i4-reversible', '4d-i8-precision',
                                  '2d-f8-reversible', '3d-f4-expert'])
def test_cut_short(name):
    """A stream that ends early is an error, never values made up from zeros."""
    src, cd, esize, nbytes = stream(STREAMS[name])
    for n in list(range(0, min(src.size, 24))) + [src.size // 2, src.size - 1]:
        r, out = decode(src[:n], cd, esize, nbytes)
        if r > 0:                                                           # (only padding was cut)
            assert sha(out) == STREAMS[name]['sha']
        else:
            assert r == -3


@pytest.mark.parametrize('name', ['2d-f4-rate', '3d-f8-accuracy', '4d-i4-reversible', '2d-f8-expert', 'special-f4-reversible'])
def test_damage_stays_inside(name):
    """Bits flipped in the stream or the header: values may go wrong, as with any
    stream that carries no checksum, but nothing is written outside the range."""
    src, cd, esize, nbytes = stream(STREAMS[name])
    rng = np.random.default_rng(32013)
    for _ in range(300):
        s = src.copy()
        for _ in range(int(rng.integers(1, 5))):
            b = int(rng.integers(0, s.size * 8))
            s[b >> 3] ^= 1 << (b & 7)
        decode(s, cd, esize, nbytes)
        c = list(cd)
        c[int(rng.integers(0, len(c)))] ^= 1 << int(rng.integers(0, 32))
        decode(src, c, esize, nbytes)
        decode(rng.integers(0, 256, int(rng.integers(0, 200)), dtype=np.uint8), cd, esize, nbytes)


def test_refused():
    src, cd, esize, nbytes = stream(STREAMS['2d-f4-rate'])
    assert decode(src, cd, esize, nbytes, nbytes - 1)[0] == -2              # no room
    assert decode(src, cd, 8, nbytes)[0] == -1                              # not the dataset's type
    assert decode(src, cd, esize, nbytes + 4)[0] == -1                      # not the chunk's size
    for n in range(4):
        assert decode(src, cd[:n], esize, nbytes)[0] == -1                  # no header
    bad = list(cd)
    bad[1] ^= 1
    assert decode(src, bad, esize, nbytes)[0] == -1                         # no magic
    bad = list(cd)
    bad[1] = (bad[1] & 0xFFFFFF) | (4 << 24)
    assert decode(src, bad, esize, nbytes)[0] == -1                         # another codec version


def test_long_header_needs_all_of_it():
    """Five of six values would read as a mode with its top bits zero, and decode."""
    src = np.frombuffer(bytes.fromhex(DS1_CHUNKS[(4, 8)]), np.uint8)
    for n in (4, 5):
        assert decode(src, DS1_CD[:n], 4, 128)[0] == -1


def corrupt_file(tmp_path, old, new):
    p = str(tmp_path / 'x.h5')
    shutil.copyfile(os.path.join(DATA, 'h5ex_d_zfp.h5'), p)
    with open(p, 'rb') as f:
        b = f.read()
    assert b.count(old) == 1
    with open(p, 'wb') as f:
        f.write(b.replace(old, new))
    return p


def test_old_codec_says_so(tmp_path):
    p = corrupt_file(tmp_path, b'zfp\x05', b'zfp\x04')
    with fastnc.File(p) as f:
        with pytest.raises(NotImplementedError, match='codec version 4'):
            f['DS1'][()]


def test_not_a_header_says_so(tmp_path):
    p = corrupt_file(tmp_path, b'zfp\x05', b'zip\x05')
    with fastnc.File(p) as f:
        with pytest.raises(IOError, match='no zfp header'):
            f['DS1'][()]
