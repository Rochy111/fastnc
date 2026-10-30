"""fastnc._pinflate: one zlib stream inflated on several threads must give
exactly what zlib (and the serial fastnc._inflate) gives -- every byte,
the same end, the same kind of error -- whatever the stream: zlib levels
0-9, every strategy, small windows and memory levels, stored blocks,
flush points, repetitive and random data, matches of distance 32768
across the cuts, tiny last blocks; partial reads (ostop); nothing written
outside the output range; corrupt and truncated input an error, never a
crash. Parts are made tiny here so that a few hundred kB already make
many cuts to get right."""
import threading
import zlib

import numpy as np
import pytest

from fastnc import _inflate, _pinflate

TINY = dict(nthreads=8, min_part=1 << 11, parts_per_thread=4)
GUARD = 64


def _data(kind, n, seed=0):
    rng = np.random.default_rng(seed)
    if kind == 'random':
        return rng.integers(0, 256, n, dtype=np.uint8).tobytes()
    if kind == 'zeros':
        return bytes(n)
    if kind == 'walk':                              # 16-bit radiances, MODIS-like
        return np.cumsum(rng.integers(-3, 4, n // 2)).astype('<i2').tobytes()
    if kind == 'text':
        words = [b'alpha', b'beta', b'gamma', b'delta', b'fill', b'65535', b'\n', b' ']
        return b''.join(words[i] for i in rng.integers(0, len(words), n // 4))[:n]
    if kind == 'period32768':                       # matches exactly 32 kB back, everywhere
        block = rng.integers(0, 256, 32768, dtype=np.uint8).tobytes()
        return (block * (n // 32768 + 1))[:n]
    if kind == 'mixed':                             # stretches of fill among noise
        parts = []
        while sum(map(len, parts)) < n:
            k = int(rng.integers(1000, 200000))
            parts.append(bytes(k) if rng.random() < 0.3 else
                         rng.integers(0, 256, k, dtype=np.uint8).tobytes() if rng.random() < 0.3 else
                         np.cumsum(rng.integers(-2, 3, k // 2)).astype('<i2').tobytes())
        return b''.join(parts)[:n]
    raise ValueError(kind)


def _compress(data, level=6, strategy=zlib.Z_DEFAULT_STRATEGY, wbits=15, memlevel=8,
              flush_every=0, flush=zlib.Z_FULL_FLUSH):
    c = zlib.compressobj(level, zlib.DEFLATED, wbits, memlevel, strategy)
    if not flush_every:
        return c.compress(data) + c.flush()
    out = []
    for i in range(0, len(data), flush_every):
        out.append(c.compress(data[i:i + flush_every]))
        out.append(c.flush(flush))
    out.append(c.flush())
    return b''.join(out)


def _serial(src, p, pend, n, o0, oend, ostop):
    out = np.zeros(oend + GUARD, np.uint8)
    tab, iws = _inflate.workspace()
    r = _inflate.zlib_inflate(src, p, pend, out, o0, oend, ostop, tab, iws)
    return r, out


def _check(comp, ref=None, ostop=None, room=0, **kw):
    """Inflate comp in parallel (output at offset GUARD of a guarded array)
    and compare with the serial inflate (and zlib's output if given)."""
    src = np.frombuffer(comp, np.uint8)            # read-only, as a file mapping
    n = len(ref) if ref is not None else 0
    o0 = GUARD
    oend = o0 + (n if ref is not None else max(1 << 20, 8 * len(comp))) + room
    out = np.full(oend + GUARD, 0xA5, np.uint8)
    stop = oend if ostop is None else o0 + ostop
    r = _pinflate.inflate_parallel(src, 0, len(comp), out, o0, oend, stop, **kw)
    assert not np.any(out[:o0] != 0xA5) and not np.any(out[oend:] != 0xA5), 'wrote outside'
    rs, sout = _serial(src, 0, len(comp), n, o0, oend, stop)
    if rs < 0:
        assert r < 0, (r, rs)
        return r
    if ostop is None:
        assert r == rs, (r, rs)
        assert out[o0:r].tobytes() == sout[o0:rs].tobytes()
        if ref is not None:
            assert r - o0 == len(ref) and out[o0:r].tobytes() == ref
    else:
        assert r >= min(stop, rs) and r <= oend
        k = min(r, rs) - o0
        assert out[o0:o0 + k].tobytes() == sout[o0:o0 + k].tobytes()
        if ref is not None:
            assert out[o0:o0 + k].tobytes() == ref[:k]
    return r


KINDS = ['random', 'zeros', 'walk', 'text', 'period32768', 'mixed']


@pytest.mark.parametrize('level', range(10))
@pytest.mark.parametrize('kind', KINDS)
def test_levels(level, kind):
    data = _data(kind, 600_000, seed=level)
    _check(_compress(data, level), data, **TINY)


@pytest.mark.parametrize('strategy', [zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED,
                                      zlib.Z_HUFFMAN_ONLY, zlib.Z_RLE, zlib.Z_FIXED])
@pytest.mark.parametrize('kind', KINDS)
def test_strategies(strategy, kind):
    data = _data(kind, 500_000, seed=strategy)
    for level in (1, 6, 9):
        _check(_compress(data, level, strategy), data, **TINY)


@pytest.mark.parametrize('wbits,memlevel', [(9, 1), (10, 2), (12, 4), (15, 1), (15, 9), (13, 9)])
def test_windows_and_block_sizes(wbits, memlevel):
    for kind in ('walk', 'text', 'mixed'):
        data = _data(kind, 400_000, seed=wbits)
        _check(_compress(data, 6, wbits=wbits, memlevel=memlevel), data, **TINY)


@pytest.mark.parametrize('flush', [zlib.Z_FULL_FLUSH, zlib.Z_SYNC_FLUSH])
@pytest.mark.parametrize('every', [1, 777, 30_000, 65_536, 200_000])
def test_flush_points(flush, every):
    """Empty stored blocks (sync markers), tiny blocks, blocks cut anywhere."""
    data = _data('mixed', 300_000 if every > 1 else 3_000, seed=every)
    _check(_compress(data, 6, flush_every=every, flush=flush), data, **TINY)


def test_stored_only_big():
    data = _data('random', 3_000_000, seed=3)
    for kw in (TINY, dict(nthreads=4)):
        _check(_compress(data, 0), data, **kw)


def test_tiny_and_empty():
    for data in (b'', b'x', b'ab' * 3, _data('random', 100), _data('text', 5000)):
        for level in (0, 1, 9):
            _check(_compress(data, level), data, **TINY)


def test_tiny_last_block():
    """A long stream ending in a block of a few bytes (flushed apart)."""
    data = _data('walk', 500_000, seed=5)
    c = zlib.compressobj(6)
    comp = c.compress(data) + c.flush(zlib.Z_FULL_FLUSH) + c.compress(b'xyz') + c.flush()
    _check(comp, data + b'xyz', **TINY)


def test_distance_32768_across_cuts():
    """Every match exactly 32 kB back, cuts every few kB: each part starts
    with matches into its unknown window only."""
    data = _data('period32768', 2_000_000, seed=7)
    for level in (1, 9):
        for mp in (1 << 11, 1 << 13, 1 << 16):
            _check(_compress(data, level), data, nthreads=8, min_part=mp, parts_per_thread=4)


def test_false_starts():
    """Stored blocks whose bytes are themselves deflate data: the parts find
    real-looking dynamic headers inside them that decode fine for a while.
    The chain must notice they are not where the stream is, and redo."""
    inner = _compress(_data('walk', 3_000_000, seed=8), 6)
    redone = 0
    for level in (0, 1):
        outer = _compress(inner, level)
        for mp in (1 << 11, 1 << 14):
            _check(outer, inner, nthreads=8, min_part=mp, parts_per_thread=4)
            info = []
            out = np.empty(len(inner), np.uint8)
            src = np.frombuffer(outer, np.uint8)
            _pinflate.inflate_parallel(src, 0, len(outer), out, 0, len(inner), nthreads=8,
                                       min_part=mp, parts_per_thread=4, info=info)
            redone += int(info[-1][0, _pinflate.SI_NREDO])
    assert redone > 0                               # the redo path did run


def test_fixed_only():
    """No dynamic block anywhere (Z_FIXED): no part finds a start, the
    stream is decoded serially by the chain."""
    data = _data('walk', 1_000_000, seed=6)
    comp = _compress(data, 6, zlib.Z_FIXED)
    _check(comp, data, nthreads=8, min_part=1 << 12)


def test_default_settings_big():
    data = _data('mixed', 8_000_000, seed=11)
    comp = _compress(data, 6)
    for nt in (1, 2, 3, 8, 24):
        _check(comp, data, nthreads=nt)


@pytest.mark.parametrize('seed', range(6))
def test_partial(seed):
    rng = np.random.default_rng(seed)
    data = _data('mixed', 1_500_000, seed=seed)
    comp = _compress(data, 6)
    for _ in range(6):
        stop = int(rng.integers(0, len(data) + 1))
        _check(comp, data, ostop=stop, **TINY)
    _check(comp, data, ostop=len(data) // 20, nthreads=8)


def test_output_room():
    """oend beyond the stream's output: the end is reported, nothing past it written."""
    data = _data('walk', 400_000, seed=2)
    comp = _compress(data, 6)
    r = _check(comp, data, room=1000, **TINY)
    assert r == GUARD + len(data)


def test_output_too_small():
    data = _data('walk', 400_000, seed=2)
    comp = _compress(data, 6)
    src = np.frombuffer(comp, np.uint8)
    for short in (1, 100, 50_000, 300_000):
        n = len(data) - short
        out = np.full(n + 2 * GUARD, 0xA5, np.uint8)
        r = _pinflate.inflate_parallel(src, 0, len(comp), out, GUARD, GUARD + n, **TINY)
        assert r == _inflate.ERR_SPACE
        assert not np.any(out[:GUARD] != 0xA5) and not np.any(out[GUARD + n:] != 0xA5)


def test_offset_in_file():
    """A stream in the middle of a bigger buffer (as a chunk in a file)."""
    data = _data('mixed', 700_000, seed=4)
    comp = _compress(data, 6)
    buf = np.frombuffer(b'\x00' * 1000 + comp + b'\xff' * 1000, np.uint8)
    out = np.zeros(len(data), np.uint8)
    r = _pinflate.inflate_parallel(buf, 1000, 1000 + len(comp), out, 0, len(data), **TINY)
    assert r == len(data) and out.tobytes() == data


def test_many_streams_two_outputs_and_extra():
    rng = np.random.default_rng(9)
    datas = [_data(k, int(rng.integers(1, 900_000)), seed=i) for i, k in enumerate(KINDS * 2)]
    comps = [_compress(d, int(rng.integers(0, 10))) for d in datas]
    src = np.frombuffer(b''.join(comps), np.uint8)
    starts = np.cumsum([0] + [len(c) for c in comps])[:-1]
    ends = starts + [len(c) for c in comps]
    which = np.arange(len(datas)) % 2
    o0 = np.zeros(len(datas), np.int64)
    for k in (0, 1):
        idx = np.nonzero(which == k)[0]
        o0[idx] = np.cumsum([0] + [len(datas[i]) for i in idx])[:-1]
    oend = o0 + [len(d) for d in datas]
    out0 = np.zeros(sum(len(datas[i]) for i in range(len(datas)) if which[i] == 0), np.uint8)
    out1 = np.zeros(sum(len(datas[i]) for i in range(len(datas)) if which[i] == 1), np.uint8)
    ran = []
    r = _pinflate.inflate_many(src, starts, ends, out0, o0, oend, None, 8, dst2=out1, which=which,
                               extra=lambda: ran.append(1), min_part=1 << 12)
    assert np.array_equal(r, oend)
    assert ran
    for i, d in enumerate(datas):
        o = out0 if which[i] == 0 else out1
        assert o[o0[i]:oend[i]].tobytes() == d


def test_threads_at_once():
    """Calls from several threads at once share the pool and the scratch."""
    datas = [_data(k, 700_000, seed=i) for i, k in enumerate(KINDS)]
    comps = [_compress(d, 6) for d in datas]
    errors = []

    def run(i):
        try:
            for _ in range(3):
                _check(comps[i], datas[i], **TINY)
        except Exception as e:                      # pragma: no cover
            errors.append(e)
    th = [threading.Thread(target=run, args=(i,)) for i in range(len(datas))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert not errors, errors


# ---------------------------------------------------------------- fuzz

def _fuzz_one(comp, rng, kw):
    b = bytearray(comp)
    how = int(rng.integers(0, 5))
    if how == 0:                                    # bit flips
        for _ in range(int(rng.integers(1, 4))):
            i = int(rng.integers(0, len(b)))
            b[i] ^= 1 << int(rng.integers(0, 8))
    elif how == 1:                                  # bytes overwritten
        for _ in range(int(rng.integers(1, 6))):
            b[int(rng.integers(0, len(b)))] = int(rng.integers(0, 256))
    elif how == 2:                                  # truncated
        b = b[:int(rng.integers(0, len(b)))]
    elif how == 3:                                  # a stretch of noise
        i = int(rng.integers(0, len(b)))
        k = int(rng.integers(1, 200))
        b[i:i + k] = rng.integers(0, 256, k, dtype=np.uint8).tobytes()
    else:                                           # a stretch moved elsewhere
        i, j = sorted(int(x) for x in rng.integers(0, len(b), 2))
        k = int(rng.integers(1, 5000))
        b[i:i + k] = bytes(b[j:j + k])
    _check(bytes(b), None, **kw)


@pytest.mark.parametrize('seed', range(8))
def test_fuzz(seed):
    rng = np.random.default_rng(100 + seed)
    kinds = KINDS
    for it in range(60):
        data = _data(kinds[it % len(kinds)], int(rng.integers(1000, 300_000)), seed=it)
        comp = _compress(data, int(rng.integers(0, 10)),
                         [zlib.Z_DEFAULT_STRATEGY, zlib.Z_RLE, zlib.Z_HUFFMAN_ONLY,
                          zlib.Z_FIXED][it % 4])
        _fuzz_one(comp, rng, dict(nthreads=8, min_part=int(rng.choice([1 << 10, 1 << 12, 1 << 15])),
                                  parts_per_thread=4))


def test_garbage():
    rng = np.random.default_rng(5)
    for _ in range(200):
        n = int(rng.integers(2, 200_000))
        g = bytearray(rng.integers(0, 256, n, dtype=np.uint8).tobytes())
        g[0], g[1] = 0x78, 0x9C                     # a valid zlib header, then noise
        _check(bytes(g), None, **TINY)
