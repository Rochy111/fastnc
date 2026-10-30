"""Write the codec test corpus, tests/data/codecs/*.h5, and the digests of
every chunk as the reference filters (hdf5plugin through HDF5) decode it,
tests/data/codecs/expected.json -- so tests/test_codecs.py checks the
kernels byte for byte without h5py or hdf5plugin. tests/data/filters.h5's
chunks are digested too.

Edge cases: one-element and odd-sized chunks, edge chunks, incompressible,
constant and highly repetitive data, every blosc codec x shuffle x type
size, blosc2 codecs x filters in 1 to 3 dimensions, bitshuffle block sizes
and leftovers, zstd levels -5 to 22 with chunks of several blocks, frames
whose window is smaller than the chunk, long matches.

    python tools/check/codecs_make.py [codec ...]            (tests/data/codecs)
    python tools/check/codecs_make.py --full DIR [codec ...]  (bigger, kept out of the repo)

The committed corpus keeps chunks small and the data compressible; --full
writes the same datasets about eight times bigger (MB chunks, floats with
all their mantissa bits) for checking with tools/check/codecs_check.py.
"""
import hashlib
import json
import os
import sys

import h5py
import hdf5plugin
import numpy as np

from codecs_ref import ROOT, chunk_offsets, decode_with_plugin, filters

ARGS = sys.argv[1:]
FULL = '--full' in ARGS
if FULL:
    OUT = ARGS[ARGS.index('--full') + 1]
    ARGS = [a for a in ARGS if a not in ('--full', OUT)]
else:
    OUT = os.path.join(ROOT, 'tests', 'data', 'codecs')


def N(n):
    """A size: as given for the full corpus, an eighth of it for the
    committed one."""
    return n if FULL else max(1, n // 8)


# blosc / blosc2 chunks of several 256 KiB blocks
BIG, BIGT, BIGK = (800_256, '<f4', 'smooth') if FULL else (300_032, '<i2', 'repeat')


def data(kind, shape, dtype, seed=0):
    """Test values: smooth (compressible floats / ints), rough (floats with
    all their mantissa bits), noise (incompressible), zeros, ramp
    (periodic), repeat (a random block repeated: long matches)."""
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    n = int(np.prod(shape))
    nb = n * dt.itemsize
    if kind == 'noise':
        b = rng.integers(0, 256, nb, dtype=np.uint8)
    elif kind == 'zeros':
        b = np.zeros(nb, np.uint8)
    elif kind == 'ramp':
        b = (np.arange(nb) % 251).astype(np.uint8)
    elif kind == 'repeat':
        blk = rng.integers(0, 256, max(1, min(nb // 4, 100_000 if FULL else 3000)), dtype=np.uint8)
        b = np.resize(blk, nb)
    elif kind in ('smooth', 'rough'):
        x = np.cumsum(rng.standard_normal(n)) * 10
        if kind == 'smooth' and not FULL:
            x = np.round(x / 2.5) / 4                   # (smaller steps, fewer mantissa bits)
        if dt.kind == 'f':
            return x.astype(dt).reshape(shape)
        if dt.kind == 'c':
            return (x + 1j * x[::-1]).astype(dt).reshape(shape)
        if dt.kind in 'iu':
            info = np.iinfo(dt)
            return np.clip(x, info.min, info.max).astype(dt).reshape(shape)
        b = np.frombuffer(np.round(x).astype('<i2').tobytes(), np.uint8)[:nb]
        b = np.resize(b, nb)
    else:
        raise ValueError(kind)
    return np.frombuffer(b.tobytes(), dt).reshape(shape).copy()


def create(g, name, kind, shape, chunks, dtype, kw, seed=0):
    try:
        g.create_dataset(name, data=data(kind, shape, dtype, seed), chunks=chunks, **kw)
    except Exception as e:
        print('   skip %s/%s: %s' % (g.name, name, e))


COMPOUND = np.dtype([('a', '<f4'), ('b', '<i4'), ('c', '<u2'), ('d', 'u1'), ('e', 'u1')])  # 12 bytes
DTYPES = ['u1', '<i2', '<f4', '<f8', '<c16', 'S3', COMPOUND]


def tname(dt):
    dt = np.dtype(dt)
    return 'cmp%d' % dt.itemsize if dt.names else dt.str.strip('<|')


def make_lzf(f):
    g = f.create_group('lzf')
    lzf = dict(compression='lzf')
    for dt in DTYPES:
        for kind in ('smooth', 'noise', 'zeros', 'ramp'):
            create(g, '%s_%s' % (tname(dt), kind), kind, (45, 70), (16, 32), dt, lzf)
    create(g, 'one', 'smooth', (1,), (1,), 'u1', lzf)
    create(g, 'rough', 'rough', (N(300_000),), (N(300_000),), '<f4', lzf)
    create(g, 'big', 'smooth', (N(300_000),), (N(300_000),), '<f8', lzf)
    create(g, 'big_repeat', 'repeat', (N(500_000),), (N(500_000),), 'u1', lzf)


def make_lz4(f):
    g = f.create_group('lz4')
    for nb in (0, 1000, 4096):
        for kind in ('smooth', 'noise', 'zeros', 'ramp', 'repeat'):
            create(g, 'b%d_%s' % (nb, kind), kind, (45, 70), (21, 37), '<f4', hdf5plugin.LZ4(nbytes=nb))
    for dt in ('u1', '<f8', 'S3'):
        create(g, 'dt_%s' % tname(dt), 'smooth', (100,), (33,), dt, hdf5plugin.LZ4())
    create(g, 'one', 'smooth', (1,), (1,), 'u1', hdf5plugin.LZ4())
    create(g, 'tiny', 'noise', (7,), (7,), 'u1', hdf5plugin.LZ4(nbytes=3))
    create(g, 'rough', 'rough', (N(250_000),), (N(250_000),), '<f4', hdf5plugin.LZ4())
    create(g, 'big', 'smooth', (N(250_000),), (N(250_000),), '<f8', hdf5plugin.LZ4())
    create(g, 'big_blocks', 'repeat', (N(500_000),), (N(500_000),), 'u1', hdf5plugin.LZ4(nbytes=N(100_000)))


def make_bitshuffle(f):
    g = f.create_group('bitshuffle')
    for cname in ('none', 'lz4', 'zstd'):
        bs = lambda **kw: hdf5plugin.Bitshuffle(cname=cname, **kw)  # noqa: E731
        for dt in DTYPES:
            for kind in ('smooth', 'noise'):
                create(g, '%s_%s_%s' % (cname, tname(dt), kind), kind, (40, 61), (13, 61), dt, bs())
        for nel in (8, 64, 1024):
            create(g, '%s_block%d' % (cname, nel), 'smooth', (5003,), (5003,), '<f4', bs(nelems=nel))
        create(g, '%s_one' % cname, 'smooth', (1,), (1,), '<f8', bs())
        create(g, '%s_seven' % cname, 'smooth', (7,), (7,), '<i2', bs())
        create(g, '%s_zeros' % cname, 'zeros', (3000,), (3000,), '<f4', bs())
        create(g, '%s_big' % cname, 'smooth', (N(300_001),), (N(300_001),), '<f4', bs())
        create(g, '%s_rough' % cname, 'rough', (N(100_001),), (N(100_001),), '<f4', bs())
    for lev in (1, 9, 19, 22):
        create(g, 'zstd_level%d' % lev, 'smooth', (N(40_000),), (N(40_000),), '<f8',
               hdf5plugin.Bitshuffle(cname='zstd', clevel=lev))


def make_zstd(f):
    g = f.create_group('zstd')
    Z = lambda lev: hdf5plugin.Zstd(clevel=lev)  # noqa: E731
    for lev in list(range(-5, 0)) + list(range(1, 23)):
        create(g, 'smooth_l%d' % lev, 'smooth', (N(160_000),), (N(160_000) // 2,), '<i2', Z(lev))
        create(g, 'text_l%d' % lev, 'smooth', (N(240_000),), (N(240_000),), 'S3', Z(lev))
    for lev in (1, 3, 9, 15, 19, 22):
        create(g, 'repeat_l%d' % lev, 'repeat', (N(3_200_000),), (N(3_200_000),), 'u1', Z(lev))
        create(g, 'multi_l%d' % lev, 'smooth', (N(1_600_000),), (N(1_600_000),), '<i2', Z(lev))
        if FULL or lev in (1, 19):
            create(g, 'rough_l%d' % lev, 'rough', (N(400_000),), (N(400_000),), '<f4', Z(lev))
    for kind in ('noise', 'zeros', 'ramp'):
        for lev in (1, 19):
            create(g, '%s_l%d' % (kind, lev), kind, (N(400_000),), (N(400_000),), 'u1', Z(lev))
    for n in (1, 2, 5, 31, 300, 70_000):
        create(g, 'size%d' % n, 'smooth', (n,), (n,), 'u1', Z(3))
    # bigger than the window of the lower levels: frames with a window descriptor
    create(g, 'window_l1', 'repeat', (2_200_000,), (2_200_000,), 'u1', Z(1))
    create(g, 'window_l3', 'repeat', (2_600_000,), (2_600_000,), 'u1', Z(3))
    if FULL:
        create(g, 'window_l3_smooth', 'smooth', (8_000_000,), (8_000_000,), '<i2', Z(3))


BLOSC_CNAMES = ('blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd')


def make_blosc(f):
    g = f.create_group('blosc')
    for cname in BLOSC_CNAMES:
        B = lambda lev=5, sh=1: hdf5plugin.Blosc(cname=cname, clevel=lev, shuffle=sh)  # noqa: E731
        for sh in (0, 1, 2):
            for dt in DTYPES:
                create(g, '%s_s%d_%s' % (cname, sh, tname(dt)), 'smooth', (20, 60), (20, 60), dt, B(sh=sh))
            create(g, '%s_s%d_noise' % (cname, sh), 'noise', (1000,), (1000,), '<f4', B(sh=sh))
            if FULL or sh == 1:
                create(g, '%s_s%d_rough' % (cname, sh), 'rough', (N(100_000),), (N(100_000),), '<f4',
                       B(sh=sh))
        for lev in (0, 1, 9):
            create(g, '%s_c%d' % (cname, lev), 'smooth', (5001,), (5001,), '<f8', B(lev))
        create(g, '%s_big' % cname, BIGK, (BIG,), (BIG,), BIGT, B())
        create(g, '%s_big_bit' % cname, 'smooth', (N(800_007),), (N(800_007),), '<i2', B(sh=2))
        create(g, '%s_zeros' % cname, 'zeros', (100_000,), (100_000,), '<f4', B())
        create(g, '%s_one' % cname, 'smooth', (1,), (1,), '<f4', B())
        create(g, '%s_edge' % cname, 'smooth', (50, 50), (32, 32), '<i2', B(sh=2))


def make_blosc2(f):
    g = f.create_group('blosc2')
    for cname in ('blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd'):
        B = lambda lev=5, flt=1: hdf5plugin.Blosc2(cname=cname, clevel=lev, filters=flt)  # noqa: E731
        for flt in (0, 1, 2, 3, 4):
            for dt in ('u1', '<i2', '<f4', '<f8', 'S3'):
                create(g, '%s_f%d_%s_1d' % (cname, flt, tname(dt)), 'smooth', (3000,), (1301,), dt, B(flt=flt))
            create(g, '%s_f%d_2d' % (cname, flt), 'smooth', (70, 90), (33, 47), '<f4', B(flt=flt))
            if FULL or flt != 4:
                create(g, '%s_f%d_3d' % (cname, flt), 'smooth', (20, 30, 40), (7, 30, 23), '<f8', B(flt=flt))
            if FULL or flt == 1 or (cname == 'lz4' and flt != 4):
                create(g, '%s_f%d_big' % (cname, flt), BIGK, (BIG,), (BIG,), BIGT,
                       B(flt=flt))
                create(g, '%s_f%d_big2d' % (cname, flt), BIGK, (BIG // 512, 512), (BIG // 512, 512), BIGT,
                       B(flt=flt))
        create(g, '%s_noise_2d' % cname, 'noise', (64, 64), (64, 64), '<f4', B())
        if FULL or cname == 'lz4':
            create(g, '%s_rough_2d' % cname, 'rough', (N(1024), 512), (N(1024), 512), '<f4', B())
        create(g, '%s_zeros_2d' % cname, 'zeros', (64, 64), (64, 64), '<f4', B())
        create(g, '%s_zeros_1d' % cname, 'zeros', (4096,), (4096,), '<f8', B())
        create(g, '%s_ramp_1d' % cname, 'ramp', (4096,), (4096,), '<f8', B())
        for lev in (0, 1, 9):
            create(g, '%s_c%d_2d' % (cname, lev), 'smooth', (100, 100), (100, 100), '<i2', B(lev, 2))
        create(g, '%s_one' % cname, 'smooth', (1, 1), (1, 1), '<f4', B())


MINE = (32000, 32001, 32004, 32008, 32015, 32026)
MAKERS = {'lzf': make_lzf, 'lz4': make_lz4, 'bitshuffle': make_bitshuffle, 'zstd': make_zstd,
          'blosc': make_blosc, 'blosc2': make_blosc2}


def digest(path):
    """{dataset: {"i,j": [mask, nbytes, sha1]}} for every chunk."""
    out = {}
    with h5py.File(path, 'r') as f:
        items = []
        f.visititems(lambda n, o: items.append(o) if isinstance(o, h5py.Dataset) else None)
        for ds in items:
            fl = filters(ds)
            if not ds.chunks or len(fl) != 1 or fl[0][0] not in MINE:
                continue
            d = {}
            for off in chunk_offsets(ds):
                mask, raw = ds.id.read_direct_chunk(off)
                want = decode_with_plugin(ds, raw, mask)
                d[','.join(map(str, off))] = [int(mask), len(want), hashlib.sha1(want).hexdigest()]
            out[ds.name] = d
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    which = [a for a in ARGS if a in MAKERS] or list(MAKERS)
    expected_path = os.path.join(OUT, 'expected.json')
    expected = json.load(open(expected_path)) if os.path.exists(expected_path) else {}
    for name in which:
        path = os.path.join(OUT, name + '.h5')
        if os.path.exists(path):
            os.remove(path)
        with h5py.File(path, 'w', libver='latest') as f:
            MAKERS[name](f)
        expected[name + '.h5'] = digest(path)
        print('  %-20s %8.1f kB  %d datasets' % (name + '.h5', os.path.getsize(path) / 1e3,
                                                  len(expected[name + '.h5'])))
    if not FULL:
        fp = os.path.join(ROOT, 'tests', 'data', 'filters.h5')
        expected['../filters.h5'] = digest(fp)
    with open(expected_path, 'w', newline='\n') as fh:
        json.dump(expected, fh, indent=0, sort_keys=True)
        fh.write('\n')
