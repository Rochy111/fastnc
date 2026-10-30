"""Single-thread decode speed of fastnc's codec kernels against the C
reference (the hdf5plugin filter function called directly through ctypes,
no HDF5 around it), chunk by chunk, on real data: WRF float32 fields and
VIIRS uint16 radiances recompressed with each filter (~1 MB chunks).

A and B alternate, `--rounds` times; reported: decoded MB/s of each (best
round) and the ratio fastnc / reference (median, with min-max over the
rounds). The reference includes its malloc of the output (it must
allocate; the kernels write in place), not the copy of its input.

    python tools/check/codecs_bench.py [--rounds 7] [--only zstd,lz4] [--h5py]

--h5py also times h5py reading each whole dataset single-threaded (HDF5's
chunk handling included) against the kernels over all its chunks into one
array.
"""
import argparse
import os
import sys
import time

import h5py
import hdf5plugin
import numpy as np

from codecs_ref import ROOT, fast_thread
from codecs_plugin import Decoder, _crt

import ctypes

REAL = os.environ.get('FASTNC_DATA', r'C:\Users\Roch\Desktop\Python\fastnc')
CACHE = os.environ.get('FASTNC_BENCH', os.path.join(os.environ.get('TEMP', '.'), 'fastnc_codecs_bench'))

FILTERS = [
    ('lzf', dict(compression='lzf')),
    ('lz4', hdf5plugin.LZ4()),
    ('bshuf-lz4', hdf5plugin.Bitshuffle(cname='lz4')),
    ('bshuf-zstd', hdf5plugin.Bitshuffle(cname='zstd', clevel=3)),
    ('zstd-1', hdf5plugin.Zstd(clevel=1)),
    ('zstd-3', hdf5plugin.Zstd(clevel=3)),
    ('zstd-9', hdf5plugin.Zstd(clevel=9)),
    ('zstd-19', hdf5plugin.Zstd(clevel=19)),
    ('blosc-lz4-shuf', hdf5plugin.Blosc(cname='lz4', shuffle=1)),
    ('blosc-blosclz-shuf', hdf5plugin.Blosc(cname='blosclz', shuffle=1)),
    ('blosc-zstd-bshuf', hdf5plugin.Blosc(cname='zstd', shuffle=2)),
    ('blosc-zlib-shuf', hdf5plugin.Blosc(cname='zlib', shuffle=1)),
    ('blosc-lz4-noshuf', hdf5plugin.Blosc(cname='lz4', shuffle=0)),
    ('blosc2-blosclz-shuf', hdf5plugin.Blosc2(cname='blosclz', filters=1)),
    ('blosc2-lz4-shuf', hdf5plugin.Blosc2(cname='lz4', filters=1)),
    ('blosc2-zstd-bshuf', hdf5plugin.Blosc2(cname='zstd', filters=2)),
]
MODULES = {32000: 'lzf', 32001: 'blosc', 32004: 'lz4', 32008: 'bitshuffle', 32015: 'zstd',
           32026: 'blosc2'}


def source_arrays():
    """name -> array: WRF 3-D float32 fields and a VIIRS uint16 radiance band."""
    out = {}
    with h5py.File(os.path.join(REAL, 'wrfout_d01_2025-10-26_06_20_00.nc'), 'r') as f:
        for k in ('T', 'QVAPOR', 'U', 'P', 'PH', 'QCLOUD'):
            out['wrf_' + k] = f[k][0]
    with h5py.File(os.path.join(REAL, 'VJ102IMG_NRT.A2026265.2006.021.nc'), 'r') as f:
        out['viirs_I05'] = f['observation_data/I05'][:2048].astype('<u2')
    return out


def chunks_for(a):
    if a.ndim == 3:
        return (max(1, (1 << 20) // (a.shape[1] * a.shape[2] * 4)), a.shape[1], a.shape[2])
    return (128, min(a.shape[1], 2048))


def build():
    os.makedirs(CACHE, exist_ok=True)
    arrays = None
    for fname, kw in FILTERS:
        path = os.path.join(CACHE, fname + '.h5')
        if os.path.exists(path):
            continue
        if arrays is None:
            arrays = source_arrays()
        with h5py.File(path + '.tmp', 'w') as f:
            for name, a in arrays.items():
                f.create_dataset(name, data=a, chunks=chunks_for(a), **kw)
        os.replace(path + '.tmp', path)


def load(fname):
    """[(dataset name, fid, cd, nbytes per chunk, [(raw bytes, mask)], h5 path)]"""
    path = os.path.join(CACHE, fname + '.h5')
    out = []
    with h5py.File(path, 'r') as f:
        for name in f:
            ds = f[name]
            pl = ds.id.get_create_plist()
            fid, flags, cd = pl.get_filter(0)[:3]
            nb = ds.dtype.itemsize * int(np.prod(ds.chunks))
            raws = []
            for i in range(ds.id.get_num_chunks()):
                off = ds.id.get_chunk_info(i).chunk_offset
                mask, raw = ds.id.read_direct_chunk(off)
                raws.append((raw, mask))
            out.append((name, fid, cd, nb, raws, path, ds.dtype.itemsize))
    return out


def time_ours(m, items, ws_cache):
    t = 0.0
    total = 0
    for name, fid, cd, nb, raws, path, es in items:
        cda = np.array([es, nb, len(cd)] + list(cd), np.int64)
        key = (m.__name__, nb)
        if key not in ws_cache:
            ws_cache[key] = (np.zeros(m.workspace_size(nb), np.uint8), np.zeros(nb, np.uint8))
        ws, dst = ws_cache[key]
        srcs = [np.frombuffer(r, np.uint8) for r, mask in raws if not mask & 1]
        t0 = time.perf_counter()
        for s in srcs:
            r = m.decode(s, 0, len(s), dst, 0, nb, cda, ws)
        t += time.perf_counter() - t0
        assert r == nb, (name, r)
        total += nb * len(srcs)
    return t, total


def time_ref(items, decs):
    t = 0.0
    total = 0
    for name, fid, cd, nb, raws, path, es in items:
        d = decs.setdefault((fid, tuple(cd)), Decoder(fid, cd))
        bufs = []
        for raw, mask in raws:
            if mask & 1:
                continue
            b = _crt.malloc(len(raw))
            ctypes.memmove(b, raw, len(raw))
            bufs.append((b, len(raw)))
        outs = []
        f = d.f
        rev = 0x0100
        t0 = time.perf_counter()
        for b, n in bufs:
            p = ctypes.c_void_p(b)
            size = ctypes.c_size_t(n)
            r = f(rev, d.ncd, d.cd, n, ctypes.byref(size), ctypes.byref(p))
            outs.append(p.value)
        t += time.perf_counter() - t0
        assert r == nb, (name, r)
        for o in outs:
            _crt.free(o)
        total += nb * len(bufs)
    return t, total


def time_h5py(path, names):
    t = 0.0
    with h5py.File(path, 'r', rdcc_nbytes=0) as f:
        for name in names:
            ds = f[name]
            out = np.empty(ds.shape, ds.dtype)
            t0 = time.perf_counter()
            ds.read_direct(out)
            t += time.perf_counter() - t0
    return t


def time_ours_whole(m, items, ws_cache):
    """All chunks of each dataset decoded into one array (cold output)."""
    t = 0.0
    for name, fid, cd, nb, raws, path, es in items:
        cda = np.array([es, nb, len(cd)] + list(cd), np.int64)
        ws = ws_cache.setdefault((m.__name__, nb), (np.zeros(m.workspace_size(nb), np.uint8), None))[0]
        srcs = [np.frombuffer(r, np.uint8) for r, mask in raws]
        out = np.empty(nb * len(srcs), np.uint8)
        t0 = time.perf_counter()
        o = 0
        for s in srcs:
            m.decode(s, 0, len(s), out, o, o + nb, cda, ws)
            o += nb
        t += time.perf_counter() - t0
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=7)
    ap.add_argument('--only', default='')
    ap.add_argument('--h5py', action='store_true')
    args = ap.parse_args()
    os.environ.setdefault('BLOSC_NTHREADS', '1')
    fast_thread()
    build()
    import importlib
    only = [s for s in args.only.split(',') if s]
    print('%-22s %8s %9s %9s   %-17s %s' % ('filter', 'ratio', 'fastnc', 'C ref', 'fastnc/C (min-max)',
                                          '| h5py whole-read ratio' if args.h5py else ''))
    for fname, kw in FILTERS:
        if only and not any(o in fname for o in only):
            continue
        items = load(fname)
        m = importlib.import_module('fastnc._' + MODULES[items[0][1]])
        ws_cache = {}
        decs = {}
        has_ref = items[0][1] != 32000
        time_ours(m, items, ws_cache)                   # compile, warm
        if has_ref:
            time_ref(items, decs)
        ratios, tos, trs = [], [], []
        for r in range(args.rounds):
            to, nbytes = time_ours(m, items, ws_cache)
            tr = time_ref(items, decs)[0] if has_ref else float('nan')
            tos.append(to)
            trs.append(tr)
            ratios.append(tr / to)
        ratios.sort()
        comp = sum(len(r) for it in items for r, mk in it[4] if not mk & 1)
        line = '%-22s %8.2f %8.0f %9.0f   %5.2f (%4.2f-%4.2f)   x%.1f' % (
            fname, ratios[len(ratios) // 2], nbytes / min(tos) / 1e6,
            nbytes / min(trs) / 1e6 if has_ref else float('nan'),
            ratios[len(ratios) // 2], ratios[0], ratios[-1], nbytes / comp)
        if args.h5py:
            names = [it[0] for it in items]
            rr = []
            for r in range(max(3, args.rounds // 2)):
                th = time_h5py(items[0][5], names)
                tw = time_ours_whole(m, items, ws_cache)
                rr.append(th / tw)
            rr.sort()
            line += '   | %5.2f (%4.2f-%4.2f)' % (rr[len(rr) // 2], rr[0], rr[-1])
        print(line, flush=True)


if __name__ == '__main__':
    main()
