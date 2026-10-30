"""Single-thread speed of fastnc's HDF5 filter decoders against HDF5's own
(through h5py), on files made here in a temporary directory (64 MB of
values per dataset, 1 MB chunks unless said otherwise).

HDF5's time for a filter is h5py reading the whole filtered dataset less
reading the same values unfiltered with the same chunks (what the filter
adds); for szip also libaec's SZ_BufftoBuffDecompress called directly on
each chunk (the code HDF5 calls). fastnc's is the decode() calls over the
chunks (bytes already in memory, as from a mapped file). A/B alternate,
`--rounds` times; medians, with the spread (min-max) of the ratio.

    python tools/check/filters_hdf_bench.py [--rounds 5] [--only szip,...] [--dir D]
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np

from filters_hdf_common import ROOT, cd_array, load, relaxed_fapl

sys.path.insert(0, os.path.join(ROOT, 'tools', 'profile'))
from common import fast_thread                                    # noqa: E402

from fastnc import _bzip2, _fletcher32, _nbit, _scaleoffset, _shuffle, _szip  # noqa: E402

MOD = {2: _shuffle, 3: _fletcher32, 4: _szip, 5: _nbit, 6: _scaleoffset, 307: _bzip2}


def field(n, dtype, seed=0):
    r = np.random.default_rng(seed)
    side = int(np.sqrt(n))
    x = np.cumsum(np.cumsum(r.standard_normal((side, side)), 0), 1) / 30
    x += r.standard_normal((side, side)) * 0.05                     # some noise
    if np.dtype(dtype).kind in 'iu':                              # (the type's range, half used)
        x = (x - x.min()) / (x.max() - x.min()) * (np.iinfo(dtype).max // 2)
    return x.astype(dtype)


def make(d):
    import h5py
    import hdf5plugin
    n = 4096 * 4096                                                # 64 MB of float32
    specs = {}
    f4 = field(n, 'f4')
    i2 = field(n, 'i2')
    i4 = field(n, 'i4')
    u1 = field(n, 'u1')
    path = os.path.join(d, 'bench.h5')
    with h5py.File(path, 'w', libver='latest') as f:
        def ds(name, data, chunks, **kw):
            f.create_dataset(name, data=data, chunks=chunks, **kw)
            f.create_dataset(name + '_plain', data=data, chunks=chunks)
            specs[name] = data.nbytes
        ds('szip_nn_f4', f4, (64, 4096), compression='szip', compression_opts=('nn', 32))
        ds('szip_ec_i2', i2, (128, 4096), compression='szip', compression_opts=('ec', 32))
        ds('szip_nn_i2', i2, (128, 4096), compression='szip', compression_opts=('nn', 32))
        ds('szip_nn_u1', u1, (256, 4096), compression='szip', compression_opts=('nn', 16))
        ds('scaleoffset_i4', i4, (64, 4096), scaleoffset=0)
        ds('scaleoffset_f4', f4, (64, 4096), scaleoffset=3)
        ds('fletcher_f4', f4, (64, 4096), fletcher32=True)
        ds('shuffle_f4', f4, (64, 4096), shuffle=True)
        ds('shuffle_i2', i2, (128, 4096), shuffle=True)
        ds('bzip2_f4', f4[:1024], (64, 4096), **hdf5plugin.BZip2())
    # n-bit: a 17-bit integer type
    path2 = os.path.join(d, 'bench_nbit.h5')
    fid = h5py.h5f.create(os.fsencode(path2), h5py.h5f.ACC_TRUNC, fapl=relaxed_fapl(latest=True))
    for name, filt in (('nbit_i4p17', True), ('nbit_i4p17_plain', False)):
        t = h5py.h5t.STD_I32LE.copy()
        t.set_precision(17)
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((64, 4096))
        if filt:
            dcpl.set_filter(h5py.h5z.FILTER_NBIT)
        did = h5py.h5d.create(fid, name.encode(), t, h5py.h5s.create_simple(i4.shape), dcpl)
        did.write(h5py.h5s.ALL, h5py.h5s.ALL, np.ascontiguousarray(i4 & 0xFFFF))
        did.close()
    fid.close()
    specs['nbit_i4p17'] = i4.nbytes
    return path, path2, specs


def h5read(path, name, nbit=False):
    import h5py
    import hdf5plugin                                               # noqa: F401
    from filters_hdf_common import h5open
    with (h5open(path) if nbit else h5py.File(path, 'r')) as f:
        d = f[name]
        out = np.empty(d.shape, d.dtype)
        t = time.perf_counter()
        d.read_direct(out)
        return time.perf_counter() - t


def ours(b, ds, chunks, out, ws):
    filt = ds.filters[0]
    cd = cd_array(ds, filt)
    nb = int(cd[1])
    t = time.perf_counter()
    o = 0
    dec = MOD[filt.id].decode
    for (offs, addr, size, mask) in chunks:
        e = dec(b, addr, addr + size, out, o, o + nb, cd, ws)
        assert e == o + nb, e
        o = e
    return time.perf_counter() - t


def libaec(b, ds, chunks):
    from filters_hdf_szlib import lib, SZ_com_t
    import ctypes
    cd = ds.filters[0].cd
    prm = SZ_com_t(cd[0], cd[2], cd[1], cd[3])         # (mask, bpp, ppb, pps)
    nb = int(np.prod(ds.chunks)) * ds.type.size
    out = ctypes.create_string_buffer(nb)
    fn = lib().SZ_BufftoBuffDecompress
    srcs = [bytes(b[a + 4:a + s]) for (o, a, s, m) in chunks]
    t = time.perf_counter()
    for s in srcs:
        n = ctypes.c_size_t(nb)
        assert fn(out, ctypes.byref(n), s, len(s), ctypes.byref(prm)) == 0
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--only', default='')
    ap.add_argument('--dir', default=None)
    a = ap.parse_args()
    fast_thread()
    d = a.dir or tempfile.mkdtemp(prefix='fastnc_fbench_')
    os.makedirs(d, exist_ok=True)
    try:
        if not os.path.exists(os.path.join(d, 'bench.h5')):
            make(d)
        path, path2 = os.path.join(d, 'bench.h5'), os.path.join(d, 'bench_nbit.h5')
        print('%-16s %8s %9s %9s %9s %9s %s' % ('dataset', 'MB', 'fastnc', 'HDF5 filt', 'h5py all',
                                              'libaec', 'ratio HDF5 filter / fastnc (spread)'))
        for p, nbit in ((path, False), (path2, True)):
            b, f, dsets = load(p)
            for name, ds in dsets.items():
                name = name.lstrip('/')
                if name.endswith('_plain') or (a.only and not any(k in name for k in a.only.split(','))):
                    continue
                chunks = ds.chunk_table()
                nb = int(np.prod(ds.chunks)) * ds.type.size
                out = np.empty(nb * len(chunks), np.uint8)
                ws = np.zeros(max(1, MOD.get(ds.filters[0].id, _szip).workspace_size(nb)), np.uint8)
                ours(b, ds, chunks[:1], out, ws)                    # (compiled)
                T = {'ours': [], 'h5': [], 'plain': [], 'aec': []}
                for r in range(a.rounds):
                    T['ours'].append(ours(b, ds, chunks, out, ws))
                    T['h5'].append(h5read(p, name, nbit))
                    T['plain'].append(h5read(p, name + '_plain', nbit))
                    if ds.filters[0].id == 4:
                        T['aec'].append(libaec(b, ds, chunks))
                mb = out.size / 1e6
                med = {k: np.median(v) if v else np.nan for k, v in T.items()}
                filt = [h - pl for h, pl in zip(T['h5'], T['plain'])]
                ratios = [fi / o for fi, o in zip(filt, T['ours'])]
                print('%-16s %8.1f %6.0f MB/s %6.0f MB/s %6.0f MB/s %6.0f MB/s  %5.1fx (%.1f-%.1f)%s' % (
                    name, mb, mb / med['ours'], mb / np.median(filt), mb / med['h5'],
                    mb / med['aec'] if T['aec'] else np.nan,
                    np.median(ratios), min(ratios), max(ratios),
                    '  vs libaec %.1fx (%.1f-%.1f)' % (
                        np.median([x / o for x, o in zip(T['aec'], T['ours'])]),
                        min(x / o for x, o in zip(T['aec'], T['ours'])),
                        max(x / o for x, o in zip(T['aec'], T['ours']))) if T['aec'] else ''))
    finally:
        if not a.dir:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    main()
