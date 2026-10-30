"""The HDF5-filter edge cases: small HDF5 files in tests/data/filters_hdf/
and the digests of HDF5's own decoding of each of their chunks, so that
tests/test_filters_hdf.py runs without HDF5.

    python tools/check/filters_hdf_make.py files     # the .h5 files (dev env: szip, hdf5plugin)
    python tools/check/filters_hdf_make.py digests   # HDF5's decoding -> expected.json (run
                                                     # in both envs: HDF5 1.14 cannot read
                                                     # full-precision scale-offset chunks,
                                                     # HDF5 2.0 has no szip; results merged,
                                                     # and must agree where both have one)
    python tools/check/filters_hdf_make.py streams   # libaec's SZ layer called directly ->
                                                     # szlib_cases.npz (any bpp / mask / pps)

Deterministic (fixed seeds). n-bit types (precision below the size)
need H5Pset_relax_file_integrity_checks, set through ctypes.
"""
import hashlib
import json
import os
import sys

import numpy as np

from filters_hdf_common import ROOT, h5open, load, ref_chunk, relaxed_fapl, set_fill_undefined

OUT = os.path.join(ROOT, 'tests', 'data', 'filters_hdf')
EXPECTED = os.path.join(OUT, 'expected.json')


def rng(seed):
    return np.random.default_rng(seed)


def smooth(shape, dtype, seed, scale=10.0):
    n = int(np.prod(shape))
    x = np.cumsum(rng(seed).standard_normal(n)) * scale
    dt = np.dtype(dtype)
    if dt.kind in 'iu':
        info = np.iinfo(dt)
        x = np.clip(x + (0 if dt.kind == 'i' else -x.min()), info.min, info.max)
    return x.reshape(shape).astype(dtype)


def patterns(shape, dtype, seed):
    """{name: array}: data meant to reach every kind of CCSDS block and
    every scale-offset range."""
    order = np.dtype(dtype).byteorder
    out = patterns_native(shape, np.dtype(dtype).newbyteorder('='), seed)
    return {k: v.astype(np.dtype(dtype)) for k, v in out.items()} if order == '>' else out


def patterns_native(shape, dtype, seed):
    dt = np.dtype(dtype)
    n = int(np.prod(shape))
    r = rng(seed)
    out = {'smooth': smooth(shape, dtype, seed)}
    if dt.kind in 'iu':
        info = np.iinfo(dt)
        lo, hi = max(info.min, -2 ** 62), min(info.max, 2 ** 62)
        out['random'] = r.integers(info.min, info.max, n, dtype=dt, endpoint=True).reshape(shape)
        out['const'] = np.full(shape, min(hi, 77), dt)
        z = np.zeros(n, dt)                                 # runs of zero blocks, a few spikes
        z[r.integers(0, n, max(1, n // 300))] = min(hi, 100)
        out['sparse'] = z.reshape(shape)
        st = (np.arange(n) // 7) % 3                        # tiny residuals: second extension
        out['stairs'] = (st + min(hi, 1000) // 2).astype(dt).reshape(shape)
        ext = np.where(r.random(n) < 0.5, np.array(info.min, dt), np.array(info.max, dt))  # the ends
        out['extremes'] = ext.reshape(shape)
        if dt.itemsize >= 2:
            w = min(13, dt.itemsize * 8 - 2)                # needs k up to 13
            out['k13'] = (r.integers(0, 2 ** w, n) + (0 if dt.kind == 'u' else -(2 ** (w - 1)))
                          ).astype(dt).reshape(shape)
    else:
        out['random'] = r.standard_normal(n).astype(dt).reshape(shape) * 1e3
        out['const'] = np.full(shape, 2.5, dt)
        z = np.zeros(n, dt)
        z[r.integers(0, n, max(1, n // 300))] = 1.5
        out['sparse'] = z.reshape(shape)
        out['bits'] = r.integers(0, 2 ** (8 * dt.itemsize - 1), n, dtype='u%d' % dt.itemsize
                                 ).view(dt).reshape(shape)  # any bit pattern (NaNs ...)
    return out


# ------------------------------------------------------------------ files

def make_szip(h5py):
    path = os.path.join(OUT, 'szip.h5')
    with h5py.File(path, 'w', libver='latest') as f:
        k = 0
        for dt in ('u1', 'i1', 'u2', 'i2', '>i2', 'u4', 'i4', '>u4', 'f4', '>f4', 'u8', 'i8', 'f8', '>f8'):
            for opt in ('nn', 'ec'):
                for ppb in (8, 16, 32):
                    for pname, data in patterns((8, 50), dt, 100 + k).items():
                        k += 1
                        if (pname != 'smooth' and ppb == 16) or (pname == 'bits' and opt == 'ec'):
                            continue
                        if dt not in ('u1', 'i2', '>u4', 'f4', 'f8') and pname not in ('smooth', 'random'):
                            continue
                        name = '%s/%s/%d/%s' % (dt.replace('>', 'be_'), opt, ppb, pname)
                        f.create_dataset(name, data=data, chunks=(4, 50),   # pps 50: padded scanlines
                                         compression='szip', compression_opts=(opt, ppb))
        # scanline shapes: pps a multiple of J, below J (the chunk's points), above 4096
        g = f.create_group('shapes')
        for dt in ('u1', 'i2', 'f4', 'f8'):
            for chunks, shape in (((16, 64), (20, 70)), ((3, 5), (7, 11)), ((40, 5), (40, 5)),
                                  ((1, 5000), (1, 5000)), ((1, 4096), (1, 4096)), ((1, 77), (3, 77)),
                                  ((5,), (23,))):
                for opt, ppb in (('nn', 8), ('ec', 32), ('nn', 32)):
                    if int(np.prod(chunks)) < ppb:
                        continue
                    for pname in ('smooth', 'sparse'):
                        data = patterns(shape, dt, 7 + k)[pname]
                        k += 1
                        g.create_dataset('%s_%s_%s_%d_%s' % (dt, 'x'.join(map(str, chunks)), opt, ppb, pname),
                                         data=data, chunks=chunks, compression='szip',
                                         compression_opts=(opt, ppb))
        # constant scanlines of 4096: zero-block runs over 64-block segments (ROS)
        for dt in ('u1', 'i2', 'u4'):
            for opt in ('nn', 'ec'):
                data = np.zeros((4, 8192), dt)
                data[1, 3000:5000] = 9
                data[2] = np.arange(8192) // 1000
                data[3, ::997] = 1
                f.create_dataset('runs/%s_%s' % (dt, opt), data=data, chunks=(2, 8192),
                                 compression='szip', compression_opts=(opt, 32))
        # odd pixels per block (even, below 8: libaec decodes them too)
        for ppb in (2, 4, 6, 10, 30):
            f.create_dataset('ppb/%d' % ppb, data=smooth((20, 33), 'i2', ppb), chunks=(10, 33),
                             compression='szip', compression_opts=('nn', ppb))
    print('  szip.h5 %.0f kB' % (os.path.getsize(path) / 1e3))


def nbit_type(h5py, base, prec, off, order='<'):
    t = base.copy()
    if order == '>':
        t.set_order(h5py.h5t.ORDER_BE)
    t.set_precision(prec)
    t.set_offset(off)
    return t


def lowlevel(h5py, f, name, ftype, shape, chunks, filters, data, mtype=None, fill=None):
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk(chunks)
    for fid, opts in filters:
        if fid == 'nbit':
            dcpl.set_filter(h5py.h5z.FILTER_NBIT)
        elif fid == 'szip':
            dcpl.set_szip(*opts)
        elif fid == 'scaleoffset':
            dcpl.set_scaleoffset(*opts)
        elif fid == 'shuffle':
            dcpl.set_shuffle()
        elif fid == 'fletcher32':
            dcpl.set_fletcher32()
        elif fid == 'deflate':
            dcpl.set_deflate(opts)
    if fill == 'undefined':
        set_fill_undefined(dcpl)
    elif fill is not None:
        dcpl.set_fill_value(np.asarray(fill, data.dtype))
    parts = name.strip('/').split('/')
    g = f
    for p in parts[:-1]:
        g = g.require_group(p)
    did = h5py.h5d.create(g.id, parts[-1].encode(), ftype, h5py.h5s.create_simple(shape), dcpl)
    did.write(h5py.h5s.ALL, h5py.h5s.ALL, np.ascontiguousarray(data), mtype=mtype)
    did.close()


def make_szip_nbit(h5py):
    """szip over types of reduced precision: bits per pixel below the size
    (HDF5 passes the precision when the offset is 0, else the size; above
    24 it rounds up to 32 / 64)."""
    path = os.path.join(OUT, 'szip_nbit.h5')
    fid = h5py.h5f.create(os.fsencode(path), h5py.h5f.ACC_TRUNC, fapl=relaxed_fapl(latest=True))
    f = h5py.File(fid)
    k = 0
    for base, prec, off, order in ((h5py.h5t.STD_U8LE, 5, 0, '<'), (h5py.h5t.STD_U16LE, 12, 0, '<'),
                                   (h5py.h5t.STD_U16LE, 12, 0, '>'), (h5py.h5t.STD_U16LE, 9, 0, '<'),
                                   (h5py.h5t.STD_U32LE, 17, 0, '<'), (h5py.h5t.STD_U32LE, 24, 0, '<'),
                                   (h5py.h5t.STD_U32LE, 24, 0, '>'), (h5py.h5t.STD_U32LE, 20, 4, '<'),
                                   (h5py.h5t.STD_U32LE, 28, 0, '<'), (h5py.h5t.STD_U64LE, 40, 0, '<'),
                                   (h5py.h5t.STD_U16LE, 1, 0, '<')):
        t = nbit_type(h5py, base, prec, off, order)
        size = t.get_size()
        for opt in (h5py.h5z.SZIP_NN_OPTION_MASK, h5py.h5z.SZIP_EC_OPTION_MASK):
            for pname in ('smooth', 'random'):
                k += 1
                x = patterns((20, 30), 'u%d' % size, 300 + k)[pname].astype('u%d' % size)
                x = x & np.array((1 << min(prec, 63)) - 1, x.dtype)
                lowlevel(h5py, f, 'p%d_o%d_%s%d/%s/%s' % (prec, off, 'be' if order == '>' else 'le', size,
                                                          'nn' if opt == 32 else 'ec', pname),
                         t, (20, 30), (10, 30), [('szip', (opt, 8))], x,
                         mtype=h5py.h5t.py_create(x.dtype))
    f.close()
    print('  szip_nbit.h5 %.0f kB' % (os.path.getsize(path) / 1e3))


def make_scaleoffset(h5py):
    path = os.path.join(OUT, 'scaleoffset.h5')
    f = h5py.File(path, 'w', libver='latest')
    k = 0
    SO_INT, SO_FLOAT_DSCALE = h5py.h5z.SO_INT, h5py.h5z.SO_FLOAT_DSCALE
    for dt in ('u1', 'i1', 'u2', 'i2', '>i2', 'u4', 'i4', '>u4', '>i4', 'u8', 'i8', '>i8'):
        ndt = np.dtype(dt).newbyteorder('=')
        info = np.iinfo(ndt)
        for pname, data in patterns((8, 37), ndt, 500 + k).items():
            if pname in ('stairs', 'k13'):
                continue
            for minbits in (0, 3, info.bits - 1):
                for fill in (None, 'undefined', 'value'):
                    k += 1
                    if minbits == 3 and pname not in ('smooth', 'sparse'):
                        continue
                    if minbits == info.bits - 1 and pname not in ('smooth', 'extremes'):
                        continue
                    if minbits and fill is not None and pname != 'smooth':
                        continue
                    if fill == 'value':
                        fv = info.min + 5 if pname != 'sparse' else 0
                        if pname == 'const':
                            fv = 77
                    else:
                        fv = fill
                    x = data.copy()
                    if fill == 'value' and pname == 'smooth':
                        x.flat[::17] = fv                   # fill values among the data
                    name = '%s/%s/m%d/%s' % (dt.replace('>', 'be_'), pname, minbits,
                                              'default' if fill is None else fill)
                    t = h5py.h5t.py_create(np.dtype(dt))
                    lowlevel(h5py, f, name, t, x.shape, (5, 37), [('scaleoffset', (SO_INT, minbits))],
                             x, mtype=h5py.h5t.py_create(ndt), fill=fv)
    for dt in ('f4', '>f4', 'f8', '>f8'):
        ndt = np.dtype(dt).newbyteorder('=')
        for pname, data in patterns((8, 37), ndt, 700 + k).items():
            if pname == 'bits':
                continue
            for D in (0, 1, 2, 3, 6, 9):
                for fill in (None, 'undefined', 'value'):
                    k += 1
                    if D in (0, 6, 9) and pname != "smooth":
                        continue
                    x = data.copy()
                    fv = fill
                    if fill == 'value':
                        fv = -999.0
                        x.flat[::13] = fv
                    if D == 9:
                        x = x / 1e4
                    name = '%s/%s/d%d/%s' % (dt.replace('>', 'be_'), pname, D,
                                              'default' if fill is None else fill)
                    t = h5py.h5t.py_create(np.dtype(dt))
                    lowlevel(h5py, f, name, t, x.shape, (5, 37), [('scaleoffset', (SO_FLOAT_DSCALE, D))],
                             x, mtype=h5py.h5t.py_create(ndt), fill=fv)
    # with deflate after it, and edge chunks
    f.create_dataset('pipeline/i4', data=smooth((30, 41), 'i4', 9), chunks=(16, 16), scaleoffset=0,
                     compression='gzip', shuffle=True)
    f.create_dataset('pipeline/f8', data=smooth((30, 41), 'f8', 9), chunks=(16, 16), scaleoffset=4,
                     compression="gzip")
    f.close()
    print('  scaleoffset.h5 %.0f kB' % (os.path.getsize(path) / 1e3))


def make_nbit(h5py):
    path = os.path.join(OUT, 'nbit.h5')
    fid = h5py.h5f.create(os.fsencode(path), h5py.h5f.ACC_TRUNC, fapl=relaxed_fapl(latest=True))
    f = h5py.File(fid)
    T = h5py.h5t
    k = 0
    ints = [(T.STD_I8LE, 3, 2), (T.STD_U8LE, 8, 0), (T.STD_U8LE, 1, 7), (T.STD_I16LE, 12, 2),
            (T.STD_U16LE, 9, 7), (T.STD_I32LE, 17, 5), (T.STD_U32LE, 31, 1), (T.STD_U32LE, 7, 20),
            (T.STD_I64LE, 40, 3), (T.STD_U64LE, 60, 2), (T.STD_I64LE, 63, 0), (T.STD_U64LE, 57, 7)]
    for base, prec, off in ints:
        for order in ('<', '>'):
            t = nbit_type(h5py, base, prec, off, order)
            size = t.get_size()
            signed = base.get_sign() == T.SGN_2
            k += 1
            lim = 1 << (prec - 1 if signed else prec)
            x = rng(k).integers(-lim if signed else 0, lim, (25, 30)).astype('i8')
            nd = np.dtype('%s%d' % ('i' if signed else 'u', size))
            lowlevel(h5py, f, 'int/%s%d_p%d_o%d_%s' % ('i' if signed else 'u', size, prec, off,
                                                       'be' if order == '>' else 'le'),
                     t, (25, 30), (10, 10), [('nbit', None)], x.astype(nd), mtype=h5py.h5t.py_create(nd))
    # floats with fewer bits: (sign pos, exp pos, exp size, mant pos, mant size), precision, offset, bias
    for size, fields, prec, off, ebias in ((4, (26, 20, 6, 0, 20), 27, 0, 31), (4, (30, 23, 7, 3, 20), 28, 3, 63),
                                           (8, (54, 44, 10, 2, 42), 53, 2, 511), (4, (31, 23, 8, 0, 23), 32, 0, 127)):
        for order in ('<', '>'):
            t = (T.IEEE_F32LE if size == 4 else T.IEEE_F64LE).copy()
            if order == '>':
                t.set_order(T.ORDER_BE)
            t.set_fields(*fields)
            t.set_offset(off)
            t.set_precision(prec)
            t.set_ebias(ebias)
            k += 1
            x = (smooth((25, 30), 'f8', k) / 100).astype('f%d' % size)
            lowlevel(h5py, f, 'float/f%d_p%d_o%d_%s' % (size, prec, off, 'be' if order == '>' else 'le'),
                     t, (25, 30), (10, 10), [('nbit', None)], x, mtype=h5py.h5t.py_create(x.dtype))
    # compound: reduced ints and floats, a full one, a string (no-op), an array, a nested compound
    i12 = nbit_type(h5py, T.STD_I16LE, 12, 2)
    u5be = nbit_type(h5py, T.STD_U8LE, 5, 1, '>')
    f20 = T.IEEE_F32LE.copy()
    f20.set_fields(26, 20, 6, 0, 20)
    f20.set_offset(0)
    f20.set_precision(27)
    f20.set_ebias(31)
    s3 = T.C_S1.copy()
    s3.set_size(3)
    arr = T.array_create(nbit_type(h5py, T.STD_U32LE, 19, 6), (3,))
    inner = T.create(T.COMPOUND, 6)
    inner.insert(b'q', 0, nbit_type(h5py, T.STD_I32LE, 22, 3, '>'))
    inner.insert(b'r', 4, nbit_type(h5py, T.STD_U16LE, 10, 0))
    ct = T.create(T.COMPOUND, 40)
    ct.insert(b'a', 0, i12)
    ct.insert(b'b', 2, u5be)
    ct.insert(b'c', 4, f20)
    ct.insert(b'd', 8, T.STD_I32LE)                         # full precision
    ct.insert(b's', 12, s3)                                 # no-op bytes
    ct.insert(b'e', 16, arr)                                # (bytes 15 and 28-29: padding)
    ct.insert(b'n', 30, inner)
    ct.insert(b'z', 36, nbit_type(h5py, T.STD_U32LE, 3, 29))
    nd = np.dtype({'names': list('abcdsenz'),
                   'formats': ['<i2', 'u1', '<f4', '<i4', 'S3', ('<u4', (3,)),
                               np.dtype([('q', '>i4'), ('r', '<u2')]), '<u4'],
                   'offsets': [0, 2, 4, 8, 12, 16, 30, 36], 'itemsize': 40})
    x = np.zeros((20, 15), nd)
    r = rng(77)
    x['a'] = r.integers(-2000, 2000, x.shape)
    x['b'] = r.integers(0, 32, x.shape)
    x['c'] = r.standard_normal(x.shape) * 10
    x['d'] = r.integers(-2 ** 31, 2 ** 31, x.shape)
    x['s'] = [[b'ab%d' % (i % 10) for i in range(15)] for _ in range(20)]
    x['e'] = r.integers(0, 2 ** 19, x.shape + (3,))
    x['n']['q'] = r.integers(-2 ** 21, 2 ** 21, x.shape)
    x['n']['r'] = r.integers(0, 1024, x.shape)
    x['z'] = r.integers(0, 8, x.shape)
    mt = h5py.h5t.py_create(nd)
    lowlevel(h5py, f, 'compound/mixed', ct, x.shape, (7, 6), [('nbit', None)], x, mtype=mt)
    # an array of compounds, an array of arrays
    at = T.array_create(inner, (2,))
    nd2 = np.dtype((np.dtype([('q', '>i4'), ('r', '<u2')]), (2,)))
    y = np.zeros((12, 9), nd2)                              # (shape (12, 9, 2))
    y['q'] = r.integers(-2 ** 21, 2 ** 21, y.shape)
    y['r'] = r.integers(0, 1024, y.shape)
    lowlevel(h5py, f, 'array/of_compound', at, (12, 9), (5, 4), [('nbit', None)], y,
             mtype=h5py.h5t.py_create(nd2))
    aa = T.array_create(T.array_create(nbit_type(h5py, T.STD_I16LE, 7, 4, '>'), (2,)), (3,))
    z = r.integers(-64, 64, (10, 10, 3, 2)).astype('>i2')
    lowlevel(h5py, f, 'array/of_array', aa, (10, 10), (4, 4), [('nbit', None)], z,
             mtype=T.array_create(T.array_create(T.STD_I16BE, (2,)), (3,)))
    # nothing to reduce: full-precision types (cd[1] = 1, stored as is)
    for dt in ('<i4', '>f8'):
        x = smooth((10, 10), dt, 5)
        lowlevel(h5py, f, 'full/%s' % dt.replace('>', 'be_').replace('<', 'le_'), h5py.h5t.py_create(np.dtype(dt)),
                 x.shape, (4, 4), [('nbit', None)], x, mtype=h5py.h5t.py_create(x.dtype))
    # n-bit then deflate
    t = nbit_type(h5py, T.STD_I32LE, 13, 0)
    x = rng(3).integers(-4000, 4000, (30, 30)).astype('i4')
    lowlevel(h5py, f, 'pipeline/i4_deflate', t, x.shape, (16, 16), [('nbit', None), ('deflate', 6)], x,
             mtype=h5py.h5t.py_create(x.dtype))
    f.close()
    print('  nbit.h5 %.0f kB' % (os.path.getsize(path) / 1e3))


def make_fletcher(h5py):
    path = os.path.join(OUT, 'fletcher.h5')
    with h5py.File(path, 'w', libver='latest') as f:
        r = rng(11)
        for n in (1, 2, 3, 7, 8, 719, 720, 721, 722, 1439, 1441, 4099, 65537):
            x = r.integers(0, 256, n, dtype='u1')
            if n == 65537:
                x[:] = 255                                  # the sums' largest values
            f.create_dataset('u1_%d' % n, data=x, chunks=(n,), fletcher32=True)
        for dt in ('>i2', 'f8'):
            f.create_dataset('odd_%s' % dt.replace('>', 'be_'), data=smooth((13, 11), dt, 3), chunks=(5, 7),
                             fletcher32=True)
        f.create_dataset('shuffle_gzip', data=smooth((50, 61), 'f4', 4), chunks=(17, 13), shuffle=True,
                         compression="gzip")
        f.create_dataset('shuffle_s3', data=np.array([b'abc', b'xyz', b'q'] * 41, 'S3'), chunks=(19,),
                         shuffle=True, fletcher32=True)
        f.create_dataset('shuffle_c16', data=(np.arange(77) * (1 + 2j)).astype('c16'), chunks=(9,),
                         shuffle=True, fletcher32=True)
        f.create_dataset('shuffle_one', data=np.arange(5, dtype='i8'), chunks=(1,), shuffle=True,
                         fletcher32=True)
        f.create_dataset('bzip2_like_odd', data=smooth((1001,), 'i2', 8), chunks=(333,), fletcher32=True)
    print('  fletcher.h5 %.0f kB' % (os.path.getsize(path) / 1e3))


def make_bzip2(h5py):
    import hdf5plugin
    path = os.path.join(OUT, 'bzip2.h5')
    with h5py.File(path, 'w', libver='latest') as f:
        for bs in (1, 9):
            f.create_dataset('smooth_%d' % bs, data=smooth((60, 70), 'f4', bs), chunks=(30, 35),
                             **hdf5plugin.BZip2(blocksize=bs))
        f.create_dataset('random', data=rng(1).integers(0, 256, 30000, dtype='u1'), chunks=(9999,),
                         **hdf5plugin.BZip2())
        f.create_dataset('zeros', data=np.zeros(5000, 'i4'), chunks=(2000,), **hdf5plugin.BZip2())
        f.create_dataset('runs', data=np.repeat(np.arange(50, dtype='u1'), 300), chunks=(15000,),
                         **hdf5plugin.BZip2())
        f.create_dataset('shuffle', data=smooth((40, 40), 'i8', 2), chunks=(20, 20), shuffle=True,
                         **hdf5plugin.BZip2())
    print('  bzip2.h5 %.0f kB' % (os.path.getsize(path) / 1e3))


# ------------------------------------------------------------------ digests

def digests():
    import h5py
    try:
        import hdf5plugin                                   # noqa: F401
    except ImportError:
        pass
    have = json.load(open(EXPECTED)) if os.path.exists(EXPECTED) else {}
    files = sorted(n for n in os.listdir(OUT) if n.endswith('.h5'))
    files = [os.path.join(OUT, n) for n in files] + [os.path.join(ROOT, 'tests', 'data', 'filters.h5')]
    ver = h5py.version.hdf5_version
    nnew = nfail = 0
    for path in files:
        key = os.path.relpath(path, os.path.join(ROOT, 'tests', 'data')).replace('\\', '/')
        b, f, dsets = load(path)
        h = h5open(path)
        out = have.setdefault(key, {})
        for name, ds in dsets.items():
            ids = {x.id for x in ds.filters}
            if not ids or not ids <= {1, 2, 3, 4, 5, 6, 307}:
                continue
            rec = out.setdefault(name, {})
            for offs, addr, size, mask in ds.chunk_table():
                ck = ','.join(map(str, offs))
                try:
                    ref = ref_chunk(h[name], offs)
                except OSError:
                    nfail += 1
                    continue
                d = hashlib.sha256(ref.tobytes()).hexdigest()[:32]
                if ck in rec and rec[ck]['sha'] != d:
                    raise SystemExit('%s %s %s: HDF5 %s disagrees with HDF5 %s' % (key, name, ck, ver,
                                                                                  rec[ck]['hdf5']))
                if ck not in rec:
                    rec[ck] = {'sha': d, 'n': int(ref.size), 'hdf5': ver}
                    nnew += 1
        h.close()
    with open(EXPECTED, 'w') as fh:
        json.dump(have, fh, indent=0, sort_keys=True)
    print('HDF5 %s: %d new digests, %d chunks it cannot decode' % (ver, nnew, nfail))


# ------------------------------------------------------------------ libaec directly

def streams():
    """SZ streams made by libaec's SZ_BufftoBuffCompress for parameters
    HDF5 never passes, with libaec's decoding of each."""
    from filters_hdf_szlib import sz_compress, sz_decompress
    cases = []
    k = 0
    for bpp in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 16, 17, 20, 23, 24, 25, 31, 32, 64):
        pixel = 1 if bpp <= 8 else (2 if bpp <= 16 else (4 if bpp <= 32 else 8))
        for mask in (128 | 32 | 8, 128 | 4 | 16, 32 | 16, 4 | 8, 1 | 2 | 32 | 8):
            for ppb, pps in ((8, 64), (16, 50), (32, 300), (8, 7), (64, 128), (10, 30)):
                k += 1
                n = 3 * pps + 5 if k % 3 else 2 * pps
                r = rng(k)
                if bpp in (32, 64):
                    v = r.integers(0, 2 ** 31, n // (bpp // 8) + 1).astype('u8') * (k % 7 + 1)
                    raw = v.astype('<u%d' % (bpp // 8)).tobytes()[:n - n % (bpp // 8)] if bpp == 32 \
                        else v.astype('<u8').tobytes()[:n - n % 8]
                else:
                    lim = 1 << bpp
                    kind = k % 4
                    if kind == 0:
                        v = (np.cumsum(r.integers(-3, 4, n)) % lim)
                    elif kind == 1:
                        v = r.integers(0, lim, n)
                    elif kind == 2:
                        v = np.zeros(n, 'i8') + (lim - 1) * (np.arange(n) // 100 % 2)
                    else:
                        v = np.minimum(np.abs(np.cumsum(r.integers(-1, 2, n))), lim - 1)
                    msb = mask & 16
                    raw = v.astype(('>' if msb else '<') + 'u%d' % pixel).tobytes()
                try:
                    st = sz_compress(raw, mask, ppb, bpp, pps)
                except ValueError:
                    continue
                dec = sz_decompress(st, len(raw), mask, ppb, bpp, pps)
                if dec is None:
                    continue
                cases.append((bpp, mask, ppb, pps, len(raw), st, dec))
    np.savez_compressed(os.path.join(OUT, 'szlib_cases.npz'),
                        params=np.array([c[:5] for c in cases], np.int64),
                        streams=np.array([np.frombuffer(c[5], np.uint8) for c in cases] + [None],
                                         dtype=object)[:-1],
                        decoded_sha=np.array([hashlib.sha256(c[6]).hexdigest()[:32] for c in cases]))
    print('  szlib_cases.npz: %d streams, %.0f kB' % (len(cases), os.path.getsize(
        os.path.join(OUT, 'szlib_cases.npz')) / 1e3))


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'files'
    os.makedirs(OUT, exist_ok=True)
    if what == 'files':
        import h5py
        make_szip(h5py)
        make_szip_nbit(h5py)
        make_scaleoffset(h5py)
        make_nbit(h5py)
        make_fletcher(h5py)
        make_bzip2(h5py)
    elif what == 'digests':
        digests()
    elif what == 'streams':
        streams()
