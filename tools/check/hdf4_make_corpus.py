"""Make the HDF4 test files (tests/data/hdf4/) with pyhdf, calling the HDF4
library directly (ctypes, the mfhdf DLL pyhdf itself loaded) for what
pyhdf does not offer: chunking (SDsetchunk), N-bit (SDsetnbitdataset) and
writing little-endian number types.

    python tools/check/hdf4_make_corpus.py [outdir] [--big DIR]

--big DIR also writes large files for timing there (not committed).
"""
import ctypes
import os
import sys

import numpy as np
from pyhdf.SD import SD, SDC

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_dll = os.path.join(sys.prefix, 'Library', 'bin', 'mfhdf.dll')
if not os.path.exists(_dll):
    _dll = 'libmfhdf.so'
lib = ctypes.CDLL(_dll)
lib.SDsetchunk.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32]
lib.SDwritedata.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

HDF_CHUNK, HDF_COMP, HDF_NBIT = 1, 3, 5
COMP = {'rle': 1, 'nbit': 2, 'skphuff': 3, 'deflate': 4, 'szip': 5}
LITEND = 0x4000


def setchunk(sds, chunks, comp=None, param=0, nbit=None):
    """SDsetchunk: chunks, optionally compressed (coder name, its parameter:
    deflate level / skipping size) or N-bit ((start_bit, bit_len, sign_ext, fill_one))."""
    buf = (ctypes.c_int32 * 128)()
    for i, c in enumerate(chunks):
        buf[i] = c
    flags = HDF_CHUNK
    if nbit is not None:
        buf[32:36] = list(nbit)
        flags = HDF_NBIT
    elif comp is not None:
        buf[32] = COMP[comp]
        buf[33] = 0
        buf[34] = param
        flags = HDF_COMP
    r = lib.SDsetchunk(ctypes.c_int32(sds._id), ctypes.addressof(buf), ctypes.c_int32(flags))
    if r != 0:
        raise RuntimeError('SDsetchunk failed')


def setnbit(sds, start_bit, bit_len, sign_ext, fill_one):
    r = lib.SDsetnbitdataset(ctypes.c_int32(sds._id), start_bit, bit_len, sign_ext, fill_one)
    if r < 0:
        raise RuntimeError('SDsetnbitdataset failed')


def write(sds, a, start=None):
    """SDwritedata of a (any number type, as stored in memory)."""
    a = np.ascontiguousarray(a)
    n = a.ndim
    st = (ctypes.c_int32 * n)(*(start or [0] * n))
    edge = (ctypes.c_int32 * n)(*a.shape)
    r = lib.SDwritedata(ctypes.c_int32(sds._id), st, None, edge, a.ctypes.data_as(ctypes.c_void_p))
    if r != 0:
        raise RuntimeError('SDwritedata failed')


TYPES = [('int8', SDC.INT8, 'i1'), ('uint8', SDC.UINT8, 'u1'), ('uchar8', SDC.UCHAR8, 'u1'),
         ('char8', SDC.CHAR8, 'S1'), ('int16', SDC.INT16, 'i2'), ('uint16', SDC.UINT16, 'u2'),
         ('int32', SDC.INT32, 'i4'), ('uint32', SDC.UINT32, 'u4'), ('float32', SDC.FLOAT32, 'f4'),
         ('float64', SDC.FLOAT64, 'f8')]


def values(code, shape, seed):
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    if code == 'S1':
        return np.frombuffer(bytes(rng.integers(32, 127, n, dtype=np.uint8)), 'S1').reshape(shape)
    dt = np.dtype(code)
    if dt.kind == 'f':
        a = rng.standard_normal(n).astype(dt) * 1000
        if n >= 4:
            a[:4] = [np.inf, -np.inf, -0.0, np.nan]
        return a.reshape(shape)
    info = np.iinfo(dt)
    a = rng.integers(info.min, info.max, n, dtype=dt, endpoint=True)
    a[:2] = [info.min, info.max]
    return a.reshape(shape)


def make_types(path):
    """Every number type (big- and little-endian), attributes of every type,
    named and unnamed dimensions, dimension scales and strings, data
    strings, calibration, range, fill values, unlimited dimensions, an
    empty variable, partly written variables."""
    f = SD(path, SDC.WRITE | SDC.CREATE | SDC.TRUNC)
    f.title = 'fastnc HDF4 types'
    f.attr('history').set(SDC.CHAR8, 'made by hdf4_make_corpus.py\0with a NUL')
    for k, (name, t, code) in enumerate(TYPES):
        if code == 'S1':
            continue
        a = f.attr('g_' + name)
        vals = values(code, (5,), 100 + k)
        a.set(t, vals.tolist())
        f.attr('g1_' + name).set(t, vals[:1].tolist())
    for k, (name, t, code) in enumerate(TYPES):
        shape = (6, 7) if k % 2 else (5, 3, 4)
        v = f.create(name, t, shape)
        a = values(code, shape, k)
        if code == 'S1':
            write(v, a.view('u1'))
            v.attr('a_' + name).set(t, 'abc')
        else:
            v[:] = a
            v.attr('a_' + name).set(t, values(code, (3,), 50 + k).tolist())
        v.long_name = 'variable of type %s' % name
        if k == 4:
            v.dim(0).setname('row')
            v.dim(1).setname('col')
        v.endaccess()
    for name, t, code in [('lfloat32', SDC.FLOAT32, '<f4'), ('lint16', SDC.INT16, '<i2'),
                          ('luint32', SDC.UINT32, '<u4'), ('lfloat64', SDC.FLOAT64, '<f8')]:
        v = f.create(name, t | LITEND, (4, 9))
        write(v, values(code[1:], (4, 9), len(name)).astype(code))
        v.endaccess()
    # dimension scales, strings, data strings, calibration, range, fill
    v = f.create('temperature', SDC.INT16, (8, 10))
    d0, d1 = v.dim(0), v.dim(1)
    d0.setname('lat')
    d1.setname('lon')
    d0.setscale(SDC.FLOAT32, list(np.linspace(-40, 40, 8)))
    d1.setscale(SDC.FLOAT64, list(np.linspace(0, 90, 10)))
    d0.setstrs('latitude', 'degrees_north', 'F6.2')
    d1.setstrs('longitude', 'degrees_east', '')
    v.setdatastrs('Temperature', 'K', 'I6', 'cartesian')
    v.setcal(0.01, 0.001, 273.15, 0.5, SDC.INT16)
    v.setrange(4000, -4000)
    v.setfillvalue(-9999)
    a = (np.arange(80, dtype='i2').reshape(8, 10) - 40) * 97
    v[0:6, :] = a[:6]                               # rows 6, 7 left as fill
    v.endaccess()
    # a second variable on the same named dimensions
    v = f.create('pressure', SDC.FLOAT32, (8, 10))
    v.dim(0).setname('lat')
    v.dim(1).setname('lon')
    v[:] = np.arange(80, dtype='f4').reshape(8, 10) * 1.5
    v.endaccess()
    # unlimited: two variables on one record dimension, different record counts
    for name, nrec in [('rec_a', 5), ('rec_b', 3)]:
        v = f.create(name, SDC.INT32, (SDC.UNLIMITED, 4))
        v.dim(0).setname('time')
        v.dim(1).setname('band')
        v[0:nrec] = np.arange(nrec * 4, dtype='i4').reshape(nrec, 4) + len(name)
        v.endaccess()
    v = f.create('rec_c', SDC.FLOAT64, (SDC.UNLIMITED,))
    for i in range(7):                               # appended record by record
        v[i:i + 1] = np.array([i * 0.5])
    v.endaccess()
    v = f.create('never_written', SDC.FLOAT32, (4, 5))
    v.endaccess()
    v = f.create('never_written_fill', SDC.UINT16, (6,))
    v.setfillvalue(7)
    v.endaccess()
    v = f.create('never_written_u8', SDC.UINT8, (3,))
    v.endaccess()
    v = f.create('partial_default_fill', SDC.FLOAT64, (4, 3))
    v[1:2, :] = np.array([[1., 2., 3.]])
    v.endaccess()
    v = f.create('chars', SDC.CHAR8, (3, 8))
    v.setfillvalue(ord('*'))
    write(v, np.frombuffer(b'abcdefgh', 'u1').reshape(1, 8))
    v.endaccess()
    v = f.create('one', SDC.FLOAT32, (1,))
    v[:] = np.array([42.0], 'f4')
    v.endaccess()
    f.end()


def make_comp(path):
    """Compression (deflate, RLE, skipping Huffman, N-bit) of whole arrays and
    of chunks; chunked arrays with edge chunks and missing chunks (HDF4
    cannot chunk an unlimited dimension)."""
    f = SD(path, SDC.WRITE | SDC.CREATE | SDC.TRUNC)
    smooth = (np.add.outer(np.arange(60), np.arange(70)) // 7).astype('i2')
    noisy = values('f4', (60, 70), 7)
    # whole-array compression
    for name, comp, arg, a in [('deflate1', SDC.COMP_DEFLATE, 1, smooth), ('deflate9', SDC.COMP_DEFLATE, 9, noisy),
                               ('rle_i2', SDC.COMP_RLE, 0, smooth), ('rle_f4', SDC.COMP_RLE, 0, noisy),
                               ('skphuff1', SDC.COMP_SKPHUFF, 1, smooth), ('skphuff4', SDC.COMP_SKPHUFF, 4, noisy),
                               ('skphuff2', SDC.COMP_SKPHUFF, 2, smooth.astype('u2'))]:
        t = {'i2': SDC.INT16, 'f4': SDC.FLOAT32, 'u2': SDC.UINT16}[a.dtype.str[1:]]
        v = f.create(name, t, a.shape)
        v.setcompress(comp, arg) if arg else v.setcompress(comp)
        v[:] = a
        v.endaccess()
    v = f.create('deflate_partial', SDC.FLOAT64, (20, 30))
    v.setcompress(SDC.COMP_DEFLATE, 6)
    v[0:5, :] = np.arange(150, dtype='f8').reshape(5, 30)
    v.endaccess()
    v = f.create('deflate_empty', SDC.INT32, (10, 10))
    v.setcompress(SDC.COMP_DEFLATE, 6)
    v.setfillvalue(-5)
    v.endaccess()
    # N-bit: (start_bit, bit_len, sign_ext, fill_one)
    for k, (t, code, params, a) in enumerate([
            (SDC.INT16, 'i2', (10, 6, 1, 0), (np.arange(-31, 32, dtype='i2') * 32)),
            (SDC.INT16, 'i2', (12, 9, 0, 1), (np.arange(0, 64, dtype='i2') * 16)),
            (SDC.UINT8, 'u1', (5, 4, 0, 0), np.arange(0, 64, dtype='u1')),
            (SDC.INT32, 'i4', (20, 13, 1, 1), (np.arange(-300, 300, 7, dtype='i4') * 256)),
            (SDC.INT32, 'i4', (31, 32, 0, 0), np.arange(-50, 50, dtype='i4') * 1234567)]):
        v = f.create('nbit%d' % k, t, a.shape)
        setnbit(v, *params)
        write(v, a)
        v.endaccess()
    # chunked
    big = (np.add.outer(np.arange(100), np.arange(90)) % 251).astype('u2')
    for name, chunks, comp, arg in [('chunk_none', (16, 32), None, 0), ('chunk_deflate', (16, 32), 'deflate', 6),
                                    ('chunk_rle', (10, 45), 'rle', 0), ('chunk_skphuff', (25, 30), 'skphuff', 2),
                                    ('chunk_whole', (100, 90), 'deflate', 1), ('chunk_rows', (1, 90), 'deflate', 4)]:
        v = f.create(name, SDC.UINT16, big.shape)
        setchunk(v, chunks, comp, arg)
        v[:] = big
        v.endaccess()
    v = f.create('chunk_nbit', SDC.INT16, (30, 40))
    setchunk(v, (8, 16), nbit=(9, 7, 1, 0))
    write(v, ((np.arange(1200, dtype='i2') % 128) - 64).reshape(30, 40) * 4)
    v.endaccess()
    v = f.create('chunk_missing', SDC.FLOAT32, (50, 60))
    v.setfillvalue(-1.5)
    setchunk(v, (20, 25), 'deflate', 5)
    v[0:20, 25:50] = np.ones((20, 25), 'f4') * 3
    v[40:50, 50:60] = np.ones((10, 10), 'f4') * 4
    v.endaccess()
    v = f.create('chunk_missing_nofill', SDC.INT32, (12, 12))
    setchunk(v, (5, 5))
    v[0:5, 0:5] = np.arange(25, dtype='i4').reshape(5, 5)
    v.endaccess()
    v = f.create('chunk_3d', SDC.FLOAT64, (7, 9, 11))
    setchunk(v, (3, 4, 5), 'deflate', 3)
    v[:] = values('f8', (7, 9, 11), 3)
    v.endaccess()
    v = f.create('chunk_empty', SDC.UINT8, (8, 8))
    setchunk(v, (4, 4), 'deflate', 1)
    v.endaccess()
    f.end()


def make_layout(outdir):
    """Contiguous data big enough for several pieces, appended records in
    linked blocks, data in an external file."""
    cwd = os.getcwd()
    os.chdir(outdir)
    try:
        f = SD('layout.hdf', SDC.WRITE | SDC.CREATE | SDC.TRUNC)
        a = (np.arange(1100 * 560, dtype='i4') * 7919 % 65536 - 32768).astype('i2').reshape(1100, 560)
        v = f.create('contig_big', SDC.INT16, a.shape)
        v[:] = a
        v.endaccess()
        v = f.create('linked', SDC.FLOAT32, (SDC.UNLIMITED, 50))
        for i in range(40):
            v[i:i + 1] = (np.arange(50, dtype='f4') + i * 100)[None]
        v.endaccess()
        v = f.create('external', SDC.INT32, (20, 30))
        v.setexternalfile('layout_ext.dat', 16)
        v[:] = np.arange(600, dtype='i4').reshape(20, 30) - 300
        v.endaccess()
        v = f.create('vector', SDC.FLOAT64, (70000,))
        v[:] = np.sin(np.arange(70000) * 0.001)
        v.endaccess()
        f.end()
    finally:
        os.chdir(cwd)


def make_many(path, n=300):
    """Many small variables, each with attributes (opening speed)."""
    f = SD(path, SDC.WRITE | SDC.CREATE | SDC.TRUNC)
    for i in range(n):
        v = f.create('var_%03d' % i, SDC.FLOAT32 if i % 3 else SDC.INT16, (4 + i % 5, 6))
        v[:] = (np.arange((4 + i % 5) * 6).reshape(4 + i % 5, 6) + i).astype('f4' if i % 3 else 'i2')
        v.units = 'u%d' % i
        v.scale_factor = 0.5 + i
        v.add_offset = float(i)
        v.dim(0).setname('d%d' % (i % 5))
        v.endaccess()
    f.end()


def make_big(outdir):
    """Large files for timing: contiguous, deflate-compressed whole (MODIS L2
    style) and deflate-chunked (MODIS L1B style), many bands."""
    rng = np.random.default_rng(0)
    base = (np.add.outer(np.arange(2030), np.arange(1354)) % 4096).astype('u2')
    for name, mode in [('big_contig.hdf', 'contig'), ('big_deflate.hdf', 'deflate'),
                       ('big_chunked.hdf', 'chunked')]:
        f = SD(os.path.join(outdir, name), SDC.WRITE | SDC.CREATE | SDC.TRUNC)
        for b in range(12):
            a = base + rng.integers(0, 64, base.shape, dtype=np.uint16)
            v = f.create('band_%02d' % b, SDC.UINT16, a.shape)
            if mode == 'deflate':
                v.setcompress(SDC.COMP_DEFLATE, 4)
            elif mode == 'chunked':
                setchunk(v, (203, 1354), 'deflate', 4)
            v[:] = a
            v.scale_factor = 0.01
            v.endaccess()
        f.end()


if __name__ == '__main__':
    args = sys.argv[1:]
    big = None
    if '--big' in args:
        i = args.index('--big')
        big = args[i + 1]
        del args[i:i + 2]
    out = args[0] if args else os.path.join(ROOT, 'tests', 'data', 'hdf4')
    os.makedirs(out, exist_ok=True)
    make_types(os.path.join(out, 'types.hdf'))
    make_comp(os.path.join(out, 'comp.hdf'))
    make_layout(out)
    make_many(os.path.join(out, 'many.hdf'))
    if big:
        os.makedirs(big, exist_ok=True)
        make_big(big)
    for fn in sorted(os.listdir(out)):
        print(fn, os.path.getsize(os.path.join(out, fn)))
