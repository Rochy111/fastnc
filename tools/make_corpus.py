"""Write the test corpus: small HDF5 / netCDF files covering every layout,
chunk index, group and attribute storage, datatype and filter this
reader handles (tests/data). Needs h5py, netCDF4 and hdf5plugin; run it
in the dev environment. The files are deterministic (fixed seeds), and
small, so they live in git: tests read them without either library.

    python tools/make_corpus.py [outdir]
"""
import os
import sys

import h5py
import hdf5plugin
import netCDF4
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests', 'data')
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261029)


def field(shape, dtype='f4'):
    """Smooth-ish values with noise: compressible, not trivial."""
    n = int(np.prod(shape))
    x = np.cumsum(rng.standard_normal(n)).reshape(shape)
    return (x * 10).astype(dtype)


def h5(name, libver=None, **kw):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        os.remove(path)
    return h5py.File(path, 'w', libver=libver, **kw)


def made(name):
    print('  %-40s %8.1f kB' % (name, os.path.getsize(os.path.join(OUT, name)) / 1e3))


# ------------------------------------------------------------ chunk indexes
with h5('idx_v1btree.h5') as f:                     # earliest format: v1 B-tree
    f.create_dataset('deep', data=field((40, 50, 60)), chunks=(2, 5, 6), compression='gzip')
    f.create_dataset('edge', data=field((37, 41), 'i2'), chunks=(8, 8))
made('idx_v1btree.h5')

with h5('idx_latest.h5', libver='latest') as f:
    f.create_dataset('fixed', data=field((30, 40)), chunks=(7, 9), compression='gzip')
    f.create_dataset('fixed_paged', data=field((64, 128), 'i4'), chunks=(2, 2))  # 2048 chunks > page
    f.create_dataset('fixed_paged_z', data=field((64, 128), 'i4'), chunks=(2, 2), compression='gzip')
    f.create_dataset('single', data=field((20, 30)), chunks=(20, 30))
    f.create_dataset('single_z', data=field((20, 30)), chunks=(20, 30), compression='gzip', shuffle=True)
    d = f.create_dataset('earray', shape=(0, 12), maxshape=(None, 12), chunks=(1, 12), dtype='f4')
    d.resize((3000, 12))                            # secondary blocks, paged data blocks
    d[...] = field((3000, 12))
    d = f.create_dataset('earray_z', shape=(0, 33), maxshape=(None, 33), chunks=(3, 8), dtype='f8',
                         compression='gzip', shuffle=True)
    d.resize((700, 33))
    d[...] = field((700, 33), 'f8')
    d = f.create_dataset('earray_mid', shape=(5, 0, 7), maxshape=(5, None, 7), chunks=(2, 3, 4), dtype='i2')
    d.resize((5, 100, 7))                           # unlimited dimension not first
    d[...] = field((5, 100, 7), 'i2')
    d = f.create_dataset('btree2', shape=(0, 0), maxshape=(None, None), chunks=(4, 5), dtype='f4')
    d.resize((90, 70))
    d[...] = field((90, 70))
    d = f.create_dataset('btree2_z', shape=(0, 0), maxshape=(None, None), chunks=(3, 4), dtype='u2',
                         compression='gzip')
    d.resize((150, 120))
    d[...] = field((150, 120), 'u2')
    d = f.create_dataset('sparse', shape=(40, 50), chunks=(10, 10), dtype='f4', fillvalue=-7.5)
    d[0:10, 0:10] = 1.0                             # most chunks never allocated
    d[25:35, 40:50] = 2.0
    d = f.create_dataset('sparse_ea', shape=(0, 10), maxshape=(None, 10), chunks=(4, 10), dtype='i4',
                         fillvalue=99)
    d.resize((40, 10))
    d[8:12] = 3
made('idx_latest.h5')

with h5('idx_implicit.h5', libver='latest') as f:
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk((5, 6))
    dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
    sp = h5py.h5s.create_simple((23, 31))
    did = h5py.h5d.create(f.id, b'implicit', h5py.h5t.NATIVE_INT32, sp, dcpl)
    ds = h5py.Dataset(did)
    ds[...] = field((23, 31), 'i4')
made('idx_implicit.h5')

# ------------------------------------------------------------ layouts
with h5('layouts.h5') as f:
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_layout(h5py.h5d.COMPACT)
    did = h5py.h5d.create(f.id, b'compact', h5py.h5t.NATIVE_FLOAT, h5py.h5s.create_simple((5, 6)), dcpl)
    h5py.Dataset(did)[...] = field((5, 6))
    f.create_dataset('contiguous', data=field((50, 60), 'f8'))
    f.create_dataset('scalar', data=np.float64(3.25))
    f.create_dataset('empty', shape=(0, 4), dtype='f4')
    f.create_dataset('empty_chunked', shape=(0, 4), maxshape=(None, 4), dtype='f4', chunks=(2, 4))
    f.create_dataset('never_written', shape=(30,), dtype='i2', fillvalue=-3)
    f.create_dataset('never_written_c', shape=(30,), dtype='i2', chunks=(7,), fillvalue=-3)
    f.create_dataset('null_space', data=h5py.Empty('f4'))
    f.create_dataset('bigendian', data=field((20, 20)).astype('>f4'), chunks=(6, 7), compression='gzip')
    f.create_dataset('bigendian_i', data=field((20, 20), '>i4'))
made('layouts.h5')

# ------------------------------------------------------------ groups and attributes
for libver, tag in ((None, 'old'), ('latest', 'new')):
    with h5('groups_%s.h5' % tag, libver=libver, track_order=(tag == 'new')) as f:
        g = f.create_group('many')
        for i in range(120):                        # multi-level symbol table / dense links
            g.create_dataset('v%03d' % (119 - i), data=np.arange(i % 5 + 1, dtype='i2'))
        a = f.create_group('attrs')
        for i in range(40):                         # dense attribute storage
            a.attrs['a%02d' % (39 - i)] = np.float32(i) * 1.5
        a.attrs['text'] = 'hello'
        a.attrs['vlen'] = h5py.special_dtype(vlen=str) and np.array(['x', 'yy'], dtype=h5py.string_dtype())
        a.attrs['big'] = np.arange(2000, dtype='f8')
        f.create_group('nest/ed/deeper').create_dataset('leaf', data=[1, 2, 3])
        f['soft'] = h5py.SoftLink('/nest/ed/deeper/leaf')
    made('groups_%s.h5' % tag)

# ------------------------------------------------------------ datatypes
with h5('types.h5') as f:
    cmp = np.dtype([('a', 'i4'), ('b', 'f8'), ('c', 'S3'), ('d', 'u1', (2,))])
    x = np.zeros(10, cmp)
    x['a'] = np.arange(10); x['b'] = np.arange(10) / 3; x['c'] = b'abc'; x['d'] = [[1, 2]] * 10
    f.create_dataset('compound', data=x, chunks=(4,), compression='gzip')
    e = h5py.enum_dtype({'clear': 0, 'cloud': 1, 'rain': 7}, basetype='u1')
    f.create_dataset('enum', data=np.array([0, 1, 7, 1, 0], 'u1'), dtype=e)
    f.create_dataset('vlen_str', data=np.array(['a', 'bc', 'déf', ''], dtype=h5py.string_dtype()))
    f.create_dataset('vlen_str_c', data=np.array(['a', 'bc', 'déf'] * 5, dtype=h5py.string_dtype()),
                     chunks=(4,), compression='gzip')
    f.create_dataset('fixed_str', data=np.array([b'ab', b'cde', b''], 'S5'))
    vl = h5py.vlen_dtype('i2')
    d = f.create_dataset('vlen_seq', shape=(3,), dtype=vl)
    d[0] = [1, 2]; d[1] = []; d[2] = [5, 6, 7]
    d = f.create_dataset('array', shape=(4,), dtype=np.dtype(('i4', (3, 2))))
    d[...] = np.arange(24, dtype='i4').reshape(4, 3, 2)
    f.create_dataset('float16', data=field((10, 10), 'f2'), chunks=(5, 5), compression='gzip')
    f.create_dataset('bool', data=np.array([True, False, True]))
    f.create_dataset('int8', data=np.arange(-50, 50, dtype='i1'))
    f.create_dataset('uint64', data=np.arange(30, dtype='u8') * (2**61))
    f.create_dataset('complex', data=(np.arange(6) + 1j).astype('c8'))
    f['named_t'] = np.dtype('f4')
    f.create_dataset('uses_named', data=field((4,)), dtype=f['named_t'])
made('types.h5')

# ------------------------------------------------------------ filters
FILTERS = {
    'gzip': dict(compression='gzip', compression_opts=6),
    'gzip_shuffle': dict(compression='gzip', shuffle=True),
    'gzip_shuffle_fletcher': dict(compression='gzip', shuffle=True, fletcher32=True),
    'fletcher': dict(fletcher32=True),
    'shuffle_only': dict(shuffle=True),
    'lzf': dict(compression='lzf'),
    'szip_nn': dict(compression='szip', compression_opts=('nn', 16)),
    'szip_ec': dict(compression='szip', compression_opts=('ec', 8)),
    'scaleoffset_int': dict(scaleoffset=0),
    'scaleoffset_float': dict(scaleoffset=3),
    'bzip2': dict(**hdf5plugin.BZip2()),
    'zstd': dict(**hdf5plugin.Zstd(clevel=5)),
    'lz4': dict(**hdf5plugin.LZ4()),
    'bitshuffle_lz4': dict(**hdf5plugin.Bitshuffle(cname='lz4')),
    'bitshuffle_zstd': dict(**hdf5plugin.Bitshuffle(cname='zstd')),
    'blosc_lz4_shuffle': dict(**hdf5plugin.Blosc(cname='lz4', shuffle=hdf5plugin.Blosc.SHUFFLE)),
    'blosc_zstd_bitshuffle': dict(**hdf5plugin.Blosc(cname='zstd', shuffle=hdf5plugin.Blosc.BITSHUFFLE)),
    'blosc_zlib_noshuffle': dict(**hdf5plugin.Blosc(cname='zlib', shuffle=hdf5plugin.Blosc.NOSHUFFLE)),
    'blosc_blosclz': dict(**hdf5plugin.Blosc(cname='blosclz')),
    'blosc2_lz4': dict(**hdf5plugin.Blosc2(cname='lz4')),
}
with h5('filters.h5') as f:
    for name, kw in FILTERS.items():
        g = f.create_group(name)
        for dt in ('f4', 'f8', 'i2', 'u1', 'i8'):
            if name.startswith('scaleoffset_float') and dt[0] != 'f':
                continue
            if name.startswith('scaleoffset_int') and dt[0] == 'f':
                continue
            if name.startswith('szip') and dt == 'i8':
                pass
            try:
                g.create_dataset(dt, data=field((45, 70), dt), chunks=(16, 32), **kw)
            except Exception as e:
                print('   skip %s %s: %s' % (name, dt, e))
    # chunks a filter could not shrink: deflate is optional, the mask bit is set
    f.create_dataset('incompressible', data=rng.standard_normal((40, 40)).astype('f8'), chunks=(10, 10),
                     compression='gzip', compression_opts=9)
    f.create_dataset('nbit', data=field((30, 30), 'i4') % 1000, chunks=(10, 10),
                     dtype=h5py.h5t.NATIVE_INT32.copy() if False else 'i4')
made('filters.h5')

# n-bit needs a reduced-precision type, set through the low-level API.
# HDF5 >= 1.14.4 refuses to create such types unless the file access
# property list relaxes its integrity checks, which h5py cannot set.
def _nbit():
    with h5('nbit.h5') as f:
        t = h5py.h5t.STD_I32LE.copy()
        t.set_precision(12)
        t.set_offset(2)
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((10, 10))
        dcpl.set_filter(h5py.h5z.FILTER_NBIT)
        did = h5py.h5d.create(f.id, b'nbit_i4', t, h5py.h5s.create_simple((25, 30)), dcpl)
        v = (field((25, 30), 'i4') % 1000).astype('i4')
        did.write(h5py.h5s.ALL, h5py.h5s.ALL, v)
        t = h5py.h5t.IEEE_F32LE.copy()
        t.set_fields(26, 20, 6, 0, 20)                  # sign, exponent pos/size, mantissa pos/size
        t.set_offset(0)
        t.set_precision(27)
        t.set_ebias(31)
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((10, 10))
        dcpl.set_filter(h5py.h5z.FILTER_NBIT)
        did = h5py.h5d.create(f.id, b'nbit_f4', t, h5py.h5s.create_simple((25, 30)), dcpl)
        did.write(h5py.h5s.ALL, h5py.h5s.ALL, field((25, 30)) / 100)
    made('nbit.h5')


try:
    _nbit()
except ValueError as e:
    print('   no n-bit file:', str(e)[:80])
    os.remove(os.path.join(OUT, 'nbit.h5'))


# ------------------------------------------------------------ netCDF
def nc(name, fmt='NETCDF4'):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        os.remove(path)
    return netCDF4.Dataset(path, 'w', format=fmt)


for fmt, tag in (('NETCDF3_CLASSIC', 'nc3_classic'), ('NETCDF3_64BIT_OFFSET', 'nc3_64bit'),
                 ('NETCDF3_64BIT_DATA', 'nc3_cdf5'), ('NETCDF4_CLASSIC', 'nc4_classic'),
                 ('NETCDF4', 'nc4')):
    with nc(tag + '.nc', fmt) as d:
        d.title = 'fastnc test file'
        d.history = 'made by tools/make_corpus.py'
        d.version = np.int32(3)
        d.ratio = np.float64(0.25)
        d.createDimension('time', None)
        d.createDimension('y', 13)
        d.createDimension('x', 17)
        d.createDimension('nchar', 6)
        t = d.createVariable('time', 'f8', ('time',))
        t.units = 'hours since 2026-01-01'
        t[:] = np.arange(5) * 6.0
        kw = dict(zlib=True, shuffle=True) if tag.startswith('nc4') else {}
        v = d.createVariable('temp', 'f4', ('time', 'y', 'x'), fill_value=-999.0, **kw)
        v.units = 'K'
        v.scale_note = 'plain'
        v[:] = field((5, 13, 17)) + 273
        v[2, 3, 4] = -999.0
        p = d.createVariable('packed', 'i2', ('time', 'y', 'x'), fill_value=-32767, **kw)
        p.scale_factor = np.float32(0.01)
        p.add_offset = np.float32(100.0)
        p.valid_min = np.int16(-30000)
        p.valid_max = np.int16(30000)
        p.set_auto_maskandscale(False)
        p[:] = field((5, 13, 17), 'i2')
        p[0, 0, :3] = -32767
        p[1, 1, 1] = 31000                          # outside valid range
        u = d.createVariable('ubyte_as_signed', 'i1', ('y', 'x'), **kw)
        u._Unsigned = 'true'
        u.missing_value = np.int8(-1)
        u[:] = (field((13, 17)) % 256).astype('u1').view('i1')
        mv = d.createVariable('missing', 'f8', ('y', 'x'), **kw)
        mv.missing_value = np.array([1e20, -1e20])
        mv.valid_range = np.array([-500.0, 500.0])
        a = field((13, 17), 'f8')
        a[0, 0] = 1e20; a[1, 1] = -1e20; a[2, 2] = 600.0
        mv[:] = a
        c = d.createVariable('names', 'S1', ('y', 'nchar'))
        c[:] = np.frombuffer(np.array(['abc', 'defghi', ''] * 4 + ['z'], 'S6').tobytes(), 'S1').reshape(13, 6)
        s = d.createVariable('rec_int', 'i4', ('time',))
        s[:] = np.arange(5) * 7
        sc = d.createVariable('scalar', 'f4')
        sc[...] = 1.5
        if tag == 'nc3_cdf5' or tag == 'nc4':
            w = d.createVariable('u64', 'u8', ('x',))
            w[:] = np.arange(17, dtype='u8') * 2**60
        if tag == 'nc4':
            sv = d.createVariable('strings', str, ('y',))
            sv[:] = np.array(['s%d' % i * (i % 3) for i in range(13)], object)
            g = d.createGroup('sub')
            g.createDimension('z', 4)
            gv = g.createVariable('deep', 'f8', ('z', 'x'), zlib=True, complevel=9)
            gv[:] = field((4, 17), 'f8')
            g.note = 'in a group'
            ct = d.createCompoundType(np.dtype([('lat', 'f4'), ('lon', 'f4'), ('n', 'i2')]), 'pos_t')
            cv = d.createVariable('pos', ct, ('y',))
            x = np.zeros(13, ct.dtype)
            x['lat'] = np.arange(13); x['lon'] = -np.arange(13); x['n'] = 3
            cv[:] = x
            et = d.createEnumType('u1', 'sky_t', {'clear': 0, 'cloudy': 1, 'missing': 255})
            ev = d.createVariable('sky', et, ('x',), fill_value=255)
            ev[:] = np.array([0, 1] * 8 + [255], 'u1')
            vt = d.createVLType('i4', 'ragged_t')
            vv = d.createVariable('ragged', vt, ('y',))
            vv[:] = np.array([np.arange(i % 4, dtype='i4') for i in range(13)], object)
            for i in range(30):
                d.setncattr('global_%02d' % i, i)   # dense attributes on the root group
            many = d.createGroup('many_vars')
            many.createDimension('n', 3)
            for i in range(40):                     # dense links
                many.createVariable('v%02d' % i, 'i2', ('n',))[:] = [i, i + 1, i + 2]
            two = d.createGroup('two_unlimited')
            two.createDimension('a', None)
            two.createDimension('b', None)
            tv = two.createVariable('ab', 'f4', ('a', 'b'), zlib=True)
            tv[:] = field((9, 11))                  # v2 B-tree chunk index
            q = d.createVariable('quantized', 'f4', ('y', 'x'), zlib=True, significant_digits=3)
            q[:] = field((13, 17))
            nf = d.createVariable('no_fill', 'f4', ('y', 'x'), fill_value=False)
            nf[:5] = 1.0
            try:
                z = d.createVariable('zstd', 'f4', ('y', 'x'), compression='zstd', complevel=4)
                z[:] = field((13, 17))
            except Exception as e:
                print('   no zstd through netCDF4:', e)
    made(tag + '.nc')

print('done ->', OUT)
