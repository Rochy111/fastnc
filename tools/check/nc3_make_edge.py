"""netCDF3 edge-case files made with netCDF4-python, for checking fastnc's
netCDF3 reader against it (tools/check/nc3_vs_netcdf4.py) and for the
expected values stored for the tests (tools/check/nc3_make_expected.py).

`make_all(folder, big=False)` writes them and returns their paths. Each
format gets: every type it has, several record variables of different
sizes (padded record slots), a lone record variable (CDF-1's unpadded
special case), char variables and text attributes with NULs and non-ASCII
bytes, an unlimited dimension with no records, scalars, a header over
64 KiB, long and non-ASCII names, values starting off 8-byte alignment;
then copies with the record count patched to "streaming" and truncated
copies. big=True adds files big enough for the threaded paths (tens of MB).
"""
import os
import sys

import numpy as np
import netCDF4

FORMATS = (('NETCDF3_CLASSIC', 'c1'), ('NETCDF3_64BIT_OFFSET', 'c2'), ('NETCDF3_64BIT_DATA', 'c5'))
CLASSIC_TYPES = ('i1', 'S1', 'i2', 'i4', 'f4', 'f8')
CDF5_TYPES = CLASSIC_TYPES + ('u1', 'u2', 'u4', 'i8', 'u8')


def _values(dt, shape, seed):
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape)) if shape else 1
    dt = np.dtype(dt)
    if dt.kind == 'S':
        v = rng.integers(32, 127, n).astype('u1').view('S1')
        v[::7] = b'\xe9'                        # non-ASCII byte
        v[::11] = b''                           # NUL
    elif dt.kind == 'f':
        v = rng.standard_normal(n).astype(dt) * 1000
        v[::13] = np.nan
        v[1::17] = np.inf
        if n > 3:
            v[2] = -0.0
    else:
        info = np.iinfo(dt)
        v = rng.integers(info.min, info.max, n, dtype=dt, endpoint=True)
        v[:1] = info.min
        v[-1:] = info.max
    return v.reshape(shape)


def _write(path, fmt, body):
    if os.path.exists(path):
        os.remove(path)
    d = netCDF4.Dataset(path, 'w', format=fmt)
    try:
        body(d)
    finally:
        d.close()
    return path


def _types_body(types):
    def body(d):
        d.createDimension('t', None)
        d.createDimension('x', 3)
        d.createDimension('y', 5)
        d.title = 'all types'
        d.empty_text = ''
        d.nul_text = 'ab\x00cd\x00\x00'
        d.setncattr('utf8_text', 'Grüße 日本')
        d.setncattr('bad_utf8', np.array([b'a', b'\xff', b'b'], 'S1').tobytes())
        for k, t in enumerate(types):
            safe = t if t != 'S1' else 'c'
            if t != 'S1':
                d.setncattr('one_' + safe, np.array([7], t))
                d.setncattr('many_' + safe, _values(t, (4,), 100 + k))
            v = d.createVariable('fix_' + safe, t, ('x', 'y'))
            v[:] = _values(t, (3, 5), k)
            r = d.createVariable('rec_' + safe, t, ('t', 'x'))
            r[0:4] = _values(t, (4, 3), 50 + k)
            s = d.createVariable('scal_' + safe, t, ())
            s[...] = _values(t, (), 70 + k)
            if t != 'S1':
                v.setncattr('one', np.array([3], t))
                v.setncattr('two', _values(t, (2,), 200 + k))
        c = d.createVariable('chars', 'S1', ('x', 'y'), fill_value=b'Z')
        c.long_name = 'text é\x00'
        c[0] = np.array(list(b'hi'), 'u1').view('S1').tolist() + [b''] * 3
    return body


def _records_body(d):
    d.createDimension('time', None)
    d.createDimension('a', 3)
    d.createDimension('b', 2)
    d.createDimension('n', 7)
    d.createVariable('fixed_before', 'i2', ('a',))[:] = [1, -2, 3]   # 6 bytes: next starts off 8
    specs = [('r_byte', 'i1', ('time', 'a')), ('r_short', 'i2', ('time',)),
             ('r_double', 'f8', ('time', 'b', 'a')), ('r_char', 'S1', ('time', 'n')),
             ('r_float', 'f4', ('time',)), ('r_int', 'i4', ('time', 'b'))]
    for k, (name, t, dims) in enumerate(specs):
        v = d.createVariable(name, t, dims)
        shape = (37,) + tuple(len(d.dimensions[x]) for x in dims[1:])
        v[:] = _values(t, shape, k)
    d.createVariable('fixed_after', 'f8', ('n',))[:] = np.arange(7) / 3


def _lone_body(t, n):
    def body(d):
        d.createDimension('time', None)
        if n:
            d.createDimension('m', n)
        v = d.createVariable('lone', t, ('time', 'm') if n else ('time',))
        v[:] = _values(t, (11, n) if n else (11,), 5)
        d.createVariable('fix', 'f4', ())[...] = 2.5
    return body


def _norecs_body(d):
    d.createDimension('time', None)
    d.createDimension('x', 4)
    d.createVariable('r1', 'f4', ('time', 'x'))
    d.createVariable('r2', 'i2', ('time',))
    d.createVariable('f', 'f8', ('x',))[:] = [1, 2, 3, 4]


def _names_body(d):
    long = 'L' * 250
    d.createDimension('déjà', 2)
    d.createDimension(long[:200], 3)
    v = d.createVariable('vär_ß日', 'f4', ('déjà', long[:200]))
    v[:] = np.arange(6).reshape(2, 3)
    v.setncattr('åttr', 'wärt')
    v.setncattr(long, np.arange(3, dtype='f8'))
    d.createVariable(long, 'i4', ())[...] = 9
    d.setncattr('globäl', 1)


def _bigheader_body(d):
    d.createDimension('x', 10)
    d.history = 'h' * 70000                    # header longer than the first 64 KiB read
    d.setncattr('many', np.arange(30000, dtype='f8'))
    v = d.createVariable('v', 'i4', ('x',))
    v[:] = np.arange(10)
    v.notes = 'nöt' * 10000


def _empty_body(d):
    pass


def _attrs_only_body(d):
    d.a = 1.5
    d.b = 'x'


def _big_body(fmt):
    def body(d):
        d.createDimension('time', None)
        d.createDimension('z', 20)
        d.createDimension('y', 300)
        d.createDimension('x', 301)
        d.createDimension('w', 2001)
        v = d.createVariable('fixed_big', 'f4', ('z', 'y', 'x'))     # 7 MB contiguous
        v[:] = _values('f4', (20, 300, 301), 1)
        g = d.createVariable('grid', 'f8', ('y', 'w'))               # 4.8 MB, strided reads
        g[:] = _values('f8', (300, 2001), 2)
        a = d.createVariable('rec_a', 'f4', ('time', 'z', 'x'))      # 24 kB a record
        b = d.createVariable('rec_b', 'i2', ('time', 'y', 'x'))      # 180 kB a record
        c = d.createVariable('rec_c', 'f8', ('time', 'w'))
        n = 40
        a[:] = _values('f4', (n, 20, 301), 3)
        b[:] = _values('i2', (n, 300, 301), 4)
        c[:] = _values('f8', (n, 2001), 5)
        if fmt == 'NETCDF3_64BIT_DATA':
            u = d.createVariable('big_u8', 'u8', ('z', 'y', 'x'))
            u[:] = _values('u8', (20, 300, 301), 6)
    return body


def _lone_big_body(d):
    d.createDimension('time', None)
    d.createDimension('y', 333)
    d.createDimension('x', 1001)
    v = d.createVariable('lone', 'i2', ('time', 'y', 'x'))          # unpadded records
    v[:] = _values('i2', (30, 333, 1001), 9)


def streaming_copy(src, dst):
    """src with its record count set to 'streaming' (all ones)."""
    raw = bytearray(open(src, 'rb').read())
    if raw[3] == 5:
        raw[4:12] = b'\xff' * 8
    else:
        raw[4:8] = b'\xff' * 4
    open(dst, 'wb').write(raw)
    return dst


def truncated_copy(src, dst, cut):
    """src's first len - cut bytes; and, as reference_for(dst), the same
    with the cut bytes zeroed. netCDF-C zero-fills a short read only up to
    its buffer's page: past that a truncated file reads as whatever its
    buffer held before (seen on records crossing the end of a big file),
    so the defined answer, zeros, is checked against the zero-padded copy."""
    raw = open(src, 'rb').read()
    open(dst, 'wb').write(raw[:len(raw) - cut])
    open(reference_for(dst), 'wb').write(raw[:len(raw) - cut] + bytes(cut))
    return dst


def reference_for(path):
    """The file netCDF4-python's values for path come from (itself, but for
    truncated copies)."""
    ref = path[:-3] + '_ref.nc'
    return ref if path.endswith('.nc') and os.path.exists(ref) else path


def make_all(folder, big=False):
    os.makedirs(folder, exist_ok=True)
    out = []
    j = lambda name: os.path.join(folder, name)
    for fmt, tag in FORMATS:
        types = CDF5_TYPES if tag == 'c5' else CLASSIC_TYPES
        out.append(_write(j('%s_types.nc' % tag), fmt, _types_body(types)))
        out.append(_write(j('%s_records.nc' % tag), fmt, _records_body))
        for t, n in (('i1', 3), ('i2', 0), ('S1', 5), ('f8', 2), ('i2', 3)):
            out.append(_write(j('%s_lone_%s_%d.nc' % (tag, t, n)), fmt, _lone_body(t, n)))
        out.append(_write(j('%s_norecs.nc' % tag), fmt, _norecs_body))
        out.append(_write(j('%s_names.nc' % tag), fmt, _names_body))
        out.append(_write(j('%s_bighdr.nc' % tag), fmt, _bigheader_body))
        out.append(_write(j('%s_empty.nc' % tag), fmt, _empty_body))
        out.append(_write(j('%s_attronly.nc' % tag), fmt, _attrs_only_body))
        rec = j('%s_records.nc' % tag)
        for cut in (1, 13, 300):
            out.append(truncated_copy(rec, j('%s_trunc%d.nc' % (tag, cut)), cut))
        out.append(truncated_copy(j('%s_lone_i1_3.nc' % tag), j('%s_lonetrunc.nc' % tag), 5))
        if big:
            out.append(_write(j('%s_big.nc' % tag), fmt, _big_body(fmt)))
            out.append(_write(j('%s_lonebig.nc' % tag), fmt, _lone_big_body))
            out.append(truncated_copy(j('%s_big.nc' % tag), j('%s_bigtrunc.nc' % tag), 7777777))
    return out


def streaming_files(folder):
    """Streaming copies (netCDF4-python cannot read these: it takes the
    count literally); their expected values are the original's."""
    pairs = []
    for fmt, tag in FORMATS:
        for name in ('records', 'lone_i1_3', 'lone_i2_3', 'norecs'):   # (not lone_i2_0:
            # 11 two-byte records padded to 24 bytes hold 12 as far as streaming can tell)
            src = os.path.join(folder, '%s_%s.nc' % (tag, name))
            dst = os.path.join(folder, '%s_%s_stream.nc' % (tag, name))
            pairs.append((streaming_copy(src, dst), src))
    return pairs


if __name__ == '__main__':
    for p in make_all(sys.argv[1], big='--big' in sys.argv):
        print(os.path.getsize(p), p)
