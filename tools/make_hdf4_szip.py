"""tests/data/hdf4/szip.hdf: HDF4 scientific data sets compressed with szip.

No HDF4 library at hand writes szip (pyhdf's builds have the decoder only),
so the file is made in two steps: pyhdf writes each data set compressed with
deflate, then each one's compression header becomes szip's and its
compressed bytes the szip stream HDF5's filter (h5py, libaec) makes of the
same values in the same byte order, HDF4's framing around it (cszip.c:
a flag byte, 1 for stored as is, the stream's length, then the stream; the
header's options carry SZ_H4_REV_2, as hcomp.c writes them). Where pyhdf's
HDF4 has szip it reads the file back as a check; the builds of conda-forge,
PyPI and Debian have not.

    python tools/make_hdf4_szip.py [out]
"""
import os
import struct
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SZ_H4_REV_2 = 0x10000           # (HDF4's mark of its second szip format: always set since 4.2)


def values():
    """name -> (values, szip coding): both codings, every sample width
    (tests/test_hdf4.py compares fastnc's reads with these)."""
    x = np.arange(60 * 80).reshape(60, 80)
    # (floats from integers over powers of two: exact, the same bits on every
    # platform -- sin / cos differ in their last bit between maths libraries)
    return {
        'i16_nn': ((x % 517 - 200).astype('>i2'), 'nn'),
        'u16_ec': ((x * 7 % 4099).astype('>u2'), 'ec'),
        'i32_nn': ((x * 1013 - 99999).astype('>i4'), 'nn'),
        'f32_nn': ((((x * 2654435761 % 1000003) - 500000) / 1024).astype('>f4'), 'nn'),
        'f64_ec': ((((x * 7919 % 999983) - 4000) * 1000 / 4096).astype('>f8'), 'ec'),
        'u8_nn': ((x % 251).astype('u1'), 'nn'),
        'noise_nn': (np.random.default_rng(3).integers(-2 ** 31, 2 ** 31 - 1, (60, 80)).astype('>i4'),
                     'nn'),                    # (does not compress: kept as is, flag 1)
    }


def szip_stream(a, opt):
    """(options mask, pixels per block, bits per pixel, pixels per scanline),
    and the szip stream, of array a as HDF5's szip filter codes it."""
    import h5py
    d = tempfile.mkdtemp()
    p = os.path.join(d, 's.h5')
    with h5py.File(p, 'w') as f:
        ds = f.create_dataset('x', data=a, chunks=a.shape, compression='szip', compression_opts=(opt, 8))
        cd = ds.id.get_create_plist().get_filter(0)[2]
        skipped, raw = ds.id.read_direct_chunk((0,) * a.ndim)
    os.remove(p)
    os.rmdir(d)
    return tuple(cd[:4]), (None if skipped & 1 else raw)    # (szip is optional: skipped where it does not pay)


def dd_records(b):
    """{(tag, ref): position of its 12-byte DD record}."""
    out = {}
    pos = 4
    while pos > 0:
        n, nxt = struct.unpack_from('>Hi', b, pos)
        for i in range(n):
            at = pos + 6 + 12 * i
            tag, ref = struct.unpack_from('>HH', b, at)
            if tag:
                out[(tag, ref)] = at
        pos = nxt
    return out


def main(out):
    from pyhdf.SD import SD, SDC
    vals = values()
    types = {'i2': SDC.INT16, 'u2': SDC.UINT16, 'i4': SDC.INT32, 'f4': SDC.FLOAT32, 'f8': SDC.FLOAT64,
             'u1': SDC.UINT8}
    sd = SD(out, SDC.WRITE | SDC.CREATE | SDC.TRUNC)
    for name, (a, _) in vals.items():
        v = sd.create(name, types[a.dtype.str[1:]], a.shape)
        v.setcompress(SDC.COMP_DEFLATE, 6)
        v[:] = a.astype(a.dtype.newbyteorder('='))
        v.endaccess()
    sd.end()
    b = bytearray(open(out, 'rb').read())
    recs = dd_records(b)
    sys.path.insert(0, ROOT)
    import fastnc
    with fastnc.Dataset(out) as d:
        refs = {n: d[n]._bv.data_ref for n in vals}     # (the data element's ref, tag 702)
    for name, (a, opt) in vals.items():
        ref = refs[name]
        hdr_at = recs[(702 | 0x4000, ref)]
        off, ln = struct.unpack_from('>ii', b, hdr_at + 4)
        code, ver, length, cref, model, coder = struct.unpack_from('>HHiHHH', b, off)
        assert code == 3 and coder == 4, (name, code, coder)
        (mask, ppb, bpp, ppsl), raw = szip_stream(a, opt)
        if raw is not None:
            size = struct.unpack_from('<I', raw, 0)[0]
            assert size == a.nbytes, (name, size, a.nbytes)
        stream = raw[4:] if raw is not None else None
        if stream is None or len(stream) >= a.nbytes:
            body = b'\x01' + struct.pack('>i', a.nbytes) + a.tobytes()
        else:
            body = b'\x00' + struct.pack('>i', len(stream)) + stream
        header = struct.pack('>HHiHHH', code, ver, length, cref, model, 5) + \
            struct.pack('>IIIBB', a.size, ppsl, mask | SZ_H4_REV_2, bpp, ppb)
        for tag, data in ((702 | 0x4000, header), (40, body)):
            r = cref if tag == 40 else ref
            at = recs[(tag, r)]
            struct.pack_into('>ii', b, at + 4, len(b), len(data))
            b += data
    open(out, 'wb').write(bytes(b))
    sd = SD(out)
    try:
        for name, (a, _) in vals.items():
            assert np.array_equal(sd.select(name)[:], a), name
        print('wrote', out, 'and pyhdf reads it back')
    except ValueError:
        # (the HDF4 of conda-forge, PyPI's pyhdf and Debian is built without szip:
        # none here reads it. The layout is cszip.c's and hcomp.c's: a flag byte,
        # 1 for stored as is, the stream's length INT32ENCODEd, then the stream)
        print('wrote', out, '(this pyhdf cannot read szip: not checked by it)')
    sd.end()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'tests', 'data', 'hdf4', 'szip.hdf'))
