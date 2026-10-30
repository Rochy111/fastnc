"""libaec's SZ compatibility layer (szip.dll / libsz), called directly
through ctypes: the code HDF5's szip filter calls, for any parameters --
bits per pixel HDF5 cannot make (n-bit types), MSB order, any scanline --
and for streams made outside HDF5.

    sz_compress(data, mask, ppb, bpp, pps) -> bytes (the SZ stream)
    sz_decompress(stream, nbytes, mask, ppb, bpp, pps) -> bytes or None
"""
import ctypes
import os
import sys


class SZ_com_t(ctypes.Structure):
    _fields_ = [('options_mask', ctypes.c_int), ('bits_per_pixel', ctypes.c_int),
                ('pixels_per_block', ctypes.c_int), ('pixels_per_scanline', ctypes.c_int)]


_lib = None


def lib():
    global _lib
    if _lib is None:
        names = ['szip.dll', 'libsz.dll', 'libszip.dll'] if sys.platform == 'win32' else \
            ['libsz.so.2', 'libsz.so', 'libsz.dylib']
        dirs = [os.path.join(sys.prefix, 'Library', 'bin'), os.path.join(sys.prefix, 'lib'), '']
        for d in dirs:
            for n in names:
                p = os.path.join(d, n) if d else n
                try:
                    _lib = ctypes.CDLL(p)
                    break
                except OSError:
                    continue
            if _lib is not None:
                break
        if _lib is None:
            raise OSError('no libaec SZ library found')
        for fn in (_lib.SZ_BufftoBuffCompress, _lib.SZ_BufftoBuffDecompress):
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p,
                           ctypes.c_size_t, ctypes.POINTER(SZ_com_t)]
            fn.restype = ctypes.c_int
    return _lib


def sz_compress(data, mask, ppb, bpp, pps):
    data = bytes(data)
    cap = len(data) * 2 + 1024
    out = ctypes.create_string_buffer(cap)
    n = ctypes.c_size_t(cap)
    prm = SZ_com_t(mask, bpp, ppb, pps)
    st = lib().SZ_BufftoBuffCompress(out, ctypes.byref(n), data, len(data), ctypes.byref(prm))
    if st != 0:
        raise ValueError('SZ_BufftoBuffCompress: %d' % st)
    return out.raw[:n.value]


def sz_decompress(stream, nbytes, mask, ppb, bpp, pps):
    stream = bytes(stream)
    out = ctypes.create_string_buffer(max(nbytes, 1))
    n = ctypes.c_size_t(nbytes)
    prm = SZ_com_t(mask, bpp, ppb, pps)
    st = lib().SZ_BufftoBuffDecompress(out, ctypes.byref(n), stream, len(stream), ctypes.byref(prm))
    if st != 0:
        return None
    return out.raw[:n.value]
