"""Call an hdf5plugin filter's C function directly (ctypes), bypassing HDF5,
so the codecs can be timed against fastnc's kernels chunk by chunk.

Each plugin DLL exports H5PLget_plugin_info(), a pointer to its
H5Z_class2_t, whose last member is the filter function
`size_t filter(unsigned flags, size_t ncd, const unsigned cd[], size_t nbytes,
size_t *buf_size, void **buf)`. Decoding (flags = H5Z_FLAG_REVERSE) it
frees the input buffer and returns a malloc'ed output, so the input is
copied into a buffer from the C runtime's malloc first (ucrtbase, the
runtime the plugins link) and the output freed with its free.
"""
import ctypes
import os
import sys

import h5py  # noqa: F401  (loads hdf5.dll, which the plugins link)
import hdf5plugin
import numpy as np

H5Z_FLAG_REVERSE = 0x0100

DLLS = {32000: None, 32001: 'libh5blosc', 32004: 'libh5lz4', 32008: 'libh5bshuf',
        32015: 'libh5zstd', 32026: 'libh5blosc2'}

FILTER = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_uint, ctypes.c_size_t,
                          ctypes.POINTER(ctypes.c_uint), ctypes.c_size_t,
                          ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_void_p))


class _Class2(ctypes.Structure):
    _fields_ = [('version', ctypes.c_int), ('id', ctypes.c_int), ('encoder_present', ctypes.c_uint),
                ('decoder_present', ctypes.c_uint), ('name', ctypes.c_char_p),
                ('can_apply', ctypes.c_void_p), ('set_local', ctypes.c_void_p),
                ('filter', FILTER)]


_crt = ctypes.CDLL('ucrtbase')
_crt.malloc.restype = ctypes.c_void_p
_crt.malloc.argtypes = [ctypes.c_size_t]
_crt.free.argtypes = [ctypes.c_void_p]
_keep = []


def plugin(fid):
    """The filter function of filter `fid` (a ctypes function)."""
    name = DLLS[fid]
    if name is None:
        raise KeyError('no plugin DLL for filter %d' % fid)
    dll = ctypes.CDLL(os.path.join(hdf5plugin.PLUGIN_PATH, name + '.dll'))
    _keep.append(dll)
    dll.H5PLget_plugin_info.restype = ctypes.POINTER(_Class2)
    return dll.H5PLget_plugin_info().contents.filter


class Decoder:
    """decode(raw bytes) -> decoded bytes with the reference plugin; time()
    measures only the filter call."""

    def __init__(self, fid, cd):
        self.f = plugin(fid)
        self.cd = (ctypes.c_uint * max(1, len(cd)))(*cd)
        self.ncd = len(cd)

    def call(self, raw, keep=True):
        n = len(raw)
        buf = _crt.malloc(max(n, 1))
        ctypes.memmove(buf, raw, n)
        p = ctypes.c_void_p(buf)
        size = ctypes.c_size_t(n)
        r = self.f(H5Z_FLAG_REVERSE, self.ncd, self.cd, n, ctypes.byref(size), ctypes.byref(p))
        out = ctypes.string_at(p.value, r) if (keep and r) else None
        if r:
            _crt.free(p.value)
        else:
            _crt.free(buf) if p.value == buf else None
        return r, out


if __name__ == '__main__':
    # check: the direct calls agree with h5py reading through HDF5
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from codecs_ref import ROOT, chunks, filters
    f = h5py.File(os.path.join(ROOT, 'tests', 'data', 'filters.h5'), 'r')
    for g in ('lz4', 'zstd', 'bitshuffle_lz4', 'bitshuffle_zstd', 'blosc_lz4_shuffle', 'blosc2_lz4'):
        ds = f[g + '/f4']
        (fid, flags, cd), = filters(ds)
        d = Decoder(fid, cd)
        ok = all(d.call(raw)[1] == want for off, raw, mask, want in chunks(ds) if not mask)
        print(g, ok)
