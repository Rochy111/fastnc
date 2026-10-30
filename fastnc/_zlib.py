"""Inflate without numba, for reads before the kernels are loaded (a fresh
process: numba's import and the kernels' cache cost most of a second).

A zlib stream is inflated straight into a numpy array through the fastest
inflater found as a shared library -- libdeflate, else zlib -- called by
ctypes, which lets go of the GIL for the call: chunks inflate on every
core with no bytes object per chunk. Any correct inflater gives the same
bytes (the format leaves nothing to choose), so values do not depend on
which one ran. With none found, `zlib.decompress` and a copy.
"""
import ctypes
import os
import sys
import threading
import zlib

_LIB = None                     # (kind, CDLL) once looked for; kind None: none found
_LOCK = threading.Lock()
_LOCAL = threading.local()


def _candidates():
    if os.name == 'nt':
        # (only where this Python's own libraries are: a DLL search of PATH
        # finds whatever other programs put there)
        for d in (os.path.join(sys.prefix, 'Library', 'bin'), os.path.join(sys.prefix, 'DLLs'),
                  sys.prefix):
            for kind, name in (('libdeflate', 'libdeflate.dll'), ('libdeflate', 'deflate.dll'),
                               ('zlib', 'zlib.dll'), ('zlib', 'zlib1.dll')):
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    yield kind, p
    else:
        import ctypes.util
        for kind, name in (('libdeflate', 'deflate'), ('zlib', 'z')):
            for d in (os.path.join(sys.prefix, 'lib'),):
                for ext in ('.so', '.dylib'):
                    p = os.path.join(d, 'lib' + name + ext)
                    if os.path.isfile(p):
                        yield kind, p
            p = ctypes.util.find_library(name)
            if p:
                yield kind, p


def _load():
    global _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        if os.environ.get('FASTNC_NO_CTYPES_ZLIB'):
            _LIB = (None, None)
            return _LIB
        for kind, p in _candidates():
            try:
                lib = ctypes.CDLL(p)
                if kind == 'libdeflate':
                    lib.libdeflate_alloc_decompressor.restype = ctypes.c_void_p
                    lib.libdeflate_alloc_decompressor.argtypes = []
                    lib.libdeflate_zlib_decompress.restype = ctypes.c_int
                    lib.libdeflate_zlib_decompress.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
                else:
                    lib.uncompress.restype = ctypes.c_int
                    lib.uncompress.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong),
                                               ctypes.c_void_p, ctypes.c_ulong]
                if _selftest(kind, lib):
                    _LIB = (kind, lib)
                    return _LIB
            except (OSError, AttributeError):
                continue
        _LIB = (None, None)
        return _LIB


def _call(kind, lib, sptr, n, dptr, cap):
    """Bytes written, or -1 (bad stream, or more than cap)."""
    if kind == 'libdeflate':
        d = getattr(_LOCAL, 'dec', None)
        if d is None:
            d = _LOCAL.dec = lib.libdeflate_alloc_decompressor()
        got = ctypes.c_size_t(0)
        r = lib.libdeflate_zlib_decompress(d, sptr, n, dptr, cap, ctypes.byref(got))
        return got.value if r == 0 else -1
    got = ctypes.c_ulong(cap)
    r = lib.uncompress(dptr, ctypes.byref(got), sptr, n)
    return got.value if r == 0 else -1


def _selftest(kind, lib):
    import numpy as np
    raw = bytes(range(256)) * 300
    z = np.frombuffer(zlib.compress(raw, 6), np.uint8)
    out = np.zeros(len(raw) + 16, np.uint8)
    n = _call(kind, lib, z.ctypes.data, z.size, out.ctypes.data, out.size)
    return n == len(raw) and out[:n].tobytes() == raw


def which():
    """'libdeflate', 'zlib' (through ctypes) or None (zlib.decompress)."""
    return _load()[0]


def inflater():
    """A function f(sptr, n, dptr, cap) -> bytes written or -1, on raw
    addresses (no per-call array lookups: for loops over many small
    chunks), or None when no library was found."""
    kind, lib = _load()
    if kind is None:
        return None
    return lambda sptr, n, dptr, cap: _call(kind, lib, sptr, n, dptr, cap)


def inflate_into(src, a, n, dst, o, cap):
    """Inflate the zlib stream src[a:a+n] (uint8 arrays) into dst[o:o+cap];
    the number of bytes it gave, or -1 if more than cap. Raises on a bad
    stream (as zlib.decompress does)."""
    kind, lib = _load()
    if kind is not None:
        r = _call(kind, lib, src.ctypes.data + a, n, dst.ctypes.data + o, cap)
        if r >= 0:
            return r
    data = zlib.decompress(src[a:a + n])        # (the reference: its error, or too long)
    if len(data) > cap:
        return -1
    dst[o:o + len(data)] = memoryview(data)
    return len(data)


def native():
    """(address of libdeflate_deflate_decompress, a decompressor of this
    thread's) as ints for the numba kernels to call, or (0, 0) without
    libdeflate."""
    kind, lib = _load()
    if kind != 'libdeflate' or os.environ.get('FASTNC_NO_NATIVE_INFLATE'):
        return 0, 0
    d = getattr(_LOCAL, 'dec', None)
    if d is None:
        d = _LOCAL.dec = lib.libdeflate_alloc_decompressor()
    return ctypes.cast(lib.libdeflate_deflate_decompress, ctypes.c_void_p).value or 0, d or 0
