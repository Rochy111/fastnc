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
    if _LIB is not None:                        # (once per chunk: no lock for that)
        return _LIB
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
                    for fn in (lib.libdeflate_zlib_decompress, lib.libdeflate_deflate_decompress):
                        fn.restype = ctypes.c_int
                        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                       ctypes.c_void_p, ctypes.c_size_t,
                                       ctypes.POINTER(ctypes.c_size_t)]
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


def _raw(lib, sptr, n, dptr, cap):
    """libdeflate on raw deflate: bytes written, or -1."""
    d = getattr(_LOCAL, 'dec', None)
    if d is None:
        d = _LOCAL.dec = lib.libdeflate_alloc_decompressor()
    got = ctypes.c_size_t(0)
    r = lib.libdeflate_deflate_decompress(d, sptr, n, dptr, cap, ctypes.byref(got))
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
        s, d = src.ctypes.data + a, dst.ctypes.data + o
        r = -1
        if kind == 'libdeflate' and n > 6:
            cmf, flg = int(src[a]), int(src[a + 1])
            if cmf & 15 == 8 and cmf >> 4 <= 7 and (cmf << 8 | flg) % 31 == 0 and not flg & 32:
                r = _raw(lib, s + 2, n - 2, d, cap)     # (the Adler-32 goes unchecked, as in _inflate)
        else:
            r = _call(kind, lib, s, n, d, cap)
        if r >= 0:
            return r
    data = zlib.decompress(src[a:a + n])        # (the reference: its error, or too long)
    if len(data) > cap:
        return -1
    dst[o:o + len(data)] = memoryview(data)
    return len(data)


def stream_inflater(src):
    """A function f(a, n, dptr, cap) -> bytes written or -1, inflating the
    zlib stream src[a:a + n] (uint8) at address dptr; or None without
    libdeflate. One Python step for each of thousands of chunks, each under
    the GIL: nothing here looks at an array or takes a lock. Anything but a
    plain zlib header gives -1, for inflate_into to deal with."""
    kind, lib = _load()
    if kind != 'libdeflate':
        return None
    base = src.ctypes.data
    head = memoryview(src)
    run = lib.libdeflate_deflate_decompress
    size_t, ref, local = ctypes.c_size_t, ctypes.byref, _LOCAL

    def f(a, n, dptr, cap):
        cmf, flg = head[a], head[a + 1]
        if n <= 6 or cmf & 15 != 8 or cmf >> 4 > 7 or (cmf << 8 | flg) % 31 or flg & 32:
            return -1
        d = getattr(local, 'dec', None)
        if d is None:
            d = local.dec = lib.libdeflate_alloc_decompressor()
        got = size_t(0)
        return got.value if run(d, base + a + 2, n - 2, dptr, cap, ref(got)) == 0 else -1
    return f


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
