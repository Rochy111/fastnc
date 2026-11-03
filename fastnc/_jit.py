"""numba, imported when the first kernel runs rather than with fastnc:
`njit`, `intrinsic` and `prange` as the modules use them, standing in
until then. Opening a file and reading its metadata -- the work that runs
no kernel -- then never pays for numba's import (a quarter of a second).
Taken from fastgrib.

The first call of any kernel imports numba, makes every kernel known so
far (each from its cache as before: the functions are the same), and puts
them in place of their stand-ins in every fastnc module, so kernels
calling kernels see numba's dispatchers. Kernels defined after that are
numba's from the start, and so is `prange` imported from here. A stand-in
called from Python works throughout.
"""
import sys
import threading

_PENDING = []                   # stand-ins not yet made
_MADE = []                      # every stand-in
_LOCK = threading.RLock()
_numba = None                   # the numba module, once imported
_IMAGE = None                   # _image.Image the stand-ins call through, if any
_NO = None                      # _image.NO


class _Kernel:
    """A stand-in for numba.njit(**opts)(fn) or numba.extending.intrinsic(fn)."""

    def __init__(self, fn, opts, kind):
        self.fn, self.opts, self.kind, self.real = fn, opts, kind, None
        self.fast = {}                          # (the image) Python types of the arguments -> unit
        self.__wrapped__, self.__name__ = fn, fn.__name__
        self.__doc__ = fn.__doc__

    def make(self, numba):
        if self.kind == 'intrinsic':
            from numba.extending import intrinsic as make
            return make(self.fn)
        return numba.njit(**self.opts)(self.fn)

    def __call__(self, *args, **kwargs):
        if self.real is None:
            if _IMAGE is None:                  # (a first read that came by no other way to it)
                from . import _chunks
                _chunks.loaded()
            if _IMAGE is not None and not kwargs:
                r = _IMAGE.call(self, args)
                if r is not _NO:
                    return r
                load()
                r = self.real(*args)
                from . import _image                # (numba's cache may now hold more)
                _image.build_later()
                return r
            load()
        return self.real(*args, **kwargs)

    def __getattr__(self, name):                # (signatures, inspect_asm, py_func ...)
        if name == 'real':
            raise AttributeError(name)
        if self.real is None:
            load()
        return getattr(self.real, name)

    def __repr__(self):
        return '<kernel %s%s>' % (self.__name__, '' if self.real is not None
                                  else ' (numba not loaded yet)')


class _Prange:
    """numba.prange's stand-in (a plain range from Python)."""

    def __call__(self, *args):
        return range(*args)


_STAND_IN = _Prange()
prange = _STAND_IN


def njit(*args, **opts):
    """numba.njit, deferred (as @njit(cache=True, nogil=True), or bare)."""
    if args and callable(args[0]):
        return _defer(args[0], {}, 'njit')
    return lambda fn: _defer(fn, opts, 'njit')


def intrinsic(fn):
    """numba.extending.intrinsic, deferred."""
    return _defer(fn, {}, 'intrinsic')


def _defer(fn, opts, kind):
    with _LOCK:
        k = _Kernel(fn, opts, kind)
        if _numba is not None:                  # numba up already: made at once
            return k.make(_numba)
        _PENDING.append(k)
        _MADE.append(k)
        return k


def image():
    return _IMAGE


def use_image(im):
    """Kernels not yet made go through im (an _image.Image) where it can."""
    global _IMAGE, _NO
    from . import _image
    _NO = _image.NO
    _IMAGE = im


def every_kernel():
    """Every stand-in made (each keeps its numba kernel in .real once made)."""
    return list(_MADE)


def bind(kernel, args, late):
    """A function calling kernel(*args), the arguments at the places `late`
    (a tuple) given anew each call: for threads sharing out one call of a
    kernel, each with its own scratch."""
    base = list(args)

    def call(*vals):
        a = base[:]
        for k, v in zip(late, vals):
            a[k] = v
        return kernel(*a)
    return call


KERNEL_FILES = ('_bitshuffle', '_blosc', '_blosc2', '_bzip2', '_chunks', '_codecs', '_fletcher32',
                '_fused', '_hdf4_codecs', '_inflate', '_intrin', '_lz4', '_lzf', '_maskscale',
                '_nbit', '_nc3', '_pinflate', '_scaleoffset', '_shuffle', '_szip', '_tree', '_tune',
                '_xarray', '_zfp', '_zstd')     # what compiled kernels can depend on
_KERNEL_STAMP = []


def kernel_stamp():
    """A stamp of what compiled kernels are made from: the kernels of
    KERNEL_FILES and everything at the top level beside them (constants), not
    the plain Python functions and classes those files also hold, nor line
    numbers. An edit of those then compiles nothing again, which numba's own
    stamp, the file's time and size, cannot tell. _intrin counts whole: its
    plain functions write code."""
    if not _KERNEL_STAMP:
        import os
        import re
        import zlib
        here = os.path.dirname(os.path.abspath(__file__))
        h = 0
        for n in KERNEL_FILES:
            with open(os.path.join(here, n + '.py'), encoding='utf-8') as f:
                text = f.read()
            decorated = False
            for part in re.split(r'(?m)^(?=[^\s#)\]}])', text):
                plain = part.startswith(('def ', 'class ')) and not decorated
                decorated = part.startswith('@')
                if not plain or n == '_intrin':
                    h = zlib.crc32(part.encode(), h)
        _KERNEL_STAMP.append('%08x' % h)
    return _KERNEL_STAMP[0]


def _locate_by_kernels():
    """numba's cache for fastnc's kernels kept by kernel_stamp, in the same
    place. (numba's private API, renamed in its time: where it differs, numba's
    own stamp holds, and any edit of these files compiles again. True if
    installed: a test holds this to the numba in use.)"""
    import os
    try:
        from numba.core import caching
        here = os.path.dirname(os.path.abspath(__file__))
        tree = getattr(caching, 'InTreeCacheLocator', None) or caching._InTreeCacheLocator
        impl = getattr(caching, 'CacheImpl', None) or caching._CacheImpl
        if any(c.__name__ == 'Locator' for c in impl._locator_classes):
            return True

        class Locator(tree):
            def get_source_stamp(self):
                return kernel_stamp()

            def get_disambiguator(self):
                return '0'

            @classmethod
            def from_function(cls, py_func, py_file):
                if os.path.dirname(os.path.abspath(py_file)) != here:
                    return None
                return super().from_function(py_func, py_file)
        impl._locator_classes.insert(1, Locator)                # (after NUMBA_CACHE_DIR's)
        return True
    except (ImportError, AttributeError, TypeError):
        return False


def source_stamp():
    """A stamp of the source files with kernels: name, size, modification
    time. (Not the others: an edit of the front end compiles nothing again.)"""
    import os
    import zlib                                     # (hashlib: 4 ms of a fresh read)
    here = os.path.dirname(os.path.abspath(__file__))
    h = 0
    for n in KERNEL_FILES:
        n += '.py'
        st = os.stat(os.path.join(here, n))
        h = zlib.crc32(('%s %d %d;' % (n, st.st_size, st.st_mtime_ns)).encode(), h)
    return '%08x' % h


def _clear_stale_cache():
    """Delete fastnc's numba cache files when any fastnc source changed.

    numba's cache checks only the source file of the kernel itself, not of
    the kernels it calls in other modules: after an edit of _inflate.py (or
    an upgrade -- pip does not remove .nbi/.nbc files), the engine kernel in
    _chunks.py kept running the old inflate from its cache. So kernel_stamp()
    is kept next to the cache; when it differs, the whole cache goes."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        stamp = kernel_stamp()
        cache = os.path.join(here, '__pycache__')
        path = os.path.join(cache, 'fastnc_kernels.stamp')
        try:
            with open(path) as f:
                if f.read() == stamp:
                    return
        except OSError:
            pass
        if os.path.isdir(cache):
            for n in os.listdir(cache):
                if n.endswith(('.nbi', '.nbc')):
                    try:
                        os.remove(os.path.join(cache, n))
                    except OSError:
                        pass
        os.makedirs(cache, exist_ok=True)
        with open(path, 'w') as f:
            f.write(stamp)
    except OSError:
        pass                                        # read-only install: numba's own checks only


def load():
    """numba imported, every pending kernel made, and each fastnc module's
    stand-ins replaced by them."""
    global _numba
    with _LOCK:
        if _numba is None:
            _clear_stale_cache()
            import numba
            _locate_by_kernels()
            for k in _PENDING:
                k.real = k.make(numba)
            _numba = numba
        if not _PENDING:
            return
        # the kernels' own modules (whatever sys.modules says now), and every
        # fastnc module (those holding a kernel imported from another)
        pkg = __name__.rpartition('.')[0]
        spaces = {id(k.fn.__globals__): k.fn.__globals__ for k in _PENDING}
        for name, mod in list(sys.modules.items()):
            if mod is not None and (name == pkg or name.startswith(pkg + '.')):
                spaces[id(vars(mod))] = vars(mod)
        # (the sentinel held in a local: replacing this module's own
        # _STAND_IN global mid-loop would hide every stand-in after it --
        # this module's `prange` among them, which a module imported later
        # then picks up)
        stand_in = _STAND_IN
        for d in spaces.values():
            for key, val in list(d.items()):
                if isinstance(val, _Kernel):
                    d[key] = val.real
                elif val is stand_in and not (d is globals() and key == '_STAND_IN'):
                    d[key] = _numba.prange
        _PENDING.clear()
