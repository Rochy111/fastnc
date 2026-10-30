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
_LOCK = threading.RLock()
_numba = None                   # the numba module, once imported


class _Kernel:
    """A stand-in for numba.njit(**opts)(fn) or numba.extending.intrinsic(fn)."""

    def __init__(self, fn, opts, kind):
        self.fn, self.opts, self.kind, self.real = fn, opts, kind, None
        self.__wrapped__, self.__name__ = fn, fn.__name__
        self.__doc__ = fn.__doc__

    def make(self, numba):
        if self.kind == 'intrinsic':
            from numba.extending import intrinsic as make
            return make(self.fn)
        return numba.njit(**self.opts)(self.fn)

    def __call__(self, *args, **kwargs):
        if self.real is None:
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
        return k


def _clear_stale_cache():
    """Delete fastnc's numba cache files when any fastnc source changed.

    numba's cache checks only the source file of the kernel itself, not of
    the kernels it calls in other modules: after an edit of _inflate.py (or
    an upgrade -- pip does not remove .nbi/.nbc files), the engine kernel in
    _chunks.py kept running the old inflate from its cache. So a stamp of
    every fastnc source file (name, size, modification time) is kept next
    to the cache; when it differs, the whole cache goes."""
    import hashlib
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        names = sorted(n for n in os.listdir(here) if n.endswith('.py'))
        h = hashlib.sha1()
        for n in names:
            st = os.stat(os.path.join(here, n))
            h.update(('%s %d %d;' % (n, st.st_size, st.st_mtime_ns)).encode())
        stamp = h.hexdigest()
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
