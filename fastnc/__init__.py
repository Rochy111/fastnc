"""fastnc: a fast netCDF / HDF5 reader (netCDF4-python and h5py, read-only,
bit-identical values, every core and optionally the GPU)."""
__version__ = '0.2.0'

from ._dataset import Dataset, Dimension, Group, Variable  # noqa: E402,F401
from ._h5py import File  # noqa: E402,F401
from ._mf import MFDataset  # noqa: E402,F401

# netCDF4-python's module-level helpers for reading, so `import fastnc as netCDF4` goes further
default_fillvals = {  # netcdf.h's NC_FILL_* (netCDF4.default_fillvals)
    'S1': '\x00', 'i1': -127, 'u1': 255, 'i2': -32767, 'u2': 65535, 'i4': -2147483647,
    'u4': 4294967295, 'i8': -9223372036854775806, 'u8': 18446744073709551614,
    'f4': 9.969209968386869e+36, 'f8': 9.969209968386869e+36,
    'c8': 9.969209968386869e+36, 'c16': 9.969209968386869e+36}


def chartostring(b, encoding='utf-8'):
    """A char array (last dimension the characters) as strings, one
    dimension fewer (netCDF4.chartostring)."""
    from ._maskscale import _chartostring
    return _chartostring(b, encoding=encoding)


def num2date(*args, **kwargs):
    """cftime.num2date, as netCDF4-python re-exports it (needs cftime)."""
    import cftime
    return cftime.num2date(*args, **kwargs)


def date2num(*args, **kwargs):
    """cftime.date2num, as netCDF4-python re-exports it (needs cftime)."""
    import cftime
    return cftime.date2num(*args, **kwargs)


def date2index(*args, **kwargs):
    """cftime.date2index, as netCDF4-python re-exports it (needs cftime)."""
    import cftime
    return cftime.date2index(*args, **kwargs)
from ._pool import set_threads, threads  # noqa: E402,F401
from ._chunks import warm  # noqa: E402,F401


def hardware():
    """What fastnc found on this machine (CPUs allowed, cores, memory,
    cache, what tune() measured) and the threads it will use."""
    from . import _hw
    return _hw.info()


def tune(gpu=True, verbose=True):
    """Time real decoding on this machine (a few seconds, once per machine)
    and keep what suits it: threads, work per thread, the numba load time,
    the GPU's bands; and link the kernels into an image, so that later
    processes need no numba. Everything works without it."""
    from . import _image, _tune
    res = _tune.tune(gpu=gpu, verbose=verbose)
    _image.build_now()
    return res


def release():
    """Free the memory fastnc keeps between reads: pooled output blocks,
    decoded chunks and regions kept for small reads, threads' scratch and, if the GPU was used, the page-locked upload buffer and cupy's pools."""
    from . import _arena, _chunks, _hold, _pinflate, _quick
    _hold.forget()
    _arena.release()
    _chunks.release_cache()
    _pinflate.release()
    _quick.forget()
    import sys
    if 'fastnc._gpu' in sys.modules:
        sys.modules['fastnc._gpu'].release()
