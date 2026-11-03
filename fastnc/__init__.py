"""fastnc: a fast netCDF / HDF5 reader (netCDF4-python and h5py, read-only,
bit-identical values, every core and optionally the GPU)."""
__version__ = '0.1.0'

from ._dataset import Dataset, Dimension, Group, Variable  # noqa: E402,F401
from ._h5py import File  # noqa: E402,F401
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
