"""fastnc: a fast netCDF / HDF5 reader (netCDF4-python and h5py, read-only,
bit-identical values, every core and optionally the GPU)."""
__version__ = '0.1.0.dev0'

from ._dataset import Dataset, Dimension, Group, Variable  # noqa: E402,F401
from ._h5py import File  # noqa: E402,F401
from ._pool import set_threads, threads  # noqa: E402,F401
from ._chunks import warm  # noqa: E402,F401


def release():
    """Free the memory fastnc keeps between reads: pooled output blocks and,
    if the GPU was used, the page-locked upload buffer and cupy's pools."""
    from . import _arena
    _arena.release()
    import sys
    if 'fastnc._gpu' in sys.modules:
        sys.modules['fastnc._gpu'].release()
