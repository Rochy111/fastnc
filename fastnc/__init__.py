"""fastnc: a fast netCDF / HDF5 reader (netCDF4-python and h5py, read-only,
bit-identical values, every core and optionally the GPU)."""
__version__ = '0.1.0.dev0'

from ._dataset import Dataset, Dimension, Group, Variable  # noqa: E402,F401
from ._h5py import File  # noqa: E402,F401
from ._pool import set_threads, threads  # noqa: E402,F401
