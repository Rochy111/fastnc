"""Inflate every chunk of a file without numba: zlib.decompress vs zlib's
uncompress vs libdeflate through ctypes, straight into a buffer; 1 and all
threads."""
import sys, time, mmap, zlib
import numpy as np
from common import fast_thread, path
sys.path.insert(0, path('.'))
from fastnc import _hdf5, _nc4, _zlib, _pool, _chunks

fast_thread()
f = sys.argv[1] if len(sys.argv) > 1 else 'VJ102IMG_NRT.A2026265.2006.021.nc'
fh = open(path(f), 'rb'); mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
src = np.frombuffer(mm, np.uint8)
nc = _nc4.NC4File(_hdf5.H5File(mm), src)
jobs = []
def walk(g):
    for v in g.variables.values():
        ds = v.ds
        if ds.filters and ds.filters[0].id == 1 and len(ds.filters) <= 2 and getattr(ds, 'chunks', None):
            idx = ds.chunk_index() if hasattr(ds, 'chunk_index') else None
    for s in g.groups.values(): walk(s)
# simpler: use the reader's plans
items = []
for g in [nc.root]:
    pass
