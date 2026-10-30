"""Shared by the filter checks (tools/check/filters_hdf_*.py) and the test
maker: fastnc's decoders run over a dataset's chunks, a whole filter
pipeline at a time, and the reference -- HDF5 itself, through h5py --
for the same chunks.

The reference for a chunk is HDF5's decoding of all of it, not only the
part inside the dataset: its raw bytes (read_direct_chunk) written with
write_direct_chunk into a dataset of the chunk's shape made with the same
creation property list, and read back whole. HDF5 recomputes the filters'
local parameters (szip's scanline, scale-offset's element count ...) for
that dataset exactly as for the original, since they depend only on the
chunk shape, type and fill value.
"""
import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from fastnc import _hdf5, _inflate                            # noqa: E402
from fastnc import _bzip2, _fletcher32, _nbit, _scaleoffset, _shuffle, _szip  # noqa: E402

MODULES = {2: _shuffle, 3: _fletcher32, 4: _szip, 5: _nbit, 6: _scaleoffset, 307: _bzip2}


def walk(f, g=None, path='', out=None):
    """{path: _hdf5.Dataset} of every dataset under group g."""
    if out is None:
        out = {}
    g = f.root if g is None else g
    for name, (kind, target) in g.links.items():
        if kind != 'hard':
            continue
        p = path + '/' + name
        h = f.object_header(target)
        if h.first(8) is not None:
            out[p] = _hdf5.Dataset(f, target, name, h)
        elif h.first(3) is None:
            walk(f, _hdf5.Group(f, target, name, h), p, out)
    return out


def cd_array(ds, filt):
    """The contract's cd: [element size, chunk bytes, n, cd...]."""
    chunk_nbytes = int(np.prod(ds.chunks)) * ds.type.size
    return np.array([ds.type.size, chunk_nbytes, len(filt.cd)] + list(filt.cd), np.int64)


def decode_chunk(src, ds, addr, size, mask, verify=True):
    """A chunk's stored bytes src[addr:addr+size] through the reverse of the
    dataset's pipeline: (decoded bytes as a uint8 array, or an error code)."""
    chunk_nbytes = int(np.prod(ds.chunks)) * ds.type.size
    buf, p, pend = src, addr, addr + size
    for i in range(len(ds.filters) - 1, -1, -1):
        filt = ds.filters[i]
        if mask & (1 << i):
            continue
        cd = cd_array(ds, filt)
        room = max(chunk_nbytes, pend - p) + 64
        out = np.zeros(room, np.uint8)
        if filt.id == 1:
            tab, iws = _inflate.workspace()
            end = _inflate.zlib_inflate(buf, p, pend, out, 0, room, room, tab, iws)
        else:
            mod = MODULES[filt.id]
            if hasattr(mod, 'decode'):
                ws = np.zeros(max(1, mod.workspace_size(chunk_nbytes)), np.uint8)
                if filt.id == 3 and not verify:
                    end = mod.strip(buf, p, pend, out, 0, room, cd, ws)
                else:
                    end = mod.decode(buf, p, pend, out, 0, room, cd, ws)
            else:
                r = mod.decode_py(bytes(buf[p:pend]), chunk_nbytes, cd)
                r = np.frombuffer(r, np.uint8)
                out[:r.size] = r
                end = r.size
        if end < 0:
            return end
        buf, p, pend = out, 0, end
    return np.array(buf[p:pend])


def load(path):
    """(file bytes as uint8, _hdf5.H5File, {path: Dataset})."""
    with open(path, 'rb') as fh:
        b = np.frombuffer(fh.read(), np.uint8)
    f = _hdf5.H5File(b)
    return b, f, walk(f)


# -- HDF5's own decoding, through h5py

_RELAX = None


def relaxed_fapl(core=False):
    """A file access property list letting HDF5 >= 1.14.4 create and open
    datatypes with unused bits (n-bit types): H5Pset_relax_file_integrity_
    checks, which h5py does not wrap, called through ctypes on the HDF5
    library h5py uses."""
    import h5py
    global _RELAX
    if _RELAX is None:
        cands = [os.path.join(os.path.dirname(h5py.__file__), 'hdf5.dll'),
                 os.path.join(sys.prefix, 'Library', 'bin', 'hdf5.dll')]
        for c in cands:
            if os.path.exists(c):
                lib = ctypes.CDLL(c)
                fn = getattr(lib, 'H5Pset_relax_file_integrity_checks', None)
                if fn is not None:
                    fn.argtypes = [ctypes.c_int64, ctypes.c_uint64]
                    fn.restype = ctypes.c_int
                    _RELAX = fn
                    break
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    if core:
        fapl.set_fapl_core(backing_store=False)
    if _RELAX is not None:
        _RELAX(fapl.id, 1)
    return fapl


def h5open(path):
    import h5py
    try:
        import hdf5plugin                       # noqa: F401 (registers bzip2 & co)
    except ImportError:
        pass
    return h5py.File(h5py.h5f.open(os.fsencode(path), h5py.h5f.ACC_RDONLY, fapl=relaxed_fapl()))


_tmp_count = [0]


def ref_chunk(dset, offset):
    """HDF5's decoding of the whole chunk at `offset` of h5py dataset dset,
    as raw bytes of the stored type (no conversion)."""
    import h5py
    import tempfile
    mask, raw = dset.id.read_direct_chunk(offset)
    _tmp_count[0] += 1
    # (on disk, closed and opened again: read in the same session HDF5 1.14
    # forgets a chunk's filter mask set by write_direct_chunk)
    name = os.path.join(tempfile.gettempdir(), 'fastnc_ref_%d_%d.h5' % (os.getpid(), _tmp_count[0]))
    fid = h5py.h5f.create(os.fsencode(name), h5py.h5f.ACC_TRUNC, fapl=relaxed_fapl())
    try:
        dcpl = dset.id.get_create_plist()
        ftype = dset.id.get_type()
        space = h5py.h5s.create_simple(dset.chunks)
        did = h5py.h5d.create(fid, b'c', ftype, space, dcpl)
        did.write_direct_chunk((0,) * len(dset.chunks), raw, mask)
        did.close()
        fid.close()
        fid = h5py.h5f.open(os.fsencode(name), h5py.h5f.ACC_RDONLY, fapl=relaxed_fapl())
        did = h5py.h5d.open(fid, b'c')
        out = np.zeros(int(np.prod(dset.chunks)) * ftype.get_size(), np.uint8)
        try:
            did.read(h5py.h5s.ALL, h5py.h5s.ALL, out, mtype=did.get_type())
        finally:
            did.close()
        return out
    finally:
        fid.close()
        os.remove(name)
