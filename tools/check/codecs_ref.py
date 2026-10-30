"""Reference chunks for the codec checks: a dataset's stored (filtered)
chunks as h5py reads them, and what the reference filter (hdf5plugin,
through HDF5) decodes each into -- the whole chunk, edge chunks included,
which reading the dataset's values cannot give.

The decoded chunk comes from a one-chunk dataset in a temporary file with
the same creation property list: the stored bytes go in with
write_direct_chunk, and reading the dataset back runs them through the
plugin. Needs h5py and hdf5plugin (the dev environment).
"""
import os
import sys
import tempfile

import h5py
import hdf5plugin  # noqa: F401  (registers the filters)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)


def filters(ds):
    """[(id, flags, cd tuple)] of a dataset's pipeline."""
    pl = ds.id.get_create_plist()
    return [pl.get_filter(i)[:3] for i in range(pl.get_nfilters())]


def cd_array(ds, which=-1):
    """The kernel's cd: [element size, chunk bytes, n, cd...] for one filter."""
    fid, flags, cd = filters(ds)[which]
    nb = ds.dtype.itemsize * int(np.prod(ds.chunks))
    return np.array([ds.dtype.itemsize, nb, len(cd)] + list(cd), np.int64)


def chunk_offsets(ds):
    n = ds.id.get_num_chunks()
    return [ds.id.get_chunk_info(i).chunk_offset for i in range(n)]


_TMP = os.path.join(tempfile.gettempdir(), 'fastnc_codecs_ref_%d.h5' % os.getpid())


def decode_with_plugin(ds, raw, mask=0):
    """The whole chunk `raw` (stored bytes) decoded by the reference
    pipeline, as bytes.

    (A new pipeline is set up with the filters' user values: the stored
    ones pass through each filter's set_local again, and bitshuffle's
    shifts its user values up three slots. The file is closed and opened
    again before reading: read straight after write_direct_chunk, HDF5
    2.x runs the pipeline on a chunk whose mask says to skip it.)"""
    pl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    pl.set_chunk(tuple(ds.chunks))
    for fid, flags, cd in filters(ds):
        pl.set_filter(fid, flags, cd[3:] if fid == 32008 else cd)
    with h5py.File(_TMP, 'w') as f:
        sp = h5py.h5s.create_simple(tuple(ds.chunks))
        d = h5py.h5d.create(f.id, b'd', ds.id.get_type(), sp, dcpl=pl)
        d.write_direct_chunk((0,) * len(ds.chunks), raw, mask)
        d.close()
    with h5py.File(_TMP, 'r') as f:
        out = f['d'][()]
    os.remove(_TMP)
    return out.tobytes()


def chunks(ds):
    """[(offset, stored bytes, filter mask, decoded bytes)] for every chunk."""
    out = []
    for off in chunk_offsets(ds):
        mask, raw = ds.id.read_direct_chunk(off)
        out.append((off, raw, mask, decode_with_plugin(ds, raw, mask)))
    return out


def fast_thread():
    sys.path.insert(0, os.path.join(ROOT, 'tools', 'profile'))
    from common import fast_thread as ft
    ft()
