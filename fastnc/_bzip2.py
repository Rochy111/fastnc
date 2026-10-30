"""bzip2 (HDF5 filter 307, hdf5plugin's / PyTables' H5Zbzip2.c): each chunk
one bzip2 stream; cd: [block size 1-9], which decoding does not need.

Decoded with Python's bz2 module (libbzip2), which releases the GIL while
it works, so the engine's threads decode chunks side by side. A numba
decoder was measured against it and did not win (docs in the report:
libbzip2's inverse BWT is bound by cache misses, as a numba one is), and
bzip2 is rare in HDF5 files; see tools/check/filters_hdf_bzip2.py.

Like H5Zbzip2.c, only the first stream counts (bytes after its end are
ignored), and a stream that ends early is an error.
"""
import bz2

import numpy as np


def decode_py(data, out_nbytes, cd):
    """The bzip2 chunk `data` (bytes-like) decoded: bytes, at most
    out_nbytes long. ValueError for a damaged stream, a truncated one, or
    one that decodes to more than out_nbytes."""
    d = bz2.BZ2Decompressor()
    try:
        out = d.decompress(data, max_length=out_nbytes + 1)
    except OSError as e:
        raise ValueError('bzip2: bad stream (%s)' % e)
    if len(out) > out_nbytes:
        raise ValueError('bzip2: chunk decodes to more than %d bytes' % out_nbytes)
    if not d.eof:
        raise ValueError('bzip2: stream ends early')
    return out


def decode_into(data, dst, o, oend):
    """decode_py into dst[o:oend]; the end of the output."""
    out = decode_py(data, oend - o, None)
    n = len(out)
    dst[o:o + n] = np.frombuffer(out, np.uint8)
    return o + n


def workspace_size(chunk_nbytes):
    return 0
