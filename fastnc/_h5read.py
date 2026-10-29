"""Raw values of an HDF5 dataset for an index: chunked through `_chunks`,
contiguous and compact straight from the mapped file, special types
(variable-length strings and sequences, references) through the parser.

Values come back in native byte order (as netCDF4-python returns them;
h5py keeps the stored order -- the values are the same).
"""
import numpy as np

from . import _chunks, _hdf5, _select


def native(dt):
    """dt with every field in native byte order."""
    if dt.fields is None:
        if dt.subdtype is not None:
            return np.dtype((native(dt.subdtype[0]), dt.subdtype[1]))
        return dt.newbyteorder('=') if dt.byteorder not in '=|' else dt
    return np.dtype({'names': list(dt.names),
                     'formats': [native(dt.fields[n][0]) for n in dt.names],
                     'offsets': [dt.fields[n][1] for n in dt.names],
                     'itemsize': dt.itemsize})


def _simple_swap(dt):
    """True for a plain big-endian number: swapped as it is placed."""
    return dt.fields is None and dt.subdtype is None and dt.byteorder == '>' and dt.itemsize > 1


class Reader:
    """Reads one dataset of an open file (src: the file's bytes as uint8)."""

    def __init__(self, f, ds, src):
        self.f, self.ds, self.src = f, ds, src
        self._index = None

    @property
    def index(self):
        if self._index is None:
            ds = self.ds
            self._index = _chunks.ChunkIndex(ds.chunk_table(), ds.chunks, ds.shape)
        return self._index

    def read(self, key=(), out=None, threads=None):
        ds = self.ds
        t = ds.type
        if ds.null:
            return None
        box, post, drop = _select.normalize(key, ds.shape)
        shape = _select.box_shape(box)
        if t.dtype is None or t.cls in (_hdf5.Datatype.VLEN, _hdf5.Datatype.REFERENCE):
            raw = self._raw_box(box, shape, np.dtype('V%d' % t.size), threads)
            vals = self.f.decode_values(t, shape, raw.tobytes()) if t.cls == _hdf5.Datatype.VLEN else raw
            return _select.finish(vals, post, drop)
        stored = t.dtype
        nat = native(stored)
        swap = _simple_swap(stored)
        direct_out = out is not None and post is None and not drop and out.shape == shape \
            and out.dtype == nat and out.flags.c_contiguous
        res = out if direct_out else np.empty(shape, nat)
        if res.size:
            self._read_into(box, res, swap, threads)
            if stored.byteorder == '>' and not swap:
                res[...] = res.view(stored).astype(nat)     # compound / array types
        res = _select.finish(res, post, drop)
        if out is not None and not direct_out:
            out[...] = res
            return out
        return res

    def _raw_box(self, box, shape, vdt, threads):
        res = np.empty(shape, vdt)
        if res.size:
            self._read_into(box, res, False, threads)
        return res

    def _read_into(self, box, res, swap, threads):
        ds = self.ds
        lay = ds.layout
        f = self.f
        esize = ds.type.size
        if lay.cls == 2:
            _chunks.read(ds, self.index, self.src, box, res, swap=swap, threads=threads)
            return
        if lay.cls == 1:
            if lay.address == f.undef:
                _chunks._fill(res, ds)
                if swap:
                    _swap(res)
                return
            n = int(np.prod(ds.shape)) if ds.shape else 1
            a = f.addr(lay.address)
            whole = np.ndarray(ds.shape, np.dtype('V%d' % esize), buffer=self.src, offset=a)
        elif lay.cls == 0:
            p, n = lay.data
            whole = np.frombuffer(f.b[p:p + n], np.dtype('V%d' % esize)).reshape(ds.shape)
        else:
            raise NotImplementedError('virtual datasets are not supported')
        sl = tuple(slice(s, e, k) for s, e, k in box)
        res.reshape(-1).view(np.dtype('V%d' % esize))[:] = whole[sl].reshape(-1)
        if swap:
            _swap(res)


def _swap(res):
    b = res.reshape(-1).view(np.uint8).reshape(-1, res.dtype.itemsize)
    b[:] = b[:, ::-1]


def read_many(readers, threads=None):
    """Every dataset of `readers` whole, chunked ones decoded together in
    one pass of the pool (one shared output buffer, biggest chunks first:
    a file's many small variables no longer wait on each other)."""
    out = [None] * len(readers)
    batch = []
    total = 0
    for i, r in enumerate(readers):
        ds = r.ds
        t = ds.type
        if (ds.layout.cls != 2 or ds.null or t.dtype is None or not ds.shape
                or t.cls in (_hdf5.Datatype.VLEN, _hdf5.Datatype.REFERENCE)
                or (t.dtype.byteorder == '>' and not _simple_swap(t.dtype))):
            continue
        n = int(np.prod(ds.shape)) * t.size
        batch.append((i, r, total, n))
        total += (n + 63) // 64 * 64                    # each array 64-byte aligned
    buf = np.empty(total, np.uint8)
    items = []
    for i, r, base, n in batch:
        ds = r.ds
        nat = native(ds.type.dtype)
        a = buf[base:base + n].view(nat).reshape(ds.shape)
        out[i] = a
        if n:
            items.append(_chunks.Item(ds, r.index, [(0, s, 1) for s in ds.shape], a,
                                      _simple_swap(ds.type.dtype), base))
    if items:
        _chunks.read_batch(items, readers[0].src, buf, threads)
    for i, r in enumerate(readers):
        if out[i] is None:
            out[i] = r.read((), threads=threads)
    return out
