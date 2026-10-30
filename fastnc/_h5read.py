"""Raw values of an HDF5 dataset for an index: chunked through `_chunks`,
contiguous and compact straight from the mapped file, special types
(variable-length strings and sequences, references) through the parser.

Values come back in native byte order (as netCDF4-python returns them;
h5py keeps the stored order -- the values are the same).
"""
import numpy as np

from . import _arena, _chunks, _hdf5, _select


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


# IEEE layouts (sign, exponent position, size, mantissa position, size, bias, implied)
_IEEE = {2: (15, 10, 5, 0, 10, 15, 2), 4: (31, 23, 8, 0, 23, 127, 2), 8: (63, 52, 11, 0, 52, 1023, 2)}


def reduced(t):
    """True for a type whose stored bits are not the native type's: an
    integer narrower than its size or off bit 0, a float with a non-IEEE
    layout (n-bit data mostly), or a compound / array holding one."""
    if t.cls in (0, 4):
        return t.precision < 8 * t.size or t.offset != 0
    if t.cls == 1:
        return t.fields is not None and t.size in _IEEE and (
            t.fields != _IEEE[t.size] or t.offset != 0 or t.precision != 8 * t.size)
    if t.cls == 6:
        return any(reduced(m[2]) for m in t.members)
    if t.cls in (8, 10):
        return reduced(t.base)
    return False


def convert_reduced(a, t):
    """Values of a reduced type as HDF5 converts them to the native type
    (what h5py and netCDF-c return): integers shifted down by their bit
    offset and sign-extended from their precision, floats re-encoded from
    their sign / exponent / mantissa fields into IEEE."""
    if t.cls == 6:
        out = a.astype(native(a.dtype))
        for name, off, mt in t.members:
            if reduced(mt):
                out[name] = convert_reduced(a[name], mt)
        return out
    if t.cls in (8, 10):
        return convert_reduced(a, t.base)
    a = a.astype(a.dtype.newbyteorder('=')) if a.dtype.byteorder == '>' else a
    u = a.view('u%d' % t.size).astype(np.uint64)
    if t.cls in (0, 4):
        mask = np.uint64((1 << t.precision) - 1)
        v = (u >> np.uint64(t.offset)) & mask
        if t.signed:
            sign = np.uint64(1 << (t.precision - 1))
            v = np.where(v & sign, v | ~mask, v)    # sign-extended, in two's complement
        return v.astype(np.uint64).view(np.int64).astype(a.dtype) if t.signed else v.astype(a.dtype)
    return _float_bits(u, t).astype(a.dtype)


def float_target(t):
    """The native float a custom float converts to (h5py's choice): the
    smallest of float32 / float64 at least its size (and holding its fields)."""
    spos, epos, esize, mpos, msize, bias, norm = t.fields
    return np.dtype('f4') if t.size <= 4 and esize <= 8 and msize <= 23 else np.dtype('f8')


def odd_float(raw, t):
    """Values of a float type numpy has no dtype for (an odd size),
    from its raw bytes (V-dtype array)."""
    b = raw.reshape(-1).view(np.uint8).reshape(-1, t.size)
    if t.order == '>':
        b = b[:, ::-1]
    pad = np.zeros((b.shape[0], 8), np.uint8)
    pad[:, :t.size] = b
    u = pad.view('<u8').reshape(raw.shape)
    return _float_bits(u, t).astype(float_target(t))


def _float_bits(u, t):
    spos, epos, esize, mpos, msize, bias, norm = t.fields
    s = ((u >> np.uint64(spos)) & np.uint64(1)).astype(bool)
    e = ((u >> np.uint64(epos)) & np.uint64((1 << esize) - 1)).astype(np.int64)
    m = ((u >> np.uint64(mpos)) & np.uint64((1 << msize) - 1)).astype(np.float64)
    emax = (1 << esize) - 1
    frac = m / float(1 << msize)
    lead = 1.0 if norm == 2 else 0.0
    val = np.where(e == 0, np.ldexp(frac, 1 - bias), np.ldexp(lead + frac, e - bias))
    val = np.where(e == emax, np.where(m == 0, np.inf, np.nan), val)
    return np.where(s, -val, val)


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
            if ds.layout.cls == 1:
                self._index = _virtual_index(ds)
            else:
                arrays = ds.chunk_arrays()
                if arrays is not None:
                    self._index = _chunks.ChunkIndex(None, ds.chunks, ds.shape, arrays)
                else:
                    self._index = _chunks.ChunkIndex(ds.chunk_table(), ds.chunks, ds.shape)
        return self._index

    @property
    def virtual(self):
        """True for a big contiguous dataset read as virtual chunks through
        the chunk engine (parallel, byte swap and selection fused)."""
        ds = self.ds
        return (ds.layout.cls == 1 and ds.layout.address != self.f.undef and len(ds.shape) >= 1
                and ds.type.dtype is not None and not ds.null
                and int(np.prod(ds.shape)) * ds.type.size >= VIRTUAL_MIN)

    def read(self, key=(), out=None, threads=None, scalars=True):
        """Raw values for key. scalars=False keeps a fully indexed result as
        a 0-d array (for callers that convert types before unwrapping)."""
        ds = self.ds
        t = ds.type
        if ds.null:
            return None
        box, post, drop = _select.normalize(key, ds.shape)
        shape = _select.box_shape(box)
        if t.cls == _hdf5.Datatype.FLOAT and t.dtype is None and t.fields is not None and t.size <= 8:
            raw = self._raw_box(box, shape, np.dtype('V%d' % t.size), threads)
            return _select.finish(odd_float(raw, t), post, drop, scalars)
        if t.dtype is None or t.cls in (_hdf5.Datatype.VLEN, _hdf5.Datatype.REFERENCE):
            raw = self._raw_box(box, shape, np.dtype('V%d' % t.size), threads)
            vals = self.f.decode_values(t, shape, raw.tobytes()) if t.cls == _hdf5.Datatype.VLEN else raw
            return _select.finish(vals, post, drop, scalars)
        stored = t.dtype
        nat = native(stored)
        swap = _simple_swap(stored)
        if t.cls in (_hdf5.Datatype.COMPOUND, _hdf5.Datatype.ARRAY) and (reduced(t) or nat != stored):
            # big-endian or reduced members: the stored bytes, then one conversion
            raw = self._raw_box(box, shape, np.dtype('V%d' % t.size), threads)
            vals = np.frombuffer(raw.tobytes(), stored)
            vals = vals.reshape(tuple(shape) + vals.shape[1:])
            vals = convert_reduced(vals, t) if reduced(t) else vals.astype(nat)
            vals = _select.finish(vals, post, drop, scalars)
            if out is not None:
                out[...] = vals
                return out
            return vals
        direct_out = out is not None and post is None and not drop and out.shape == shape \
            and out.dtype == nat and out.flags.c_contiguous
        res = out if direct_out else _arena.empty(shape, nat)
        if res.size:
            self._read_into(box, res, swap, threads)
            if stored.byteorder == '>' and not swap:
                res[...] = res.view(stored).astype(nat)     # compound / array types
        if reduced(t) and res.size:
            res = convert_reduced(res, t)
        res = _select.finish(res, post, drop, scalars)
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
        if lay.cls == 2 or (lay.cls == 1 and self.virtual):
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
            if f.source is not None:                    # a remote file: fetch the selection's runs
                f.source.fetch(*contiguous_runs(a, ds.shape, esize, box))
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


VIRTUAL_MIN = 256 << 10         # contiguous datasets this big go through the engine
VIRTUAL_CHUNK = 1 << 20


def _virtual_index(ds):
    """A contiguous dataset as chunks of about a megabyte, each one run of
    the file: whole trailing dimensions, and a count along the first
    dimension whose trailing part fits (dimensions before it taken one at
    a time). Their addresses follow from the shape."""
    shape = tuple(ds.shape)
    esize = ds.type.size
    rank = len(shape)
    k = 0
    inner = esize * int(np.prod(shape[1:]))
    while k < rank - 1 and inner > VIRTUAL_CHUNK:
        k += 1
        inner = esize * int(np.prod(shape[k + 1:]))
    chunks = [1] * k + [max(1, VIRTUAL_CHUNK // max(inner, 1))] + list(shape[k + 1:])
    chunks[k] = min(chunks[k], shape[k]) or 1
    grid = [-(-s // c) for s, c in zip(shape, chunks)]
    n = int(np.prod(grid))
    idx = np.indices(grid).reshape(rank, n).T * np.array(chunks)       # chunk origins
    strides = np.ones(rank, np.int64)
    for d in range(rank - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    base = ds.file.addr(ds.layout.address)
    addr = base + (idx * strides).sum(axis=1) * esize
    csize = int(np.prod(chunks)) * esize
    table = [(tuple(o), int(a), csize, 0) for o, a in zip(idx.tolist(), addr.tolist())]
    return _chunks.ChunkIndex(table, tuple(chunks), shape)


def contiguous_runs(a, shape, esize, box, most=1 << 20):
    """(starts, sizes) in bytes of the runs of a contiguous array at byte a
    that a box selects: one per row along the last dimension (stepped
    selections: from the first to the last element of each row), or one
    range over all when there would be more than `most`."""
    rank = len(shape)
    if rank == 0:
        return np.array([a], np.int64), np.array([esize], np.int64)
    stride = [esize] * rank
    for d in range(rank - 2, -1, -1):
        stride[d] = stride[d + 1] * shape[d + 1]
    first = [s for s, e, k in box]
    last = [s + (len(range(s, e, k)) - 1) * k for s, e, k in box]
    nrows = 1
    for s, e, k in box[:-1]:
        nrows *= len(range(s, e, k))
    run = (last[-1] - first[-1] + 1) * esize
    if nrows > most:
        lo = a + sum(f * st for f, st in zip(first, stride))
        hi = a + sum(l * st for l, st in zip(last, stride)) + esize
        return np.array([lo], np.int64), np.array([hi - lo], np.int64)
    off = np.zeros(1, np.int64) + a + first[-1] * esize
    for d in range(rank - 1):
        s, e, k = box[d]
        off = (off[:, None] + np.arange(s, e, k, dtype=np.int64)[None, :] * stride[d]).ravel()
    return off, np.full(len(off), run, np.int64)


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
        if ((ds.layout.cls != 2 and not r.virtual) or ds.null or t.dtype is None or not ds.shape
                or t.cls in (_hdf5.Datatype.VLEN, _hdf5.Datatype.REFERENCE)
                or (t.dtype.byteorder == '>' and not _simple_swap(t.dtype))):
            continue
        n = int(np.prod(ds.shape)) * t.size
        batch.append((i, r, total, n))
        total += (n + 63) // 64 * 64                    # each array 64-byte aligned
    buf = _arena.empty((total,), np.uint8)
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
