"""Raw values of an HDF5 dataset for an index: chunked through `_chunks`,
contiguous and compact straight from the mapped file, special types
(variable-length strings and sequences, references) through the parser,
virtual datasets and external storage through `_vds`.

Values come back in native byte order (as netCDF4-python returns them;
h5py keeps the stored order -- the values are the same).
"""
import math
import threading
from itertools import product

import numpy as np

from . import _arena, _chunks, _hdf5, _hold, _select, _tree


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


_IEEE = _hdf5.IEEE


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
    return _float_bits(u, t).astype(float_target(t))


def float_target(t):
    """The native float a custom float converts to (h5py's choice): the
    smallest that is at least its size and holds its exponents and mantissa."""
    spos, epos, esize, mpos, msize, bias, norm = t.fields
    for dt, mantissa, top in (('f2', 10, 16), ('f4', 23, 128), ('f8', 52, 1024)):
        if np.dtype(dt).itemsize >= t.size and msize <= mantissa and 1 << (esize - 1) <= top:
            return np.dtype(dt)
    return np.dtype('f8')


def int_target(t):
    """The native integer an integer of an odd width converts to: the next that holds it."""
    return np.dtype('%s%d' % ('i' if t.signed else 'u', next(n for n in (1, 2, 4, 8) if n >= t.size)))


def odd_int(raw, t):
    """Integers of 3, 5, 6 or 7 bytes, from their raw bytes (a V-dtype array)."""
    b = raw.reshape(-1).view(np.uint8).reshape(-1, t.size)
    if t.order == '>':
        b = b[:, ::-1]
    pad = np.zeros((b.shape[0], 8), np.uint8)
    pad[:, :t.size] = b
    if t.signed:
        pad[b[:, -1] >= 128, t.size:] = 255
    return pad.view('<i8' if t.signed else '<u8').reshape(raw.shape).astype(int_target(t))


def _bit_field(b, at, n):
    """n <= 64 bits from byte `at // 8`, bit `at % 8` of rows of little-endian bytes."""
    first, shift = at // 8, at % 8
    last = (at + n - 1) // 8
    out = np.zeros(len(b), np.uint64)
    for k in range(first, min(last, first + 7) + 1):
        out |= b[:, k].astype(np.uint64) << np.uint64(8 * (k - first))
    out >>= np.uint64(shift)
    if last > first + 7:                            # (it straddles a ninth byte)
        out |= b[:, last].astype(np.uint64) << np.uint64(64 - shift)
    return out if n == 64 else out & np.uint64((1 << n) - 1)


def odd_float(raw, t):
    """Values of a float type numpy has no dtype for (an odd size, wider than
    a double, or a VAX's byte order), from its raw bytes (V-dtype array)."""
    b = raw.reshape(-1).view(np.uint8).reshape(-1, t.size)
    if t.order == '>':
        b = b[:, ::-1]
    elif t.order == 'V':                            # words of two bytes, the last word first
        b = b[:, np.arange(t.size).reshape(-1, 2)[::-1].ravel()]
    if t.size > 8:
        return _wide_float(b, t).reshape(raw.shape)
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
    if t.order == 'V':                              # (no infinities or NaNs: an exponent of 0 is 0)
        val = np.where(e == 0, 0.0, np.ldexp(lead + frac, e - bias))
    else:
        val = np.where(e == emax, np.where(m == 0, np.inf, np.nan), val)
    return np.where(s, -val, val)


def _wide_float(b, t):
    """Floats wider than a double as doubles: x87's 80 bits (a mantissa with its
    leading bit, msb-set) and binary128 (without, implied), whatever their padding."""
    spos, epos, esize, mpos, msize, bias, norm = t.fields
    s = _bit_field(b, spos, 1).astype(bool)
    e = _bit_field(b, epos, esize).astype(np.int64)
    n = min(msize, 64)
    m = _bit_field(b, mpos + msize - n, n).astype(np.float64)
    frac = m / 2.0 ** n
    emax = (1 << esize) - 1
    # (as doubles the exponent is clipped: past a double's range is an infinity or zero anyway)
    shift = np.clip(e - bias, -2000, 2000)
    val = np.ldexp(1.0 + frac if norm == 2 else 2.0 * frac, shift)
    val = np.where(e == 0, np.ldexp(frac if norm == 2 else 2.0 * frac, int(max(1 - bias, -2000))), val)
    val = np.where(e == emax, np.where(frac == 0 if norm == 2 else 2.0 * frac == 1.0, np.inf, np.nan), val)
    return np.where(s, -val, val)


def _simple_swap(dt):
    """True for a plain big-endian number: swapped as it is placed (not a
    complex one: two numbers, each swapped by itself)."""
    return dt.fields is None and dt.subdtype is None and dt.byteorder == '>' and dt.itemsize > 1 and dt.kind != 'c'


def _small_read(ds, index, src, box, out, swap):
    from ._quick import small_read                  # (it needs this module)
    return small_read(ds, index, src, box, out, swap)


SCATTERED = 4096                # most places of a list read one by one
WANTS = 4                       # ... where its box has this many times what it wants
BIG_BOX = 1 << 24               # ... and is this big: below that, a copy of it costs less
LOOK_FROM = 256                 # chunks in a dataset from which its index is looked into, not read
LOOK_MOST = 64                  # chunks a read may look up before the whole table is worth it
LOOK_SHARE = 6                  # ... or this share of the chunks, as a shift (a lookup: ~60 entries of a table)


class _Few:
    """The chunks of a box, looked up, as _chunks' plans want an index."""
    complete = False

    def __init__(self, cshape, chunks, offsets, addr, size, mask):
        self.cshape, self.chunks = cshape, chunks
        self.offsets, self.addr, self.size, self.mask = offsets, addr, size, mask


class LazyIndex(_chunks.ChunkIndex):
    """A chunk index looked into, not read. A dataset's chunks can number
    millions, and reading all of them for one (60 ms for a million in a B-tree,
    against 0.1 for h5py going down it) grew with the file. Here the table is
    made when something asks for it (a read of many chunks, or many reads of
    a few: past a share of the chunks, lookups cost what the table does), and
    until then a read of a few finds those. `index.rows[n]` is the row in
    `index.looked` of the chunk numbered n on the grid, or -1 (for reads of
    plain values, see _quick)."""

    def __init__(self, ds, find):
        self._frame(ds.chunks, ds.shape)
        self._ds, self._find = ds, find
        self.seen = {}                              # grid place -> what _find says
        self.looked = []                            # addresses of the rows given out: only added to
        self.rows = _Rows(self)
        self.lock = threading.RLock()

    def __getattr__(self, name):                    # (only what has not been set)
        if name in ('offsets', 'addr', 'size', 'mask', 'complete'):
            with self.lock:
                if name not in self.__dict__:
                    ds = self._ds
                    arrays = ds.chunk_arrays()
                    self._tables(ds.chunk_table() if arrays is None else None, arrays)
                    self.seen = None
                return self.__dict__[name]
        raise AttributeError(name)

    def _at(self, g):
        seen = self.seen
        if seen is None:                            # (the table made, a reader of rows still asks)
            return self._row(g)
        got = seen.get(g, 0)
        if got == 0:
            if len(seen) >= max(LOOK_MOST >> 2, self.ngrid >> LOOK_SHARE):
                self.addr                           # (this many asked, more will be: as dear as the table)
                return self._row(g)
            got = seen[g] = self._find(tuple(x * c for x, c in zip(g, self.cshape)))
        return got

    def _row(self, g):
        t = self.table
        if t is None:
            t = self.table = self._grid_table()
        if t is False:
            return self._find(tuple(x * c for x, c in zip(g, self.cshape)))
        lin = 0
        for x, n in zip(g, self.gridlist):
            lin = lin * n + x
        row = t[lin]
        return None if row < 0 else (int(self.addr[row]), int(self.size[row]), int(self.mask[row]))

    def part(self, sel):
        if self.seen is None:
            return None
        count = len(self.seen) or 1
        spans = []
        most = max(LOOK_MOST, self.ngrid >> LOOK_SHARE)
        for (s, e, k), c, n in zip(sel, self.cshape, self.gridlist):
            a, b = s // c, min((e - 1) // c, n - 1)
            if k != 1 or a > b:
                return None
            count *= b - a + 1
            if count > most:
                return None
            spans.append(range(a, b + 1))
        offsets, addr, size, mask = [], [], [], []
        for g in product(*spans):
            got = self._at(g)
            if self.seen is None:
                return None                         # (the table was made: use it)
            if got is not None:
                offsets.extend(x * c for x, c in zip(g, self.cshape))
                addr.append(got[0])
                size.append(got[1])
                mask.append(got[2])
        rank = len(self.cshape)
        return _Few(self.cshape, self.chunks, np.array(offsets, np.int64).reshape(-1, rank),
                    np.array(addr, np.int64), np.array(size, np.int64), np.array(mask, np.int64))



class _Rows(dict):
    """Rows of a LazyIndex's chunks by their number on the grid: in the kind
    of loop this is for, found once and then asked for over and over, at the
    speed of a dict. Threads read both without a lock, so a row is given out
    only once its address is there, and a row given out stays right for good."""

    def __init__(self, index):
        self.index = index

    def __missing__(self, lin):
        index = self.index
        with index.lock:
            row = self.get(lin)
            if row is not None:
                return row
            if index.seen is None:                  # (the table is made: all of them, in one go)
                t = index.table
                if t is None:
                    t = index.table = index._grid_table()
                if t is not False:
                    base = len(index.looked)
                    index.looked.extend(index.addr.tolist())
                    self.update(enumerate(np.where(t < 0, -1, t + base).tolist()))
                    return self[lin]
            g = []
            n = lin
            for m in reversed(index.gridlist):
                n, x = divmod(n, m)
                g.append(x)
            got = index._at(tuple(reversed(g)))
            if got is not None:
                index.looked.append(got[0])
            row = self[lin] = -1 if got is None else len(index.looked) - 1
            return row


class Reader:
    """Reads one dataset of an open file (src: the file's bytes as uint8)."""

    def __init__(self, f, ds, src):
        self.f, self.ds, self.src = f, ds, src
        self._index = None
        self._q = None                              # (see _boxed)
        self.asked = False                          # (see _quick.reader)
        self.strings = str                          # what variable-length strings come as (h5py's: bytes)
        self.plain = ()                             # classes of type _quick may give as they lie, but numbers
        self.record = None                          # ... and records in this dtype: netCDF's, the same but aligned
        f.readers.append(self)

    @property
    def index(self):
        if self._index is None:
            ds = self.ds
            if self.src is None:
                raise ValueError('the file of dataset %s is closed' % ds.name)
            find = None
            if ds.layout.cls == 1:
                self._index = _virtual_index(ds)
                return self._index
            if math.prod(-(-s // c) for s, c in zip(ds.shape, ds.chunks)) >= LOOK_FROM and not ds.layout.flags & 1:
                find = self.f.chunk_finder(ds)
            if find is not None:
                self._index = LazyIndex(ds, find)
            else:
                arrays = ds.chunk_arrays()
                if arrays is not None:
                    self._index = _chunks.ChunkIndex(None, ds.chunks, ds.shape, arrays)
                else:
                    self._index = _chunks.ChunkIndex(ds.chunk_table(), ds.chunks, ds.shape)
                if ds.layout.flags & 1 and ds.filters:      # partial edge chunks stored unfiltered
                    i = self._index
                    i.mask = i.mask.copy()
                    i.mask[((i.offsets + i.chunks) > np.array(ds.shape)).any(axis=1)] = (1 << len(ds.filters)) - 1
        return self._index

    @property
    def virtual(self):
        """True for a big contiguous dataset read as virtual chunks through
        the chunk engine (parallel, byte swap and selection fused)."""
        ds = self.ds
        return (ds.layout.cls == 1 and ds.layout.address != self.f.undef and len(ds.shape) >= 1
                and ds.type.dtype is not None and not ds.null
                and math.prod(ds.shape) * ds.type.size >= VIRTUAL_MIN)

    def read(self, key=(), out=None, threads=None, scalars=True):
        """Raw values for key. scalars=False keeps a fully indexed result as
        a 0-d array (for callers that convert types before unwrapping)."""
        ds = self.ds
        t = ds.type
        if ds.null:
            return None
        if ds.layout.cls == 3:                          # virtual: its sources' values
            from . import _vds
            return _vds.read(self, key, out, threads, scalars)
        box, post, drop = _select.normalize(key, ds.shape)
        shape = _select.box_shape(box)
        if t.cls == _hdf5.Datatype.FLOAT and t.dtype is None and t.fields is not None:
            raw = self._raw_box(box, shape, np.dtype('V%d' % t.size), threads)
            return _select.finish(odd_float(raw, t), post, drop, scalars)
        if t.cls in (_hdf5.Datatype.FIXED, _hdf5.Datatype.BITFIELD) and t.dtype is None and t.size < 8:
            raw = self._raw_box(box, shape, np.dtype('V%d' % t.size), threads)
            return _select.finish(odd_int(raw, t), post, drop, scalars)
        if t.dtype is None or t.cls in (_hdf5.Datatype.VLEN, _hdf5.Datatype.REFERENCE):
            raw = self._raw_box(box, shape, np.dtype('V%d' % t.size), threads)
            vals = self.f.decode_values(t, shape, raw.tobytes(), self.strings) if t.holds_vlen() else raw
            return _select.finish(vals, post, drop, scalars)
        stored = t.dtype
        nat = native(stored)
        swap = _simple_swap(stored)
        if t.cls == _hdf5.Datatype.ARRAY or (t.cls == _hdf5.Datatype.COMPOUND and (reduced(t) or nat != stored)):
            # array types, big-endian or reduced members: the stored bytes,
            # then one conversion
            raw = self._raw_box(box, shape, np.dtype('V%d' % t.size), threads)
            vals = np.frombuffer(raw.tobytes(), stored)
            vals = vals.reshape(tuple(shape) + vals.shape[1:])
            vals = convert_reduced(vals, t) if reduced(t) else vals.astype(native(vals.dtype))
            vals = _select.finish(vals, post, drop, scalars)
            if out is not None:
                out[...] = vals
                return out
            return vals
        if post is not None and out is None:                # (lists, masks: all of a box to pick from)
            res = self._boxed(box)
            if res is not None:
                return _select.finish(res, post, drop, scalars)
            if not ds.filters:
                res = self._scattered(box, post, shape, nat, swap, threads)
                if res is not None:
                    return _select.finish(res[0], res[1], drop, scalars)
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

    def _boxed(self, box):
        """A box by the way small reads go, if it is one: a plan cost more than
        the list it was for. None if not."""
        q = self._q
        if q is None:
            from . import _quick
            q = self._q = _quick.reader(self)
        return q(tuple(slice(s, e, k) for s, e, k in box)) if q else None

    def _scattered(self, box, post, shape, nat, swap, threads):
        """A list along one axis, of values stored as they are, that a big box
        mostly does not want: place by place, not all of the box to pick from.
        Those values and what is left to do; None if the box will do."""
        at = [d for d, p in enumerate(post) if isinstance(p, np.ndarray)]
        if len(at) != 1:
            return None
        d = at[0]
        p = post[d]
        if p.size > SCATTERED or p.size * WANTS > shape[d] or math.prod(shape) * nat.itemsize < BIG_BOX:
            return None
        res = np.empty(shape[:d] + (p.size,) + shape[d + 1:], nat)
        one = shape[:d] + (1,) + shape[d + 1:]
        lo = box[d][0]
        place = (slice(None),) * d
        for i, q in enumerate(p.tolist()):
            part = box[:d] + [(lo + q, lo + q + 1, 1)] + box[d + 1:]
            if d == 0:
                self._read_into(part, res[i:i + 1], swap, threads)
            else:
                v = np.empty(one, nat)
                self._read_into(part, v, swap, threads)
                res[place + (i,)] = v[place + (0,)]
        rest = post[:d] + (slice(None),) + post[d + 1:]
        return res, None if all(isinstance(k, slice) and k == slice(None) for k in rest) else rest

    def _raw_box(self, box, shape, vdt, threads):
        res = np.empty(shape, vdt)
        if res.size:
            self._read_into(box, res, False, threads)
        return res

    def _read_into(self, box, res, swap, threads):
        if self.src is None:
            raise ValueError('the file of dataset %s is closed' % self.ds.name)
        ds = self.ds
        lay = ds.layout
        f = self.f
        esize = ds.type.size
        if lay.cls == 2 or (lay.cls == 1 and self.virtual):
            if not ((ds.filters and self.f.source is None and _small_read(ds, self.index, self.src, box, res, swap))
                    or _hold.read(ds, self.index, self.src, box, res, swap, threads)):
                _chunks.read(ds, self.index, self.src, box, res, swap=swap, threads=threads)
            return
        if lay.cls == 1:
            if ds.external:                             # in raw files of their own
                from . import _vds
                _vds.read_external(ds, box, res)
                if swap:
                    _swap(res)
                return
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
            raise _hdf5.FormatError('layout class %d read as raw bytes' % lay.cls)
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
    # (the last ones stop where the dataset does: it may be where the file does)
    sizes = chunks[k] * inner - np.maximum(idx[:, k] + chunks[k] - shape[k], 0) * inner
    table = [(tuple(o), int(a), int(z), 0) for o, a, z in zip(idx.tolist(), addr.tolist(), sizes.tolist())]
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


def read_many(readers, threads=None, probes=None):
    """Every dataset of `readers` whole, chunked ones decoded together in
    one pass of the pool (one shared output buffer, biggest chunks first:
    a file's many small variables no longer wait on each other). probes:
    a _maskscale.Probe or None for each reader, tested while decoding."""
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
        n = math.prod(ds.shape) * t.size
        batch.append((i, r, total, n))
        total += (n + 63) // 64 * 64                    # each array 64-byte aligned
    buf = _arena.empty((total,), np.uint8)
    items = []
    made = _tree.indexes([b[1] for b in batch])
    for (i, r, base, n), index in zip(batch, made):
        ds = r.ds
        nat = native(ds.type.dtype)
        a = buf[base:base + n].view(nat).reshape(ds.shape)
        out[i] = a
        if n:
            items.append(_chunks.Item(ds, r.index if index is None else index, [(0, s, 1) for s in ds.shape], a,
                                      _simple_swap(ds.type.dtype), base,
                                      probes[i] if probes is not None else None))
    if items:
        _chunks.read_batch(items, readers[0].src, buf, threads)
    for i, r in enumerate(readers):
        if out[i] is None:
            out[i] = r.read((), threads=threads)
    return out
