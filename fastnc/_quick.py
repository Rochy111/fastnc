"""Small reads with nothing to work out: a function made once for a dataset.

Everything a read goes through (reducing the index, an output array, planning,
converting) is there for reads that need it. `x[i]` in a loop needs none of it:
a dataset's layout does not change, so where a row lies is a sum, and the row a
copy of bytes that are already in memory, in the mapped file or in a decoded
chunk. `reader` makes that sum into a closure, with what it depends on held in
its variables: a row in about 1.5 us, against 2 to 4 for h5py's C path and 6 to
16 through the general one; and a row across chunks piece by piece, where
planning it took 90 us.

It answers only what it is sure of, and returns None for anything else: a chunk
not decoded or not stored, a big read that should go to the pool, any index but
integers and plain slices. The general path then does what it always did (and
leaves decoded chunks in the cache for this one).
"""
import math
import zlib
from itertools import product

import numpy as np

from . import _chunks, _h5read

BIG = 1 << 21                   # bytes from which a read is worth the pool's threads
SMALL = 1 << 16                 # chunks up to this size are decoded on the spot, one at a time
MOST = 32                       # chunks from which the kernel's one pass is worth planning
MOST_BARE = 256                 # ... where they need no decoding: a copy each
GATHER = 4                      # chunks from which those are worth the kernel's call (see gather)
WHOLE = slice(None)
KEEP = 1 << 23                  # bytes of them kept at hand, and no more than
KEEP_CHUNKS = 512               # ... this many (a dataset's index is in each key)
_NDARRAY = np.ndarray
_recent = {}                    # (chunk index, number on the grid) -> decoded chunk, typed
_held = 0


def forget():
    global _held
    _recent.clear()
    _held = 0


def _chunk(key, index, place, swap, nat):
    """The decoded chunk at a grid place, typed, if the cache has it: at hand
    from then on, past the lock and the view. (Decoded values do not change:
    one still here after the cache has let it go is as good.)"""
    h = _chunks.cached_chunk(index, place, swap)
    if h is None:
        return None
    return _keep(key, _typed(h, nat))


def _keep(key, h):
    global _held
    _held += h.nbytes
    if _held > KEEP or len(_recent) >= KEEP_CHUNKS:
        _recent.clear()
        _held = h.nbytes
    _recent[key] = h
    return h


def _typed(h, nat):
    try:
        return h.view(nat)[..., 0]                  # (bytes last: the cache's form, any dtype)
    except ValueError:                              # (numpy before 1.23, a part of a bigger box)
        return h.copy().view(nat)[..., 0]


def small_read(ds, index, src, box, out, swap):
    """All of a small dataset of one or a few small chunks, decoded here. The
    batch is for many chunks on many threads; for a file of fifty variables of
    a few kB each, as a GLM file is, it was most of a read of them all. False:
    not such a read. (Not for part of one: the cache of chunks serves those.)"""
    ids = [x.id for x in ds.filters]
    shuffled = ids == [_chunks.F_SHUFFLE, _chunks.F_DEFLATE]
    if not shuffled and ids != [_chunks.F_DEFLATE] or not _chunks.fusable(ds):
        return False
    esize = ds.type.size
    if out.nbytes > SMALL or out.size != math.prod(ds.shape) or not out.size:
        return False
    p = _chunks.plan(index, ds.shape, box)
    if p.fill_needed or p.mask.any() or len(p.addr) > 4:
        return False
    cshape = index.cshape + (esize,)
    bytes_out = out.reshape(-1).view(np.uint8).reshape(out.shape + (esize,))
    ranges = zip(p.addr.tolist(), p.size.tolist(), p.cstart.tolist(), p.count.tolist(), p.ostart.tolist())
    for a, n, first, count, at in ranges:
        data = zlib.decompress(src[a:a + n])
        if shuffled and esize > 1:
            data = np.frombuffer(data, np.uint8).reshape(esize, -1).T.tobytes()
        h = np.frombuffer(data, np.uint8).reshape(cshape)[tuple(slice(f, f + c) for f, c in zip(first, count))]
        bytes_out[tuple(slice(o, o + c) for o, c in zip(at, count))] = h[..., ::-1] if swap else h
    return True


def _few(index):
    """The table of grid places to rows that _grid_table leaves out for a few
    chunks, as a list; None for a sparse grid, or a chunk it cannot place."""
    if len(index.addr) >= 16 or index.ngrid > 1 << 16:
        return None
    grid = index.gridlist
    t = [-1] * index.ngrid
    for row, off in enumerate(index.offsets.tolist()):
        lin = 0
        for x, c, g in zip(off, index.cshape, grid):
            x //= c
            if x < 0 or x >= g:
                return None                         # (past the edge of a dataset that shrank)
            lin = lin * g + x
        if t[lin] >= 0:
            return None
        t[lin] = row
    return t


def _record(t):
    """Whether a compound's values are what h5py gives as they lie: numbers and
    strings that need no trimming, in compounds of the same."""
    if t.complex():
        return False
    for _, _, m in t.members:
        if not (m.cls in (0, 1) and m.dtype is not None and not m.odd() or m.cls == 3 and m.strpad == 1
                or m.cls == 6 and _record(m)):
            return False
    return True


def reader(r):
    """A function of an index giving r's values for it, or None if it cannot
    say; False if no read of this dataset can go this way."""
    if not isinstance(r, _h5read.Reader):
        return False
    if r.src is None:                               # (the file closed: the general path says so)
        return None
    ds = r.ds
    t = ds.type
    if not r.asked and math.prod(ds.shape) * t.size <= SMALL:
        r.asked = True                              # (a small dataset is mostly read once:
        return None                                 # the next read makes it, if there is one)
    f = r.f
    records = t.cls == 6 and 6 in r.plain and t.dtype is not None and _record(t)
    chars = t.cls == 3 and 3 in r.plain and t.size == 1
    if (getattr(f, 'source', None) is not None or ds.null or (t.cls not in (0, 1) and not records and not chars)
            or t.dtype is None or _h5read.reduced(t) or not ds.shape):
        return False
    stored = t.dtype
    nat = _h5read.native(stored)
    swap = _h5read._simple_swap(stored)
    if stored.byteorder == '>' and not swap or records and nat != stored:
        return False
    if records and r.record is not None:
        stored = nat = r.record
    shape = tuple(int(n) for n in ds.shape)
    rank = len(shape)
    lay = ds.layout
    decoded = False
    index = table = addrs = None
    if lay.cls == 2:
        index = r.index
        cs = index.cshape
        if len(cs) != rank:
            return False
        if ds.filters:
            decoded = True
        elif index.seen is not None:
            table, addrs = index.rows, index.looked
        else:
            table = index.table
            if table is None:
                table = index.table = index._grid_table()
            if table is False:
                table = _few(index)
                if table is None:
                    return False
            addrs = index.addr.tolist()
        grid = [-(-n // c) for n, c in zip(shape, cs)]
    elif lay.cls == 1 and not ds.external and lay.address != f.undef:
        cs, grid = shape, [1] * rank
        addrs = [f.addr(lay.address)]
    elif lay.cls == 0:
        cs, grid = shape, [1] * rank
        addrs = [lay.data[0]]
    else:
        return False
    esize = stored.itemsize
    wide = [esize] * rank                           # bytes per element, along each dimension
    for d in range(rank - 2, -1, -1):
        wide[d] = wide[d + 1] * cs[d + 1]
    after = [esize] * (rank + 1)                    # bytes under each dimension of the dataset
    for d in range(rank - 1, -1, -1):
        after[d] = after[d + 1] * shape[d]
    alone = [g == 1 for g in grid]
    plan = list(zip(shape, cs, grid, wide))
    tail = [(shape[d:], tuple(wide[d:])) for d in range(rank + 1)]
    whole = tuple(cs)
    cbytes = wide[0] * cs[0]
    wants_swap = bool(swap)
    ids = [x.id for x in ds.filters]
    shuffled = ids == [_chunks.F_SHUFFLE, _chunks.F_DEFLATE]
    look = None
    if decoded and cbytes <= SMALL and (shuffled or ids == [_chunks.F_DEFLATE]) and (
            not shuffled or not ds.filters[0].cd or ds.filters[0].cd[0] == esize):
        look = getattr(index, '_at', None)
    spans = {}                                      # what a slice of a dimension comes to in
    #                                                 chunks: the same for every row of a loop
    joins = {}                                      # ... and what slices come to together (see pieces)
    below = [1] * rank                              # chunks under each dimension of the grid
    for d in range(rank - 2, -1, -1):
        below[d] = below[d + 1] * grid[d + 1]

    def whole_record(v):
        """Records of ours to keep: an element is a view of what it comes from, and
        numpy copies them field by field, past the bytes between fields."""
        out = np.array(v.tobytes(), 'V%d' % v.nbytes).view(nat) if v.__class__ is not _NDARRAY or not v.ndim \
            else np.frombuffer(bytearray(v.tobytes()), nat).reshape(v.shape)
        return out if out.ndim else out[()]

    def gather(rows, shp, count):
        """Many chunks stored as they are, by one kernel with nothing to plan:
        a copy each, not a Python loop. None: if the table of them is not worth
        making yet, or one is not stored."""
        src = r.src
        if src is None or not _chunks.loaded():
            return None
        if getattr(index, 'seen', None) is not None:
            if count <= max(_h5read.LOOK_MOST, index.ngrid >> _h5read.LOOK_SHARE):
                return None                         # (a few of very many: looked up)
            index.addr                              # (made now, as lookups would cost more)
        t = index.table
        if t is None:
            t = index.table = index._grid_table()
        if t is False:
            return None
        out = np.empty(tuple(shp), nat)
        stride = esize
        for row in reversed(rows):
            if row[7]:
                row[7] = stride
                stride *= (row[3] - row[2] - 1) // row[8] + 1
        ok = _chunks._gather(src, index.addr, t, np.array(rows, np.int64), rank, esize, wants_swap,
                             out.reshape(-1).view(np.uint8))
        return out if ok else None

    def dots(key):
        """An index with its Ellipsis written out."""
        i = next(j for j, k in enumerate(key) if k is Ellipsis)
        return key[:i] + (WHOLE,) * (rank - len(key) + 1) + key[i + 1:]

    def spread(key):
        """The same over several chunks, piece by piece: a row across square
        chunks, a small variable of a few. Steps go by the kernel, or are left
        to the general path."""
        parts, shp, rows, mid = [], [], [], []
        count = 1
        size = esize
        stepped = False
        m = len(key)
        lin = 0
        pre = post = ()                             # the indexes in chunks before the first slice, after the last
        for d, (n, b, g, w) in enumerate(plan):
            k = key[d] if d < m else WHOLE
            if k.__class__ is int:
                if k < 0:
                    k += n
                if k < 0 or k >= n:
                    return None
                p = k // b
                if parts:
                    parts.append(((p, k - p * b, None),))
                    mid.append(k)
                else:
                    lin += p * below[d]
                    pre += (k - p * b,)
                if not decoded:
                    rows.append([p, p, k, k + 1, b, below[d], w, 0, 1])
                continue
            if k.__class__ is not slice:
                return spread(dots(key)) if k is Ellipsis else None
            s, e, st = k.indices(n)
            if st <= 0 or s >= e or st != 1 and decoded:
                return None
            first, last = s // b, (e - 1) // b
            count *= last - first + 1
            if decoded:
                if count > MOST:
                    return None
            else:
                rows.append([first, last, s, e, b, below[d], w, 1, st])
            if st != 1:
                stepped = True
                cnt = (e - s - 1) // st + 1
                shp.append(cnt)
                size *= cnt
                continue
            one = spans.get((d, s, e))
            if one is None:
                one = []
                for p in range(first, last + 1):
                    q = p * b
                    lo, hi = max(s, q), min(e, q + b)
                    one.append((p, slice(lo - q, hi - q), slice(lo - s, hi - s)))
                if len(spans) >= 64:
                    spans.clear()
                spans[d, s, e] = one
            parts.append(one)
            mid.append((s, e))
            shp.append(e - s)
            size *= e - s
        if size > BIG:
            return None
        if not decoded and (count >= GATHER or stepped):
            out = gather(rows, shp, count)
            if out is not None:
                return out
            if count > MOST_BARE or stepped:
                return None
        d = rank
        while parts and mid[-1].__class__ is int:
            d -= 1
            p, i, _ = parts.pop()[0]
            mid.pop()
            lin += p * below[d]
            post = (i,) + post
        out = np.empty(tuple(shp), nat)
        for at, inner, where in pieces(d - len(parts), tuple(mid), parts):
            at += lin
            if decoded:
                h = _recent.get((index, at))
                if h is None:
                    h = _chunk((index, at), index, places(at), wants_swap, nat)
                    if h is None:
                        return None
            else:
                row = table[at]
                if row < 0 or r.src is None:
                    return None
                h = np.ndarray(whole, stored, r.src, addrs[row])
            out[where] = h[pre + inner + post]
        return out

    def pieces(first, mid, parts):
        """What dimensions from first on, the slices and any integer between
        them, make of a read: each chunk's number on the grid, its slices, where
        they go. The same for every row of a loop, which took 5.6 us to work
        out five of them anew, against 3.9 for all of h5py's read."""
        kept = joins.get((first, mid))
        if kept is None:
            kept = []
            under = grid[first:first + len(parts)]
            scale = below[first + len(parts) - 1] if parts else 1
            for part in product(*parts):
                at = 0
                inner = []
                where = []
                for (p, i, o), g in zip(part, under):
                    at = at * g + p
                    inner.append(i)
                    if o is not None:
                        where.append(o)
                kept.append((at * scale, tuple(inner), tuple(where)))
            if len(joins) >= 64:
                joins.clear()
            joins[first, mid] = kept
        return kept

    def read(key):
        c = key.__class__
        if c is int or c is slice:
            key = (key,)
        elif c is not tuple:
            return None
        m = len(key)
        if m > rank:
            return None
        lin = at = 0
        shp, step, inner = [], [], []
        size = after[m]
        for k, (n, b, g, w) in zip(key, plan):
            c = k.__class__
            if c is int:
                if k < 0:
                    k += n
                if k < 0 or k >= n:
                    return None
                p = k // b
                k -= p * b
                inner.append(k)
                at += k * w
            elif c is slice:
                s, e, st = k.indices(n)
                if st <= 0 or s >= e:
                    return None
                p = s // b
                if (e - 1) // b != p:
                    return spread(key)
                q = p * b
                s -= q
                e -= q
                inner.append(slice(s, e, st))
                at += s * w
                cnt = (e - s - 1) // st + 1
                shp.append(cnt)
                step.append(st * w)
                size *= cnt
            else:
                if isinstance(k, np.integer):
                    return read(tuple(int(q) if isinstance(q, np.integer) else q for q in key))
                return read(dots(key)) if k is Ellipsis else None
            lin = lin * g + p
        for d in range(m, rank):
            if not alone[d]:
                return spread(key)
        if decoded:
            h = _recent.get((index, lin))
            if h is None:
                if size == cbytes:                  # all of a chunk: if no read has seen it, ours
                    h = _chunks.take_chunk(index, places(lin), wants_swap)
                    if h is not None:
                        return _typed(h, nat)[tuple(inner)]
                h = _chunk((index, lin), index, places(lin), wants_swap, nat)
                if h is None:
                    if look is None:
                        return None
                    h = small(lin)
                    if h is None:
                        return None
            v = h[tuple(inner)]
            if records:
                return whole_record(v)
            return v.copy() if v.__class__ is _NDARRAY else v
        if size > BIG:
            return None
        row = 0
        if table is not None:
            row = table[lin]
            if row < 0:
                return None
        src = r.src
        if src is None:                             # (the file closed: the general path says so)
            return None
        v = np.ndarray(tuple(shp) + tail[m][0], stored, src, addrs[row] + at, tuple(step) + tail[m][1])
        if swap:
            v = v.astype(nat)
            return v if v.ndim else v[()]
        if records:
            return whole_record(v)
        return v.copy() if v.ndim else v[()]

    def small(lin):
        """A small chunk nothing has decoded, on its own: the reads that bring
        one here (a pixel here, a pixel there) took 130 us through the batch
        for what zlib does in two. Not one next to the last: a read going
        along is better served by the general path, which decodes ahead."""
        place = places(lin)
        src = r.src
        if src is None or _chunks._along(index, [place]) is not None:
            return None
        got = look(place)
        if got is None or got[2]:
            return None
        _chunks._went(index, [place], rank)
        try:
            data = zlib.decompress(src[got[0]:got[0] + got[1]])
        except zlib.error:
            return None                             # (the general path says what is wrong)
        if len(data) != cbytes:
            return None
        if shuffled and esize > 1:
            data = np.frombuffer(data, np.uint8).reshape(esize, -1).T.tobytes()
        h = np.frombuffer(data, stored).reshape(whole)
        return _keep((index, lin), h.astype(nat) if swap else h)

    def places(lin):
        """A chunk's linear number on the grid as its place."""
        out = [0] * rank
        for d in range(rank - 1, -1, -1):
            lin, out[d] = divmod(lin, grid[d])
        return tuple(out)
    return read
