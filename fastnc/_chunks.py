"""Reading a selection of a chunked HDF5 dataset: which chunks it touches,
and a kernel that decodes them on every core straight into the output.

A read is planned in Python with numpy (no Python per chunk): the chunk
table, parsed once per dataset, is intersected with the selection one
dimension at a time, giving each chunk's first selected element, count
and place in the output. Then `_pool.share` runs `_work` on the caller
and the pool's threads; each takes chunks off an atomic counter and

1. inflates (or otherwise decodes) the chunk into its own scratch buffer
   -- stopping as soon as the last element the selection needs is out, so
   a window of a MODIS band does not inflate the rest of a 17 MB chunk;
2. places the selected elements into the output in one pass over the
   scratch while it is still in cache, undoing HDF5's byte shuffle and
   swapping big-endian to native as it goes (no separate passes).

A chunk whose bytes need no transform and land contiguously in the output
is inflated directly there. Chunks never written read as the fill value.

Chunks of more than PAR_MIN compressed bytes (MODIS keeps bands in 33 MB
chunks) would leave every core but one idle for most of a read: they are
inflated first, each cut into parts decoded on all the pool's threads at
once (`_pinflate.inflate_many`: speculative decoding from guessed block
starts, every part verified against where the one before it ended), the
other chunks run by each thread once the parts are taken.
"""
import math
import sys
import threading
import time

import numpy as np

from . import _hw, _inflate, _pinflate, _pio, _pool
from ._intrin import ld64, st64, take
from ._jit import bind, njit

# pipeline stage kinds the kernel knows (HDF5 filter ids)
F_DEFLATE = 1
F_SHUFFLE = 2
F_FLETCHER = 3

MAXDIM = 32


class Plan:
    """The chunks a selection touches and where each goes. wide: cstart,
    count and ostart with a column for every dimension of a batch, if made so."""
    __slots__ = ('addr', 'size', 'mask', 'cstart', 'count', 'ostart', 'needend', 'fill_needed', 'wide')

    def __init__(self):
        self.wide = None


class ChunkIndex:
    """A dataset's chunk table as arrays: element offsets (n, rank),
    absolute addresses, stored sizes, filter masks; and a lookup from the
    chunk's grid position to its row."""

    seen = None                                     # see _h5read.LazyIndex

    def __init__(self, table, chunks, shape, arrays=None):
        self._frame(chunks, shape)
        self._tables(table, arrays)

    def part(self, sel):
        """The chunks of a box without the table, if there is a way to."""
        return None

    def _tables(self, table, arrays):
        rank = len(self.cshape)
        if arrays is not None:                      # (offsets, addr, size, mask) already
            self.offsets, self.addr, self.size, self.mask = arrays
            n = len(self.addr)
        else:
            n = len(table)
            self.offsets = np.array([t[0] for t in table], np.int64).reshape(n, rank)
            self.addr = np.array([t[1] for t in table], np.int64)
            self.size = np.array([t[2] for t in table], np.int64)
            self.mask = np.array([t[3] for t in table], np.int64)
        self.complete = n >= self.ngrid             # every chunk stored: no fill needed

    def _frame(self, chunks, shape):
        rank = len(chunks)
        self.chunks = np.array(chunks, np.int64)
        self.grid = np.array([-(-s // c) for s, c in zip(shape, chunks)], np.int64)
        self.gridlist = self.grid.tolist()
        self.ngrid = int(np.prod(self.grid)) if rank else 1
        self.plans = {}                             # box -> Plan, for repeated reads
        self.ways = {}                              # thread -> its _Way through this
        self.coming = {}                            # grid place -> Event: being decoded by a read
        self.gone = set()                           # grid places threads have read and not kept
        self.fresh = set()                          # grid places decoded ahead and not yet read
        self.owner = None                           # the file, once anything of it is cached
        self.cshape = tuple(int(c) for c in chunks)
        self.table = None                           # row of each grid place (False: no use here)
        self.bare = None                            # see _bare: None until looked at
        self._cstrides = None

    def cstrides(self, esize):
        """A chunk's strides in bytes, as ints."""
        s = self._cstrides
        if s is None:
            out = [esize] * len(self.cshape)
            for d in range(len(out) - 2, -1, -1):
                out[d] = out[d + 1] * self.cshape[d + 1]
            s = self._cstrides = tuple(out)
        return s

    def near(self, sel):
        """The rows of the chunks a box touches, without going through all of
        them: planning a small read over the whole table took 90 us a row, in
        a loop over a dataset's rows. None: look at all of them."""
        t = self.table
        C = self.cshape
        if t is None:
            t = self.table = self._grid_table()
        if t is False:
            return None
        lin = None
        for (s, e, _), c, n in zip(sel, C, self.gridlist):
            a, b = s // c, min((e - 1) // c, n - 1)
            if a > b:
                return np.zeros(0, np.int32)
            axis = np.arange(a, b + 1, dtype=np.int64)
            lin = axis if lin is None else (lin[:, None] * n + axis[None, :]).ravel()
        rows = t[lin]
        return rows[rows >= 0]

    def _grid_table(self):
        n = len(self.addr)
        self.gridlist = self.grid.tolist()
        if not len(self.cshape) or n < 16 or self.ngrid > 4 * n + (1 << 16):
            return False                            # (small, or sparse: the table is not worth it)
        g = self.offsets // self.chunks
        if ((g < 0) | (g >= self.grid)).any():
            return False                            # (a chunk past the edge of a dataset that shrank)
        lin = g[:, 0].copy()
        for d in range(1, g.shape[1]):
            lin = lin * self.grid[d] + g[:, d]
        t = np.full(self.ngrid, -1, np.int32)
        t[lin] = np.arange(n, dtype=np.int32)
        return t if (t >= 0).sum() == n else False  # (two chunks in one place)


_STRIDES = {}                   # shape -> its strides in elements: files repeat a few shapes


def strides(shape):
    """The C-order strides of a shape (a tuple), in elements: don't write to
    them. Kept, because a batch asks twice for each of hundreds of variables."""
    s = _STRIDES.get(shape)
    if s is None:
        out = [1] * len(shape)
        for d in range(len(shape) - 2, -1, -1):
            out[d] = out[d + 1] * shape[d + 1]
        s = np.array(out, np.int64)
        s.flags.writeable = False
        if len(_STRIDES) >= 1024:
            _STRIDES.clear()
        _STRIDES[shape] = s
    return s


def plan(index, shape, sel):
    """Chunks touched by `sel` (one (start, stop, step) per dimension, step >
    0, start < stop) as a Plan, kept on the index for the next read of the
    same box (planning a small read costs more than decoding it)."""
    key = tuple(tuple(s) for s in sel)
    p = index.plans.get(key)
    if p is None:
        p = _plan(index, shape, sel)
        if len(index.plans) >= 16:
            index.plans.pop(next(iter(index.plans)))
        index.plans[key] = p
    return p


def _plan(index, shape, sel):
    rank = len(shape)
    few = index.part(sel) if rank else None
    if few is not None:
        return (_few_plan if len(few.addr) <= 8 else _many_plan)(few, shape, sel, few.offsets, None)
    off = index.offsets
    C = index.chunks
    if rank and all(s == 0 and e >= n and k == 1 for (s, e, k), n in zip(sel, shape)):
        # the whole dataset (a load): every chunk inside the shape kept as it is
        sh = np.array(shape, np.int64)
        if bool((off < sh).all()) and bool((off >= 0).all()):
            p = Plan()
            p.addr, p.size, p.mask = index.addr, index.size, index.mask
            p.cstart = np.zeros_like(off)
            p.count = np.ascontiguousarray(np.minimum(C, sh - off))
            p.ostart = np.ascontiguousarray(off)
            p.needend = ((p.count - 1) * strides(tuple(C.tolist()))).sum(axis=1) + 1
            p.fill_needed = not index.complete and len(off) < index.ngrid
            return p
    rows = index.near(sel) if rank else None
    if rows is not None:
        off = off[rows]
    if rank and len(off) <= 8 and (index.complete or all(s[2] == 1 for s in sel)):
        return _few_plan(index, shape, sel, off, rows)
    return _many_plan(index, shape, sel, off, rows)


def _few_plan(index, shape, sel, off, rows):
    """_many_plan's sums in plain integers, for a few chunks: numpy's fixed
    cost on each of some forty small arrays came to 70 us, a read of two
    small chunks planning three times more than it took to decode them."""
    C = index.cshape
    rank = len(C)
    keep, cstart, count, ostart, needend = [], [], [], [], []
    cstr = [1] * rank
    for d in range(rank - 2, -1, -1):
        cstr[d] = cstr[d + 1] * C[d + 1]
    for i, o in enumerate(off.tolist()):
        cs, ct, os_ = [], [], []
        end = 1
        for (s, e, k), c0, c, n, st in zip(sel, o, C, shape, cstr):
            lo = c0 if c0 > s else s
            hi = min(c0 + c, e, n)
            first = s + -(-(lo - s) // k) * k           # first selected index >= lo
            if first >= hi:
                break
            m = (hi - 1 - first) // k + 1
            cs.append(first - c0)
            ct.append(m)
            os_.append((first - s) // k)
            end += (first - c0 + (m - 1) * k) * st
        else:
            keep.append(i)
            cstart.extend(cs)
            count.extend(ct)
            ostart.extend(os_)
            needend.append(end)
    idx = keep if rows is None else rows[keep].tolist()
    p = Plan()
    p.addr = index.addr[idx]
    p.size = index.size[idx]
    p.mask = index.mask[idx]
    p.cstart = np.array(cstart, np.int64).reshape(-1, rank)
    p.count = np.array(count, np.int64).reshape(-1, rank)
    p.ostart = np.array(ostart, np.int64).reshape(-1, rank)
    p.needend = np.array(needend, np.int64)
    if index.complete:
        p.fill_needed = False
    else:                                           # (steps of 1: see _plan)
        touched = 1
        for (s, e, _), c, n in zip(sel, C, shape):
            touched *= (min(e, n) - 1) // c - s // c + 1
        p.fill_needed = len(idx) < touched
    return p


def _many_plan(index, shape, sel, off, rows):
    rank = len(shape)
    C = index.chunks
    keep = np.ones(len(off), bool)
    cstart = np.empty_like(off)
    count = np.empty_like(off)
    ostart = np.empty_like(off)
    for d in range(rank):
        s, e, k = sel[d]
        c0 = off[:, d]
        lo = np.maximum(c0, s)
        hi = np.minimum(np.minimum(c0 + C[d], e), shape[d])
        first = s + -(-(lo - s) // k) * k           # first selected index >= lo
        n = np.where(first < hi, (hi - 1 - first) // k + 1, 0)
        keep &= n > 0
        cstart[:, d] = first - c0
        count[:, d] = n
        ostart[:, d] = (first - s) // k
    kept = np.nonzero(keep)[0]
    idx = kept if rows is None else rows[kept]
    p = Plan()
    p.addr = index.addr[idx]
    p.size = index.size[idx]
    p.mask = index.mask[idx]
    p.cstart = np.ascontiguousarray(cstart[kept])
    p.count = np.ascontiguousarray(count[kept])
    p.ostart = np.ascontiguousarray(ostart[kept])
    # the decoded bytes each chunk needs: up to its last selected element
    last = p.cstart + (p.count - 1) * np.array([s[2] for s in sel], np.int64)
    cstr = np.ones(rank, np.int64)
    for d in range(rank - 2, -1, -1):
        cstr[d] = cstr[d + 1] * C[d + 1]
    p.needend = (last * cstr).sum(axis=1) + 1 if rank else np.ones(len(idx), np.int64)
    # stored chunks cover the selection unless some grid chunk is absent
    if index.complete:
        p.fill_needed = False
    else:
        touched = 1
        for d in range(rank):
            s, e, k = sel[d]
            first_c = s // C[d]
            last_c = (min(e, shape[d]) - 1) // C[d]
            # grid chunks along d holding at least one selected index
            cs = np.arange(first_c, last_c + 1)
            lo = np.maximum(cs * C[d], s)
            hi = np.minimum(cs * C[d] + C[d], e)
            first = s + -(-(lo - s) // k) * k
            touched *= int(np.count_nonzero(first < hi))
        p.fill_needed = len(idx) < touched
    return p


# ---------------------------------------------------------------- kernels

# Undoing the byte shuffle eight elements at a time: a word of each byte
# plane loaded, the planes' bytes interleaved with shifts and masks (bytes
# spread to every other byte, then 16-bit units to every other unit, ...),
# whole words stored. LLVM vectorizes these; byte by byte the same work ran
# at 3-4 GB/s (a third of a shuffled read's decoding on one thread), this
# at 17-25 GB/s.

@njit(cache=True, nogil=True)
def _sp8(x):
    """The low four bytes of x spread to the even bytes of a word."""
    u = np.uint64
    x &= u(0xFFFFFFFF)
    x = (x | (x << u(16))) & u(0x0000FFFF0000FFFF)
    return (x | (x << u(8))) & u(0x00FF00FF00FF00FF)


@njit(cache=True, nogil=True)
def _sp16(x):
    """The low two 16-bit units of x spread to the even units of a word."""
    u = np.uint64
    x &= u(0xFFFFFFFF)
    return (x | (x << u(16))) & u(0x0000FFFF0000FFFF)


@njit(cache=True, nogil=True)
def _unshuffle2(sc, p0, p1, out, b, n):
    """n 2-byte elements to out[b:], byte k of each from plane sc[pk:]."""
    u = np.uint64
    j = 0
    while j + 8 <= n:
        x = ld64(sc, p0 + j)
        y = ld64(sc, p1 + j)
        st64(out, b, _sp8(x) | (_sp8(y) << u(8)))
        st64(out, b + 8, _sp8(x >> u(32)) | (_sp8(y >> u(32)) << u(8)))
        b += 16
        j += 8
    while j < n:
        out[b] = sc[p0 + j]
        out[b + 1] = sc[p1 + j]
        b += 2
        j += 1


@njit(cache=True, nogil=True)
def _unshuffle4(sc, p0, p1, p2, p3, out, b, n):
    """n 4-byte elements to out[b:], byte k of each from plane sc[pk:]."""
    u = np.uint64
    j = 0
    while j + 8 <= n:
        x0 = ld64(sc, p0 + j)
        x1 = ld64(sc, p1 + j)
        x2 = ld64(sc, p2 + j)
        x3 = ld64(sc, p3 + j)
        a0 = _sp8(x0) | (_sp8(x1) << u(8))                 # bytes 0-1 of elements 0-3
        a1 = _sp8(x0 >> u(32)) | (_sp8(x1 >> u(32)) << u(8))
        c0 = _sp8(x2) | (_sp8(x3) << u(8))                 # bytes 2-3
        c1 = _sp8(x2 >> u(32)) | (_sp8(x3 >> u(32)) << u(8))
        st64(out, b, _sp16(a0) | (_sp16(c0) << u(16)))
        st64(out, b + 8, _sp16(a0 >> u(32)) | (_sp16(c0 >> u(32)) << u(16)))
        st64(out, b + 16, _sp16(a1) | (_sp16(c1) << u(16)))
        st64(out, b + 24, _sp16(a1 >> u(32)) | (_sp16(c1 >> u(32)) << u(16)))
        b += 32
        j += 8
    while j < n:
        out[b] = sc[p0 + j]
        out[b + 1] = sc[p1 + j]
        out[b + 2] = sc[p2 + j]
        out[b + 3] = sc[p3 + j]
        b += 4
        j += 1


@njit(cache=True, nogil=True)
def _unshuffle8(sc, p0, nel, swap, out, b, n):
    """n 8-byte elements to out[b:], byte k of each from plane k (planes nel
    bytes apart from sc[p0:]; reversed if swap)."""
    u = np.uint64
    lo = u(0xFFFFFFFF)
    hi = u(0xFFFFFFFF00000000)
    d = -nel if swap else nel                  # from plane to plane
    q0 = p0 + 7 * nel if swap else p0
    j = 0
    while j + 8 <= n:
        x0 = ld64(sc, q0 + 0 * d + j)
        x1 = ld64(sc, q0 + 1 * d + j)
        x2 = ld64(sc, q0 + 2 * d + j)
        x3 = ld64(sc, q0 + 3 * d + j)
        x4 = ld64(sc, q0 + 4 * d + j)
        x5 = ld64(sc, q0 + 5 * d + j)
        x6 = ld64(sc, q0 + 6 * d + j)
        x7 = ld64(sc, q0 + 7 * d + j)
        a0 = _sp8(x0) | (_sp8(x1) << u(8))                 # bytes 0-1, elements 0-3
        a1 = _sp8(x0 >> u(32)) | (_sp8(x1 >> u(32)) << u(8))
        c0 = _sp8(x2) | (_sp8(x3) << u(8))                 # bytes 2-3
        c1 = _sp8(x2 >> u(32)) | (_sp8(x3 >> u(32)) << u(8))
        e0 = _sp8(x4) | (_sp8(x5) << u(8))                 # bytes 4-5
        e1 = _sp8(x4 >> u(32)) | (_sp8(x5 >> u(32)) << u(8))
        g0 = _sp8(x6) | (_sp8(x7) << u(8))                 # bytes 6-7
        g1 = _sp8(x6 >> u(32)) | (_sp8(x7 >> u(32)) << u(8))
        L0 = _sp16(a0) | (_sp16(c0) << u(16))              # bytes 0-3, elements 0-1
        L1 = _sp16(a0 >> u(32)) | (_sp16(c0 >> u(32)) << u(16))
        L2 = _sp16(a1) | (_sp16(c1) << u(16))
        L3 = _sp16(a1 >> u(32)) | (_sp16(c1 >> u(32)) << u(16))
        H0 = _sp16(e0) | (_sp16(g0) << u(16))              # bytes 4-7
        H1 = _sp16(e0 >> u(32)) | (_sp16(g0 >> u(32)) << u(16))
        H2 = _sp16(e1) | (_sp16(g1) << u(16))
        H3 = _sp16(e1 >> u(32)) | (_sp16(g1 >> u(32)) << u(16))
        st64(out, b, (L0 & lo) | (H0 << u(32)))
        st64(out, b + 8, (L0 >> u(32)) | (H0 & hi))
        st64(out, b + 16, (L1 & lo) | (H1 << u(32)))
        st64(out, b + 24, (L1 >> u(32)) | (H1 & hi))
        st64(out, b + 32, (L2 & lo) | (H2 << u(32)))
        st64(out, b + 40, (L2 >> u(32)) | (H2 & hi))
        st64(out, b + 48, (L3 & lo) | (H3 << u(32)))
        st64(out, b + 56, (L3 >> u(32)) | (H3 & hi))
        b += 64
        j += 8
    while j < n:
        for k in range(8):
            out[b + k] = sc[q0 + k * d + j]
        b += 8
        j += 1


@njit(cache=True, nogil=True)
def _swapped(sc, a, out, b, n, esize):
    """n elements of sc[a:] into out[b:] with their bytes reversed (the same
    place will do), a word at a time: byte by byte was 1.2 GB/s, and h5py read
    big-endian chunks with no shuffle faster than this did."""
    u = np.uint64
    nb = n * esize
    k = 0
    m8 = u(0x00FF00FF00FF00FF)
    m16 = u(0x0000FFFF0000FFFF)
    if esize == 2:
        while k + 8 <= nb:
            x = ld64(sc, a + k)
            st64(out, b + k, ((x & m8) << u(8)) | ((x >> u(8)) & m8))
            k += 8
    elif esize == 4:
        while k + 8 <= nb:
            x = ld64(sc, a + k)
            x = ((x & m8) << u(8)) | ((x >> u(8)) & m8)
            st64(out, b + k, ((x & m16) << u(16)) | ((x >> u(16)) & m16))
            k += 8
    elif esize == 8:
        while k + 8 <= nb:
            x = ld64(sc, a + k)
            x = ((x & m8) << u(8)) | ((x >> u(8)) & m8)
            x = ((x & m16) << u(16)) | ((x >> u(16)) & m16)
            st64(out, b + k, (x << u(32)) | (x >> u(32)))
            k += 8
    while k < nb:
        for j in range(esize // 2):                 # (both ends of one at a time: it may be in place)
            t = sc[a + k + j]
            out[b + k + j] = sc[a + k + esize - 1 - j]
            out[b + k + esize - 1 - j] = t
        if esize & 1:
            out[b + k + esize // 2] = sc[a + k + esize // 2]
        k += esize


@njit(cache=True, nogil=True)
def vlen_pack(src, raw, n, step, base, width, out, ends):
    """The variable-length values that raw's n elements (step bytes each: a
    length, a collection's address and an object's number) refer to, one after
    another in out, each ending at ends[i]. Offsets and lengths of 8 bytes. A
    collection is walked once for each run of values in it: they are written
    together, so a run is all of them. The bytes written, or -1 if anything is
    not where it should be."""
    u = np.uint64
    pos = np.empty(65536, np.int64)                 # (up to top: past that, nothing looks)
    size = np.empty(65536, np.int64)
    pos[0] = -1
    cap = out.shape[0]
    last = src.shape[0]
    cur = -1
    top = 0
    w = 0
    for i in range(n):
        q = i * step
        coll = np.int64(ld64(raw, q + 4))
        if coll != 0:
            a = coll + base
            if a != cur:
                if a < 0 or a + 16 > last or ld64(src, a) & u(0xFFFFFFFF) != u(0x4C4F4347):     # 'GCOL'
                    return -1
                end = a + np.int64(ld64(src, a + 8))
                if end > last:
                    return -1
                top = 0
                p = a + 16
                while p + 16 <= end:
                    k = np.int64(ld64(src, p) & u(0xFFFF))
                    if k == 0:
                        break
                    m = np.int64(ld64(src, p + 8))
                    if m < 0 or p + 16 + m > end:
                        return -1
                    for j in range(top + 1, k):
                        pos[j] = -1
                    pos[k] = p + 16
                    size[k] = m
                    if k > top:
                        top = k
                    p += 16 + (m + 7) // 8 * 8
                cur = a
            idx = np.int64(ld64(raw, q + 8) >> u(32))
            if idx > top or pos[idx] < 0:
                return -1
            take = min(np.int64(ld64(raw, q) & u(0xFFFFFFFF)) * width, size[idx])
            if w + take > cap:
                return -1
            at = pos[idx]
            k = 0
            while k + 8 <= take:
                st64(out, w + k, ld64(src, at + k))
                k += 8
            while k < take:
                out[w + k] = src[at + k]
                k += 1
            w += take
        ends[i] = w
    return w


@njit(cache=True, nogil=True)
def _gather(src, addr, table, spec, rank, esize, swap, out):
    """A box across chunks stored as they are, into out: nothing to plan, each
    chunk a copy (see _quick). spec has a row for each dimension: first and last
    chunk, the box's start and end, the chunk's size, strides of the grid in
    chunks, of a chunk and of out in bytes, and the box's step. False if a
    chunk it wants is not stored."""
    last = rank - 1
    g = np.zeros(MAXDIM, np.int64)
    idx = np.zeros(MAXDIM, np.int64)
    n = np.zeros(MAXDIM, np.int64)
    for d in range(rank):
        g[d] = spec[d, 0]
    cbytes = spec[0, 4] * spec[0, 6]
    end = src.shape[0]
    tail = spec[last, 8]
    while True:
        lin = 0
        c0 = 0
        o0 = 0
        some = True
        for d in range(rank):
            lin += g[d] * spec[d, 5]
            q = g[d] * spec[d, 4]
            s = spec[d, 2]
            st = spec[d, 8]
            lo = max(s, q)
            lo = s + (lo - s + st - 1) // st * st           # the first it wants here
            hi = min(spec[d, 3], q + spec[d, 4])
            if lo >= hi:
                some = False                                # (steps past this chunk)
                break
            n[d] = (hi - lo - 1) // st + 1
            c0 += (lo - q) * spec[d, 6]
            o0 += (lo - s) // st * spec[d, 7]
        if some:
            row = table[lin]
            if row < 0:
                return False
            a = addr[row]
            if a < 0 or a + cbytes > end:
                return False
            a += c0
            run = n[last] * esize
            while True:
                s = a
                o = o0
                for d in range(last):
                    s += idx[d] * spec[d, 8] * spec[d, 6]
                    o += idx[d] * spec[d, 7]
                if tail != 1:
                    for j in range(n[last]):
                        if swap:
                            _swapped(src, s + j * tail * esize, out, o + j * esize, 1, esize)
                        else:
                            for k in range(esize):
                                out[o + j * esize + k] = src[s + j * tail * esize + k]
                elif swap:
                    _swapped(src, s, out, o, n[last], esize)
                else:
                    k = 0
                    while k + 8 <= run:
                        st64(out, o + k, ld64(src, s + k))
                        k += 8
                    while k < run:
                        out[o + k] = src[s + k]
                        k += 1
                d = last - 1
                while d >= 0:
                    idx[d] += 1
                    if idx[d] < n[d]:
                        break
                    idx[d] = 0
                    d -= 1
                if d < 0:
                    break
        d = last
        while d >= 0:
            g[d] += 1
            if g[d] <= spec[d, 1]:
                break
            g[d] = spec[d, 0]
            d -= 1
        if d < 0:
            return True


@njit(cache=True, nogil=True)
def _place(sc, s0, nel, out, cstart, count, ostart, cstr, ostr, step, rank, esize,
           shuffled, swap):
    """Copy the selected elements of a decoded chunk (sc[s0:], nel elements,
    C order, chunk strides cstr in elements) into out (output strides ostr
    in elements), undoing a byte shuffle and swapping byte order if asked."""
    idx = np.zeros(MAXDIM, np.int64)
    last = rank - 1
    n_last = count[last]
    st_last = step[last]
    while True:
        # element offsets of this run's first element
        se = 0
        de = 0
        for d in range(rank):
            se += (cstart[d] + idx[d] * step[d]) * cstr[d]
            de += (ostart[d] + idx[d]) * ostr[d]
        if not shuffled and not swap:
            if st_last == 1:
                a = s0 + se * esize
                b = de * esize
                nb = n_last * esize
                k = 0
                while k + 8 <= nb:
                    st64(out, b + k, ld64(sc, a + k))
                    k += 8
                while k < nb:
                    out[b + k] = sc[a + k]
                    k += 1
            else:
                for j in range(n_last):
                    a = s0 + (se + j * st_last) * esize
                    b = (de + j) * esize
                    for k in range(esize):
                        out[b + k] = sc[a + k]
        elif shuffled:
            if st_last == 1 and (esize == 2 or esize == 4 or esize == 8):
                # whole words at a time (a swap is the byte planes reversed)
                p0 = s0 + se
                b = de * esize
                if esize == 2:
                    if swap:
                        _unshuffle2(sc, p0 + nel, p0, out, b, n_last)
                    else:
                        _unshuffle2(sc, p0, p0 + nel, out, b, n_last)
                elif esize == 4:
                    if swap:
                        _unshuffle4(sc, p0 + 3 * nel, p0 + 2 * nel, p0 + nel, p0, out, b, n_last)
                    else:
                        _unshuffle4(sc, p0, p0 + nel, p0 + 2 * nel, p0 + 3 * nel, out, b, n_last)
                else:
                    _unshuffle8(sc, p0, nel, swap, out, b, n_last)
            elif swap:
                for j in range(n_last):
                    e = se + j * st_last
                    b = (de + j) * esize
                    for k in range(esize):
                        out[b + esize - 1 - k] = sc[s0 + k * nel + e]
            elif esize == 4:
                p0 = s0 + se
                p1 = p0 + nel
                p2 = p1 + nel
                p3 = p2 + nel
                b = de * 4
                for j in range(n_last):
                    q = j * st_last
                    out[b] = sc[p0 + q]
                    out[b + 1] = sc[p1 + q]
                    out[b + 2] = sc[p2 + q]
                    out[b + 3] = sc[p3 + q]
                    b += 4
            elif esize == 2:
                p0 = s0 + se
                p1 = p0 + nel
                b = de * 2
                for j in range(n_last):
                    q = j * st_last
                    out[b] = sc[p0 + q]
                    out[b + 1] = sc[p1 + q]
                    b += 2
            elif esize == 8:
                p0 = s0 + se
                b = de * 8
                for j in range(n_last):
                    q = p0 + j * st_last
                    for k in range(8):
                        out[b + k] = sc[q + k * nel]
                    b += 8
            else:
                for j in range(n_last):
                    e = se + j * st_last
                    b = (de + j) * esize
                    for k in range(esize):
                        out[b + k] = sc[s0 + k * nel + e]
        elif st_last == 1:
            _swapped(sc, s0 + se * esize, out, de * esize, n_last, esize)
        else:
            for j in range(n_last):
                a = s0 + (se + j * st_last) * esize
                b = (de + j) * esize
                for k in range(esize):
                    out[b + esize - 1 - k] = sc[a + k]
        # next run (odometer over all but the last dimension)
        d = last - 1
        while d >= 0:
            idx[d] += 1
            if idx[d] < count[d]:
                break
            idx[d] = 0
            d -= 1
        if d < 0:
            return


# per-variable parameter table columns
V_ESIZE, V_RANK, V_NEL, V_CBYTES, V_SWAP, V_NST, V_OUT, V_STAGE0 = range(8)
MAXSTAGE = 8
V_PROBE = V_STAGE0 + MAXSTAGE   # the byte plane to test + 1 (0: none), then two byte values
V_COLS = V_PROBE + 3

_LO = np.uint64(0x0101010101010101)
_HI = np.uint64(0x8080808080808080)
_NONE = np.uint64(0)


@njit(cache=True, nogil=True)
def _holds(a, lo, n, b0, b1):
    """Whether a[lo:lo + n] holds the byte b0 or b1, a word at a time (the
    usual test for a zero byte: exact on whether there is one)."""
    w0 = np.uint64(b0) * _LO
    w1 = np.uint64(b1) * _LO
    seen = _NONE
    n8 = n // 8
    for k in range(n8):
        x = ld64(a, lo + k * 8)
        y0 = x ^ w0
        y1 = x ^ w1
        seen |= ((y0 - _LO) & ~y0) | ((y1 - _LO) & ~y1)
    if (seen & _HI) != _NONE:
        return True
    for k in range(lo + n8 * 8, lo + n):
        if a[k] == b0 or a[k] == b1:
            return True
    return False


@njit(cache=True, nogil=True)
def _box_holds(a, p0, C, cstart, count, cstr, step, rank, b0, b1):
    """_holds over what a selection takes of the byte plane at a[p0:] (a
    chunk's corner past the variable's edge holds fill values: no value of
    the variable). Each run goes from its first to its last element, steps
    or not; whole trailing dimensions make longer runs."""
    k = rank - 1
    while k > 0 and count[k] == C[k] and step[k] == 1:
        k -= 1
    run = ((count[k] - 1) * step[k] + 1) * cstr[k]
    idx = np.zeros(rank, np.int64)
    while True:
        e = cstart[k] * cstr[k]
        for d in range(k):
            e += (cstart[d] + idx[d] * step[d]) * cstr[d]
        if _holds(a, p0 + e, run, b0, b1):
            return True
        d = k - 1
        while d >= 0:
            idx[d] += 1
            if idx[d] < count[d]:
                break
            idx[d] = 0
            d -= 1
        if d < 0:
            return False


@njit(cache=True, nogil=True)
def holds_value(a, lo, hi, w0, w1, low):
    """Whether a value in a[lo:hi] (hi - lo a multiple of its size) equals
    one of two, bit for bit (value_words gives w0, w1 and low). A value is
    zero after the xor exactly if ~(((y & low) + low) | y | low) has a bit
    in its place: no carry leaves one. Under a word in all it cannot tell:
    True."""
    n = hi - lo
    if n < 8:
        return True
    seen = _NONE
    n8 = n // 8
    for k in range(n8):
        x = ld64(a, lo + k * 8)
        y = x ^ w0
        z = x ^ w1
        seen |= ~(((y & low) + low) | y | low) | ~(((z & low) + low) | z | low)
    # (what is left over, in the word ending at hi: its places line up)
    x = ld64(a, hi - 8)
    y = x ^ w0
    z = x ^ w1
    seen |= ~(((y & low) + low) | y | low) | ~(((z & low) + low) | z | low)
    return seen != _NONE


def value_words(values):
    """(the first value repeated over a word, the last, and a word with all
    but the top bit of each value's place set) for holds_value."""
    es = values.dtype.itemsize
    b = values.tobytes()
    low = np.full(8 // es, (1 << (8 * es - 1)) - 1, 'u%d' % es).tobytes()
    return tuple(np.frombuffer(w, np.uint64)[0] for w in (b[:es] * (8 // es), b[-es:] * (8 // es), low))


@njit(cache=True, nogil=True)
def _work(src, out, V, VD, addr, size, mask, var, needend, direct, cstart, count, ostart,
          counter, status, scratch, tab, iws, band, progress, clean):
    """Take chunks off counter[0] until none is left: decode each into
    scratch (or straight into out) and place its selected elements.

    One call serves a batch of variables: chunk i belongs to variable
    var[i], whose parameters are row V[var[i]] (element size, rank, chunk
    elements and bytes, byte swap, the filter ids in decoding order, the
    byte offset of its output in out) and VD[var[i]] (chunk shape, chunk
    and output strides in elements, selection steps). status[i]: 0 ok,
    < 0 a decoder's error, 1 a filter this kernel does not run.
    band (if not empty): chunk i's band of the output, progress[band[i]]
    counted up once it is placed (the GPU path uploads each band when all
    its chunks are in).
    clean[v] counts variable v's chunks whose probed byte plane (V_PROBE)
    was found free of both byte values.
    """
    n = addr.shape[0]
    track = band.shape[0] > 0
    while True:
        i = take(counter, 0)
        if i >= n:
            return
        _work_one(src, out, V, VD, addr, size, mask, var, needend, direct, cstart, count,
                  ostart, status, scratch, tab, iws, i, clean)
        if track:
            take(progress, band[i])


@njit(cache=True, nogil=True)
def _work_one(src, out, V, VD, addr, size, mask, var, needend, direct, cstart, count, ostart,
              status, scratch, tab, iws, i, clean):
    """Chunk i of _work's batch."""
    v = var[i]
    esize = V[v, V_ESIZE]
    rank = V[v, V_RANK]
    nel = V[v, V_NEL]
    cbytes = V[v, V_CBYTES]
    swap = V[v, V_SWAP] == 1
    nst = V[v, V_NST]
    obase = V[v, V_OUT]
    p = addr[i]
    pend = p + size[i]
    m = mask[i]
    # a byte shuffle (applied before compression when written) spreads
    # every byte plane over the chunk: no early stop; placement unshuffles
    shuffled = False
    for k in range(nst):
        if V[v, V_STAGE0 + k] == F_SHUFFLE and not (m >> (nst - 1 - k)) & 1:
            shuffled = esize > 1
    in_scratch = False
    ok = 0
    placed = False
    # fletcher32 listed after deflate in decoding order was applied first
    # when writing: its checksum trails the decompressed bytes
    # (and if it came before the shuffle too, the byte planes include it)
    tail = 0
    planes = nel
    seen_deflate = False
    seen_shuffle = False
    for k in range(nst):
        if (m >> (nst - 1 - k)) & 1:
            continue
        fk = V[v, V_STAGE0 + k]
        if fk == F_DEFLATE:
            seen_deflate = True
        elif fk == F_SHUFFLE:
            seen_shuffle = True
        elif fk == F_FLETCHER and seen_deflate:
            tail = 4
            if seen_shuffle:
                planes = (cbytes + 4) // esize
    for k in range(nst):
        f = V[v, V_STAGE0 + k]
        if (m >> (nst - 1 - k)) & 1:
            continue                            # filter skipped for this chunk
        if f == F_FLETCHER:
            if not in_scratch:
                pend -= 4                       # checksum of the stored bytes
        elif f == F_SHUFFLE:
            pass
        elif f == F_DEFLATE:
            if in_scratch:
                ok = 1                          # two compressors: not here
                break
            if direct[i] >= 0 and tail == 0:
                o = obase + direct[i]
                r = _inflate.zlib_inflate(src, p, pend, out, o, o + cbytes, o + cbytes, tab, iws)
                ok = r if r < 0 else 0
                placed = True
                break
            stop = cbytes + tail
            need = needend[i] * esize
            if not shuffled and need < cbytes:
                stop = need                     # the rest of the chunk is not wanted
            r = _inflate.zlib_inflate(src, p, pend, scratch, 0, cbytes + tail, stop, tab, iws)
            if r < 0:
                ok = r
                break
            in_scratch = True
        else:
            ok = 1
            break
    status[i] = ok
    if ok != 0 or placed:
        return
    if in_scratch:
        # the chunk's byte planes are here: a value's top byte missing from
        # its plane means the value is nowhere in the chunk
        if shuffled and V[v, V_PROBE] != 0:
            pl = (V[v, V_PROBE] - 1) * planes
            b0 = V[v, V_PROBE + 1]
            b1 = V[v, V_PROBE + 2]
            if not _holds(scratch, pl, nel, b0, b1) or not _box_holds(
                    scratch, pl, VD[v, 0], cstart[i], count[i], VD[v, 1], VD[v, 3], rank, b0, b1):
                take(clean, v)
        _place(scratch, 0, planes, out[obase:], cstart[i], count[i], ostart[i], VD[v, 1],
               VD[v, 2], VD[v, 3], rank, esize, shuffled, swap)
    else:
        _place(src, p, nel, out[obase:], cstart[i], count[i], ostart[i], VD[v, 1],
               VD[v, 2], VD[v, 3], rank, esize, shuffled, swap)


@njit(cache=True, nogil=True)
def _swap_inplace(a, esize, counter, n):
    """Reverse each element's bytes in a[0:n*esize], in blocks taken off a counter."""
    blk = 65536
    nb = (n + blk - 1) // blk
    while True:
        i = take(counter, 0)
        if i >= nb:
            return
        e0 = i * blk
        _swapped(a, e0 * esize, a, e0 * esize, min(n, e0 + blk) - e0, esize)


# ---------------------------------------------------------------- driver

_TLS = threading.local()


def _scratch(nbytes):
    box = getattr(_TLS, 'box', None)
    if box is None:
        box = _TLS.box = [None, 0]
    s = _pool.kept(box, nbytes)
    if getattr(_TLS, 'tab', None) is None:
        _TLS.tab, _TLS.iws = _inflate.workspace()
        if NATIVE_INFLATE:
            from . import _zlib
            _TLS.iws[_inflate.NATIVE:_inflate.NATIVE + 2] = _zlib.native()
    return s, _TLS.tab, _TLS.iws


NATIVE_INFLATE = True           # whole chunks through libdeflate where it is installed


SUPPORTED = (F_DEFLATE, F_SHUFFLE, F_FLETCHER)
_EMPTY = np.zeros(0, np.int64)
_EMPTY1 = np.zeros(1, np.int64)


class Item:
    """One dataset's part of a batch: its chunk index, the box to read, the
    output array (one dimension per dataset dimension, C-contiguous, in a
    shared byte buffer at byte offset `base`), whether to swap bytes, and
    what to test while decoding (a _maskscale.Probe, or None)."""
    __slots__ = ('ds', 'index', 'box', 'out', 'swap', 'base', 'probe')

    def __init__(self, ds, index, box, out, swap, base=0, probe=None):
        self.ds, self.index, self.box, self.out, self.swap, self.base = ds, index, box, out, swap, base
        self.probe = probe


_READY = threading.Event()      # numba and the kernels loaded
_LOADING = []

# When the kernels load. Loading them (numba's import, then the kernels from
# its cache) takes most of a second of a fresh process, holding the GIL
# nearly throughout; loaded in the background it slowed a first read
# threefold (the pool's threads queue for the GIL after every chunk). So
# reads do without them -- zlib or libdeflate on every core, numpy
# (_read_python, _maskscale.run_plan_numpy) -- until those have taken as
# long as loading takes on this machine, then load them, on the reading
# thread, before the next read: a script reading a few files never pays for
# them; a session that keeps reading pays once, after it has spent about as
# long doing without them (the ski-rental rule: never more than twice the
# better of the two). The load time is noted whenever a load happens.
LOAD_AFTER = None               # seconds (None: this machine's load time)
LOAD_GUESS = 1.0                # ... before any load was timed here
LOAD_LEAST = 0.25               # ... at least (no import of numba is shorter; a mistaken
                                # timing here would load the kernels in the middle of reads)
LOAD_MOST = 3.0                 # ... and at most (a compile is not a load)
_SLOW = [0.0]                   # seconds read without the kernels so far


def _load_after():
    if LOAD_AFTER is not None:
        return LOAD_AFTER
    return max(LOAD_LEAST, min(_hw.tuned().get('load_s') or LOAD_GUESS, LOAD_MOST))


def slow(seconds):
    """Count time spent reading without the kernels."""
    _SLOW[0] += seconds


IMAGE = True                    # kernels from this machine's image (_image), where it has one


def loaded():
    """Whether kernels can be called now without waiting: numba's are loaded,
    or an image of them is there, which takes no time."""
    if _READY.is_set():
        return True
    if IMAGE:
        from . import _image
        if _image.open_image() is not None:
            _READY.set()
            return True
    return False


def ready():
    """Whether to use the kernels now: once loaded -- and loading them
    here if reading without them has taken as long as that would."""
    if loaded():
        return True
    if _SLOW[0] >= _load_after():
        if IMAGE:
            from . import _image
            if _image.coming():
                return False
        from . import _jit
        if not _jit.can_load():                     # (no numba, no image: numpy and zlib it is)
            return False
        warm(wait=True)
        return True
    return False


def _warm():
    """Import numba and load (from cache, or compile) the batch kernel with
    the argument types reads use, off the caller's thread; the time it
    took is kept for this machine (the shortest seen: a compile or a busy
    moment is not what a load costs -- and neither is a load in a process
    that had numba imported already, which is a fraction of one)."""
    cold = 'numba' not in sys.modules
    t0 = time.perf_counter()
    try:
        i64 = np.zeros(1, np.int64)
        i2 = np.zeros((1, 1), np.int64)
        u8 = np.zeros(16, np.uint8)
        ro = np.frombuffer(bytes(16), np.uint8)    # the file: a read-only view of its mapping
        tab, iws = _inflate.workspace()
        _work(ro, u8, np.zeros((1, V_COLS), np.int64), np.zeros((1, 4, 1), np.int64),
              i64[:0], i64[:0], i64[:0], i64[:0], i64[:0], i64[:0], i2[:0], i2[:0], i2[:0],
              np.zeros(1, np.int64), i64[:0], u8, tab, iws, i64[:0], i64[:1], i64[:1])
        _pinflate.warm()
        took = round(time.perf_counter() - t0, 3)
        if cold and took < LOAD_MOST and took < (_hw.tuned().get('load_s') or LOAD_MOST):
            _hw.save_tuned(cpu={'load_s': took})
    finally:
        _READY.set()
    if IMAGE:
        from . import _image
        _image.build_later()                        # the next process need not do this


def _start_warm():
    if not _LOADING:
        _LOADING.append(threading.Thread(target=_warm, name='fastnc-warm', daemon=True))
        _LOADING[0].start()


def warm(wait=True):
    """Load numba and fastnc's kernels now (otherwise they load once reading
    without them has taken a second in all: see LOAD_AFTER). For a session
    that will read a lot, or worker processes:
    ProcessPoolExecutor(initializer=fastnc.warm)."""
    from . import _jit
    if loaded() or not _jit.can_load():             # (no numba here: an image or nothing)
        return
    _start_warm()
    if wait:
        _READY.wait()


PAR_MIN = 4 << 20              # compressed bytes from which a chunk is inflated on many threads


def _stages(V, v, m):
    """Chunk with filter mask m of variable v: (deflate runs on it, its
    compressed bytes end this many bytes before the stored end (fletcher32
    checksum), its bytes come out shuffled, a checksum trails its
    decompressed bytes)."""
    nst = V[v, V_NST]
    esize = V[v, V_ESIZE]
    deflate, cut, shuffled, tail = False, 0, False, False
    for k in range(nst):
        if (m >> (nst - 1 - k)) & 1:
            continue                                # filter skipped for this chunk
        f = V[v, V_STAGE0 + k]
        if f == F_FLETCHER:
            if deflate:
                tail = True
            else:
                cut += 4
        elif f == F_SHUFFLE:
            shuffled = esize > 1
        elif f == F_DEFLATE:
            if deflate:
                return False, 0, False, False       # twice: left to _work (which refuses it)
            deflate = True
    return deflate, cut, shuffled, tail


def _big(V, var, size, mask, least):
    """Which chunks (sorted biggest first) are deflated and at least `least`
    compressed bytes: inflated on many threads (a checksum after the
    deflate keeps a chunk on the serial path)."""
    big = np.zeros(len(size), bool)
    for i in range(len(size)):
        if size[i] < least:
            break
        deflate, cut, _, tail = _stages(V, var[i], mask[i])
        big[i] = deflate and not tail and size[i] - cut >= least
    return big


def _read_big(src, buf, V, VD, bigs, nthreads, items, extra, total):
    """The big chunks inflated together, each cut into parts decoded at
    once on the pool, `extra` (the other chunks) run by each thread once
    the parts are taken: straight into the output when a chunk is one run
    of it, else into a scratch buffer (stopping once the selection's last
    element is out, unless shuffled) then placed, in bands on the pool."""
    addr, size, mask, var, needend, direct, cstart, count, ostart = bigs
    rows = []                                       # p, pend, o, oend, ostop, which
    tmpn = 0
    for i in range(len(addr)):
        v = var[i]
        cbytes = int(V[v, V_CBYTES])
        _, cut, shuffled, _ = _stages(V, v, mask[i])
        p, pend = int(addr[i]), int(addr[i] + size[i] - cut)
        if direct[i] >= 0:
            o = int(V[v, V_OUT] + direct[i])
            rows.append((p, pend, o, o + cbytes, o + cbytes, 0))
        else:
            need = int(needend[i] * V[v, V_ESIZE])
            stop = cbytes if shuffled or need >= cbytes else need
            rows.append((p, pend, tmpn, tmpn + cbytes, tmpn + stop, 1))
            tmpn += cbytes
    tmp = _bigtmp(tmpn) if tmpn else buf
    a = np.array(rows, np.int64)
    try:
        r = _pinflate.inflate_many(src, a[:, 0], a[:, 1], buf, a[:, 2], a[:, 3], a[:, 4],
                                   nthreads, dst2=tmp, which=a[:, 5], extra=extra, total=total)
        bad = np.nonzero(r < 0)[0]
        if len(bad):
            i = bad[0]
            raise IOError('chunk at byte %d of %s could not be decoded (error %d)'
                          % (addr[i], items[var[i]].ds.name, r[i]))
        if tmpn:
            _place_big(buf, tmp, a, V, VD, var, mask, cstart, count, ostart, nthreads)
    finally:
        if tmpn:
            _bigtmp_keep(tmp)


def _place_big(buf, tmp, a, V, VD, var, mask, cstart, count, ostart, nthreads):
    """The big chunks inflated into tmp (rows of `a` with which = 1) placed:
    each chunk's selection cut into bands along its longest dimension but
    the last (rows stay whole), about two per thread in all, on the pool."""
    ind = np.nonzero(a[:, 5])[0]
    whole = sum(int(np.prod(count[i])) for i in ind)
    bands = []
    for i in ind:
        c = count[i]
        rank = int(V[var[i], V_RANK])
        d = int(np.argmax(c[:rank - 1])) if rank > 1 else 0
        nb = max(1, min(int(c[d]), int(round(2 * nthreads * np.prod(c) / max(1, whole)))))
        for b in range(nb):
            r0, r1 = c[d] * b // nb, c[d] * (b + 1) // nb
            if r1 > r0:
                bands.append((i, d, r0, r1))

    def place(band):
        i, d, r0, r1 = band
        v = var[i]
        cs, cn, os_ = cstart[i].copy(), count[i].copy(), ostart[i].copy()
        cs[d] += r0 * VD[v, 3, d]
        cn[d] = r1 - r0
        os_[d] += r0
        _, _, shuffled, _ = _stages(V, v, mask[i])
        _place(tmp, a[i, 2], V[v, V_NEL], buf[V[v, V_OUT]:], cs, cn, os_, VD[v, 1], VD[v, 2],
               VD[v, 3], V[v, V_RANK], V[v, V_ESIZE], shuffled, V[v, V_SWAP] == 1)
    _pool.map_items(place, bands)


_BIGTMP_LOCK = threading.Lock()
_BIGTMP = [None]                # big chunks' inflate buffer, kept for the next read
BIGTMP_KEEP = 256 << 20


def _bigtmp(n):
    with _BIGTMP_LOCK:
        t, _BIGTMP[0] = _BIGTMP[0], None
    return t if t is not None and t.size >= n else np.empty(n, np.uint8)


def _bigtmp_keep(t):
    if t.size <= _hw.scaled(BIGTMP_KEEP):
        with _BIGTMP_LOCK:
            if _BIGTMP[0] is None or _BIGTMP[0].size < t.size:
                _BIGTMP[0] = t


def read_batch(items, src, buf, threads=None, bands=None):
    """Decode the chunks of every item in one pass of the pool, biggest
    chunks first, into buf (uint8; each item's output starts at its base).
    bands (one item): (band of each of its plan's chunks, progress counts)
    -- chunks then go in band order, and progress[b] counts band b's
    chunks as they are placed (_gpu uploads each band once complete)."""
    if src is None:
        raise ValueError('the file is closed')      # (closed under a read: never None into a kernel)
    fused = [fusable(it.ds) for it in items]
    if not all(fused):
        # other filters (szip, scale-offset, n-bit, bzip2, ...), or more of them
        # than the kernels' tables hold: stage by stage
        from . import _staged
        fetch_chunks(items)                         # (all: one round with the fast ones)
        _staged.read_items([it for it, k in zip(items, fused) if not k], src, buf, threads)
        items = [it for it, k in zip(items, fused) if k]
        if not items:
            return
    if all(f.id in SUPPORTED for it in items for f in it.ds.filters) and not ready():
        t = time.perf_counter()
        _read_python(items, src, threads)
        slow(time.perf_counter() - t)
        return
    # (the tables as plain rows, made into arrays once: writing them into
    # arrays a variable at a time was a dozen numpy calls for each of hundreds)
    plans = []
    maxrank = max(1, max(len(it.ds.shape) for it in items))
    zeros = [0] * maxrank
    rows, cshapes, cstrides, ostrides, steps = [], [], [], [], []
    straight = []                                   # (place in the batch, offsets of chunks to inflate in place)
    cmax = n = 0
    for it in items:
        ds, index, box = it.ds, it.index, it.box
        rank = len(ds.shape)
        esize = ds.type.size
        stages = [f.id for f in reversed(ds.filters)]
        p = plan(index, ds.shape, box)
        if p.fill_needed:
            _fill(it.out, ds, it.swap)
        oshape = it.out.shape
        chunks = tuple(index.chunks.tolist())
        ostr = strides(oshape)
        step = [s[2] for s in box]
        nel = math.prod(chunks) if rank else 1
        row = [esize, rank, nel, nel * esize, int(bool(it.swap)), len(stages), it.base] + stages
        row += [0] * (V_COLS - len(row))
        if it.probe is not None and it.probe.values.dtype.itemsize == esize:
            # the top byte of each value, and the plane top bytes are stored in
            b = it.probe.values.tobytes()
            little = sys.byteorder == 'little'
            tops = b[esize - 1::esize] if little else b[::esize]
            row[V_PROBE:V_PROBE + 3] = (esize if little != bool(it.swap) else 1, tops[0], tops[-1])
        rows.append(row)
        rest = zeros[rank:]
        cshapes.append(list(chunks) + rest)
        cstrides.append(strides(chunks).tolist() + rest)
        ostrides.append(ostr.tolist() + rest)
        steps.append(step + rest)
        cmax = max(cmax, nel * esize)
        count = len(p.addr)
        # chunks inflated straight into the output: whole chunk selected,
        # step 1, no shuffle, no swap, and the chunk one run of the output
        # (extent 1 in the leading dimensions, the output's after the first
        # other): their bytes are the output's
        if count and rank and not it.swap and F_SHUFFLE not in stages and all(k == 1 for k in step):
            k1 = next((d for d in range(rank) if chunks[d] > 1), rank - 1)
            if chunks[k1 + 1:] == tuple(oshape[k1 + 1:]):
                full = np.all(p.count == index.chunks, axis=1) & np.all(p.cstart == 0, axis=1)
                lin = (p.ostart * ostr).sum(axis=1) * esize
                straight.append((n, full, lin))
        n += count
        plans.append((p, None))
    if n == 0:
        return
    V = np.array(rows, np.int64)
    VD = np.empty((len(items), 4, maxrank), np.int64)
    VD[:, 0], VD[:, 1], VD[:, 2], VD[:, 3] = cshapes, cstrides, ostrides, steps
    addr = np.concatenate([p.addr for p, _ in plans])
    size = np.concatenate([p.size for p, _ in plans])
    mask = np.concatenate([p.mask for p, _ in plans])
    needend = np.concatenate([p.needend for p, _ in plans])
    direct = np.full(n, -1, np.int64)
    for at, full, lin in straight:
        direct[at:at + len(full)][full] = lin[full]
    var = np.repeat(np.arange(len(plans), dtype=np.int64), [len(p.addr) for p, _ in plans])

    def pad(a):
        if a.shape[1] == maxrank:
            return a
        z = np.zeros((a.shape[0], maxrank), np.int64)
        z[:, :a.shape[1]] = a
        return z
    wide = [p.wide if p.wide is not None and p.wide[0].shape[1] == maxrank
            else (pad(p.cstart), pad(p.count), pad(p.ostart)) for p, _ in plans]
    cstart, count, ostart = (np.concatenate([w[k] for w in wide]) for k in range(3))
    fetch_chunks(items, [p for p, _ in plans])
    band = _EMPTY if bands is None else np.asarray(bands[0], np.int64)
    progress = _EMPTY1 if bands is None else bands[1]
    if n > 1:
        if bands is None:
            order = np.argsort(-size, kind='stable')     # biggest first: no straggler at the end
        else:
            order = np.lexsort((-size, band))            # bands in order, biggest first in each
            band = band[order]
        addr, size, mask, needend, direct, var = (addr[order], size[order], mask[order],
                                                  needend[order], direct[order], var[order])
        cstart, count, ostart = cstart[order], count[order], ostart[order]
    filtered = any(it.ds.filters for it in items)
    src, addr = maybe_stage(items[0].ds, src, addr, size, threads, filtered)
    # a thread for each ~32 kB of output at most: handing tiny reads to the
    # pool costs more than it saves
    outbytes = sum(math.prod(it.out.shape) * it.ds.type.size for it in items)
    allthreads = threads or _pool.threads()
    big = None
    if allthreads > 1 and size[0] >= PAR_MIN:
        # chunks much bigger than a thread's share of the work: cut into parts
        total = int(size.sum())
        big = _big(V, var, size, mask, max(PAR_MIN, total // allthreads))
        if big.any():
            bi = np.nonzero(big)[0]
            bigs = (addr[bi], size[bi], mask[bi], var[bi], needend[bi], direct[bi], cstart[bi],
                    count[bi], ostart[bi])
            keep = ~big
            addr, size, mask, needend, direct, var = (addr[keep], size[keep], mask[keep],
                                                      needend[keep], direct[keep], var[keep])
            if band.size:
                band = band[keep]                   # (big chunks: no progress; the caller
                                                    # sees them when read_batch returns)
            cstart, count, ostart = cstart[keep], count[keep], ostart[keep]
            n = len(addr)
        else:
            big = None
    counter = np.zeros(1, np.int64)
    status = np.zeros(n, np.int64)
    clean = np.zeros(len(items), np.int64)
    # (the work is the decoding: a column of a variable has little output
    # but every chunk to inflate; compressed bytes count double, about
    # what they inflate to. With nothing to decode it is copying)
    if filtered:
        nthreads = _pool.work_threads(max(outbytes, 2 * int(size.sum())), min(allthreads, n))
    else:
        nthreads = _pool.memory_threads(outbytes, min(allthreads, n))
    call = bind(_work, (src, buf, V, VD, addr, size, mask, var, needend, direct, cstart, count,
                        ostart, counter, status) + _scratch(cmax + 8) + (band, progress, clean),
                (15, 16, 17))

    def work():
        call(*_scratch(cmax + 8))
    if big is not None:                             # the other chunks once a thread's parts are done
        _read_big(src, buf, V, VD, bigs, allthreads, items, work if n else None, total)
    else:
        _pool.share(work, nthreads)
    if status.any():
        bad = np.nonzero(status)[0][0]
        raise IOError('chunk at byte %d of %s could not be decoded (error %d)'
                      % (addr[bad], items[var[bad]].ds.name, status[bad]))
    for v, it in enumerate(items):
        # (every chunk has to have been tested: one decoded any other way, a
        # big one in parts, or a part of the selection with no chunk is not)
        if it.probe is not None:
            p = plans[v][0]
            it.probe.proven = not p.fill_needed and 0 < len(p.addr) == clean[v]


def place_staged(it, p, sel, src, addr, size, mask, stages, buf, threads=None):
    """Place chunks already decoded into a staging buffer `src` (chunk i's
    bytes at addr[i], size[i]): the batch kernel with only the remaining
    stages (a final byte shuffle, mask bit per chunk), for the plan rows sel."""
    ds, index = it.ds, it.index
    rank = len(ds.shape)
    esize = ds.type.size
    V = np.zeros((1, V_COLS), np.int64)
    VD = np.zeros((1, 4, max(rank, 1)), np.int64)
    oshape = it.out.shape
    cstr = np.ones(rank, np.int64)
    ostr = np.ones(rank, np.int64)
    for d in range(rank - 2, -1, -1):
        cstr[d] = cstr[d + 1] * index.chunks[d + 1]
        ostr[d] = ostr[d + 1] * oshape[d + 1]
    nel = int(np.prod(index.chunks)) if rank else 1
    V[0, :V_STAGE0] = (esize, rank, nel, nel * esize, int(bool(it.swap)), len(stages), it.base)
    V[0, V_STAGE0:V_STAGE0 + len(stages)] = stages
    VD[0, 0, :rank] = index.chunks
    VD[0, 1, :rank] = cstr
    VD[0, 2, :rank] = ostr
    VD[0, 3, :rank] = [s[2] for s in it.box]
    n = len(sel)
    var = np.zeros(n, np.int64)
    direct = np.full(n, -1, np.int64)
    needend = np.ascontiguousarray(p.needend[sel])
    cstart = np.ascontiguousarray(p.cstart[sel])
    count = np.ascontiguousarray(p.count[sel])
    ostart = np.ascontiguousarray(p.ostart[sel])
    counter = np.zeros(1, np.int64)
    status = np.zeros(n, np.int64)
    addr = np.ascontiguousarray(addr, np.int64)
    size = np.ascontiguousarray(size, np.int64)
    mask = np.ascontiguousarray(mask, np.int64)

    def work():
        sc, tab, iws = _scratch(8)
        _work(src, buf, V, VD, addr, size, mask, var, needend, direct, cstart, count, ostart,
              counter, status, sc, tab, iws, _EMPTY, _EMPTY1, _EMPTY1)
    nbytes = int(np.prod(it.out.shape)) * esize
    _pool.share(work, _pool.work_threads(nbytes, min(threads or _pool.threads(), n)))
    if status.any():
        raise IOError('a staged chunk of %s could not be placed' % ds.name)


def maybe_stage(ds, src, addr, size, threads, filtered):
    """(src, addr), the chunks' bytes staged first if that is the better way:
    a big unfiltered batch from a local file (positional reads into warm
    memory beat faulting in the freshly mapped file's pages; with a
    decompressor at work the faults hide behind it), or a big compressed one
    from storage that makes faults dear (_pio.Reader.slow)."""
    if int(size.sum()) < _pio.STAGE_MIN:
        return src, addr
    f = getattr(ds, 'file', None)                   # (HDF4's datasets have none)
    if filtered and (f is None or getattr(f, 'source', None) is not None
                     or not _pio.slow(src, addr, threads or _pool.threads())):
        return src, addr                            # (no reader made: a handle per file adds up)
    io = _stage_reader(ds)
    return (src, addr) if io is None else io.stage(addr, size, threads)


def _stage_reader(ds):
    """The local file's positional reader (made on first use), or None for
    remote sources and files opened without a path."""
    f = getattr(ds, 'file', None)                   # (HDF4's datasets have none)
    if f is None or getattr(f, 'source', None) is not None or getattr(f, 'path', None) is None:
        return None
    r = getattr(f, '_pio', None)
    if r is None:
        try:
            r = f._pio = _pio.Reader(f.path)
        except OSError:
            f.path = None
            return None
    return r


def fetch_chunks(items, plans=None):
    """The hook for files read from elsewhere (`_remote`): the stored bytes
    of every planned chunk (plans: one Plan per item; None: planned here)
    are fetched into src before anything reads them -- of an unfiltered
    chunk only from its first to its last selected element. Local files:
    one attribute check."""
    source = getattr(getattr(items[0].ds, 'file', None), 'source', None)
    if source is None:
        return
    addr, size = [], []
    for k, it in enumerate(items):
        p = plans[k] if plans is not None else plan(it.index, it.ds.shape, it.box)
        if it.ds.filters or not len(p.addr):
            addr.append(p.addr)
            size.append(p.size)
            continue
        C = it.index.chunks
        cstr = np.ones(len(C), np.int64)
        for d in range(len(C) - 2, -1, -1):
            cstr[d] = cstr[d + 1] * C[d + 1]
        esize = it.ds.type.size
        lo = (p.cstart * cstr).sum(axis=1) * esize if len(C) else np.zeros(len(p.addr), np.int64)
        addr.append(p.addr + lo)
        size.append(np.minimum(p.needend * esize, p.size) - lo)
    source.fetch(np.concatenate(addr), np.concatenate(size))


def read(ds, index, src, sel, out, swap=False, threads=None, bands=None):
    """Decode the chunks of dataset ds touched by sel into out: a C-contiguous
    array with one dimension per dataset dimension, sized to the selection
    (integer indexes as length one). swap: reverse each element's bytes as
    it is placed (big-endian numbers read as native)."""
    if bands is None and (_plain(ds, index, src, sel, out) or (_CACHE and _hit(index, sel, out, swap))
                          or _cached(ds, index, src, sel, out, swap, threads)):
        return out
    read_batch([Item(ds, index, sel, out, swap, 0)], src, out.reshape(-1).view(np.uint8), threads,
               bands)
    return out


CACHE_BYTES = 32 << 20          # decoded chunks kept, all datasets together
CACHE_CHUNKS = 16               # reads touching at most this many chunks go through it
_CACHE = {}                     # (chunk index, address, swap) -> decoded chunk; LRU order
_CACHE_HELD = [0]
_SEEN = {}                      # (id of chunk index, address) of chunks read once (an id
                                # reused after a file is gone: only a chunk kept early)
_CACHE_LOCK = threading.Lock()


def release_cache():
    with _CACHE_LOCK:
        _CACHE.clear()
        _SEEN.clear()
        _CACHE_HELD[0] = 0


def _hit(index, sel, out, swap):
    """A read inside one chunk already decoded, with no plan: all that going
    through a dataset row by row needs for most rows. False: not such a read."""
    C = index.cshape
    if len(sel) != len(C) or out.ndim != len(C):
        return False
    place, inner = [], []
    for (s, e, k), c in zip(sel, C):
        g = s // c
        if (e - 1) // c != g:
            return False
        place.append(g)
        g *= c
        inner.append(slice(s - g, e - g, k))
    key = (index, tuple(place), bool(swap))
    with _CACHE_LOCK:
        h = _CACHE.pop(key, None)
        if h is None:
            return False
        _CACHE[key] = h                             # (now the most recent)
    out.reshape(-1).view(np.uint8).reshape(out.shape + h.shape[-1:])[...] = h[tuple(inner)]
    return True


def cached_chunk(index, place, swap):
    """A decoded chunk at a grid place, now the most recent, or None."""
    key = (index, place, swap)
    with _CACHE_LOCK:
        h = _CACHE.pop(key, None)
        if h is not None:
            _CACHE[key] = h
            index.fresh.discard(place)
    return h


def take_chunk(index, place, swap):
    """A chunk decoded ahead that no read has seen, out of the cache: the
    caller's now, to hand on without a copy. None if there is no such chunk."""
    if place not in index.fresh:
        return None
    index.fresh.discard(place)
    if len(index.ways) > 1:
        _gone(index, [place])
    with _CACHE_LOCK:
        h = _CACHE.pop((index, place, swap), None)
        if h is not None:
            _CACHE_HELD[0] -= h.nbytes
    return h


def _trim_cache():
    """Chunks decoded ahead of a read that has stopped: past the cache's own
    budget only while it is going on. True while they may still be wanted."""
    with _CACHE_LOCK:
        size = _CACHE_HELD[0]
        seen = _pool.CACHE_SEEN
        if size <= _hw.scaled(CACHE_BYTES):
            seen[0] = 0
            return False
        if size != seen[0]:                         # (read from since the last look: going on)
            seen[0] = size
            return True
        while _CACHE_HELD[0] > _hw.scaled(CACHE_BYTES) and _CACHE:
            _CACHE_HELD[0] -= _CACHE.pop(next(iter(_CACHE))).nbytes
        seen[0] = 0
    return False


def drop_file(file):
    """Lets go of the chunks of a closed file: nothing can read them again."""
    with _CACHE_LOCK:
        for k in [k for k in _CACHE if k[0].owner is file]:
            _CACHE_HELD[0] -= _CACHE.pop(k).nbytes


def _bare(ds):
    """The dtype of a dataset whose chunks are its values as they stand in a
    local file, else False."""
    from ._h5read import reduced
    f = getattr(ds, 'file', None)                   # (HDF4's datasets have none)
    if f is None or getattr(f, 'source', None) is not None or ds.filters:
        return False
    t = ds.type
    if t.cls not in (0, 1) or t.dtype is None or reduced(t):
        return False
    return t.dtype


def _plain(ds, index, src, sel, out):
    """A read inside one chunk of uncompressed values: their place in the file
    is arithmetic, and copying them all there is to it. Through the plan and
    the kernel such a read took 95 us, in a loop over rows. False: not such a
    read (or a chunk not stored)."""
    dt = index.bare
    if dt is None:
        dt = index.bare = _bare(ds)
    if dt is False or len(sel) != len(index.cshape) or out.ndim != len(sel) or index.seen is not None:
        return False
    t = index.table
    if t is None:
        t = index.table = index._grid_table()
    lin = at = 0
    steps = []
    for (s, e, k), c, n, st in zip(sel, index.cshape, index.gridlist, index.cstrides(dt.itemsize)):
        g = s // c
        if (e - 1) // c != g:
            return False
        lin = lin * n + g
        at += (s - g * c) * st
        steps.append(k * st)
    if t is False:
        if len(index.addr) != 1 or lin or index.offsets.any():
            return False                            # (a lone chunk need not be the first)
        row = 0
    else:
        row = t[lin]
        if row < 0:
            return False
    out[...] = np.ndarray(out.shape, dt, src, int(index.addr[row]) + at, tuple(steps))
    return True


def _beside(places, rank):
    """{a grid place next to these: (the dimension, the direction) to it}."""
    return {g[:d] + (g[d] + k,) + g[d + 1:]: (d, k) for g in places for d in range(rank) for k in (-1, 1)}


class _Way:
    """One thread's way through a dataset. Threads going through a file by
    turns had one between them: each read looked like a jump from the other's,
    nothing was decoded ahead, and two threads took twice as long as one."""
    __slots__ = 'beside', 'last', 'ahead'

    def __init__(self):
        self.beside = {}                            # grid place next to the chunks last decoded ->
        #                                             the way to it (dimension, direction)
        self.last = None                            # the place a read last started at (see _went)
        self.ahead = 0                              # chunks to decode ahead of a read going along


def _way(index):
    """This thread's _Way through a dataset."""
    ways = index.ways
    me = threading.get_ident()
    w = ways.get(me)
    if w is None:
        if len(ways) >= 64:                         # (threads come and go)
            ways.clear()
        w = ways[me] = _Way()
    return w


def _along(index, places):
    """The way to these from any thread's last read, or None: one thread's
    reads may carry on from another's."""
    for w in list(index.ways.values()):
        beside = w.beside
        for g in places:
            if g in beside:
                return beside[g]
    return None


def _went(index, places, rank):
    """Where this thread's read of these places may go next: next to them, or
    as far along one dimension again as they are from its last read's. Every
    sixth step of a variable is a loop like any other, and went a chunk at a
    time on one thread: 2 ms a step, against 0.5 for every step."""
    w = _way(index)
    way = _beside(places, rank)
    was, w.last = w.last, places[0]
    if was is not None:
        moved = [(d, b - a) for d, (a, b) in enumerate(zip(was, places[0])) if a != b]
        if len(moved) == 1 and abs(moved[0][1]) > 1:
            d, k = moved[0]
            way.update({g[:d] + (g[d] + k,) + g[d + 1:]: (d, k) for g in places})
    w.beside = way


def _ahead(index, shape, origins, way, count):
    """The boxes of the chunks a read going on along way (a dimension, a
    direction, a stride) comes to next, count steps of them past the chunks at
    origins; none past the end. One box if the chunks are next to each other,
    so that they are decoded in one pass of the kernel, however many and
    however small: a row each, as in the Ku-band file, cost 100 us a row one
    at a time."""
    if count < 1:
        return []
    d, k = way
    C = index.chunks
    lo, hi = origins.min(axis=0), (origins + C).max(axis=0)
    box = [(int(a), min(int(b), n), 1) for a, b, n in zip(lo, hi, shape)]
    if abs(k) > 1:
        boxes = []
        for j in range(1, count + 1):
            a, b = max(int(lo[d] + j * k * C[d]), 0), min(int(hi[d] + j * k * C[d]), shape[d])
            if a >= b:
                break
            boxes.append(box[:d] + [(a, b, 1)] + box[d + 1:])
        return boxes
    a, b = (int(hi[d]), int(hi[d] + count * C[d])) if k > 0 else (int(lo[d] - count * C[d]), int(lo[d]))
    a, b = max(a, 0), min(b, shape[d])
    if a >= b:
        return []
    box[d] = (a, b, 1)
    return [box]


def _claim(index, places):
    """These places are this read's to decode: others wait for them, and do
    not decode them ahead. What _unclaim needs to say it is done."""
    import os
    done = threading.Event()
    done.pid = os.getpid()
    for g in places:
        index.coming[g] = done
    return done


def _unclaim(index, places, done):
    with _CACHE_LOCK:
        for g in places:
            index.coming.pop(g, None)
    done.set()


def _gone(index, places):
    """These were read and are in no cache: a thread behind the others is not
    to decode them ahead. (Only some 16,000: a later pass may read them again.)"""
    if len(index.gone) >= 1 << 14:
        index.gone.clear()
    index.gone.update(places)


def _places(box, C):
    """The grid places of the chunks a box covers."""
    from itertools import product
    return list(product(*[range(s // c, (e - 1) // c + 1) for (s, e, _), c in zip(box, C)]))


def _unclaimed(index, boxes, way, swap):
    """Of boxes to decode ahead, what is neither decoded nor being decoded: the
    run of chunks nearest the read along way. Threads going through one file by
    turns decoded the same chunks: 610 at 8 threads, for 240 read. (Holding
    the cache's lock.)"""
    if not way:
        return boxes
    d, k = way
    C = index.cshape

    def taken(box, j):
        slab = _places(box[:d] + [(j * C[d], (j + 1) * C[d], 1)] + box[d + 1:], C)
        return any(g in index.coming or g in index.gone or (index, g, swap) in _CACHE for g in slab)
    out = []
    for box in boxes:
        free = []
        lo, hi = box[d][0] // C[d], (box[d][1] - 1) // C[d] + 1
        for j in range(lo, hi) if k > 0 else range(hi - 1, lo - 1, -1):
            if not taken(box, j):
                free.append(j)
            elif free:
                break
        if free:
            a, b = min(free) * C[d], (max(free) + 1) * C[d]
            out.append(box[:d] + [(max(a, box[d][0]), min(b, box[d][1]), 1)] + box[d + 1:])
    return out


def _keep_box(index, shape, box, buf, swap, way):
    """The decoded box's chunks into the cache, each a view of it (they go
    out together: they were put in together, and are read in turn). Past them
    a read is still going the same way, not jumping: decoded twice over and
    with no chunks ahead, otherwise, at the end of every box."""
    C = index.cshape
    q = plan(index, shape, box)
    first = [b[0] for b in box]
    places = (q.ostart + np.array(first, np.int64) - q.cstart) // index.chunks
    made = 0
    kept = []
    with _CACHE_LOCK:
        for g in places.tolist():
            s = tuple(slice(p * c - f, min((p + 1) * c, n) - f) for p, c, f, n in zip(g, C, first, shape))
            h = buf[s]
            g = tuple(g)
            kept.append(g)
            key = (index, g, swap)
            old = _CACHE.pop(key, None)
            if old is not None:
                _CACHE_HELD[0] -= old.nbytes
            _CACHE[key] = h
            made += h.nbytes
        _CACHE_HELD[0] += made
    index.fresh.update(kept)
    beside = _way(index).beside
    beside.update(_beside(kept, len(C)))
    d, k = way
    if abs(k) > 1:
        beside.update({g[:d] + (g[d] + k,) + g[d + 1:]: way for g in kept})


def _cached(ds, index, src, sel, out, swap, threads):
    """Small parts of a few shuffled chunks, read through a cache of decoded
    chunks: a loop reading points, or one level of a variable, decoded the
    whole chunk again on every read (a byte shuffle allows no early stop).
    The chunks not in the cache are decoded whole by one read_batch (the
    same values, on as many threads) and kept (least recently used out
    first, CACHE_BYTES in all; fastnc.release() empties it), and the
    selection is copied out of them. False: not such a read."""
    rank = len(ds.shape)
    esize = ds.type.size
    if rank == 0 or out.ndim != rank or not ds.filters:
        return False
    whole = out.size == math.prod(ds.shape)
    if whole and _SEEN.setdefault((id(index), 'all'), False) is False:
        _SEEN[id(index), 'all'] = True              # (all of it: only ever worth keeping if read again)
        return False
    C = index.chunks
    cbytes = int(np.prod(C)) * esize
    budget = _hw.scaled(CACHE_BYTES)
    if cbytes * 2 > budget:
        return False
    p = plan(index, ds.shape, sel)
    n = len(p.addr)
    if n == 0 or n > CACHE_CHUNKS or n * cbytes * 2 > budget:
        return False
    start = np.array([s[0] for s in sel], np.int64)
    step = np.array([s[2] for s in sel], np.int64)
    origin = start + p.ostart * step - p.cstart         # each chunk's first element
    swap = bool(swap)
    hits = [None] * n
    places = [tuple(g) for g in (origin // C).tolist()]
    if index.coming:                                    # (another thread is decoding what this wants)
        import os                                       # (a claim from before a fork: its thread is not here)
        me = os.getpid()
        with _CACHE_LOCK:
            wait = {e for e in map(index.coming.get, places) if e is not None and e.pid == me}
        if wait:
            for e in wait:
                e.wait()
            return _cached(ds, index, src, sel, out, swap, threads)
    if out.nbytes * 2 > n * cbytes and _along(index, places) is None and not any(g in index.fresh for g in places):
        # most of its chunks: nothing to keep them for, unless it is going
        # through them, a chunk a read, when the next ones are worth decoding,
        # or it read just these before (a small variable, read over and over)
        with _CACHE_LOCK:
            again = whole or all((id(index), g) in _SEEN for g in places)
            for g in places:
                _SEEN[id(index), g] = True
        if not again:
            _went(index, places, rank)
            if len(index.ways) < 2:
                return False
            done = _claim(index, places)                # (not for another thread to decode ahead)
            try:
                read_batch([Item(ds, index, sel, out, swap, 0)], src, out.reshape(-1).view(np.uint8), threads)
            finally:
                _unclaim(index, places, done)
            _gone(index, places)
            return True
    if p.fill_needed:
        _fill(out, ds, swap)
    with _CACHE_LOCK:
        for i in range(n):
            key = (index, places[i], swap)
            h = _CACHE.pop(key, None)
            if h is not None:
                _CACHE[key] = h                         # (now the most recent)
                hits[i] = h
    miss = [i for i in range(n) if hits[i] is None]
    if miss:
        # a chunk is kept from its second read on: a read made once (a fresh
        # open's first window) costs what it did, not a whole-chunk decode. One
        # next to the last chunks read is kept at once: a program going through
        # a dataset row by row comes back to it, reads at random do not. So are
        # a few small ones: decoded all the same, keeping them costs less than
        # this read (many, as a crop across rows of a chunk each, cost more)
        grid = [places[i] for i in miss]
        going = _along(index, grid)
        _went(index, grid, rank)
        # ... and the chunks after it are decoded now, on the threads that
        # would otherwise wait: as many as were right so far, twice over
        w = _way(index)
        w.ahead = 0 if going is None else max(1, 2 * w.ahead)
        with _CACHE_LOCK:
            new = [i for i in miss if _SEEN.setdefault((id(index), places[i]), False) is False]
            for i in new:
                _SEEN[id(index), places[i]] = True
            if len(_SEEN) > 1 << 16:
                _SEEN.clear()
        if new and going is None and (cbytes > 1 << 18 or n > 4):
            return False
        todo = [([int(x) for x in origin[i]], places[i], i) for i in miss]
        boxes = []
        index.owner = getattr(ds, 'file', None)     # (HDF4's datasets have none)
        if going is not None:
            # (as far as this machine's cores use them, not the cache's budget:
            # chunks of megabytes would come three at a pass, on 24 threads)
            from . import _arena
            per = cbytes * len(miss)
            room = max(budget // 2, min(_arena.BUDGET // 8, _pool.AHEAD * _pool.threads() * per))
            boxes = _ahead(index, ds.shape, origin[miss], going, min(w.ahead, room // per - 1))
            budget += room
            _arena.wake(_trim_cache)
        mine, done = [], None
        if len(index.ways) > 1:                         # (threads going through one file: each its own)
            with _CACHE_LOCK:
                boxes = _unclaimed(index, boxes, going, swap)
                mine = grid + [g for b in boxes for g in _places(b, index.cshape)]
                done = _claim(index, mine)
        try:
            items, shapes, base = [], [], 0
            for o, _, _ in todo:
                box = [(o[d], min(o[d] + int(C[d]), ds.shape[d]), 1) for d in range(rank)]
                shapes.append(tuple(e - s for s, e, _ in box))
                items.append((box, base))
                base += (int(np.prod(shapes[-1])) * esize + 63) // 64 * 64
            buf = np.empty(base, np.uint8)
            batch = []
            for (box, b0), shape in zip(items, shapes):
                nb = int(np.prod(shape)) * esize
                batch.append(Item(ds, index, box, buf[b0:b0 + nb].view(out.dtype).reshape(shape),
                                  swap, b0))
            read_batch(batch, src, buf, threads)
            if boxes:
                sizes = [tuple(e - s for s, e, _ in b) for b in boxes]
                ends = np.cumsum([0] + [(math.prod(s) * esize + 63) // 64 * 64 for s in sizes]).tolist()
                buf2 = np.empty(ends[-1], np.uint8)
                read_batch([Item(ds, index, b, buf2[o:o + math.prod(s) * esize].view(out.dtype).reshape(s), swap, o)
                            for b, s, o in zip(boxes, sizes, ends)], src, buf2, threads)
                for b, s, o in zip(boxes, sizes, ends):
                    _keep_box(index, ds.shape, b, buf2[o:o + math.prod(s) * esize].reshape(s + (esize,)), swap, going)
            with _CACHE_LOCK:
                for (box, b0), shape, (_, a, i) in zip(items, shapes, todo):
                    nb = int(np.prod(shape)) * esize
                    h = buf[b0:b0 + nb].reshape(shape + (esize,))
                    h = h.copy() if len(todo) > 1 else h    # (each freed on its own)
                    if i is not None:
                        hits[i] = h
                    key = (index, a, swap)
                    old = _CACHE.pop(key, None)
                    if old is not None:
                        _CACHE_HELD[0] -= old.nbytes
                    _CACHE[key] = h
                    _CACHE_HELD[0] += nb
                while _CACHE_HELD[0] > budget and _CACHE:
                    _CACHE_HELD[0] -= _CACHE.pop(next(iter(_CACHE))).nbytes
        finally:
            if done is not None:
                _unclaim(index, mine, done)
    ob = out.reshape(-1).view(np.uint8).reshape(out.shape + (esize,))
    for i in range(n):
        cs, cn, os_ = p.cstart[i], p.count[i], p.ostart[i]
        ob[tuple(slice(int(os_[d]), int(os_[d] + cn[d])) for d in range(rank))] = \
            hits[i][tuple(slice(int(cs[d]), int(cs[d] + (cn[d] - 1) * step[d] + 1), int(step[d]))
                          for d in range(rank))]
    return True


def fusable(ds):
    """Whether the kernels' one pass can decode this dataset: each filter once
    and one of theirs, undone in the one order it goes by (checksum of what
    is stored, inflate, shuffle: any other, such as netCDF-C's checksum under
    the shuffle, is left to the stages, as a chunk may go without a filter that
    is optional), and a shuffle by the size of its elements (its own says:
    PyTables has shuffled 16-byte variable-length records by 8)."""
    filters = ds.filters
    order = [f.id for f in reversed(filters)]
    return order == [x for x in (F_FLETCHER, F_DEFLATE, F_SHUFFLE) if x in order] and all(
        f.id != F_SHUFFLE or not f.cd or f.cd[0] == ds.type.size for f in filters)


def _fill(out, ds, swap=False):
    """The fill value in out, native if its chunks are being swapped as placed
    (HDF5 keeps it in the file's order; HDF4's datasets, native already, have no `file`)."""
    if ds.fill is not None and len(ds.fill) == ds.type.size:
        swap = swap and getattr(ds, 'file', None) is not None
        fill_with(out, ds.fill[::-1] if swap else ds.fill)
    else:
        out.reshape(-1).view(np.uint8)[:] = 0


def fill_with(out, elem):
    """out (C-contiguous) holding the element `elem` (its bytes, as they
    lie in memory) everywhere: a word at a time for elements of 1, 2, 4 or 8
    bytes (numpy's fill: 38 MB of 2-byte fill took 48 ms a byte column at a
    time, 1.4 ms so), a byte at a time for any other."""
    flat = out.reshape(-1).view(np.uint8)
    n = len(elem)
    if n in (1, 2, 4, 8):
        word = 'u%d' % n
        flat.view(word).fill(np.frombuffer(bytes(elem), word)[0])
    else:
        flat.reshape(-1, n)[:] = np.frombuffer(bytes(elem), np.uint8)


def swap_native(out, threads=None):
    """Make a big-endian array native in place, on the pool."""
    esize = out.dtype.itemsize
    n = out.size
    if esize == 1 or n == 0:
        return out.view(out.dtype.newbyteorder('='))
    b = out.reshape(-1).view(np.uint8)
    counter = np.zeros(1, np.int64)
    nb = -(-n // 65536)
    _pool.share(lambda: _swap_inplace(b, esize, counter, n), min(threads or _pool.threads(), nb))
    return out.view(out.dtype.newbyteorder('='))


_PYTLS = threading.local()


def _pybuf(nbytes):
    """This thread's inflate buffer for _quick, at least nbytes."""
    box = getattr(_PYTLS, 'box', None)
    if box is None:
        box = _PYTLS.box = [None, 0]
    return _pool.kept(box, nbytes)


class _Quick:
    """_read_python's common chunks done fast, for one item: deflate (then
    a byte shuffle; a fletcher32 checksum of the stored bytes first)
    inflated by _zlib straight into this thread's buffer -- or into the
    output, when the chunk is a whole contiguous part of it -- and placed
    by numpy copies, one per byte plane when shuffled. Everything per chunk
    that numpy can work out for all chunks at once is worked out here, as
    plain Python numbers: the per-chunk Python runs under the GIL, so on
    24 threads it was the bottleneck (GOES-19 C02's 3,616 chunks)."""

    def __init__(self, it, p, src):
        from . import _zlib
        self.inflate = _zlib.inflate_into
        self.quick = _zlib.stream_inflater(src)
        ds = it.ds
        self.esize = esize = ds.type.size
        self.C = C = tuple(int(c) for c in it.index.chunks)
        rank = len(C)
        nel = int(np.prod(C)) if C else 1
        self.cbytes = cbytes = nel * esize
        self.swap = it.swap
        self.filters = [f.id for f in ds.filters]
        self.specs = {}
        out = it.out
        self.out = out
        step = [s[2] for s in it.box]
        self.step = step
        self.addr = p.addr.tolist()
        self.size = p.size.tolist()
        self.mask = p.mask.tolist()
        self.cs = p.cstart.tolist()
        self.cn = p.count.tolist()
        self.os = p.ostart.tolist()
        whole = np.zeros(len(p.addr), bool)
        if (not it.swap and rank and all(k == 1 for k in step)
                and all(C[d] == out.shape[d] for d in range(1, rank))):
            whole = np.all(p.cstart == 0, axis=1) & np.all(p.count == np.array(C), axis=1)
        self.whole = whole.tolist()
        self.wofs = (p.ostart[:, 0] * (cbytes // C[0])).tolist() if rank else [0] * len(p.addr)
        self.flat = out.reshape(-1).view(np.uint8)
        self.faddr = self.flat.ctypes.data
        if esize > 1:
            self.ob = out.view(np.uint8).reshape(out.shape + (esize,))
        if esize in (1, 2, 4, 8):
            u = np.dtype('u%d' % esize)
            self.u = u.newbyteorder('S') if it.swap else u
            self.ou = out.view(u)

    def spec(self, m):
        """(shuffled, checksum bytes to cut) for filter mask m, or None."""
        sp = self.specs.get(m, False)
        if sp is False:
            f = self.filters
            stages = [f[k] for k in range(len(f) - 1, -1, -1) if not (m >> k) & 1]
            cut = 0
            if stages[:1] == [F_FLETCHER]:
                cut = 4
                stages = stages[1:]
            if stages == [F_DEFLATE]:
                sp = (False, cut)
            elif stages == [F_DEFLATE, F_SHUFFLE]:
                sp = (self.esize > 1, cut)
            else:
                sp = None
            self.specs[m] = sp
        return sp

    def run(self, i, src):
        """Chunk i done; False if not such a chunk (or not of the size
        expected): the general path does it."""
        sp = self.spec(self.mask[i])
        if sp is None:
            return False
        shuffled, cut = sp
        cbytes = self.cbytes
        a, n = self.addr[i], self.size[i] - cut
        f = self.quick
        if self.whole[i] and not shuffled:
            if f is not None and f(a, n, self.faddr + self.wofs[i], cbytes) == cbytes:
                return True
            return self.inflate(src, a, n, self.flat, self.wofs[i], cbytes) == cbytes
        buf = _pybuf(cbytes)
        if f is None or f(a, n, _PYTLS.box[1], cbytes) != cbytes:
            if self.inflate(src, a, n, buf, 0, cbytes) != cbytes:
                return False
        cs, cn, os_ = self.cs[i], self.cn[i], self.os[i]
        src_sl = tuple([slice(c0, c0 + (k - 1) * s + 1, s) for c0, k, s in zip(cs, cn, self.step)])
        dst_sl = tuple([slice(o0, o0 + k) for o0, k in zip(os_, cn)])
        esize, C = self.esize, self.C
        if shuffled:
            planes = buf[:cbytes].reshape((esize,) + C)
            ob = self.ob
            swap = self.swap
            for k in range(esize):
                ob[dst_sl + (esize - 1 - k if swap else k,)] = planes[k][src_sl]
        elif esize in (1, 2, 4, 8):
            self.ou[dst_sl] = buf[:cbytes].view(self.u).reshape(C)[src_sl]
        else:
            sel = buf[:cbytes].reshape(C + (esize,))[src_sl]
            self.ob[dst_sl] = sel[..., ::-1] if self.swap else sel
        return True


GRID_MIN = 32                   # chunks from which an unfiltered dataset is gathered at once
GRID_MAX = 256 << 20            # ... and at most this many bytes of them
ZGRID_CHUNK = 64 << 10          # deflated chunks up to this size are inflated into one
ZGRID_MAX = 64 << 20            # staging buffer (at most this big) and placed all at once
ZGRID_SOLO = 16 << 10           # ... on one thread when they average less than this compressed
ZGRID_BAND = 4 << 20            # ... placed in bands of at least this many bytes


def _zgrid(it, p, src, threads):
    """_read_python for a deflated (and maybe shuffled) dataset of many
    small chunks (the Ku-band file: 15,000 chunks of 3.5 kB, where a numpy
    step per chunk was most of the time): each chunk inflated straight
    into a staging buffer in grid order -- one ctypes call per chunk, on
    every core -- then the byte planes and the chunk grid's axes put in
    order by numpy, band by band on the pool. False if not such a case."""
    from . import _zlib
    ds = it.ds
    C = tuple(int(c) for c in it.index.chunks)
    rank = len(C)
    n = len(p.addr)
    esize = ds.type.size
    nel = int(np.prod(C)) if C else 1
    cb = nel * esize
    if (rank == 0 or n < GRID_MIN or cb > ZGRID_CHUNK or n * cb > _hw.scaled(ZGRID_MAX) or p.fill_needed
            or not ds.filters):
        return False
    f = [x.id for x in ds.filters]
    if f not in ([F_DEFLATE], [F_SHUFFLE, F_DEFLATE], [F_DEFLATE, F_FLETCHER],
                 [F_SHUFFLE, F_DEFLATE, F_FLETCHER]) or p.mask.any():
        return False
    inflate = _zlib.inflater()
    if inflate is None:
        return False
    shuffled = F_SHUFFLE in f and esize > 1
    cut = 4 if F_FLETCHER in f else 0
    start = np.array([b[0] for b in it.box], np.int64)
    step = np.array([b[2] for b in it.box], np.int64)
    Ca = np.array(C, np.int64)
    g = (start + p.ostart * step - p.cstart) // Ca
    g0 = g.min(axis=0)
    gn = g.max(axis=0) - g0 + 1
    if int(np.prod(gn)) != n:
        return False
    order = np.argsort(np.ravel_multi_index(tuple((g - g0).T), tuple(gn)))
    staged = np.empty(n * cb, np.uint8)
    base = staged.ctypes.data
    sbase = src.ctypes.data
    sa = (p.addr[order] + sbase).tolist()
    sn = (p.size[order] - cut).tolist()
    bad = []
    # (each ctypes call lets go of the GIL and takes it back: for chunks of
    # a few hundred bytes the threads only queue for it -- measured on the
    # Ku-band file, 1 thread 32 ms, 4 threads 60, 24 threads 73)
    mean = int(p.size.sum()) // n
    nthr = 1 if mean < ZGRID_SOLO else max(1, min(threads or _pool.threads(), n // 16))
    per = -(-n // nthr)

    def inflate_part(t):
        o = base + t * per * cb
        for j in range(t * per, min(n, t * per + per)):
            if inflate(sa[j], sn[j], o, cb) != cb:
                bad.append(j)
                return
            o += cb
    _pool.map_items(inflate_part, range(nthr))
    if bad:
        return False
    # (grid rows, chunk dims, element bytes) with the planes undone, then
    # the grid's axes interleaved: (g0, c0, g1, c1, ..., esize)
    if shuffled:
        A = staged.reshape(tuple(gn) + (esize,) + C)
        A = np.moveaxis(A, rank, 2 * rank)
    else:
        A = staged.reshape(tuple(gn) + C + (esize,))
    perm = [x for d in range(rank) for x in (d, rank + d)] + [2 * rank]
    A = A.transpose(perm)
    ob = it.out.view(np.uint8).reshape(it.out.shape + (esize,))
    if it.swap:
        A = A[..., ::-1]
    # split along the first grid dimension: each band one numpy copy
    nb = max(1, min(int(gn[0]), threads or _pool.threads(), (n * cb) // ZGRID_BAND))
    rows = -(-int(gn[0]) // nb)
    s0, k0 = int(it.box[0][0]), int(it.box[0][2])
    m0 = it.out.shape[0]
    lo0 = s0 - int(g0[0]) * C[0]
    rest = tuple(slice(int(b[0] - g0[d] * C[d]), int(b[0] - g0[d] * C[d]) + (int(m) - 1) * int(b[2]) + 1,
                       int(b[2]))
                 for d, (b, m) in enumerate(zip(it.box, it.out.shape)) if d > 0)

    def place(t):
        r0, r1 = t * rows, min(int(gn[0]), t * rows + rows)
        if r0 >= r1:
            return
        sub = A[r0:r1].reshape(((r1 - r0) * C[0],) + tuple(int(gn[d]) * C[d] for d in range(1, rank))
                               + (esize,))
        # output rows j with lo0 + j * k0 in [r0 * C0, r1 * C0)
        a0 = r0 * C[0]
        j0 = max(0, -(-(a0 - lo0) // k0))
        j1 = min(m0, -(-(r1 * C[0] - lo0) // k0))
        if j1 <= j0:
            return
        ob[j0:j1] = sub[(slice(lo0 + j0 * k0 - a0, lo0 + (j1 - 1) * k0 - a0 + 1, k0),) + rest]
    _pool.map_items(place, range(nb))
    return True


def _gather_grid(it, p, src):
    """_read_python for a dataset without filters and many chunks (Metop-SG:
    19,000 one-row chunks; a numpy step per chunk would cost 25 us each):
    the chunks' bytes concatenated in grid order, the chunk grid's axes
    interleaved by one transposing copy, and the selection assigned from
    that. False if not such a case (a chunk not stored, too big): the
    chunk-by-chunk path then."""
    ds = it.ds
    C = tuple(int(c) for c in it.index.chunks)
    rank = len(C)
    n = len(p.addr)
    esize = ds.type.size
    cb = int(np.prod(C)) * esize
    if ds.filters or rank == 0 or n < GRID_MIN or n * cb > _hw.scaled(GRID_MAX) or p.fill_needed:
        return False
    start = np.array([b[0] for b in it.box], np.int64)
    step = np.array([b[2] for b in it.box], np.int64)
    Ca = np.array(C, np.int64)
    g = (start + p.ostart * step - p.cstart) // Ca      # each chunk's grid position
    g0 = g.min(axis=0)
    gn = g.max(axis=0) - g0 + 1
    if int(np.prod(gn)) != n:
        return False
    order = np.argsort(np.ravel_multi_index(tuple((g - g0).T), tuple(gn)))
    parts = [src[a:a + cb] for a in p.addr[order].tolist()]
    for k, x in enumerate(parts):
        if len(x) < cb:
            # (a contiguous dataset read as row blocks, ending the file: its last
            # block runs past the end, where nothing is selected)
            parts[k] = np.concatenate([x, np.zeros(cb - len(x), np.uint8)])
    staged = np.concatenate(parts)
    if esize in (1, 2, 4, 8):
        u = np.dtype('u%d' % esize)
        A = staged.view(u.newbyteorder('S') if it.swap else u).reshape(tuple(gn) + C)
        tail = ()
    else:
        A = staged.reshape(tuple(gn) + C + (esize,))
        tail = (rank * 2,)
    perm = [x for d in range(rank) for x in (d, rank + d)] + list(tail)
    R = A.transpose(perm).reshape(tuple(int(gn[d]) * C[d] for d in range(rank)) + A.shape[2 * rank:])
    sl = tuple(slice(int(b[0] - g0[d] * C[d]), int(b[0] - g0[d] * C[d]) + (int(m) - 1) * int(b[2]) + 1,
                     int(b[2]))
               for d, (b, m) in enumerate(zip(it.box, it.out.shape)))
    if esize in (1, 2, 4, 8):
        it.out.view(u)[...] = R[sl]
    else:
        sel = R[sl]
        it.out.view(np.uint8).reshape(it.out.shape + (esize,))[...] = sel[..., ::-1] if it.swap else sel
    return True


def _read_python(items, src, threads=None):
    """Reads before the kernels are loaded: zlib and numpy -- the reference
    path itself, so the values are the same -- chunk by chunk on the pool."""
    import zlib
    jobs = []
    plans = []
    for it in items:
        ds, index, box, out = it.ds, it.index, it.box, it.out
        p = plan(index, ds.shape, box)
        if p.fill_needed:
            _fill(out, ds, it.swap)
        plans.append(p)
    if any(len(p.addr) for p in plans):
        fetch_chunks(items, plans)
    quick = {}
    sizes = []
    for k, (it, p) in enumerate(zip(items, plans)):
        if len(p.addr) and not _gather_grid(it, p, src) and not _zgrid(it, p, src, threads):
            quick[k] = _Quick(it, p, src)
            jobs.extend((k, i) for i in range(len(p.addr)))
            sizes.append(p.size)
    if jobs:
        order = np.argsort(-np.concatenate(sizes), kind='stable').tolist()
        jobs = [jobs[j] for j in order]

    def one(job):
        k, i = job
        if quick[k].run(i, src):
            return
        it, p = items[k], plans[k]
        ds = it.ds
        esize = ds.type.size
        C = tuple(int(c) for c in it.index.chunks)
        nel = int(np.prod(C)) if C else 1
        filters = ds.filters
        a, n, m = int(p.addr[i]), int(p.size[i]), int(p.mask[i])
        data = src[a:a + n]
        for k in range(len(filters) - 1, -1, -1):  # each stage in decoding order
            if (m >> k) & 1:
                continue
            fid = filters[k].id
            if fid == F_FLETCHER:
                data = data[:-4]
            elif fid == F_DEFLATE:
                data = zlib.decompress(data)
            elif fid == F_SHUFFLE and esize > 1:
                q = len(data) // esize              # HDF5 leaves the leftover bytes at the end
                data = (np.frombuffer(data, np.uint8, q * esize).reshape(esize, q).T.tobytes()
                        + bytes(data[q * esize:]))
        if len(data) < nel * esize:                 # (a virtual chunk past the end of the file)
            data = bytes(data) + bytes(nel * esize - len(data))
        b = np.frombuffer(data, np.uint8, nel * esize)
        chunk = np.ascontiguousarray(b).view(np.dtype('V%d' % esize)).reshape(C)
        step = [s[2] for s in it.box]
        src_sl = tuple(slice(int(c0), int(c0) + (int(cn) - 1) * k + 1, k)
                       for c0, cn, k in zip(p.cstart[i], p.count[i], step))
        dst_sl = tuple(slice(int(o0), int(o0) + int(cn)) for o0, cn in zip(p.ostart[i], p.count[i]))
        ov = it.out.view(np.dtype('V%d' % esize))
        if it.swap:
            sel = np.ascontiguousarray(chunk[src_sl])
            swapped = sel.view(np.uint8).reshape(sel.shape + (esize,))[..., ::-1].copy()
            ov[dst_sl] = swapped.view(np.dtype('V%d' % esize)).reshape(sel.shape)
        else:
            ov[dst_sl] = chunk[src_sl]
    if len(jobs) <= 1:
        for j in jobs:
            one(j)
    else:
        _pool.map_items(one, jobs)
