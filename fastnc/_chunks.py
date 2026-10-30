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
"""
import threading

import numpy as np

from . import _inflate, _pool
from ._intrin import ld64, st64, take
from ._jit import njit

# pipeline stage kinds the kernel knows (HDF5 filter ids)
F_DEFLATE = 1
F_SHUFFLE = 2
F_FLETCHER = 3

MAXDIM = 32


class Plan:
    """The chunks a selection touches and where each goes."""
    __slots__ = ('addr', 'size', 'mask', 'cstart', 'count', 'ostart', 'needend', 'fill_needed')


class ChunkIndex:
    """A dataset's chunk table as arrays: element offsets (n, rank),
    absolute addresses, stored sizes, filter masks; and a lookup from the
    chunk's grid position to its row."""

    def __init__(self, table, chunks, shape):
        n = len(table)
        rank = len(chunks)
        self.offsets = np.array([t[0] for t in table], np.int64).reshape(n, rank)
        self.addr = np.array([t[1] for t in table], np.int64)
        self.size = np.array([t[2] for t in table], np.int64)
        self.mask = np.array([t[3] for t in table], np.int64)
        self.chunks = np.array(chunks, np.int64)
        self.grid = np.array([-(-s // c) for s, c in zip(shape, chunks)], np.int64)
        self.ngrid = int(np.prod(self.grid)) if rank else 1
        self.complete = n >= self.ngrid             # every chunk stored: no fill needed


def plan(index, shape, sel):
    """Chunks touched by `sel` (one (start, stop, step) per dimension, step >
    0, start < stop) as a Plan; None when the selection is empty."""
    rank = len(shape)
    off = index.offsets
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
    idx = np.nonzero(keep)[0]
    p = Plan()
    p.addr = index.addr[idx]
    p.size = index.size[idx]
    p.mask = index.mask[idx]
    p.cstart = np.ascontiguousarray(cstart[idx])
    p.count = np.ascontiguousarray(count[idx])
    p.ostart = np.ascontiguousarray(ostart[idx])
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
            if swap:
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
V_COLS = V_STAGE0 + MAXSTAGE


@njit(cache=True, nogil=True)
def _work(src, out, V, VD, addr, size, mask, var, needend, direct, cstart, count, ostart,
          counter, status, scratch, tab, iws):
    """Take chunks off counter[0] until none is left: decode each into
    scratch (or straight into out) and place its selected elements.

    One call serves a batch of variables: chunk i belongs to variable
    var[i], whose parameters are row V[var[i]] (element size, rank, chunk
    elements and bytes, byte swap, the filter ids in decoding order, the
    byte offset of its output in out) and VD[var[i]] (chunk shape, chunk
    and output strides in elements, selection steps). status[i]: 0 ok,
    < 0 a decoder's error, 1 a filter this kernel does not run."""
    n = addr.shape[0]
    while True:
        i = take(counter, 0)
        if i >= n:
            return
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
        for k in range(nst):
            f = V[v, V_STAGE0 + k]
            if (m >> (nst - 1 - k)) & 1:
                continue                            # filter skipped for this chunk
            if f == F_FLETCHER:
                pend -= 4
            elif f == F_SHUFFLE:
                pass
            elif f == F_DEFLATE:
                if in_scratch:
                    ok = 1                          # two compressors: not here
                    break
                if direct[i] >= 0:
                    o = obase + direct[i]
                    r = _inflate.zlib_inflate(src, p, pend, out, o, o + cbytes, o + cbytes, tab, iws)
                    ok = r if r < 0 else 0
                    placed = True
                    break
                stop = cbytes
                need = needend[i] * esize
                if not shuffled and need < cbytes:
                    stop = need                     # the rest of the chunk is not wanted
                r = _inflate.zlib_inflate(src, p, pend, scratch, 0, cbytes, stop, tab, iws)
                if r < 0:
                    ok = r
                    break
                in_scratch = True
            else:
                ok = 1
                break
        status[i] = ok
        if ok != 0 or placed:
            continue
        if in_scratch:
            _place(scratch, 0, nel, out[obase:], cstart[i], count[i], ostart[i], VD[v, 1],
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
        e1 = min(n, e0 + blk)
        for e in range(e0, e1):
            q = e * esize
            for k in range(esize // 2):
                t = a[q + k]
                a[q + k] = a[q + esize - 1 - k]
                a[q + esize - 1 - k] = t


# ---------------------------------------------------------------- driver

_TLS = threading.local()


def _scratch(nbytes):
    s = getattr(_TLS, 'scratch', None)
    if s is None or s.size < nbytes:
        s = np.empty(max(nbytes, 1 << 16), np.uint8)
        _TLS.scratch = s
    if getattr(_TLS, 'tab', None) is None:
        _TLS.tab, _TLS.iws = _inflate.workspace()
    return s, _TLS.tab, _TLS.iws


SUPPORTED = (F_DEFLATE, F_SHUFFLE, F_FLETCHER)


class Item:
    """One dataset's part of a batch: its chunk index, the box to read, the
    output array (one dimension per dataset dimension, C-contiguous, in a
    shared byte buffer at byte offset `base`), and whether to swap bytes."""
    __slots__ = ('ds', 'index', 'box', 'out', 'swap', 'base')

    def __init__(self, ds, index, box, out, swap, base=0):
        self.ds, self.index, self.box, self.out, self.swap, self.base = ds, index, box, out, swap, base


def read_batch(items, src, buf, threads=None):
    """Decode the chunks of every item in one pass of the pool, biggest
    chunks first, into buf (uint8; each item's output starts at its base)."""
    plans = []
    V = np.zeros((len(items), V_COLS), np.int64)
    maxrank = max(1, max(len(it.ds.shape) for it in items))
    VD = np.zeros((len(items), 4, maxrank), np.int64)
    cmax = 0
    for v, it in enumerate(items):
        ds, index, box = it.ds, it.index, it.box
        rank = len(ds.shape)
        esize = ds.type.size
        stages = [f.id for f in reversed(ds.filters)]
        for fid in stages:
            if fid not in SUPPORTED:
                raise NotImplementedError('filter %d is not supported yet' % fid)
        if len(stages) > MAXSTAGE:
            raise NotImplementedError('more than %d filters' % MAXSTAGE)
        p = plan(index, ds.shape, box)
        if p.fill_needed:
            _fill(it.out, ds)
        oshape = it.out.shape
        cstr = np.ones(rank, np.int64)
        ostr = np.ones(rank, np.int64)
        for d in range(rank - 2, -1, -1):
            cstr[d] = cstr[d + 1] * index.chunks[d + 1]
            ostr[d] = ostr[d + 1] * oshape[d + 1]
        step = np.array([s[2] for s in box], np.int64)
        nel = int(np.prod(index.chunks)) if rank else 1
        V[v, :V_STAGE0] = (esize, rank, nel, nel * esize, int(bool(it.swap)), len(stages), it.base)
        V[v, V_STAGE0:V_STAGE0 + len(stages)] = stages
        VD[v, 0, :rank] = index.chunks
        VD[v, 1, :rank] = cstr
        VD[v, 2, :rank] = ostr
        VD[v, 3, :rank] = step
        cmax = max(cmax, nel * esize)
        # chunks inflated straight into the output: whole chunk selected,
        # step 1, chunk extent equal to the output's in every dimension but
        # the first, no shuffle, no swap: their bytes are the output's
        direct = np.full(len(p.addr), -1, np.int64)
        if (len(p.addr) and rank and not it.swap and F_SHUFFLE not in stages and np.all(step == 1)
                and (rank == 1 or np.all(index.chunks[1:] == np.array(oshape[1:])))):
            full = np.all(p.count == index.chunks, axis=1) & np.all(p.cstart == 0, axis=1)
            lin = (p.ostart * ostr).sum(axis=1) * esize
            direct[full] = lin[full]
        plans.append((p, direct))
    n = sum(len(p.addr) for p, _ in plans)
    if n == 0:
        return
    addr = np.concatenate([p.addr for p, _ in plans])
    size = np.concatenate([p.size for p, _ in plans])
    mask = np.concatenate([p.mask for p, _ in plans])
    needend = np.concatenate([p.needend for p, _ in plans])
    direct = np.concatenate([d for _, d in plans])
    var = np.concatenate([np.full(len(p.addr), v, np.int64) for v, (p, _) in enumerate(plans)])

    def pad(a):
        if a.shape[1] == maxrank:
            return a
        z = np.zeros((a.shape[0], maxrank), np.int64)
        z[:, :a.shape[1]] = a
        return z
    cstart = np.concatenate([pad(p.cstart) for p, _ in plans])
    count = np.concatenate([pad(p.count) for p, _ in plans])
    ostart = np.concatenate([pad(p.ostart) for p, _ in plans])
    if n > 1:
        order = np.argsort(-size, kind='stable')     # biggest first: no straggler at the end
        addr, size, mask, needend, direct, var = (addr[order], size[order], mask[order],
                                                  needend[order], direct[order], var[order])
        cstart, count, ostart = cstart[order], count[order], ostart[order]
    counter = np.zeros(1, np.int64)
    status = np.zeros(n, np.int64)
    # a thread for each ~32 kB of output at most: handing tiny reads to the
    # pool costs more than it saves
    outbytes = int(sum(int(np.prod(it.out.shape)) * it.ds.type.size for it in items))
    nthreads = max(1, min(threads or _pool.threads(), n, outbytes >> 15))

    def work():
        sc, tab, iws = _scratch(cmax + 8)
        _work(src, buf, V, VD, addr, size, mask, var, needend, direct, cstart, count, ostart,
              counter, status, sc, tab, iws)
    _pool.share(work, nthreads)
    if status.any():
        bad = np.nonzero(status)[0][0]
        raise IOError('chunk at byte %d of %s could not be decoded (error %d)'
                      % (addr[bad], items[var[bad]].ds.name, status[bad]))


def read(ds, index, src, sel, out, swap=False, threads=None):
    """Decode the chunks of dataset ds touched by sel into out: a C-contiguous
    array with one dimension per dataset dimension, sized to the selection
    (integer indexes as length one). swap: reverse each element's bytes as
    it is placed (big-endian numbers read as native)."""
    read_batch([Item(ds, index, sel, out, swap, 0)], src, out.reshape(-1).view(np.uint8), threads)
    return out


def _fill(out, ds):
    if ds.fill is not None and len(ds.fill) == ds.type.size:
        out.reshape(-1).view(np.uint8).reshape(-1, ds.type.size)[:] = np.frombuffer(ds.fill, np.uint8)
    else:
        out.reshape(-1).view(np.uint8)[:] = 0


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
