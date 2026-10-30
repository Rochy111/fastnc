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
import threading

import numpy as np

from . import _inflate, _pinflate, _pool
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
        self.plans = {}                             # box -> Plan, for repeated reads


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
            continue
        if in_scratch:
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


_READY = threading.Event()      # numba and the kernels loaded
WARM_AFTER = 1 << 20           # a read bigger than this starts loading them
_LOADING = []


def _warm():
    """Import numba and load (from cache, or compile) the batch kernel with
    the argument types reads use, off the caller's thread."""
    try:
        i64 = np.zeros(1, np.int64)
        i2 = np.zeros((1, 1), np.int64)
        u8 = np.zeros(16, np.uint8)
        ro = np.frombuffer(bytes(16), np.uint8)    # the file: a read-only view of its mapping
        tab, iws = _inflate.workspace()
        _work(ro, u8, np.zeros((1, V_COLS), np.int64), np.zeros((1, 4, 1), np.int64),
              i64[:0], i64[:0], i64[:0], i64[:0], i64[:0], i64[:0], i2[:0], i2[:0], i2[:0],
              np.zeros(1, np.int64), i64[:0], u8, tab, iws)
        _pinflate.warm()
    finally:
        _READY.set()


def _start_warm():
    if not _LOADING:
        _LOADING.append(threading.Thread(target=_warm, name='fastnc-warm', daemon=True))
        _LOADING[0].start()


def warm(wait=True):
    """Load numba and fastnc's kernels now (they load on their own in the
    background at the first read; reads until then use zlib). For worker
    processes: ProcessPoolExecutor(initializer=fastnc.warm)."""
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
    if t.size <= BIGTMP_KEEP:
        with _BIGTMP_LOCK:
            if _BIGTMP[0] is None or _BIGTMP[0].size < t.size:
                _BIGTMP[0] = t


def read_batch(items, src, buf, threads=None):
    """Decode the chunks of every item in one pass of the pool, biggest
    chunks first, into buf (uint8; each item's output starts at its base)."""
    staged = [it for it in items if not all(f.id in SUPPORTED for f in it.ds.filters)]
    if staged:
        # other filters (szip, scale-offset, n-bit, bzip2, ...): stage by stage
        from . import _staged
        for it in staged:
            if not _staged.handles(it.ds.filters):
                bad = [f.id for f in it.ds.filters if not _staged.handles([f])]
                raise NotImplementedError('filter %d is not supported yet' % bad[0])
        fetch_chunks(items)                         # (all: one round with the fast ones)
        _staged.read_items(staged, src, buf, threads)
        items = [it for it in items if all(f.id in SUPPORTED for f in it.ds.filters)]
        if not items:
            return
    if not _READY.is_set() and all(f.id in SUPPORTED for it in items for f in it.ds.filters):
        # the kernels load in the background (~0.5-1 s, once per process),
        # started by the first read big enough to want them (the import holds
        # the GIL: not during the small reads of an xarray open); until then
        # zlib, which releases the GIL, decodes on the pool
        if sum(int(np.prod(it.out.shape)) * it.ds.type.size for it in items) > WARM_AFTER:
            _start_warm()
        _read_python(items, src, threads)
        return
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
        # step 1, no shuffle, no swap, and the chunk one run of the output
        # (extent 1 in the leading dimensions, the output's after the first
        # other): their bytes are the output's
        direct = np.full(len(p.addr), -1, np.int64)
        k1 = next((d for d in range(rank) if index.chunks[d] > 1), rank - 1)
        if (len(p.addr) and rank and not it.swap and F_SHUFFLE not in stages and np.all(step == 1)
                and np.all(index.chunks[k1 + 1:] == np.array(oshape[k1 + 1:], np.int64))):
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
    fetch_chunks(items, [p for p, _ in plans])
    if n > 1:
        order = np.argsort(-size, kind='stable')     # biggest first: no straggler at the end
        addr, size, mask, needend, direct, var = (addr[order], size[order], mask[order],
                                                  needend[order], direct[order], var[order])
        cstart, count, ostart = cstart[order], count[order], ostart[order]
    # a thread for each ~32 kB of output at most: handing tiny reads to the
    # pool costs more than it saves
    outbytes = int(sum(int(np.prod(it.out.shape)) * it.ds.type.size for it in items))
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
            cstart, count, ostart = cstart[keep], count[keep], ostart[keep]
            n = len(addr)
        else:
            big = None
    counter = np.zeros(1, np.int64)
    status = np.zeros(n, np.int64)
    nthreads = max(1, min(allthreads, n, outbytes >> 15))

    def work():
        sc, tab, iws = _scratch(cmax + 8)
        _work(src, buf, V, VD, addr, size, mask, var, needend, direct, cstart, count, ostart,
              counter, status, sc, tab, iws)
    if big is not None:                             # the other chunks once a thread's parts are done
        _read_big(src, buf, V, VD, bigs, allthreads, items, work if n else None, total)
    else:
        _pool.share(work, nthreads)
    if status.any():
        bad = np.nonzero(status)[0][0]
        raise IOError('chunk at byte %d of %s could not be decoded (error %d)'
                      % (addr[bad], items[var[bad]].ds.name, status[bad]))


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
              counter, status, sc, tab, iws)
    nbytes = int(np.prod(it.out.shape)) * esize
    _pool.share(work, max(1, min(threads or _pool.threads(), n, nbytes >> 15)))
    if status.any():
        raise IOError('a staged chunk of %s could not be placed' % ds.name)


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
            _fill(out, ds)
        jobs.extend((it, p, i) for i in range(len(p.addr)))
        plans.append(p)
    if jobs:
        fetch_chunks(items, plans)
    jobs.sort(key=lambda j: -int(j[1].size[j[2]]))

    def one(job):
        it, p, i = job
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
