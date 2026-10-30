"""Masked / scaled reads with the masking and scaling done chunk by chunk
as each is decoded, while its values are still in the core's cache.

Done after the read, mask/scale reads the whole raw array back from memory
and writes the floats and the mask: for big scaled variables (GOES, VIIRS,
MODIS) about as long as the read itself, nearly all of it memory traffic.
Here each chunk is inflated (and unshuffled, byte swapped, cut to the
selection) into the thread's scratch, and _maskscale's block kernels turn
that into the floats and the mask in place in the result: the raw values
never go to memory. Same operations, same types, same values as
_maskscale.run_plan (and so as netCDF4-python): only where they happen
moves.

Used for one scaled variable read on the CPU once the kernels are loaded,
when the chunks are deflate / shuffle / fletcher32 (or none), stored (no
fill) and at least one per thread, the index simple slices and the result
big; else the usual read and _maskscale.apply. Measured
(tools/profile/fused_check.py): GOES-19 C02 Rad 140 -> 122 ms, C13 10.8 ->
9.0, VIIRS I05 13.7 -> 12.3; slower for a few big chunks or mask-only
floats (WRF), which keep the usual path.
"""
import numpy as np

from . import _arena, _chunks, _maskscale, _pool, _select
from ._chunks import (F_DEFLATE, F_FLETCHER, F_SHUFFLE, MAXDIM, V_CBYTES, V_ESIZE, V_NEL,
                      V_NST, V_RANK, V_STAGE0, V_SWAP, _place)
from ._inflate import zlib_inflate
from ._intrin import take
from ._jit import njit
from ._maskscale import _fix_block, _flag_hits, _scale_block


def read(var, key, threads=None):
    """Variable.read(key) masked and scaled, fused with the decoding; None
    when this read is not one for it (the caller then reads as usual)."""
    from . import _h5read, _hdf5
    if not _chunks._READY.is_set():
        return None
    bv = var._bv
    r = getattr(bv, '_reader', None)
    if not isinstance(r, _h5read.Reader):
        return None
    ds = r.ds
    t = ds.type
    if t.cls not in (0, 1) or t.dtype is None or ds.null or _h5read.reduced(t):
        return None
    if not (ds.layout.cls == 2 or (ds.layout.cls == 1 and r.virtual)):
        return None
    if any(f.id not in _chunks.SUPPORTED for f in ds.filters) or len(ds.filters) > _chunks.MAXSTAGE:
        return None
    stored = t.dtype
    nat = _h5read.native(stored)
    swap = _h5read._simple_swap(stored)
    if stored.byteorder == '>' and not swap:
        return None
    k = key if isinstance(key, tuple) else (key,)
    box, post, drop = _select.normalize(k, ds.shape)
    if post is not None:
        return None
    shape = tuple(_select.box_shape(box))
    final = tuple(s for d, s in enumerate(shape) if d not in drop) if drop else shape
    n = int(np.prod(shape)) if shape else 1
    if not final or n < _maskscale.FAST_MIN_SIZE:
        return None
    dt = var.datatype
    var_dtype = None if isinstance(dt, np.dtype) or dt is str else dt
    kind = _maskscale._kind(bv.dtype, var_dtype)
    if kind not in ('primitive', 'enum') or np.dtype(bv.dtype) != nat:
        return None
    index = r.index
    p = _chunks.plan(index, ds.shape, box)
    if p.fill_needed or not len(p.addr):
        return None
    nthr = threads or _pool.threads()
    if nthr > 1 and (int(p.size.max()) >= _chunks.PAR_MIN or len(p.addr) < nthr):
        return None                                 # (fewer chunks than threads: the usual
                                                    # read, then mask/scale on every thread)
    # (last: the plan warns about odd attributes, once per read)
    pl = _maskscale.make_plan(nat, bv.attrs, bv.dtype, kind, var._mask, var._scale,
                              getattr(bv, 'no_fill', False))
    if pl is None or pl.out_dtype is None or pl.nan_const:
        return None                                 # (mask only: the read goes straight into
                                                    # the result, then a parallel pass: faster)
    if r.f.source is not None:
        _chunks.fetch_chunks([_chunks.Item(ds, index, box, None, swap)], [p])
    return _run(var, r, ds, index, box, shape, final, nat, swap, pl, p, nthr)


def _run(var, r, ds, index, box, shape, final, nat, swap, pl, p, nthr):
    rank = len(shape)
    esize = ds.type.size
    stages = [f.id for f in reversed(ds.filters)]
    C = index.chunks
    V = np.zeros((1, _chunks.V_COLS), np.int64)
    nel = int(np.prod(C))
    V[0, :V_STAGE0] = (esize, rank, nel, nel * esize, int(bool(swap)), len(stages), 0)
    V[0, V_STAGE0:V_STAGE0 + len(stages)] = stages
    cstr = np.ones(rank, np.int64)
    ostr = np.ones(rank, np.int64)
    for d in range(rank - 2, -1, -1):
        cstr[d] = cstr[d + 1] * C[d + 1]
        ostr[d] = ostr[d + 1] * shape[d + 1]
    VD = np.zeros((5, rank), np.int64)
    VD[0], VD[1], VD[2] = C, cstr, ostr
    VD[3] = [s[2] for s in box]
    VD[4] = shape
    order = np.argsort(-p.size, kind='stable')
    addr, size, mask, needend = p.addr[order], p.size[order], p.mask[order], p.needend[order]
    cstart, count, ostart = p.cstart[order], p.count[order], p.ostart[order]
    xdt = pl.xdt
    maskonly = pl.out_dtype is None
    out = _arena.empty(shape, nat if maskonly else pl.out_dtype)
    of = out.reshape(-1)
    if maskonly:
        of = of.view(xdt)
    msk = _arena.empty(shape, np.bool_) if pl.mask else np.empty(1, np.bool_)
    m = msk.reshape(-1)
    nch = len(addr)
    stats = np.zeros((nch, 4), np.int64)
    status = np.zeros(nch, np.int64)
    ctr = np.zeros(2, np.int64)
    maxsel = int(np.prod(count, axis=1).max())
    src = r.src
    xproto = np.zeros(1, xdt)
    outbytes = int(np.prod(shape)) * max(esize, out.dtype.itemsize)
    nthr = max(1, min(nthr, nch, outbytes >> 15))

    def work():
        sc, tab, iws = _chunks._scratch(nel * esize + 8)
        _work_ms(src, V, VD, addr, size, mask, needend, cstart, count, ostart, ctr, status,
                 sc, tab, iws, xproto, maxsel, of, m, maskonly, pl.eq, pl.eqcat, pl.neq,
                 pl.lohi, pl.opts, pl.sc, pl.sc32, stats)
    _pool.share(work, nthr)
    if status.any():
        bad = np.nonzero(status)[0][0]
        raise IOError('chunk at byte %d of %s could not be decoded (error %d)'
                      % (addr[bad], ds.name, status[bad]))
    tot = stats.sum(axis=0)
    if pl.mask and tot[3] and (stats[:, 3] == 0).any():
        _pool.share(lambda: _zero_clean(m, VD, count, ostart, stats, ctr), nthr)
    if final != shape:
        out = out.reshape(final)
        if pl.mask:
            msk = msk.reshape(final)
    data = out.view(pl.view) if (maskonly and pl.view is not None) else out
    return _maskscale.finish(pl, data, None if maskonly else out, msk, tot, var._always_mask)


@njit(cache=True, nogil=True)
def _decode(src, V, p, pend, m, needend_i, scratch, tab, iws):
    """Chunk 0-variable V's stored bytes src[p:pend] (filter mask m) decoded
    as far as needed: (status, in scratch, shuffled, byte planes); status 0
    ok, < 0 a decoder's error, 1 a filter combination not handled here.
    _chunks._work's stage logic."""
    esize = V[0, V_ESIZE]
    nel = V[0, V_NEL]
    cbytes = V[0, V_CBYTES]
    nst = V[0, V_NST]
    shuffled = False
    for k in range(nst):
        if V[0, V_STAGE0 + k] == F_SHUFFLE and not (m >> (nst - 1 - k)) & 1:
            shuffled = esize > 1
    tail = 0
    planes = nel
    seen_deflate = False
    seen_shuffle = False
    for k in range(nst):
        if (m >> (nst - 1 - k)) & 1:
            continue
        fk = V[0, V_STAGE0 + k]
        if fk == F_DEFLATE:
            seen_deflate = True
        elif fk == F_SHUFFLE:
            seen_shuffle = True
        elif fk == F_FLETCHER and seen_deflate:
            tail = 4
            if seen_shuffle:
                planes = (cbytes + 4) // esize
    in_scratch = False
    for k in range(nst):
        f = V[0, V_STAGE0 + k]
        if (m >> (nst - 1 - k)) & 1:
            continue
        if f == F_FLETCHER:
            if not in_scratch:
                pend -= 4
        elif f == F_SHUFFLE:
            pass
        elif f == F_DEFLATE:
            if in_scratch:
                return 1, in_scratch, shuffled, planes
            stop = cbytes + tail
            need = needend_i * esize
            if not shuffled and need < cbytes:
                stop = need
            r = zlib_inflate(src, p, pend, scratch, 0, cbytes + tail, stop, tab, iws)
            if r < 0:
                return r, in_scratch, shuffled, planes
            in_scratch = True
        else:
            return 1, in_scratch, shuffled, planes
    return 0, in_scratch, shuffled, planes


@njit(cache=True, nogil=True)
def _ms_run(xb, ob, mb, s, maskonly, eq, eqcat, neq, lo, hi, nan_on, nan_mv, nan_fv, mode,
            l1f4, l2f4, sc, sc32, cnt):
    """_maskscale.maskscale_blocks' work on one run: the values scaled into
    ob, flagged; the mask and the values under it written only if some are
    masked. Returns how many are."""
    if not maskonly and mode != 0:
        _scale_block(xb, ob, mode, l1f4, l2f4, sc, sc32)
    c = 0
    if mb.size:
        c = _flag_hits(xb, eq, neq, lo, hi, nan_on, s)
        if c:
            _fix_block(xb, ob, mb, s, eq, eqcat, neq, nan_mv, nan_fv,
                       (not maskonly) and mode != 0, l1f4, l2f4, cnt)
    return c


@njit(cache=True, nogil=True)
def _work_ms(src, V, VD, addr, size, mask, needend, cstart, count, ostart, ctr, status,
             scratch, tab, iws, xproto, maxsel, of, m, maskonly, eq, eqcat, neq, lohi, opts,
             sc, sc32, stats):
    """Chunks taken off ctr[0]: each decoded, its selection placed in this
    thread's scratch (rs, in the values' type), then masked and scaled run
    by run into of / m (mask only: the values copied to of as they are).
    stats[i] as _maskscale's per block, for chunk i."""
    n = addr.shape[0]
    rank = V[0, V_RANK]
    esize = V[0, V_ESIZE]
    swap = V[0, V_SWAP] == 1
    last = rank - 1
    cstr = VD[1]
    ostr = VD[2]
    step = VD[3]
    oshape = VD[4]
    domask = opts[0] != 0
    nan_mv = opts[1] != 0
    nan_fv = opts[2] != 0
    nan_on = nan_mv or nan_fv
    mode = opts[3]
    l1f4 = opts[4] != 0
    l2f4 = opts[5] != 0
    lo = lohi[0]
    hi = lohi[1]
    rs = np.empty(maxsel, xproto.dtype)
    rsb = rs.view(np.uint8)
    s = np.empty(maxsel + 8 if domask else 8, np.uint8)
    cnt = np.zeros(3, np.int64)
    zero = np.zeros(MAXDIM, np.int64)
    rstr = np.ones(MAXDIM, np.int64)
    idx = np.zeros(MAXDIM, np.int64)
    nomask = m[:0]
    while True:
        i = take(ctr, 0)
        if i >= n:
            return
        p = addr[i]
        st, in_scratch, shuffled, planes = _decode(src, V, p, p + size[i], mask[i], needend[i],
                                                   scratch, tab, iws)
        status[i] = st
        if st != 0:
            continue
        cnt_i = count[i]
        os_i = ostart[i]
        # the selection's runs: contiguous in the result when every
        # dimension after the first counted is whole there
        nsel = 1
        for d in range(rank):
            nsel *= cnt_i[d]
        for d in range(rank - 2, -1, -1):
            rstr[d] = rstr[d + 1] * cnt_i[d + 1]
        if in_scratch:
            _place(scratch, 0, planes, rsb, cstart[i], cnt_i, zero, cstr, rstr, step,
                   rank, esize, shuffled, swap)
        else:
            _place(src, p, V[0, V_NEL], rsb, cstart[i], cnt_i, zero, cstr, rstr, step,
                   rank, esize, shuffled, swap)
        whole = True
        k0 = 0
        while k0 < rank and cnt_i[k0] == 1:
            k0 += 1
        for d in range(k0 + 1, rank):
            if cnt_i[d] != oshape[d]:
                whole = False
        if whole:
            run = nsel
        else:
            run = cnt_i[last]
        nruns = nsel // run
        c_chunk = 0
        first_hit = -1
        for d in range(rank):
            idx[d] = 0
        for r in range(nruns):
            if whole:
                de = 0
                for d in range(rank):
                    de += os_i[d] * ostr[d]
            else:
                de = 0
                rem = r
                for d in range(last - 1, -1, -1):
                    idx[d] = rem % cnt_i[d]
                    rem //= cnt_i[d]
                for d in range(rank):
                    de += (os_i[d] + (idx[d] if d < last else 0)) * ostr[d]
            ob = of[de:de + run]
            xb = rs[r * run:r * run + run]
            if maskonly:
                for q in range(run):
                    ob[q] = xb[q]
            mb = m[de:de + run] if domask else nomask
            c = _ms_run(xb, ob, mb, s, maskonly, eq, eqcat, neq, lo, hi, nan_on, nan_mv, nan_fv,
                        mode, l1f4, l2f4, sc, sc32, cnt)
            if c:
                if first_hit < 0:
                    first_hit = r
                c_chunk += c
            elif domask and first_hit >= 0:
                m[de:de + run] = False              # (a clean run after a masked one)
        if first_hit > 0:
            # clean runs before the chunk's first masked one: their mask too
            for r in range(first_hit):
                de = 0
                rem = r
                for d in range(last - 1, -1, -1):
                    idx[d] = rem % cnt_i[d]
                    rem //= cnt_i[d]
                for d in range(rank):
                    de += (os_i[d] + (idx[d] if d < last else 0)) * ostr[d]
                m[de:de + run] = False
        if c_chunk:
            stats[i, 0] = cnt[0]
            stats[i, 1] = cnt[1]
            stats[i, 2] = cnt[2]
        stats[i, 3] = c_chunk


@njit(cache=True, nogil=True)
def _zero_clean(m, VD, count, ostart, stats, ctr):
    """The mask False over the chunks _work_ms found nothing to mask in
    (taken off ctr[1])."""
    n = count.shape[0]
    rank = count.shape[1]
    last = rank - 1
    ostr = VD[2]
    idx = np.zeros(MAXDIM, np.int64)
    while True:
        i = take(ctr, 1)
        if i >= n:
            return
        if stats[i, 3] != 0:
            continue
        cnt_i = count[i]
        run = cnt_i[last]
        nruns = 1
        for d in range(last):
            nruns *= cnt_i[d]
        for r in range(nruns):
            rem = r
            for d in range(last - 1, -1, -1):
                idx[d] = rem % cnt_i[d]
                rem //= cnt_i[d]
            de = 0
            for d in range(rank):
                de += (ostart[i, d] + (idx[d] if d < last else 0)) * ostr[d]
            m[de:de + run] = False
