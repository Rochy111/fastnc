"""Chunks whose filter pipeline is not the deflate / shuffle / fletcher32
fast path (szip, scale-offset, n-bit, bzip2, and the other codecs).

The pipeline runs stage by stage over every chunk of a group at once,
each stage one codec's batch driver (`_codecs`) on the pool, between two
staging buffers of one chunk-sized slot per chunk. fletcher32 only
shortens the input (its checksum is verified on the stored bytes); a
shuffle that comes last is left to the placement, which undoes it while
scattering the selected elements (the fast path's `_work` kernel, fed the
staged chunks as its "file"). Chunks are taken in groups of at most
GROUP_BYTES decoded, so staging stays bounded on big reads.
"""
import threading

import numpy as np

from . import _chunks, _codecs, _fletcher32, _hw, _inflate, _pool
from ._chunks import F_FLETCHER, F_SHUFFLE

GROUP_BYTES = 256 << 20
NOTHING_TO_UNDO = (32022, 32023)        # BitGroom, Granular BitRound: they round what is written
_TLS = threading.local()


def handles(filters):
    return all(f.id in _codecs.DRIVERS or f.id in NOTHING_TO_UNDO or f.id == F_FLETCHER for f in filters)


def _ws(fid, size):
    cache = getattr(_TLS, 'ws', None)
    if cache is None:
        cache = _TLS.ws = {}
    box = cache.get(fid)
    if box is None:
        box = cache[fid] = [None, 0]
    return _pool.kept(box, size)


def read_items(items, src, buf, threads=None):
    """Decode and place every chunk of these items (each an `_chunks.Item`
    whose dataset needs the staged path)."""
    for it in items:
        ds = it.ds
        p = _chunks.plan(it.index, ds.shape, it.box)
        if p.fill_needed:
            _chunks._fill(it.out, ds, it.swap)
        n = len(p.addr)
        if not n:
            continue
        cbytes = int(np.prod(it.index.chunks)) * ds.type.size
        per = max(1, _hw.scaled(GROUP_BYTES) // max(cbytes, 1))
        order = np.argsort(-p.size, kind='stable')
        for g0 in range(0, n, per):
            sel = order[g0:g0 + per]
            _group(it, p, sel, cbytes, src, buf, threads)


def _group(it, p, sel, cbytes, src, buf, threads, room=0):
    ds = it.ds
    filters = ds.filters
    nf = len(filters)
    n = len(sel)
    addr = p.addr[sel].astype(np.int64)
    size = p.size[sel].astype(np.int64)
    mask = p.mask[sel].astype(np.int64)
    coders = sum(f.id not in (F_FLETCHER, F_SHUFFLE) + NOTHING_TO_UNDO for f in filters)
    if not room:                                    # (a coder's output can be more than the chunk it
        room = 2 * cbytes + 4096 if coders > 1 else cbytes + 64     # holds, and the one around it gives that back)
    slot = (room + 7) // 8 * 8
    ooff = np.arange(n, dtype=np.int64) * slot
    olen = np.full(n, room, np.int64)
    cur, ioff, ilen = src, addr, size
    bufs = [np.empty(n * slot, np.uint8), None]
    which = 0
    # a thread per ~64 kB of decoded chunk at most: tiny datasets stay on the caller
    nthreads = max(1, min(threads or _pool.threads(), n, (n * cbytes) >> 16))
    shuffle_last = None
    stages = list(range(nf - 1, -1, -1))             # decoding order
    last = filters[stages[-1]] if stages else None
    if last is not None and last.id == F_SHUFFLE and (not last.cd or last.cd[0] == ds.type.size):
        shuffle_last = stages.pop()                 # left to the placement
    for k in stages:
        f = filters[k]
        skip = ((mask >> k) & 1).astype(np.uint8)
        if f.id in NOTHING_TO_UNDO:
            continue
        if f.id != F_FLETCHER and f.id not in _codecs.DRIVERS:
            if skip.all():                          # (an optional filter these chunks went without)
                continue
            raise NotImplementedError('filter %d is not supported yet' % f.id)
        if f.id == F_FLETCHER:
            bad = _verify(cur, ioff, ilen, skip)
            if bad >= 0:
                raise IOError('fletcher32 checksum mismatch in a chunk of %s (at byte %d)'
                              % (ds.name, addr[bad]))
            ilen = np.where(skip == 1, ilen, ilen - 4)
            continue
        driver, mod = _codecs.DRIVERS[f.id]
        params = list(f.cd)
        if f.id == 32008 and len(params) > 4 and params[4] not in (2, 3):
            params[4] = 0                           # (a second bitshuffle's parameters landed in the
                                                    # first's: to it, what is not LZ4 or Zstd is nothing)
        cd = np.array([ds.type.size, cbytes, len(params)] + params, np.int64)
        if bufs[which] is None:
            bufs[which] = np.empty(n * slot, np.uint8)
        dst = bufs[which]
        counter = np.zeros(1, np.int64)
        status = np.zeros(n, np.int64)
        outlen = np.zeros(n, np.int64)
        wsize = mod.workspace_size(cbytes)

        def work(driver=driver, cur=cur, ioff=ioff, ilen=ilen, dst=dst, skip=skip, cd=cd,
                 counter=counter, status=status, outlen=outlen, fid=f.id, wsize=wsize):
            ws = _ws(fid, wsize)
            if fid == 1:
                tab, iws = _chunks._scratch(0)[1:]
                driver(cur, ioff, ilen, dst, ooff, olen, skip, cd, ws, counter, status, outlen, tab, iws)
            else:
                driver(cur, ioff, ilen, dst, ooff, olen, skip, cd, ws, counter, status, outlen)
        _pool.share(work, nthreads)
        if status.any():
            py = getattr(mod, 'decode_py', None)
            for i in np.nonzero(status)[0]:
                if py is None:
                    raise IOError('chunk at byte %d of %s could not be decoded (filter %d, error %d)'
                                  % (addr[i], ds.name, f.id, status[i]))
                data = bytes(cur[ioff[i]:ioff[i] + ilen[i]])
                res = np.frombuffer(bytes(py(data, cbytes, cd)), np.uint8)
                dst[ooff[i]:ooff[i] + res.size] = res
                outlen[i] = res.size
        if coders > 1 and k != stages[-1] and (outlen >= room).any() and room < 1 << 34:
            return _group(it, p, sel, cbytes, src, buf, threads, 4 * room)  # (filled it: what was cut off)
        cur, ioff, ilen = dst, ooff, outlen
        which = 1 - which
    # placement (unshuffling if the shuffle came last) by the fast path's kernel
    item = _chunks.Item(ds, it.index, it.box, it.out, it.swap, it.base)
    sh_mask = ((mask >> shuffle_last) & 1) if shuffle_last is not None else np.zeros(n, np.int64)
    stages_left = [F_SHUFFLE] if shuffle_last is not None else []
    _chunks.place_staged(item, p, sel, cur, ioff, ilen, sh_mask, stages_left, buf, threads)


def _verify(cur, ioff, ilen, skip):
    for i in range(len(ioff)):
        if not skip[i] and _fletcher32.verify(cur, int(ioff[i]), int(ioff[i] + ilen[i])) != 0:
            return i
    return -1
