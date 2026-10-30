"""Reading straight to the GPU (cupy arrays), identical to the CPU result.

Measured on HDF5's deflated chunks, decompressing on the card loses: even
nVIDIA's nvCOMP decodes each chunk serially and took 142 ms for VIIRS'
1 GB (the CPU's 24 threads: ~75 ms), 338 ms for wrfout (CPU: 16 ms) and
4.3 s for MODIS' 33 MB chunks (CPU: 0.1 s). So the work is split where
each side is fast:

1. the CPU cores decode chunks (inflate, unshuffle) straight into
   page-locked host memory, band by band along the first dimension;
2. each band goes up over PCIe asynchronously (45-50 GB/s from pinned
   memory on PCIe 5) while the cores decode the next one;
3. on the card: byte order swapped where the file is big-endian, and the
   mask/scale step -- so what crosses the bus is the packed integers
   (uint16 radiances), not the float32 they become.

The page-locked buffer is kept and reused between reads (locking memory
costs far more than the copy); `release()` frees it.
"""
import threading

import numpy as np

from . import _pool, _select

_LOCK = threading.Lock()
_PINNED = None                  # (cupy pinned memory, size)
_STREAM = None
BAND_BYTES = 32 << 20           # upload granularity: overlap without tiny copies


def _cupy():
    import cupy
    return cupy


def _pinned(nbytes):
    """A page-locked host buffer of at least nbytes (kept for the next read)."""
    global _PINNED
    cp = _cupy()
    if _PINNED is None or _PINNED[1] < nbytes:
        _PINNED = None
        size = max(nbytes, 64 << 20)
        _PINNED = (cp.cuda.alloc_pinned_memory(size), size)
    return np.frombuffer(_PINNED[0], np.uint8, nbytes)


def release():
    """Free the page-locked buffer and cupy's cached device memory."""
    global _PINNED, _STREAM
    with _LOCK:
        _PINNED = None
        _STREAM = None
        try:
            cp = _cupy()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except ImportError:
            pass


def _bands(bv, box):
    """The box split along its first dimension into bands for pipelining:
    at chunk boundaries, each band big enough to keep every core busy."""
    shape = _select.box_shape(box)
    if not shape or shape[0] < 2:
        return [box]
    rowbytes = int(np.prod(shape[1:])) * _itemsize(bv)
    total = rowbytes * shape[0]
    chunks = getattr(bv, 'chunks', None)
    s, e, k = box[0]
    if k != 1:
        return [box]
    step = chunks[0] if chunks else max(1, BAND_BYTES // max(rowbytes, 1))
    nchunk_rows = -(-shape[0] // step)
    # enough chunks per band for the pool, and bands of at least BAND_BYTES
    per = max(1, BAND_BYTES // max(rowbytes * step, 1))
    if chunks:
        nper_row = int(np.prod([-(-n // c) for n, c in zip(shape[1:], chunks[1:])])) or 1
        per = max(per, -(-2 * _pool.threads() // nper_row))
    if nchunk_rows <= per or total < 2 * BAND_BYTES:
        return [box]
    out = []
    first = (s // step) * step
    r = s
    while r < e:
        nxt = min(e, first + ((r - first) // step + per) * step)
        out.append([(r, nxt, 1)] + list(box[1:]))
        r = nxt
    return out


def _itemsize(bv):
    dt = bv.dtype
    return np.dtype(dt).itemsize if dt is not str else 8


def read(var, key, threads=None):
    """var[key] (a fastnc Variable) as a cupy array, masked/scaled as var is
    set to, identical to the CPU path's values."""
    cp = _cupy()
    bv = var._bv
    dt = bv.dtype
    if dt is str or np.dtype(dt).kind in 'OSUV' or np.dtype(dt).fields is not None:
        return _host_then_upload(var, key, threads)
    shape = var.shape
    box, post, drop = _select.normalize(key if isinstance(key, tuple) else (key,), shape)
    if post is not None:
        return _host_then_upload(var, key, threads)
    oshape = _select.box_shape(box)
    dt = np.dtype(dt)
    nbytes = int(np.prod(oshape)) * dt.itemsize if oshape else dt.itemsize
    global _STREAM
    with _LOCK:
        host = _pinned(nbytes).view(dt).reshape(oshape)
        if _STREAM is None:
            _STREAM = cp.cuda.Stream(non_blocking=True)
        stream = _STREAM
        dev = cp.empty(oshape, dt)
        bands = _bands(bv, box)
        r0 = box[0][0] if box else 0
        for b in bands:
            if oshape:
                lo = b[0][0] - r0
                hi = lo + _select.box_shape(b)[0]
                hb, db = host[lo:hi], dev[lo:hi]
            else:
                hb, db = host, dev
            bv.read(tuple(slice(s, e, k) for s, e, k in b), out=hb, threads=threads)
            db.set(hb, stream=stream)               # async from pinned memory
        stream.synchronize()
    if drop:
        dev = dev.reshape([n for d, n in enumerate(oshape) if d not in drop])
    return _post(var, dev)


def _post(var, dev):
    """The mask / scale step on the card."""
    bv = var._bv
    if not (var._mask or var._scale):
        return dev
    try:
        from . import _maskscale_gpu
    except ImportError:
        # until the GPU kernel is in: the CPU rule, uploaded (same values)
        from . import _maskscale
        cp = _cupy()
        res = _maskscale.apply(cp.asnumpy(dev), bv.attrs, bv.dtype, mask=var._mask,
                               scale=var._scale, always_mask=var._always_mask,
                               chartostring=var._chartostring)
        return _upload_result(res)
    return _maskscale_gpu.apply(dev, bv.attrs, bv.dtype, mask=var._mask, scale=var._scale,
                                always_mask=var._always_mask)


def _upload_result(res):
    cp = _cupy()
    if np.ma.isMaskedArray(res):
        return cp.asarray(np.ma.filled(res, np.nan) if res.dtype.kind == 'f' else res.data), \
            cp.asarray(np.ma.getmaskarray(res))
    return cp.asarray(res)


def _host_then_upload(var, key, threads):
    """Types the card has no use for as such (strings, compounds) and fancy
    indexes: read on the CPU, then upload what numpy can hold."""
    cp = _cupy()
    old = var._device
    var.__dict__['_device'] = 'cpu'
    try:
        res = var.read(key, threads=threads)
    finally:
        var.__dict__['_device'] = old
    if isinstance(res, np.ndarray) and res.dtype.kind not in 'OSUV':
        return cp.asarray(res)
    return res
