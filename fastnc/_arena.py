"""Output memory kept between reads, so its pages are not faulted in again.

The first write to fresh memory costs a page fault per 4 kB, and on
Windows those are slow: of a whole-file VIIRS read (1 GB of values), 64 ms
is decoding and 54 ms is the operating system mapping fresh pages into
the new arrays; wrfout d02 (netCDF-3) reads in 80 ms into warm memory and
108 ms into fresh. A program that reads file after file -- a WRF run, a
day of granules, an xarray loop -- pays that every time.

So big outputs (>= 1 MB) are carved from anonymous memory blocks that are
kept once touched. An array is a view of `base = np.frombuffer(block)`;
numpy keeps `base` as the base of every view derived from it (the block
itself is not an ndarray, so numpy's base collapsing stops there), and a
finalizer on `base` returns the block to the pool only when the last view
is gone. The next read of a similar size writes into pages already mapped.
The pool holds at most a quarter of physical memory, half of what is
available at import, and 4 GB; `fastnc.release()` empties it.
"""
import mmap
import threading
import weakref

import numpy as np

from . import _hw

MIN_BYTES = 1 << 20             # smaller arrays: plain numpy (malloc reuses those)
ALIGN = 1 << 16
_LOCK = threading.Lock()
_FREE = []                      # [(size, block)], blocks no array uses
_held = 0


def _budget():
    total, avail = _hw.memory()
    if total is None:
        total, avail = 8 << 30, 4 << 30
    return min(total // 4, avail // 2, 4 << 30)


BUDGET = _budget()


def empty(shape, dtype):
    """An uninitialized array, from the pool when big."""
    dt = np.dtype(dtype)
    n = dt.itemsize
    for s in shape:
        n *= s
    if n < MIN_BYTES or BUDGET <= 0:
        return np.empty(shape, dt)
    size = (n + ALIGN - 1) // ALIGN * ALIGN
    block = None
    global _held
    with _LOCK:
        best = -1
        for i, (bs, b) in enumerate(_FREE):
            if size <= bs <= 2 * size and (best < 0 or bs < _FREE[best][0]):
                best = i
        if best >= 0:
            size, block = _FREE.pop(best)
            _held -= size
    if block is None:
        block = mmap.mmap(-1, size)
    base = np.frombuffer(block, np.uint8)
    weakref.finalize(base, _give_back, block, size)
    return base[:n].view(dt).reshape(shape)


def _give_back(block, size):
    global _held
    with _LOCK:
        if _held + size <= BUDGET:
            _FREE.append((size, block))
            _held += size
            return
        # over budget: let the oldest blocks go (closed when unreferenced)
        _FREE.append((size, block))
        _held += size
        while _held > BUDGET and _FREE:
            s, _ = _FREE.pop(0)
            _held -= s


def release():
    """Drop every pooled block (their memory goes back to the system)."""
    global _held
    with _LOCK:
        _FREE.clear()
        _held = 0


def held():
    """Bytes kept in the pool now."""
    return _held
