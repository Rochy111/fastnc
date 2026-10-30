"""Chunk bytes read with positional reads instead of through the mapping.

A file opened afresh is mapped afresh, and the first touch of each mapped
page costs a fault even when the file sits in the OS cache: 22 ms per
113 MB on this machine, worse when many threads fault small pieces at once
(Metop-SG's 6 kB row chunks: a whole-file read 6.5 ms into warm pages, 38
ms into fresh ones). Positional reads from the cache into memory that is
already mapped (the output arena's) run at ~21 GB/s on 8 threads and fault
nothing. So a batch's chunks are read that way when they add up to more
than STAGE_MIN: their byte ranges merged (gaps under GAP read through),
read in pieces on the pool into one pooled buffer, the chunk addresses
moved into it. The kernels then decode from that buffer as from the file.

The readers are _nc3's: one overlapped handle shared by every thread on
Windows (a handle opened the usual way serializes its readers), preadv
elsewhere.
"""
import threading

import numpy as np

from . import _arena, _pool

STAGE_MIN = 1 << 20             # batches smaller than this read through the mapping
GAP = 256 << 10                 # merge ranges closer than this (read the gap too)
PIECE = 4 << 20                 # one positional read at most this big


class _Shim:
    """What _nc3's readers want of a file: its path and a descriptor."""

    def __init__(self, path):
        self.path = path
        self._fh = open(path, 'rb', buffering=0)

    def close(self):
        self._fh.close()


class Reader:
    """Positional reads of one local file, shared by the threads."""

    def __init__(self, path):
        from ._nc3 import _IO
        self._shim = _Shim(path)
        self._io = _IO(self._shim)
        self._lock = threading.Lock()

    def close(self):
        try:
            self._io.close()
        finally:
            self._shim.close()

    def stage(self, addr, size, threads=None):
        """The bytes of every range (addr[i], size[i]) in one read-only
        buffer, and each range's position in it (addr remapped)."""
        n = len(addr)
        order = np.argsort(addr, kind='stable')
        a = addr[order]
        e = a + size[order]
        # runs of ranges no further apart than GAP
        brk = np.ones(n, bool)
        if n > 1:
            runend = np.maximum.accumulate(e)
            brk[1:] = a[1:] > runend[:-1] + GAP
        rid = np.cumsum(brk) - 1
        starts = a[brk]
        ends = np.zeros(len(starts), np.int64)
        np.maximum.at(ends, rid, e)
        lens = ends - starts
        offs = np.zeros(len(starts), np.int64)
        offs[1:] = np.cumsum(lens)[:-1]
        total = int(lens.sum())
        buf = _arena.empty((total + 16,), np.uint8)
        base = buf.ctypes.data
        pieces = []
        for s, ln, o in zip(starts.tolist(), lens.tolist(), offs.tolist()):
            for k in range(0, ln, PIECE):
                pieces.append((s + k, min(PIECE, ln - k), o + k))
        io = self._io
        short = []

        def one(pc):
            off, ln, o = pc
            got = io.read(buf, base, o, ln, off)
            if got < ln:
                short.append(pc)
        _pool.map_items(one, pieces)
        if short:
            raise IOError('file shorter than its chunk index says (read %r)' % (short[0],))
        new = np.empty(n, np.int64)
        new[order] = a - starts[rid] + offs[rid]
        buf.flags.writeable = False                 # the type kernels see for a mapped file
        return buf, new
