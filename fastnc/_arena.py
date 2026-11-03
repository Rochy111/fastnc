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

What it saves is small: faults cost some 10 ms a GB on all threads, more on few
(see _Pages). So it is worth it to a loop, not to a process that read something a
while ago and has gone on to other things, which would hold gigabytes for
nothing. Blocks go when they have not been
used for IDLE seconds, and so does what else is kept for the next read (see
`idle`), by a timer that runs only while something is kept.
"""
import mmap
import sys
import threading
import time
import weakref

import numpy as np

from . import _hw

MIN_BYTES = 1 << 20             # smaller arrays: plain numpy (malloc reuses those)
ALIGN = 1 << 16
IDLE = 5.0                      # seconds kept memory may go unused
_LOCK = threading.Lock()
_FREE = []                      # [(size, block, when it came back)], blocks no array uses
_held = 0
_timer = None
_again = False                  # wake() was called since the timer last looked
_idle = []                      # functions to call when time passes: true while they still keep something
_KERNEL32 = None
_boxes = {}                     # id: what a thread keeps for its next read (see `kept`)
_used = False                   # one of them was asked for since the timer last looked


class _Pages:
    """Memory of this process's own, on Windows, which numpy takes as it is
    (and keeps as the base of what it makes: not an ndarray, see above). What
    mmap gives there is a section of the page file: first writes to a GB took
    101 ms on 24 threads, 531 on one, against 43 and 169 here and 31 and 48 for
    memory already touched. On Linux, a private mapping is the same thing."""
    __slots__ = ('at', 'free', '__array_interface__')

    def __init__(self, size):
        global _KERNEL32
        if _KERNEL32 is None:
            import ctypes
            k = ctypes.WinDLL('kernel32', use_last_error=True)
            k.VirtualAlloc.restype = ctypes.c_void_p
            k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
            k.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
            _KERNEL32 = k
        self.free = _KERNEL32.VirtualFree
        self.at = _KERNEL32.VirtualAlloc(None, size, 0x3000, 4)     # MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE
        if not self.at:
            raise MemoryError('cannot allocate %d bytes' % size)
        self.__array_interface__ = {'data': (self.at, False), 'shape': (size,), 'typestr': '|u1', 'version': 3}

    def __del__(self):
        if getattr(self, 'at', None):
            self.free(self.at, 0, 0x8000)           # MEM_RELEASE


def _block(size):
    """A new block, and the array of bytes that is it."""
    if sys.platform == 'win32':
        block = _Pages(size)
        return block, np.asarray(block)
    block = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    return block, np.frombuffer(block, np.uint8)


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
        for i, (bs, b, _) in enumerate(_FREE):
            if size <= bs <= 2 * size and (best < 0 or bs < _FREE[best][0]):
                best = i
        if best >= 0:
            size, block, _ = _FREE.pop(best)
            _held -= size
    if block is None:
        block, base = _block(size)
    else:
        base = np.asarray(block) if sys.platform == 'win32' else np.frombuffer(block, np.uint8)
    weakref.finalize(base, _give_back, block, size)
    return base[:n].view(dt).reshape(shape)


def _give_back(block, size):
    global _held
    with _LOCK:
        _FREE.append((size, block, time.monotonic()))
        _held += size
        # over budget: let the oldest blocks go (closed when unreferenced)
        while _held > BUDGET and _FREE:
            _held -= _FREE.pop(0)[0]
    wake()


def wake(keeper=None):
    """Something is kept: see to it that time passing lets it go. keeper: a
    function to call then, true while it still keeps something."""
    global _timer, _again
    with _LOCK:
        if keeper is not None and keeper not in _idle:
            _idle.append(keeper)
        _again = True
        if _timer is None:
            _timer = threading.Timer(IDLE, _tick)
            _timer.daemon = True
            _timer.start()


def _tick():
    """What has gone unused lets go; the timer goes on while anything is left."""
    global _held, _timer, _again
    old = time.monotonic() - IDLE
    with _LOCK:
        keep = [x for x in _FREE if x[2] > old]
        _held = sum(x[0] for x in keep)
        _FREE[:] = keep
        keepers = list(_idle)
        _again = False
    still = [f() for f in keepers]
    with _LOCK:
        _timer = None
        # (_again: something was kept after its keeper had answered)
        more = bool(_FREE) or _again or any(still)
    if more:
        wake()


def kept(box, nbytes):
    """box[0], a buffer of nbytes or more that one thread keeps between reads,
    made if it is not there; box[1] its address. A thread cannot reach another's
    locals, so the box is what time passing empties: a read under way has the
    buffer in a reference of its own and is not disturbed. Each of 24 threads
    keeps a chunk's room and pages touched stay: 100 MB of the 224 left after a
    loop over OISST's time steps."""
    global _used
    _used = True
    s = box[0]
    if s is None or s.size < nbytes:
        s = box[0] = np.empty(max(nbytes, 1 << 16), np.uint8)
        box[1] = s.ctypes.data
        with _LOCK:
            _boxes[id(box)] = box
        wake(_empty_boxes)
    return s


def _empty_boxes(now=False):
    global _used
    if _used and not now:
        _used = False
        return True
    with _LOCK:
        boxes = list(_boxes.values())
        _boxes.clear()
    for b in boxes:
        b[0] = None
    return False


def release():
    """Drop every pooled block and what threads keep (their memory goes back
    to the system)."""
    global _held
    with _LOCK:
        _FREE.clear()
        _held = 0
    _empty_boxes(True)


def held():
    """Bytes kept in the pool now."""
    return _held
