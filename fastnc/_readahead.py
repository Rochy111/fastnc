"""Readahead across variables: whole variables read one after another in
the file's order (xarray's Dataset.load(), `for v in ds.variables:
ds[v][:]`) are decoded several at a time.

Read alone, a WRF variable of four chunks keeps four of 24 cores busy; read
together, every chunk of every variable shares the pool (_h5read.read_many).
So once two big variables of a group have been read whole in a row, the
variables that follow them are decoded together into a window -- 32 MB at
first, doubling up to 256 MB while reads keep consuming it in order -- and
the next whole reads are served from it. A read out of order drops the
window: what is held is at most one window, never the file.

Only raw values are held (mask / scale stay with the read that asks), so
every result is the read's own result; only when its decoding happened
moves.
"""
import math
import threading

import numpy as np

from . import _hw

START = 64 << 20              # first window (bytes of raw values; 32, 64, 128, 256 MB: WRF xarray load 64, 59, 59, 57 ms)
MAX = 256 << 20                 # the window grows by doubling up to this
BIG = 256 << 10                 # reads of variables at least this big drive the pattern
MISSES = 3                      # reads missing a full window after which a group reads none
ENABLED = True


class Readahead:
    """One group's readahead state."""

    def __init__(self, group):
        self.group = group
        self.lock = threading.Lock()
        self.order = None           # [front-end Variable] in the group's order
        self.pos = None             # id(bv) -> index in order
        self.first = None           # the first big variable read, until the order is wanted
        self.last = -1              # index of the last big variable read whole
        self.streak = 0
        self.window = _hw.scaled(START)
        self.cache = {}             # index -> (raw array, proved to hold nothing masked)
        self.missed = 0             # reads the window held something else for

    def _index(self):
        if self.order is None:
            self.order = list(self.group.variables.values())
            self.pos = {id(v._bv): i for i, v in enumerate(self.order)}
        return self.pos

    def take(self, var):
        """(var's whole raw values, whether the read proved none of them is
        masked) if the window holds them (then not any more), else None."""
        with self.lock:
            if not self.cache:
                return None
            i = self._index().get(id(var._bv))
            got = self.cache.pop(i, None) if i is not None else None
            if got is None:
                self.missed += 1                # (a window read for nothing: see _fetch)
                return None
            if _nbytes(var) >= BIG:
                self.last = i
            if not self.cache:                  # consumed in order: a bigger one next
                self.window = min(2 * self.window, _hw.scaled(MAX))
                self._fetch(i + 1)
            return got

    def note(self, var):
        """var was just read whole (not from the window). (A first read is
        only remembered: the group's order takes every variable's metadata,
        which one read of a file should not pay for.)"""
        if _nbytes(var) < BIG:
            return
        with self.lock:
            if self.order is None and self.first is None:
                self.first = var
                return
            first, self.first = self.first, None
            if first is not None:
                if hasattr(self.group, 'arrange'):      # (a group that can be walked in two orders)
                    self.group.arrange(first, var)
                self._note(first)
            self._note(var)

    def _note(self, var):
        i = self._index().get(id(var._bv))
        if i is None:
            return
        nxt = next((j for j in range(self.last + 1, len(self.order))
                    if _nbytes(self.order[j]) >= BIG), None)
        if self.last >= 0 and i == nxt:
            self.streak += 1
        else:
            self.streak = 1
            self.cache.clear()
            self.window = _hw.scaled(START)
        self.last = i
        if self.streak >= 2:
            self._fetch(i + 1)

    def _fetch(self, start):
        """The variables from order[start] on decoded together, up to a
        window of bytes (only those read_many decodes in one pass)."""
        from ._h5read import read_many
        self.cache.clear()
        if self.missed >= MISSES:
            return                              # (this program's reads are not what it took them for)
        picked = []
        total = 0
        for j in range(start, len(self.order)):
            v = self.order[j]
            if not _batchable(v):
                if picked:
                    break                       # (the window stays one run of the order)
                continue
            n = _nbytes(v)
            if picked and total + n > self.window:
                break
            picked.append(j)
            total += n
        if len(picked) < 2:
            return
        probes = [self.order[j]._probe() for j in picked]
        raws = read_many([self.order[j]._bv._reader for j in picked], probes=probes)
        for j, a, pr in zip(picked, raws, probes):
            self.cache[j] = (a, pr is not None and pr.proven)


def _nbytes(v):
    bv = v._bv
    try:
        dt = np.dtype(bv.dtype)
    except TypeError:
        return 0
    return math.prod(bv.shape) * dt.itemsize if bv.shape else dt.itemsize


def _batchable(v):
    """A variable read_many decodes in its one pass, whole raw values being
    exactly what v's own read gives."""
    bv = v._bv
    r = getattr(bv, '_reader', None)
    if r is None or type(r).__name__ != 'Reader' or not bv.shape:
        return False
    ds = r.ds
    t = ds.type
    if t.cls not in (0, 1) or t.dtype is None or ds.null:
        return False
    from . import _h5read
    if _h5read.reduced(t) or (t.dtype.byteorder == '>' and not _h5read._simple_swap(t.dtype)):
        return False
    if not (ds.layout.cls == 2 or (ds.layout.cls == 1 and r.virtual)):
        return False
    if tuple(ds.shape) != tuple(bv.shape) or v._short() or np.dtype(bv.dtype) != _h5read.native(t.dtype):
        return False
    return True


def whole(key, shape):
    """Whether key (a tuple) reads all of an array of this shape, as is."""
    if not key or (len(key) == 1 and key[0] is Ellipsis):
        return True
    if len(key) > len(shape):
        return False
    for k, n in zip(key, shape):
        if k is Ellipsis:
            return len(key) == 1
        if not (isinstance(k, slice) and k.step in (None, 1) and k.start in (None, 0)
                and (k.stop is None or (isinstance(k.stop, int) and k.stop >= n))):
            return False
    return True
