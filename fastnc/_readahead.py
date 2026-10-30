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
import threading

import numpy as np

START = 64 << 20                # first window (bytes of raw values; 32, 64, 128, 256 MB: WRF xarray load 64, 59, 59, 57 ms)
MAX = 256 << 20                 # the window grows by doubling up to this
BIG = 256 << 10                 # reads of variables at least this big drive the pattern
ENABLED = True


class Readahead:
    """One group's readahead state."""

    def __init__(self, group):
        self.group = group
        self.lock = threading.Lock()
        self.order = None           # [front-end Variable] in the group's order
        self.pos = None             # id(bv) -> index in order
        self.last = -1              # index of the last big variable read whole
        self.streak = 0
        self.window = START
        self.cache = {}             # index -> raw array

    def _index(self):
        if self.order is None:
            self.order = list(self.group.variables.values())
            self.pos = {id(v._bv): i for i, v in enumerate(self.order)}
        return self.pos

    def take(self, var):
        """var's whole raw values if the window holds them (then not any
        more), else None."""
        with self.lock:
            if not self.cache:
                return None
            i = self._index().get(id(var._bv))
            raw = self.cache.pop(i, None) if i is not None else None
            if raw is None:
                return None
            if _nbytes(var) >= BIG:
                self.last = i
            if not self.cache:                  # consumed in order: a bigger one next
                self.window = min(2 * self.window, MAX)
                self._fetch(i + 1)
            return raw

    def note(self, var):
        """var was just read whole (not from the window)."""
        if _nbytes(var) < BIG:
            return
        with self.lock:
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
                self.window = START
            self.last = i
            if self.streak >= 2:
                self._fetch(i + 1)

    def _fetch(self, start):
        """The variables from order[start] on decoded together, up to a
        window of bytes (only those read_many decodes in one pass)."""
        from ._h5read import read_many
        self.cache.clear()
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
        raws = read_many([self.order[j]._bv._reader for j in picked])
        for j, a in zip(picked, raws):
            self.cache[j] = a


def _nbytes(v):
    bv = v._bv
    try:
        dt = np.dtype(bv.dtype)
    except TypeError:
        return 0
    return int(np.prod(bv.shape)) * dt.itemsize if bv.shape else dt.itemsize


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
    if (tuple(ds.shape) != tuple(bv.shape) or tuple(v.shape) != tuple(bv.shape)
            or np.dtype(bv.dtype) != _h5read.native(t.dtype)):
        return False
    return True


def whole(key, shape):
    """Whether key (a tuple) reads all of an array of this shape, as is."""
    if key in ((), (Ellipsis,)):
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
