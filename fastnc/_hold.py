"""Small reads that each decode the same chunks: a region kept, decoded.

Chunks of one time step hold every place of it, so a time series at one place
decodes all the variable's chunks and keeps one value of each. The next series,
at the place beside it, decodes them all again: thirty stations, thirty times
the variable, for a few kilobytes. The cache of chunks is no help (a read of
hundreds would go through Python hundreds of times, and it holds 32 MB).

So: when a read of a small part of many chunks is followed by another of the
same chunks, a region around them is decoded once, into one array, and from then
on such a read is a copy out of it. Two reads cost what they did, about; from the
third on a read takes microseconds. Only the second pays, with at most what
reading the chunks takes, and a program making one read never gets here. A read
past the region's edge grows it, which is a pass over the chunks again.

A region is kept per dataset, within what the output pool may hold on this
machine (`_arena.BUDGET`), oldest out first, until the file is closed,
`fastnc.release()`, or it goes unused for `_arena.IDLE` seconds.
"""
import math
import threading
import time

from . import _arena, _chunks, _hw

WANTED = 4                      # a read this small a part of its region counts
SHARE = 4                       # a side kept is at least this share of the chunks' (see _around)
MOST = 8                        # past this share of the chunks' values, all of them (two places
                                # picked at random fall within it one time in ten)
_held = {}                      # chunk index -> Held, oldest first
_lock = threading.Lock()


class Held:
    __slots__ = ('file', 'first', 'stop', 'values', 'swap', 'used')

    def __init__(self, file, first, stop, values, swap):
        self.file, self.first, self.stop, self.values, self.swap = file, first, stop, values, swap
        self.used = time.monotonic()


def forget(file=None):
    """Lets go of the regions of a file, or all, and a closed file's chunks."""
    with _lock:
        for k in [k for k, h in _held.items() if file is None or h.file is file]:
            del _held[k]
    if file is not None:
        _chunks.drop_file(file)


def _expire():
    """What has gone unused lets go; true while something is still kept."""
    old = time.monotonic() - _arena.IDLE
    with _lock:
        for k in [k for k, h in _held.items() if h.used < old]:
            del _held[k]
        return bool(_held)


def _copy(h, box, out):
    h.used = time.monotonic()
    out[...] = h.values[tuple(slice(s - a, e - a, k) for (s, e, k), a in zip(box, h.first))]
    return True


def _around(box, spans, first, stop):
    """What to keep of the chunks first..stop for reads so far (spans) and this
    one: twice what they span, a quarter of each side at least, or all of it
    if that is nearly as much. Places near each other, as stations in a region
    are, then keep a sixteenth of a whole grid, and decode only as far as they
    need; places all over keep everything at the first go, and pay no second."""
    lo, hi, whole = [], [], 1
    for (s, e, _), (was, end), a, b in zip(box, spans, first, stop):
        s, e = min(s, was), max(e, end)
        side = min(max(2 * (e - s), (b - a) // SHARE), b - a)
        at = min(max((s + e - side) // 2, a), b - side)
        lo.append(at)
        hi.append(at + side)
        whole *= b - a
    if math.prod(b - a for a, b in zip(lo, hi)) * MOST > whole:
        return first, stop
    return lo, hi


def _chunks_of(ds, index, box, itemsize, nbytes):
    """(first, stop, size) of the chunks a read of nbytes in box touches, when
    it is one to keep a region for: a small part of them, and not the cache
    of chunks' to serve (else None)."""
    if not ds.filters or not nbytes:
        return None
    first, stop = [], []
    count = 1
    for (s, e, _), c, n in zip(box, index.cshape, ds.shape):
        a, b = s // c, (e - 1) // c + 1
        count *= b - a
        first.append(a * c)
        stop.append(min(b * c, n))
    if count <= _chunks.CACHE_CHUNKS and \
            count * math.prod(index.cshape) * itemsize * 2 <= _hw.scaled(_chunks.CACHE_BYTES):
        return None                                 # (the cache of chunks is for those, unless
                                                    # they are too big for it, as _cached judges:
                                                    # GridSat's images, one 20-40 MB chunk each;
                                                    # a window across four 5 MB chunks)
    size = math.prod(b - a for a, b in zip(first, stop)) * itemsize
    if nbytes * WANTED > size and (size > _hw.scaled(_chunks.CACHE_BYTES) or all(
            s == 0 and e == n for (s, e, _), n in zip(box, ds.shape))):
        return None                                 # (most of its chunks: kept only if they
                                                    # are few bytes, as the cache would keep them,
                                                    # and not all of the dataset, read again as
                                                    # fast decoded: TRMM's windows across 17 small
                                                    # chunks)
    return first, stop, size


def holds(ds, index, box, itemsize):
    """Whether a read of box is this module's: its region is kept, or would
    be from a read again. A read that decodes its chunks itself (_fused)
    leaves such a read here, or every repeat of it decodes them again."""
    h = _held.get(index)
    if h is not None and all(s >= a and e <= b for (s, e, _), a, b in zip(box, h.first, h.stop)):
        return True
    nbytes = math.prod((e - s + k - 1) // k for s, e, k in box) * itemsize
    got = _chunks_of(ds, index, box, itemsize, nbytes)
    if got is None or got[2] > _arena.BUDGET:
        return False
    first, stop, size = got
    if nbytes * WANTED <= size:
        return True                                 # (a small part: a region from the second read)
    asked = first, stop                             # (most of its chunks: the caller's way the
    if getattr(index, 'asked', None) == asked:      # first time, noted, and a region if read again)
        return True
    index.asked, index.spans = asked, [(s, e) for s, e, _ in box]
    return False


def read(ds, index, src, box, out, swap, threads):
    """The box read into out from its dataset's region; False: not so read."""
    h = _held.get(index)
    if h is not None and h.swap == swap and h.values.dtype == out.dtype and all(
            s >= a and e <= b for (s, e, _), a, b in zip(box, h.first, h.stop)):
        return _copy(h, box, out)
    if out.ndim != len(box):
        return False
    got = _chunks_of(ds, index, box, out.dtype.itemsize, out.nbytes)
    if got is None:
        return False
    first, stop, size = got
    asked = first, stop
    if getattr(index, 'asked', None) != asked:
        index.asked, index.spans = asked, [(s, e) for s, e, _ in box]
        return False
    spans = zip(h.first, h.stop) if h is not None else index.spans  # (grown from what it held)
    first, stop = _around(box, spans, first, stop)
    size = math.prod(b - a for a, b in zip(first, stop)) * out.dtype.itemsize
    if size > _arena.BUDGET:
        return False
    with _lock:
        _held.pop(index, None)
        room = _arena.BUDGET - size
        for k in list(_held):
            if sum(x.values.nbytes for x in _held.values()) <= room:
                break
            del _held[k]
    region = [(a, b, 1) for a, b in zip(first, stop)]
    values = _arena.empty([b - a for a, b in zip(first, stop)], out.dtype)
    _chunks.read(ds, index, src, region, values, swap=swap, threads=threads)
    h = Held(ds.file, first, stop, values, swap)
    with _lock:
        _held[index] = h
    _arena.wake(_expire)
    return _copy(h, box, out)
