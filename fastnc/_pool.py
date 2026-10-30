"""fastnc's threads: one pool, a thread per logical CPU unless set_threads()
or FASTNC_THREADS says otherwise, kept between calls (fastgrib's design).

Work is shared out by a counter, not by futures per item: `share(work, n)`
runs `work()` on the caller and n - 1 pool threads at once, and each takes
items off an atomic counter (`_intrin.take`) inside a nogil kernel until
none is left -- thousands of 3 kB chunks cost no Python per chunk, and a
thread that starts late or sits on an efficiency core just takes fewer.
"""
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

_POOL = None
_POOL_LOCK = threading.Lock()


def _forked():
    """In a child of fork(): the parent's threads did not come along, so its
    pool is a new one's to make (else every worker waits on threads that do
    not exist)."""
    global _POOL, _POOL_LOCK
    _POOL = None
    _POOL_LOCK = threading.Lock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_forked)


def _threads_from_env():
    v = os.environ.get('FASTNC_THREADS', '').strip()
    if not v:
        return 0
    try:
        return max(1, int(v))
    except ValueError:
        import warnings
        warnings.warn('FASTNC_THREADS=%r is not a number of threads: ignored' % v)
        return 0


_THREADS = _threads_from_env()


def threads():
    """How many threads reading uses: set_threads()'s, else FASTNC_THREADS,
    else one per logical CPU."""
    return _THREADS or os.cpu_count() or 1


def set_threads(n=None):
    """Read on n threads (None: one per logical CPU). Calls already running
    finish on the threads they have."""
    global _THREADS, _POOL
    if n is not None and (int(n) != n or n < 1):
        raise ValueError('set_threads: a whole number of threads, 1 or more (or None)')
    with _POOL_LOCK:
        _THREADS = 0 if n is None else int(n)
        old, _POOL = _POOL, None
    if old is not None:
        old.shutdown(wait=False)


def share(work, n):
    """work() on the calling thread and on up to n - 1 pool threads at once,
    for a nogil kernel whose callers take items off a shared counter. The
    caller works too and, once done, calls off helpers not yet started
    (there is nothing left for them): it never waits on a busy pool."""
    n = max(1, min(n, threads()))
    if n == 1:
        work()
        return
    futs = [pool().submit(work) for _ in range(n - 1)]
    try:
        work()
    finally:
        for f in futs:
            if not f.cancel():
                f.result()


def map_items(fn, items):
    """fn(item) for every item on the pool, the caller helping; for work done
    item by item in Python (that releases the GIL inside)."""
    items = list(items)
    if len(items) <= 1:
        for it in items:
            fn(it)
        return
    lock = threading.Lock()
    nxt = [0]

    def work():
        while True:
            with lock:
                i = nxt[0]
                nxt[0] += 1
            if i >= len(items):
                return
            fn(items[i])
    share(work, len(items))


def pool():
    """The threads, made on first use."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(threads(), initializer=fast_thread)
    return _POOL


def fast_thread():
    """Declare this thread's work as the user's (high QoS) on Windows: a
    process without a focused window (a script from a terminal) counts as
    background work there, and on a hybrid CPU its busy threads go to the
    efficiency cores -- timings then swing 30-50%."""
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        from ctypes import wintypes

        class State(ctypes.Structure):
            _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG),
                        ('StateMask', wintypes.ULONG)]
        k = ctypes.windll.kernel32
        k.GetCurrentThread.restype = wintypes.HANDLE
        st = State(1, 1, 0)      # execution speed: managed, not throttled
        k.SetThreadInformation(wintypes.HANDLE(k.GetCurrentThread()), 3,  # ThreadPowerThrottling
                               ctypes.byref(st), ctypes.sizeof(st))
    except Exception:
        pass
