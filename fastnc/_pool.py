"""fastnc's threads: one pool, a thread per logical CPU unless set_threads()
or FASTNC_THREADS says otherwise, kept between calls (fastgrib's design).

Work is shared out by a counter, not by futures per item: `share(work, n)`
runs `work()` on the caller and n - 1 pool threads at once, and each takes
items off an atomic counter (`_intrin.take`) inside a nogil kernel until
none is left -- thousands of 3 kB chunks cost no Python per chunk, and a
thread that starts late or sits on an efficiency core just takes fewer.

The threads are fastnc's own, not a ThreadPoolExecutor's: each parks on a
lock of its own and is woken by releasing it, and a helper that wakes
after the work is done does nothing and is not waited for. Handing work
to 23 helpers cost ~100 us through futures and a queue (4.4 us each) --
for a WRF file read variable by variable, 160 such hand-offs were 16 ms.
"""
import os
import sys
import threading

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
_AUTO = 0                       # threads() when nothing set them, worked out once
_WORK = 0                       # work_threads()'s bytes per thread, worked out once
THREAD_BYTES = 1 << 15          # ... before fastnc.tune() has measured this machine's


def threads():
    """How many threads reading uses: set_threads()'s, else FASTNC_THREADS,
    else what fastnc.tune() found best here, else every CPU this process
    may use (its affinity and a container's quota count: _hw)."""
    global _AUTO
    if _THREADS:
        return _THREADS
    if not _AUTO:
        from . import _hw
        n = _hw.tuned().get('threads') or _hw.cpus()
        _AUTO = max(1, min(int(n), _hw.cpus()))
    return _AUTO


def work_threads(nbytes, most):
    """Threads worth waking for nbytes of output, at most `most` (and at
    least one): a helper joins some time after it is woken, so each must
    have more than that time's work left -- THREAD_BYTES, or what
    fastnc.tune() measured here."""
    global _WORK
    if not _WORK:
        from . import _hw
        _WORK = int(_hw.tuned().get('thread_bytes') or THREAD_BYTES)
    return max(1, min(most, nbytes // _WORK))


def set_threads(n=None):
    """Read on n threads (None: as many as suit this machine). Calls already
    running finish on the threads they have."""
    global _THREADS, _POOL
    if n is not None and (int(n) != n or n < 1):
        raise ValueError('set_threads: a whole number of threads, 1 or more (or None)')
    with _POOL_LOCK:
        _THREADS = 0 if n is None else int(n)
        old, _POOL = _POOL, None
    if old is not None:
        old.shutdown()


class _Job:
    """One share(): helpers join until the caller closes it; the caller then
    waits only for those that joined."""
    __slots__ = ('work', 'lock', 'active', 'closed', 'done', 'error')

    def __init__(self, work):
        self.work = work
        self.lock = threading.Lock()
        self.active = 0
        self.closed = False
        self.done = threading.Lock()
        self.done.acquire()
        self.error = None

    def run(self):
        with self.lock:
            if self.closed:
                return
            self.active += 1
        try:
            self.work()
        except BaseException as e:                  # (raised by the caller)
            self.error = e
        finally:
            with self.lock:
                self.active -= 1
                last = self.closed and self.active == 0
            if last:
                self.done.release()

    def close(self):
        with self.lock:
            self.closed = True
            wait = self.active > 0
        if wait:
            self.done.acquire()
        if self.error is not None:
            raise self.error


class _Worker:
    __slots__ = ('pool', 'wake', 'job', 'thread')

    def __init__(self, pool):
        self.pool = pool
        self.wake = threading.Lock()
        self.wake.acquire()
        self.job = None
        self.thread = threading.Thread(target=self.loop, name='fastnc', daemon=True)
        self.thread.start()

    def loop(self):
        fast_thread()
        pool = self.pool
        while True:
            self.wake.acquire()
            job, self.job = self.job, None
            if job is None:
                return                              # (the pool shut down)
            job.run()
            with pool.lock:
                if pool.stopped:
                    return
                pool.idle.append(self)


class _Pool:
    def __init__(self, n):
        self.lock = threading.Lock()
        self.stopped = False
        self.idle = []
        self.idle = [_Worker(self) for _ in range(n)]

    def share(self, work, n):
        job = _Job(work)
        with self.lock:
            k = min(n - 1, len(self.idle))
            helpers = self.idle[len(self.idle) - k:]
            del self.idle[len(self.idle) - k:]
        for w in helpers:
            w.job = job
            w.wake.release()
        try:
            work()
        finally:
            job.close()

    def shutdown(self):
        with self.lock:
            self.stopped = True
            idle, self.idle = self.idle, []
        for w in idle:
            w.job = None
            w.wake.release()


def share(work, n):
    """work() on the calling thread and on up to n - 1 pool threads at once,
    for a nogil kernel whose callers take items off a shared counter. The
    caller works too and, once done, closes the job: helpers not yet
    started do nothing (there is nothing left for them), and it waits only
    for those already working -- never on a busy pool. Called from a pool
    thread (work within work), it gets whichever threads are idle."""
    n = max(1, min(n, threads()))
    if n == 1:
        work()
        return
    pool().share(work, n)


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
            _POOL = _Pool(threads())
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
