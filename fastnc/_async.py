"""Reads in the background: `read_async` / `load_async` hand a read to a
worker of fastnc's and return a concurrent.futures.Future of its result, so a
program can be using one array while the next is read. Two workers: each read
already decodes on every core, so more at once only queue for them; what a
second worker buys is the next read starting while the last one is collected.
The workers are made on first use and declared fast to the OS, as any thread
that reads is (_pool.fast_thread)."""
import threading
from concurrent.futures import ThreadPoolExecutor

from . import _pool

WORKERS = 2
_executor = None
_lock = threading.Lock()


def submit(fn, *args, **kwargs):
    """fn(*args, **kwargs) on a worker: its Future."""
    global _executor
    ex = _executor
    if ex is None:
        with _lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(WORKERS, thread_name_prefix='fastnc-read',
                                               initializer=_pool.fast_thread)
            ex = _executor
    return ex.submit(fn, *args, **kwargs)
