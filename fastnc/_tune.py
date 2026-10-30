"""fastnc.tune(): real work timed on this machine, kept for the next process
(a few seconds, once per machine; everything works without it, on what
_hw finds and on defaults that scale).

    threads        how many threads decode fastest: the performance cores,
                   all cores, or every logical CPU (the fewest within 3% of
                   the best: threads that add nothing only use up the
                   machine)
    thread_bytes   the work worth waking one more thread for: the least at
                   which two threads beat one, timed
    load_s        the time numba and the kernels take to load, in a fresh
                   process (when reading without them stops paying)
    band_bytes     the GPU: what crosses the bus in about the time an
                   upload takes to get going

The workload is deflated data of the shape real files have (a byte
shuffled noisy field, 128 kB chunks), in memory: no file is needed.
"""
import os
import subprocess
import sys
import time
import zlib

import numpy as np

from . import _chunks, _hw, _inflate, _pool
from ._intrin import take
from ._jit import njit

CHUNK = 1 << 17
KINDS = 16                      # distinct chunks
BATCH = 512                     # chunks a run decodes
BAND_TIME = 180e-6              # seconds of bus time a band of the GPU pipeline is
BAND_LEAST, BAND_MOST = 2 << 20, 64 << 20
SMALL = 1 << 14                 # the chunks of the work-per-thread trial
WORK_STEPS = tuple(1 << k for k in range(14, 21))   # 16 kB .. 1 MB of work per thread


@njit(cache=True, nogil=True)
def _decode_all(src, starts, ends, out, ctr, tab, iws):
    """Inflate streams taken off ctr[0], each into this thread's out."""
    n = starts.size
    while True:
        i = take(ctr, 0)
        if i >= n:
            return
        _inflate.zlib_inflate(src, starts[i], ends[i], out, 0, out.size, out.size, tab, iws)


def _chunks_of(nbytes):
    """KINDS deflated chunks of nbytes: (the bytes, starts, sizes)."""
    rng = np.random.default_rng(1)
    blobs = []
    for _ in range(KINDS):
        n = nbytes // 2
        x = np.cumsum(rng.integers(-40, 41, n)).astype(np.int16)
        s = np.concatenate([x.view(np.uint8)[0::2], x.view(np.uint8)[1::2]])     # shuffled
        blobs.append(zlib.compress(s.tobytes(), 4))
    src = np.frombuffer(b''.join(blobs), np.uint8)
    sizes = np.array([len(b) for b in blobs], np.int64)
    return src, np.concatenate([[0], np.cumsum(sizes)[:-1]]), sizes


def _workload():
    src, pos, sizes = _chunks_of(CHUNK)
    pick = np.arange(BATCH) % KINDS
    return src, pos[pick], pos[pick] + sizes[pick]


def _rate(src, starts, ends, n):
    """Bytes a second decoded on n threads (best of three)."""
    best = 0.0
    for _ in range(3):
        ctr = np.zeros(1, np.int64)

        def work():
            sc, tab, iws = _chunks._scratch(CHUNK + 8)
            _decode_all(src, starts, ends, sc[:CHUNK], ctr, tab, iws)
        t = time.perf_counter()
        _pool.share(work, n)
        best = max(best, BATCH * CHUNK / (time.perf_counter() - t))
    return best


def _timed(src, starts, ends, n):
    """Best time of a small job on n threads (the helpers given time to park
    between runs, as between a program's reads)."""
    best = 1e9
    for _ in range(40):
        time.sleep(0.0005)
        ctr = np.zeros(1, np.int64)

        def work():
            sc, tab, iws = _chunks._scratch(SMALL + 8)
            _decode_all(src, starts, ends, sc[:SMALL], ctr, tab, iws)
        t = time.perf_counter()
        _pool.share(work, n)
        best = min(best, time.perf_counter() - t)
    return best


def _thread_bytes():
    """The least work per thread at which a second thread pays (by 10%):
    timed, since what it takes a helper to join depends on the OS, the GIL
    and how many are woken at once."""
    src, pos, sizes = _chunks_of(SMALL)
    for per in WORK_STEPS:
        k = 2 * per // SMALL
        pick = np.arange(k) % KINDS
        s, e = pos[pick], pos[pick] + sizes[pick]
        if _timed(src, s, e, 2) < 0.9 * _timed(src, s, e, 1):
            return per
    return WORK_STEPS[-1]


def _load_time():
    """Load numba and the kernels in a fresh process: _chunks notes it."""
    code = 'import fastnc; fastnc.warm()'
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, timeout=600)


def _gpu():
    """{'key', 'upload_bytes_s', 'band_bytes'} of the current GPU, or None."""
    try:
        import cupy as cp
        p = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = p['name'].decode() if isinstance(p['name'], bytes) else str(p['name'])
        n = 256 << 20
        pin = cp.cuda.alloc_pinned_memory(n)
        host = np.frombuffer(pin, np.uint8, n)
        host[::4096] = 1
        dev = cp.empty(n, np.uint8)
        best = 0.0
        for _ in range(3):
            t = time.perf_counter()
            dev.set(host)
            cp.cuda.Device().synchronize()
            best = max(best, n / (time.perf_counter() - t))
    except Exception:
        return None
    band = int(min(BAND_MOST, max(BAND_LEAST, best * BAND_TIME)))
    return dict(key=_hw.machine_key(name), name=name, upload_bytes_s=int(best), band_bytes=band)


def tune(gpu=True, verbose=True):
    """Time real decoding here and keep what suits this machine (see the
    module's docstring). Returns the measurements."""
    _chunks.warm()
    say = print if verbose else (lambda *a: None)
    keep = _pool._THREADS
    _pool._THREADS = 0                              # (everything allowed, for the trial)
    _pool._AUTO = _hw.cpus()
    try:
        src, starts, ends = _workload()
        topo = _hw.topology()
        tries = sorted({min(c, _hw.cpus()) for c in
                        (topo['perf_cores'], topo['cores'], topo['logical'], _hw.cpus())})
        rates = {}
        for n in tries:
            rates[n] = _rate(src, starts, ends, n)
            say('%2d threads: %6.2f GB/s decoded' % (n, rates[n] / 1e9))
        top = max(rates.values())
        threads = min(n for n, r in rates.items() if r >= 0.97 * top)
        work = _thread_bytes()
        say('a second thread pays from %d kB of work per thread' % (work >> 10))
    finally:
        _pool._THREADS = keep
        _pool._AUTO = 0
        _pool._WORK = 0
    out = dict(threads=threads, thread_bytes=work, decode_bytes_s=int(top))
    _hw.save_tuned(cpu=out)
    _load_time()
    g = _gpu() if gpu else None
    if g is not None:
        _hw.save_tuned(gpu=g)
        say('%s: %.1f GB/s upload, %d MB bands' % (g['name'], g['upload_bytes_s'] / 1e9,
                                                  g['band_bytes'] >> 20))
    out = dict(_hw.tuned())
    if g is not None:
        out['gpu'] = g
    say('kept in', _hw._tune_path())
    return out
