"""Quick measurements that decide which new ideas are worth building.

  dupes   how much of each file's work is chunks repeated byte for byte
  smt     does a second hardware thread on one core add decoding speed?
  adler   libdeflate with and without its Adler-32 check
  faults  fresh memory: page by page, on many threads, or in one call
  nc3     netCDF-3 reads against the thread count

    python tools/profile/ideas_a.py dupes smt adler faults nc3
"""
import ctypes
import hashlib
import mmap
import os
import sys
import threading
import time
import zlib

import numpy as np
from common import fast_thread, path, real_files

sys.path.insert(0, path('.'))
import fastnc                                       # noqa: E402
from fastnc import _hdf5, _nc4, _pool, _zlib        # noqa: E402

GOES = ['extra/OR_ABI-L1b-RadF-M6C02_G19_s20263021200207_e20263021209516_c20263021209551.nc',
        'extra/OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc']


def chunks_of(f):
    """[(addr, size, raw bytes, filters)] of every deflated chunk of a file,
    and the file as an array."""
    fh = open(path(f), 'rb')
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    src = np.frombuffer(mm, np.uint8)
    nc = _nc4.NC4File(_hdf5.H5File(mm), src)
    out = []

    def walk(g):
        for v in g.variables.values():
            ds = v.ds
            if ds.type.dtype is None or not ds.shape or ds.layout.cls != 2:
                continue
            fs = [x.id for x in ds.filters]
            if 1 not in fs:
                continue
            idx = v._reader.index
            cb = int(np.prod(idx.chunks)) * ds.type.size
            cut = 4 if fs[-1] == 3 else 0
            out.extend((int(a), int(s) - cut, cb) for a, s in zip(idx.addr, idx.size))
        for s in g.groups.values():
            walk(s)
    walk(nc.root)
    return src, out


def dupes():
    print('%-22s %7s | %6s %6s %6s | %8s %8s' % ('file', 'chunks', 'dup%', 'zb%', 'raw%', 'all ms', 'uniq ms'))
    for f in real_files() + GOES:
        if open(path(f), 'rb').read(3) == b'CDF':
            continue
        src, ch = chunks_of(f)
        if not ch:
            continue
        seen = {}
        d = dz = dr = tz = tr = 0
        uniq = []
        for a, s, cb in ch:
            h = hashlib.blake2b(src[a:a + s], digest_size=16).digest()
            tz += s
            tr += cb
            if h in seen:
                d += 1
                dz += s
                dr += cb
            else:
                seen[h] = 1
                uniq.append((a, s, cb))
        kind, lib = _zlib._load()
        fp = _zlib.inflater()
        buf = np.empty(max(c[2] for c in ch) + 64, np.uint8)
        base = src.ctypes.data
        bp = buf.ctypes.data

        def run(lst):
            t = time.perf_counter()
            for a, s, cb in lst:
                fp(base + a, s, bp, cb)
            return (time.perf_counter() - t) * 1e3
        ta = min(run(ch) for _ in range(2))
        tu = min(run(uniq) for _ in range(2))
        print('%-22s %7d | %5.1f%% %5.1f%% %5.1f%% | %8.1f %8.1f' % (
            f[-22:], len(ch), 100 * d / len(ch), 100 * dz / tz, 100 * dr / tr, ta, tu), flush=True)


def _pin(mask):
    k = ctypes.windll.kernel32
    k.GetCurrentThread.restype = ctypes.c_void_p
    k.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k.SetThreadAffinityMask.restype = ctypes.c_size_t
    return k.SetThreadAffinityMask(k.GetCurrentThread(), mask)


def smt():
    """Decode on logical CPUs 0, 0+1 (one core's two threads), 0+2 (two cores)."""
    from fastnc import _tune
    fastnc.warm()
    src, pos, sizes = _tune._chunks_of(1 << 17)
    fp = _zlib.inflater()
    base = src.ctypes.data
    a = (pos + base).tolist()
    s = sizes.tolist()
    REP = 40

    def run(cpus):
        out = [0.0] * len(cpus)
        start = threading.Barrier(len(cpus))

        def w(k):
            _pin(1 << cpus[k])
            fast_thread()
            buf = np.empty((1 << 17) + 64, np.uint8)
            bp = buf.ctypes.data
            start.wait()
            t = time.perf_counter()
            for _ in range(REP):
                for x, y in zip(a, s):
                    fp(x, y, bp, 1 << 17)
            out[k] = time.perf_counter() - t
        ts = [threading.Thread(target=w, args=(k,)) for k in range(len(cpus))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return len(cpus) * REP * len(a) * (1 << 17) / max(out) / 1e9
    for name, c in (('one thread', [0]), ('two threads, one core', [0, 1]), ('two threads, two cores', [0, 2]),
                    ('E core alone', [16]), ('two E cores', [16, 17])):
        print('%-24s %6.2f GB/s' % (name, max(run(c) for _ in range(3))), flush=True)


def adler():
    from fastnc import _tune
    kind, lib = _zlib._load()
    if kind != 'libdeflate':
        print('no libdeflate here')
        return
    lib.libdeflate_deflate_decompress.restype = ctypes.c_int
    lib.libdeflate_deflate_decompress.argtypes = lib.libdeflate_zlib_decompress.argtypes
    d = lib.libdeflate_alloc_decompressor()
    for f in ('wrfout_d01_2025-10-26_06_20_00.nc', 'VJ102IMG_NRT.A2026265.2006.021.nc', GOES[0]):
        src, ch = chunks_of(f)
        buf = np.empty(max(c[2] for c in ch) + 64, np.uint8)
        base, bp = src.ctypes.data, buf.ctypes.data
        got = ctypes.c_size_t(0)
        ref = ctypes.byref(got)

        def zl():
            for a, s, cb in ch:
                lib.libdeflate_zlib_decompress(d, base + a, s, bp, cb, ref)

        def raw():
            for a, s, cb in ch:
                lib.libdeflate_deflate_decompress(d, base + a + 2, s - 6, bp, cb, ref)

        def best(fn):
            r = []
            for _ in range(5):
                t = time.perf_counter()
                fn()
                r.append(time.perf_counter() - t)
            return min(r) * 1e3
        a, b = best(zl), best(raw)
        print('%-24s with Adler-32 %8.1f ms, without %8.1f ms (%.1f%% less)' % (f[-24:], a, b, 100 * (1 - b / a)), flush=True)


def faults():
    if sys.platform != 'win32':
        print('Windows only')
        return
    k = ctypes.windll.kernel32
    k.VirtualAlloc.restype = ctypes.c_void_p
    k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    k.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    k.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k.VirtualUnlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessWorkingSetSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    N = 1 << 30
    COMMIT_RESERVE, RW, RELEASE = 0x3000, 4, 0x8000

    def fresh():
        return k.VirtualAlloc(None, N, COMMIT_RESERVE, RW)

    def done(p):
        k.VirtualFree(p, 0, RELEASE)

    def touch1(p):
        a = np.ctypeslib.as_array((ctypes.c_ubyte * N).from_address(p))
        a[::4096] = 1

    def touchn(p, n):
        a = np.ctypeslib.as_array((ctypes.c_ubyte * N).from_address(p))
        step = N // n

        def one(i):
            a[i * step:(i + 1) * step:4096] = 1
        _pool.map_items(one, range(n))

    def lock(p):
        if not k.VirtualLock(p, N):
            print('VirtualLock failed', ctypes.GetLastError())
        k.VirtualUnlock(p, N)

    ok = k.SetProcessWorkingSetSize(k.GetCurrentProcess(), N * 2, N * 4)
    print('working set raised:', bool(ok))

    def trial(name, fn):
        r = []
        for _ in range(5):
            p = fresh()
            t = time.perf_counter()
            fn(p)
            r.append(time.perf_counter() - t)
            done(p)
        print('%-28s %7.1f ms per GB' % (name, min(r) * 1e3), flush=True)
    trial('touch, 1 thread', touch1)
    for n in (4, 8, 24):
        trial('touch, %d threads' % n, lambda p, n=n: touchn(p, n))
    trial('VirtualLock + Unlock', lock)

    def lock_then_touch(p):
        lock(p)
        touch1(p)
    trial('... then touch every page', lock_then_touch)

    def lockn(p, n=8):
        step = N // n

        def one(i):
            k.VirtualLock(ctypes.c_void_p(p + i * step), step)
            k.VirtualUnlock(ctypes.c_void_p(p + i * step), step)
        _pool.map_items(one, range(n))
    trial('VirtualLock on 8 threads', lockn)


def pin():
    """Decoding on n threads, placed by the OS or pinned one per logical CPU
    (nogil, in the real kernel: no GIL between chunks)."""
    from fastnc import _chunks, _tune
    fastnc.warm()
    src, starts, ends = _tune._workload()
    N = (1 << 17) * len(starts)

    def run(cpus):
        ctr = np.zeros(1, np.int64)
        go = threading.Barrier(len(cpus) + 1)
        out = [0.0] * len(cpus)

        def w(k):
            if cpus[k] is not None:
                _pin(1 << cpus[k])
            fast_thread()
            sc, tab, iws = _chunks._scratch((1 << 17) + 8)
            go.wait()
            _tune._decode_all(src, starts, ends, sc[:1 << 17], ctr, tab, iws)
            out[k] = time.perf_counter()
        ts = [threading.Thread(target=w, args=(k,)) for k in range(len(cpus))]
        for t in ts:
            t.start()
        go.wait()
        t0 = time.perf_counter()
        for t in ts:
            t.join()
        return N / (max(out) - t0) / 1e9

    def best(c):
        return max(run(c) for _ in range(5))
    P = list(range(0, 16, 2))
    S = list(range(1, 16, 2))
    E = list(range(16, 24))
    for name, c in (('8 OS placed', [None] * 8), ('8 pinned P', P),
                    ('16 OS placed', [None] * 16), ('16 pinned P+E', P + E), ('16 pinned P+HT', P + S),
                    ('24 OS placed', [None] * 24), ('24 pinned all', P + S + E),
                    ('8 pinned E', E)):
        print('%-16s %6.2f GB/s' % (name, best(c)), flush=True)


def pin2():
    """A long real decode (every chunk of GOES-19 C02, ~1 GB, into scratch,
    several times) on n threads: placed by the OS, pinned, or with an
    ideal-processor hint only."""
    from fastnc import _chunks, _tune
    fastnc.warm()
    src, ch = chunks_of(GOES[0])
    cb = max(c[2] for c in ch)
    REP = 3
    starts = np.array([c[0] for c in ch] * REP, np.int64)
    ends = starts + np.array([c[1] for c in ch] * REP, np.int64)
    N = sum(c[2] for c in ch) * REP
    k = ctypes.windll.kernel32
    k.GetCurrentThread.restype = ctypes.c_void_p
    k.SetThreadIdealProcessor.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    def run(cpus, how):
        ctr = np.zeros(1, np.int64)
        go = threading.Barrier(len(cpus) + 1)
        end = [0.0] * len(cpus)

        def w(i):
            if how == 'pin':
                _pin(1 << cpus[i])
            elif how == 'hint':
                k.SetThreadIdealProcessor(k.GetCurrentThread(), cpus[i])
            fast_thread()
            sc, tab, iws = _chunks._scratch(cb + 8)
            go.wait()
            _tune._decode_all(src, starts, ends, sc[:cb], ctr, tab, iws)
            end[i] = time.perf_counter()
        ts = [threading.Thread(target=w, args=(i,)) for i in range(len(cpus))]
        for t in ts:
            t.start()
        go.wait()
        t0 = time.perf_counter()
        for t in ts:
            t.join()
        return N / (max(end) - t0) / 1e9

    def best(c, how):
        return max(run(c, how) for _ in range(3))
    P = list(range(0, 16, 2))
    S = list(range(1, 16, 2))
    E = list(range(16, 24))
    for name, c in (('8: P cores', P), ('16: P+E', P + E), ('16: P+HT', P + S), ('24: all', P + S + E)):
        print('%-12s OS %6.2f   hint %6.2f   pinned %6.2f GB/s' % (
            name, best(c, 'os'), best(c, 'hint'), best(c, 'pin')), flush=True)


def nc3map():
    """A netCDF-3 file's bytes copied and swapped straight out of a mapping
    (one fused pass, first touch and later), on n threads, against the
    positional reads then a swap in place."""
    fastnc.warm()
    fast_thread()
    from fastnc._intrin import take
    from fastnc._jit import njit

    @njit(nogil=True)
    def copy_swap(src, dst, ctr, piece):
        s4 = src
        n = s4.size
        np_ = (n + piece - 1) // piece
        while True:
            b = take(ctr, 0)
            if b >= np_:
                return
            for i in range(b * piece, min(n, b * piece + piece)):
                x = s4[i]
                dst[i] = ((x >> np.uint32(24)) | ((x >> np.uint32(8)) & np.uint32(0xFF00))
                          | ((x << np.uint32(8)) & np.uint32(0xFF0000)) | (x << np.uint32(24)))

    p = path('wrfout_d02_2024-11-06_01_30_00')
    size = os.path.getsize(p) // 4 * 4
    out = np.empty(size // 4, np.uint32)
    out[::1024] = 0

    def once(mm_arr, n):
        ctr = np.zeros(1, np.int64)
        t = time.perf_counter()
        _pool.share(lambda: copy_swap(mm_arr, out, ctr, 1 << 18), n)
        return (time.perf_counter() - t) * 1e3
    for n in (8, 16, 24):
        firsts, laters = [], []
        for _ in range(4):
            fh = open(p, 'rb')
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            a = np.frombuffer(mm, np.uint32, size // 4)
            firsts.append(once(a, n))
            laters.append(min(once(a, n) for _ in range(3)))
            del a
        print('%2d threads: first touch of the mapping %6.1f ms, later %6.1f ms (%d MB)' % (
            n, min(firsts), min(laters), size >> 20), flush=True)


def nc3():
    fastnc.warm()
    fast_thread()
    p = path('wrfout_d02_2024-11-06_01_30_00')
    d = fastnc.Dataset(p)
    d.set_auto_maskandscale(False)

    def best(fn, k=7):
        r = []
        for _ in range(k):
            t = time.perf_counter()
            fn()
            r.append(time.perf_counter() - t)
        return min(r) * 1e3
    for n in (4, 8, 12, 16, 20, 24):
        fastnc.set_threads(n)
        print('%2d threads: load %6.1f ms' % (n, best(lambda: d.load())), flush=True)
    fastnc.set_threads(None)


if __name__ == '__main__':
    fast_thread()
    for what in sys.argv[1:] or ['dupes', 'smt', 'adler', 'pin', 'faults', 'nc3']:
        print('---', what)
        globals()[what]()
