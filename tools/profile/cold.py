"""Reads of files not in the OS cache, and how many requests reach the disk
(Windows; no admin rights needed).

    python tools/profile/cold.py [file-substring ...]

Evicting a file takes care. Opening it with FILE_FLAG_NO_BUFFERING looks like
it works, and does for a file only ever read, but pages of a file that has been
mapped survive it: a table made that way had VIIRS "cold" in 45 ms with 0 bytes
leaving the disk. So each run reads a copy written with FILE_FLAG_NO_BUFFERING,
which never enters the cache, rewritten before every run, and the disk's own
counters (IOCTL_DISK_PERFORMANCE) say what was read. Copies go to the scratch
directory given by FASTNC_COLD_DIR, else the system's temporary directory, and
are removed. Trust a row only if its MB is the file's size.

Each file: warm, cold, cold with compressed chunks staged (_pio), cold h5py.
Median of 5 runs.
"""
import ctypes
import ctypes.wintypes as w
import mmap
import os
import sys
import tempfile
import time

from common import fast_thread, path

sys.path.insert(0, path('.'))
fast_thread()
import fastnc                                       # noqa: E402
from fastnc import _pio                             # noqa: E402

k = ctypes.WinDLL('kernel32', use_last_error=True)
k.CreateFileW.restype = ctypes.c_void_p
k.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p, w.DWORD, w.DWORD, ctypes.c_void_p]
k.CloseHandle.argtypes = [ctypes.c_void_p]
k.WriteFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD), ctypes.c_void_p]
k.SetEndOfFile.argtypes = [ctypes.c_void_p]
k.SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, w.DWORD]
k.DeviceIoControl.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong,
                              ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p]
GENERIC_WRITE, NO_BUFFERING, WRITE_THROUGH = 0x40000000, 0x20000000, 0x80000000
INVALID = ctypes.c_void_p(-1).value
SECTOR = 4096
STEP = 8 << 20
FILES = [('VIIRS', 'VJ102IMG_NRT.A2026265.2006.021.nc'),
         ('MODIS', 'MOD021KM.A2025301.1445.007.2025301204859.nc'),
         ('WRF d01', 'wrfout_d01_2025-10-26_06_20_00.nc')]


class Performance(ctypes.Structure):
    _fields_ = [('BytesRead', ctypes.c_longlong), ('BytesWritten', ctypes.c_longlong),
                ('ReadTime', ctypes.c_longlong), ('WriteTime', ctypes.c_longlong),
                ('IdleTime', ctypes.c_longlong), ('ReadCount', ctypes.c_ulong),
                ('WriteCount', ctypes.c_ulong), ('QueueDepth', ctypes.c_ulong),
                ('SplitCount', ctypes.c_ulong), ('QueryTime', ctypes.c_longlong),
                ('StorageDeviceNumber', ctypes.c_ulong), ('StorageManagerName', ctypes.c_wchar * 8)]


def disk():
    """(bytes read, requests) so far, every disk together."""
    got = [0, 0]
    for i in range(8):
        h = k.CreateFileW('\\\\.\\PhysicalDrive%d' % i, 0, 7, None, 3, 0, None)
        if h == INVALID:
            continue
        p = Performance()
        n = ctypes.c_ulong()
        if k.DeviceIoControl(h, 0x70020, None, 0, ctypes.byref(p), ctypes.sizeof(p), ctypes.byref(n), None):
            got[0] += p.BytesRead
            got[1] += p.ReadCount
        k.CloseHandle(h)
    return got


class Cold:
    """A copy of a file that can be made cold again."""

    def __init__(self, src):
        self.size = os.path.getsize(src)
        where = os.environ.get('FASTNC_COLD_DIR') or tempfile.gettempdir()
        self.path = os.path.join(where, 'fastnc_cold_%d.nc' % os.getpid())
        self.buf = mmap.mmap(-1, (self.size + SECTOR - 1) // SECTOR * SECTOR)   # (aligned)
        with open(src, 'rb', buffering=0) as f:
            v = memoryview(self.buf)
            at = 0
            while at < self.size:
                n = f.readinto(v[at:at + STEP])
                if not n:
                    break
                at += n
            v.release()
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))

    def evict(self):
        h = k.CreateFileW(self.path, GENERIC_WRITE, 7, None, 4, NO_BUFFERING | WRITE_THROUGH, None)
        if h == INVALID:
            raise ctypes.WinError(ctypes.get_last_error())
        done, total, wrote = 0, len(self.buf), w.DWORD()
        while done < total:
            if not k.WriteFile(h, self.addr + done, min(STEP, total - done), ctypes.byref(wrote), None):
                raise ctypes.WinError(ctypes.get_last_error())
            done += wrote.value
        k.CloseHandle(h)
        h = k.CreateFileW(self.path, GENERIC_WRITE, 7, None, 3, 0, None)        # its true length
        k.SetFilePointerEx(h, self.size, None, 0)
        k.SetEndOfFile(h)
        k.CloseHandle(h)

    def remove(self):
        try:
            os.remove(self.path)
        except OSError:
            pass


def load(p):
    with fastnc.Dataset(p) as d:
        d.set_auto_maskandscale(False)
        d.load(recursive=True)


def h5(p):
    import h5py
    with h5py.File(p, 'r') as f:
        def visit(n, o):
            if isinstance(o, h5py.Dataset) and o.shape:
                o[()]
        f.visititems(visit)


def run(c, fn, label, cold, n=5):
    got = []
    for _ in range(n):
        if cold:
            c.evict()
            time.sleep(0.2)
        a = disk()
        t = time.perf_counter()
        fn(c.path)
        t = (time.perf_counter() - t) * 1e3
        b = disk()
        got.append((t, (b[0] - a[0]) / 1e6, b[1] - a[1]))
    got.sort()
    t, mb, req = got[len(got) // 2]
    print('  %-14s %8.1f ms  %6.0f MB from disk  %6d requests' % (label, t, mb, req), flush=True)


def main():
    fastnc.warm()
    want = sys.argv[1:]
    for label, f in FILES:
        if want and not any(s in label for s in want):
            continue
        c = Cold(path(f))
        try:
            print('%s (%.0f MB)' % (label, c.size / 1e6))
            c.evict()
            load(c.path)
            run(c, load, 'warm', False)
            run(c, load, 'cold', True)
            real = _pio.slow
            _pio.slow = lambda src, addr, threads: True
            run(c, load, 'cold, staged', True)
            _pio.slow = real
            run(c, h5, 'cold h5py', True)
        finally:
            c.remove()


if __name__ == '__main__':
    main()
