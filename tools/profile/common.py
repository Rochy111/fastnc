"""Shared by the profiling scripts: where the real files are, and keeping
timing threads off the efficiency cores of a hybrid CPU on Windows."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('FASTNC_DATA', os.path.dirname(os.path.dirname(HERE)))


def real_files():
    """The real netCDF/HDF5 files in DATA (byte-identical "(1)" copies left out)."""
    return sorted(f for f in os.listdir(DATA)
                  if (f.endswith('.nc') or f.startswith('wrfout_')) and '(1)' not in f)


def path(name):
    return os.path.join(DATA, name)


def fast_thread():
    """Ask Windows to schedule this thread on performance cores (a script run
    from a terminal is background work there; timings swing 30-50% otherwise)."""
    if sys.platform != 'win32':
        return
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG),
                    ('StateMask', wintypes.ULONG)]
    k = ctypes.windll.kernel32
    k.GetCurrentThread.restype = wintypes.HANDLE
    st = State(1, 1, 0)
    k.SetThreadInformation(wintypes.HANDLE(k.GetCurrentThread()), 3,
                           ctypes.byref(st), ctypes.sizeof(st))
