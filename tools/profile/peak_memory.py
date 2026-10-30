"""Peak working set of a fresh process reading every variable of a file
(masked, as netCDF4-python reads by default; raw for h5py), each read kept
until the end -- what reading a file into memory costs."""
import ctypes, os, subprocess, sys
from ctypes import wintypes
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CODE = r'''
import ctypes, sys
from ctypes import wintypes
class PMC(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t), ('a', ctypes.c_size_t), ('b', ctypes.c_size_t), ('c', ctypes.c_size_t),
                ('d', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
k32 = ctypes.windll.kernel32
k32.GetCurrentProcess.restype = wintypes.HANDLE
k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
def peak():
    m = PMC(); m.cb = ctypes.sizeof(PMC)
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(m), m.cb)
    return m.PeakWorkingSetSize, m.PeakPagefileUsage
lib, p = sys.argv[1], sys.argv[2]
import numpy as np
base = peak()
keep = []
if lib == 'fastnc':
    import fastnc
    d = fastnc.Dataset(p); keep.append(d.load(recursive=True))
elif lib == 'netCDF4':
    import netCDF4
    d = netCDF4.Dataset(p)
    def walk(g):
        for v in g.variables.values():
            try: keep.append(v[...])
            except Exception: pass
        for s in g.groups.values(): walk(s)
    walk(d)
else:
    import h5py
    f = h5py.File(p, 'r')
    f.visititems(lambda n, o: keep.append(o[()]) if isinstance(o, h5py.Dataset) and o.shape else None)
pk = peak()
total = 0
def size(x):
    global total
    if isinstance(x, dict):
        for y in x.values(): size(y)
    elif isinstance(x, np.ndarray):
        total += x.nbytes + (np.ma.getmaskarray(x).nbytes if np.ma.isMaskedArray(x) and x.mask is not np.ma.nomask else 0)
for x in keep: size(x)
print('%.0f %.0f %.0f' % (pk[0] / 1e6, pk[1] / 1e6, total / 1e6))
'''
for f in sys.argv[1:]:
    for lib in ('fastnc', 'netCDF4', 'h5py'):
        out = subprocess.run([sys.executable, '-c', CODE, lib, os.path.abspath(f)], capture_output=True,
                             text=True, cwd=ROOT)
        try:
            ws, commit, held = out.stdout.split()
            print('%-28s %-8s peak working set %6s MB, peak commit %6s MB, results %6s MB' % (f[:28], lib, ws, commit, held))
        except ValueError:
            print(f[:28], lib, 'failed', out.stderr[-200:])
