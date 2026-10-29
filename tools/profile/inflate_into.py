from common import DATA, fast_thread, path, real_files
import sys, os, time, ctypes, h5py, numpy as np, threading
from ctypes import wintypes, c_ulong, c_void_p, byref, c_int
from concurrent.futures import ThreadPoolExecutor
fast_thread()
z = ctypes.CDLL(os.environ.get('FASTNC_ZLIB', os.path.join(sys.prefix, 'Library', 'bin', 'zlib.dll')))
unc = z.uncompress; unc.argtypes=[c_void_p, ctypes.POINTER(c_ulong), c_void_p, c_ulong]; unc.restype=c_int
p = path(sys.argv[1])
h = h5py.File(p,'r'); raw = np.fromfile(p, np.uint8)
ds=[]; h.visititems(lambda n,o: ds.append(o) if isinstance(o,h5py.Dataset) and o.chunks and o.size else None)
jobs=[]; total=0
for d in ds:
    dc=d.id.get_create_plist(); F=[dc.get_filter(i)[0] for i in range(dc.get_nfilters())]
    if 1 not in F: continue
    csz = int(np.prod(d.chunks))*d.dtype.itemsize
    L=[]; d.id.chunk_iter(L.append)
    for ci in L: jobs.append((ci.byte_offset, ci.size, csz)); total += csz
print(len(jobs),'chunks', total/1e6,'MB out')
out = np.empty(total, np.uint8)   # stands for the output arrays (chunk-shaped here)
outs=[]; o=0
for off,sz,csz in jobs: outs.append(o); o+=csz
base_in = raw.ctypes.data; base_out = out.ctypes.data
def run_range(i0, i1):
    n=c_ulong()
    for i in range(i0,i1):
        off,sz,csz=jobs[i]; n.value=csz
        r=unc(base_out+outs[i], byref(n), base_in+off, sz)
        assert r==0, r
counter = [0]; lock=threading.Lock()
def worker(_=None):
    fast_thread(); n=c_ulong()
    while True:
        with lock:
            i=counter[0]; counter[0]+=1
        if i>=len(jobs): return
        off,sz,csz=jobs[i]; n.value=csz
        unc(base_out+outs[i], byref(n), base_in+off, sz)
pool=ThreadPoolExecutor(24)
order = sorted(range(len(jobs)), key=lambda i:-jobs[i][2])
jobs=[jobs[i] for i in order]; outs=[outs[i] for i in order]
out.fill(0)
for nt in (1,4,8,16,24):
    best=9
    for _ in range(4):
        counter[0]=0; t=time.perf_counter()
        fs=[pool.submit(worker) for _ in range(nt-1)]; worker(); [f.result() for f in fs]
        best=min(best,time.perf_counter()-t)
    print(f'{nt:2d}T inflate-into-place {best*1e3:7.1f} ms  {total/best/1e6:6.0f} MB/s out')
