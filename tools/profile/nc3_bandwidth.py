from common import DATA, fast_thread, path, real_files
import os, time, mmap, ctypes, numpy as np, netCDF4
from concurrent.futures import ThreadPoolExecutor
fast_thread()
p = path('wrfout_d02_2024-11-06_01_30_00')
N = os.path.getsize(p)
f = netCDF4.Dataset(p); print(f.data_model, 'numrecs', len(f.dimensions['Time']), 'vars', len(f.variables),
  'record vars', sum(1 for v in f.variables.values() if v.dimensions and v.dimensions[0]=='Time'))
f.close()
def best(fn, n=5):
    r=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); r.append(time.perf_counter()-t)
    return min(r)
src = np.ones(N//4, np.float32); dst = np.empty_like(src)
pool = ThreadPoolExecutor(24, initializer=fast_thread)
def par(fn, nt):
    step = -(-len(src)//nt)
    list(pool.map(lambda i: fn(slice(i*step,(i+1)*step)), range(nt)))
t=best(lambda: np.copyto(dst,src)); print(f'memcpy 1T   {N/t/1e9:.1f} GB/s')
for nt in (4,8,16,24):
    t=best(lambda: par(lambda s: np.copyto(dst[s],src[s]), nt)); print(f'memcpy {nt}T  {N/t/1e9:.1f} GB/s')
sw = src.view('>f4')
t=best(lambda: np.copyto(dst,sw)); print(f'byteswap copy 1T {N/t/1e9:.1f} GB/s')
for nt in (8,16):
    t=best(lambda: par(lambda s: np.copyto(dst[s],sw[s]), nt)); print(f'byteswap copy {nt}T {N/t/1e9:.1f} GB/s')
# fresh allocation cost (page faults)
t=best(lambda: np.empty(N//4,np.float32).fill(0)); print(f'fresh alloc+touch 1T {N/t/1e9:.1f} GB/s')
t=best(lambda: (lambda a: par(lambda s: a[s].fill(0), 16))(np.empty(N//4,np.float32))); print(f'fresh alloc+touch 16T {N/t/1e9:.1f} GB/s')
# file cache -> user buffer
buf = bytearray(N)
def readall():
    with open(p,'rb',buffering=0) as fh: fh.readinto(buf)
t=best(readall); print(f'readinto (cached) 1T {N/t/1e9:.1f} GB/s')
def readpar(nt):
    step=-(-N//nt); mv=memoryview(buf)
    def one(i):
        with open(p,'rb',buffering=0) as fh:
            fh.seek(i*step); fh.readinto(mv[i*step:min(N,(i+1)*step)])
    list(pool.map(one, range(nt)))
for nt in (4,8,16):
    t=best(lambda: readpar(nt)); print(f'readinto (cached) {nt}T {N/t/1e9:.1f} GB/s')
fh=open(p,'rb'); mm=mmap.mmap(fh.fileno(),0,access=mmap.ACCESS_READ)
def mmcopy():
    m=mmap.mmap(fh.fileno(),0,access=mmap.ACCESS_READ); a=np.frombuffer(m,np.uint8); np.copyto(np.frombuffer(buf,np.uint8),a); del a; m.close()
t=best(mmcopy); print(f'fresh mmap+copy 1T {N/t/1e9:.1f} GB/s')
def mmpar(nt):
    m=mmap.mmap(fh.fileno(),0,access=mmap.ACCESS_READ); a=np.frombuffer(m,np.uint8); b=np.frombuffer(buf,np.uint8)
    step=-(-N//nt); list(pool.map(lambda i: np.copyto(b[i*step:(i+1)*step],a[i*step:(i+1)*step]), range(nt))); del a; m.close()
for nt in (8,16): t=best(lambda: mmpar(nt)); print(f'fresh mmap+copy {nt}T {N/t/1e9:.1f} GB/s')
def ncraw():
    f=netCDF4.Dataset(p); f.set_auto_maskandscale(False)
    for v in f.variables.values(): v[...]
    f.close()
t=best(ncraw,3); print(f'netCDF4-python raw all vars {t:.3f}s {N/t/1e9:.2f} GB/s')
