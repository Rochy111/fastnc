from common import DATA, fast_thread, path, real_files
import sys, os, time, ctypes, h5py, netCDF4, numpy as np
fast_thread()
files = sys.argv[1:] or real_files()
def h5_all(p):
    t=time.perf_counter(); h=h5py.File(p,'r'); topen=time.perf_counter()-t
    ds=[]; h.visititems(lambda n,o: ds.append(o) if isinstance(o,h5py.Dataset) else None)
    tmeta=time.perf_counter()-t-topen
    n=0
    for d in ds:
        if d.shape==() or d.size==0: continue
        a=d[()]; n+=a.nbytes
    tot=time.perf_counter()-t; h.close(); return topen,tmeta,tot,n
def nc_all(p, ms):
    t=time.perf_counter(); f=netCDF4.Dataset(p); topen=time.perf_counter()-t
    vs=[]
    def walk(g):
        vs.extend(g.variables.values())
        for s in g.groups.values(): walk(s)
    walk(f)
    for v in vs: v.set_auto_maskandscale(ms)
    tmeta=time.perf_counter()-t-topen
    n=0
    for v in vs:
        try: a=v[...]; n+=np.asarray(a).nbytes
        except Exception as e: pass
    tot=time.perf_counter()-t; f.close(); return topen,tmeta,tot,n
for f in files:
    p=path(f)
    print(f'\n{f[:60]}')
    for name,fn in [('h5py',lambda: h5_all(p)),('nc4 raw',lambda: nc_all(p,False)),('nc4 m+s',lambda: nc_all(p,True))]:
        try:
            rs=[fn() for _ in range(4)]
            best=min(rs[1:],key=lambda r:r[2])
            print(f'  {name:8s} first {rs[0][2]:.3f}s  best {best[2]:.3f}s (open {best[0]*1e3:.1f}ms, walk {best[1]*1e3:.1f}ms)  {best[3]/1e6:.0f}MB  {best[3]/1e6/best[2]:.0f}MB/s')
        except Exception as e: print('  ',name,'ERR',repr(e)[:100])
