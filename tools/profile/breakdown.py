# Where does the time go: chunk index walk, raw I/O, inflate, unshuffle, placement.
from common import DATA, fast_thread, path, real_files
import sys, os, time, zlib, ctypes, h5py, numpy as np
from concurrent.futures import ThreadPoolExecutor
fast_thread()
print('zlib', zlib.ZLIB_RUNTIME_VERSION)
files = sys.argv[1:] or real_files()
pool = ThreadPoolExecutor(16, initializer=fast_thread)
for f in files:
    p=path(f); h=h5py.File(p,'r')
    ds=[]; h.visititems(lambda n,o: ds.append(o) if isinstance(o,h5py.Dataset) and o.chunks and o.size else None)
    t=time.perf_counter()
    infos=[]
    for d in ds:
        n=d.id.get_num_chunks()
        infos.append([d.id.get_chunk_info(i) for i in range(n)])
    t_index=time.perf_counter()-t
    # alternative: chunk_iter (h5py>=3.8, HDF5>=1.12.3) single call
    t=time.perf_counter()
    for d in ds:
        L=[]; d.id.chunk_iter(L.append)
    t_iter=time.perf_counter()-t
    raw=open(p,'rb').read()
    t=time.perf_counter(); blobs=[]
    for d,inf in zip(ds,infos):
        for ci in inf: blobs.append((d,ci.byte_offset,ci.size,ci.filter_mask))
    # read via h5py read_direct_chunk
    t=time.perf_counter()
    for d,inf in zip(ds,infos):
        for ci in inf: d.id.read_direct_chunk(ci.chunk_offset)
    t_rdc=time.perf_counter()-t
    t=time.perf_counter()
    comp=[]
    for d,off,sz,fm in blobs: comp.append((d, raw[off:off+sz]))
    t_slice=time.perf_counter()-t
    # inflate
    def filt(d): 
        dc=d.id.get_create_plist(); return [dc.get_filter(i)[0] for i in range(dc.get_nfilters())]
    F={id(d):filt(d) for d in ds}
    work=[(d,b) for d,b in comp if 1 in F[id(d)]]
    def inflate(item): return zlib.decompress(item[1])
    t=time.perf_counter(); outs=[inflate(w) for w in work]; t_inf=time.perf_counter()-t
    t=time.perf_counter(); list(pool.map(inflate, work)); t_inf_mt=time.perf_counter()-t
    # unshuffle
    t=time.perf_counter()
    for (d,_),o in zip(work,outs):
        if 2 in F[id(d)] and d.dtype.itemsize>1:
            n=d.dtype.itemsize; a=np.frombuffer(o,np.uint8)
            m=len(a)//n; u=np.empty(m*n,np.uint8).reshape(m,n); u[:]=a[:m*n].reshape(n,m).T
    t_shuf=time.perf_counter()-t
    biggest=max(len(o) for o in outs) if outs else 0
    inb=sum(len(w[1]) for w in work); outb=sum(len(o) for o in outs)
    t=time.perf_counter()
    for d in ds: d[()]
    t_h5=time.perf_counter()-t
    print(f'\n{f[:55]}  chunked vars={len(ds)} chunks={len(blobs)} deflated={len(work)} biggest={biggest/1e6:.1f}MB')
    print(f'  h5py read chunked vars: {t_h5*1e3:7.1f} ms')
    print(f'  chunk index get_chunk_info: {t_index*1e3:7.1f} ms  chunk_iter: {t_iter*1e3:7.1f} ms')
    print(f'  read_direct_chunk all:  {t_rdc*1e3:7.1f} ms   slice from bytes: {t_slice*1e3:.1f} ms')
    if work:
        print(f'  zlib inflate 1 thread:  {t_inf*1e3:7.1f} ms ({inb/1e6:.1f}->{outb/1e6:.1f} MB, {outb/1e6/t_inf:.0f} MB/s out)   16 threads: {t_inf_mt*1e3:.1f} ms')
        print(f'  numpy unshuffle:        {t_shuf*1e3:7.1f} ms')
