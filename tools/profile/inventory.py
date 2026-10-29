from common import DATA, fast_thread, path, real_files
import sys, os, h5py, numpy as np, collections, json
files = real_files()
out = {}
for f in sorted(files):
    p = path(f)
    try:
        h = h5py.File(p, 'r')
    except Exception as e:
        print(f, 'not hdf5', e); continue
    sb = h.id.get_create_plist().get_version()
    vars_ = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            d = obj
            dc = d.id.get_create_plist()
            layout = {0:'compact',1:'contig',2:'chunked',3:'virtual'}[dc.get_layout()]
            filt = [dc.get_filter(i)[0] for i in range(dc.get_nfilters())]
            nch = 0; stor = d.id.get_storage_size()
            idx = None
            if layout == 'chunked':
                try: nch = d.id.get_num_chunks()
                except Exception: pass
            vars_.append(dict(name=name, dtype=str(d.dtype), shape=d.shape, chunks=d.chunks,
                              layout=layout, filters=filt, nchunks=nch, stored=stor,
                              raw=d.size*d.dtype.itemsize, nattrs=len(d.attrs)))
    h.visititems(visit)
    ngroups = [0]
    h.visit(lambda n: ngroups.__setitem__(0, ngroups[0] + isinstance(h[n], h5py.Group)))
    tot_raw = sum(v['raw'] for v in vars_); tot_st = sum(v['stored'] for v in vars_)
    lay = collections.Counter(v['layout'] for v in vars_)
    fl = collections.Counter(tuple(v['filters']) for v in vars_)
    nch = [v['nchunks'] for v in vars_ if v['layout']=='chunked']
    chs = [ (v['raw']/max(v['nchunks'],1)) for v in vars_ if v['layout']=='chunked' and v['nchunks']]
    print(f"\n== {f}  superblock v{sb[0]}  groups={ngroups[0]} vars={len(vars_)} raw={tot_raw/1e6:.1f}MB stored={tot_st/1e6:.1f}MB")
    print("  layouts", dict(lay), " filters", dict(fl))
    if nch: print(f"  chunks total={sum(nch)} median chunk raw={np.median(chs)/1e3:.1f}kB  min={min(chs)/1e3:.1f}kB max={max(chs)/1e3:.1f}kB")
    big = sorted(vars_, key=lambda v: -v['raw'])[:6]
    for v in big:
        print(f"   {v['name'][:50]:50s} {v['dtype']:8s} {str(v['shape']):22s} ch={str(v['chunks']):20s} n={v['nchunks']:5d} {v['filters']} {v['raw']/1e6:.1f}MB->{v['stored']/1e6:.1f}MB")
    out[f] = vars_
json.dump(out, open('inventory.json','w'), default=str)
