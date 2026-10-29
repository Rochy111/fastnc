"""fastnc's inflate against zlib on every deflated chunk of the real files:
byte-for-byte, and single-thread speed."""
import os, sys, time, zlib
import h5py, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from common import fast_thread, path, real_files
from fastnc import _inflate

fast_thread()
tab, iws = _inflate.workspace()
for f in (sys.argv[1:] or real_files()):
    p = path(f)
    try:
        h = h5py.File(p, 'r')
    except OSError:
        continue
    raw = np.fromfile(p, np.uint8)
    jobs = []
    def visit(n, d):
        if isinstance(d, h5py.Dataset) and d.chunks and d.size:
            dc = d.id.get_create_plist()
            if 1 in [dc.get_filter(i)[0] for i in range(dc.get_nfilters())]:
                L = []; d.id.chunk_iter(L.append)
                csz = int(np.prod(d.chunks)) * d.dtype.itemsize
                jobs.extend((c.byte_offset, c.size, csz) for c in L)
    h.visititems(visit)
    if not jobs:
        continue
    total = sum(j[2] for j in jobs)
    out = np.empty(total, np.uint8)
    offs = np.cumsum([0] + [j[2] for j in jobs])
    _inflate.zlib_inflate(raw, jobs[0][0], jobs[0][0] + jobs[0][1], out, 0, jobs[0][2], jobs[0][2], tab, iws)
    best = 9e9
    for rep in range(3):
        t = time.perf_counter()
        for (off, sz, csz), o in zip(jobs, offs):
            r = _inflate.zlib_inflate(raw, off, off + sz, out, o, o + csz, o + csz, tab, iws)
            if r != o + csz:
                print('  FAIL', f, off, r - o if r >= 0 else r); break
        best = min(best, time.perf_counter() - t)
    t = time.perf_counter()
    ref = b''.join(zlib.decompress(raw[off:off + sz].tobytes()) for off, sz, _ in jobs)
    tz = time.perf_counter() - t
    ok = ref == out.tobytes()
    print(f'{f[:40]:40s} {len(jobs):6d} chunks {total/1e6:7.1f} MB  identical={ok}  '
          f'fastnc {best*1e3:7.1f} ms ({total/best/1e6:5.0f} MB/s)   zlib.decompress {tz*1e3:7.1f} ms ({total/tz/1e6:5.0f} MB/s)')
