"""Where a whole-file read's time goes now: kernel work vs page faults on
fresh output memory vs faults on a freshly mapped file."""
import sys, time, mmap
import numpy as np
from common import fast_thread, path
sys.path.insert(0, path('.'))
import fastnc
from fastnc import _hdf5, _h5read, _chunks, _nc4

fast_thread()
_chunks.warm()
f = sys.argv[1] if len(sys.argv) > 1 else 'VJ102IMG_NRT.A2026265.2006.021.nc'
p = path(f)


def readers(mm):
    src = np.frombuffer(mm, np.uint8)
    nc = _nc4.NC4File(_hdf5.H5File(mm), src)
    out = []
    def walk(g):
        for v in g.variables.values():
            if v.ds.type.cls in (0, 1) and v.ds.shape:
                out.append(v._reader)
        for s in g.groups.values():
            walk(s)
    walk(nc.root)
    return out


def best(fn, n=5):
    r = []
    for _ in range(n):
        t = time.perf_counter(); fn(); r.append(time.perf_counter() - t)
    return min(r) * 1e3


fh = open(p, 'rb')
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
rds = readers(mm)
total = sum(int(np.prod(r.ds.shape)) * r.ds.type.size for r in rds)
print('%s: %d variables, %.0f MB of values' % (f[:30], len(rds), total / 1e6))
# 1. same mapping, fresh outputs each time (what a second read in a session costs)
t1 = best(lambda: _h5read.read_many(rds))
# 2. same mapping, outputs reused (decode only: no fresh pages)
outs = _h5read.read_many(rds)
buf = np.empty(sum((o.nbytes + 63) // 64 * 64 for o in outs), np.uint8); buf[:] = 0
def into():
    items = []; base = 0
    for r, o in zip(rds, outs):
        a = buf[base:base + o.nbytes].view(o.dtype).reshape(o.shape)
        items.append(_chunks.Item(r.ds, r.index, [(0, s, 1) for s in r.ds.shape], a, False, base))
        base += (o.nbytes + 63) // 64 * 64
    _chunks.read_batch(items, rds[0].src, buf)
t2 = best(into)
# 3. fresh mapping each time (a new open), fresh outputs
def fresh():
    m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    _h5read.read_many(readers(m))
t3 = best(fresh)
# 4. page faulting alone: allocate and touch the outputs in parallel
def touch():
    a = np.empty(total, np.uint8)
    n = 24; step = -(-total // n)
    fastnc._pool.map_items(lambda i: a[i * step:(i + 1) * step].fill(0), range(n))
t4 = best(touch)
fastnc.set_threads(1)
t5 = best(into, 2)
fastnc.set_threads(None)
print('decode only (outputs reused)      %7.1f ms' % t2)
print('decode, fresh outputs             %7.1f ms   (+%.1f)' % (t1, t1 - t2))
print('decode, fresh outputs, new mmap   %7.1f ms   (+%.1f)' % (t3, t3 - t1))
print('touching that much fresh memory   %7.1f ms   (24 threads)' % t4)
print('decode only, 1 thread             %7.1f ms   (%.1fx from threads)' % (t5, t5 / t2))
