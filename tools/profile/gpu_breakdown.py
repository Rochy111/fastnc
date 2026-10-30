"""Where a GPU read's time goes: the CPU read alone, the read into pinned
memory band by band with the uploads, and the mask/scale on the card."""
import sys, time
from common import fast_thread, path
sys.path.insert(0, path('.'))
import numpy as np, cupy as cp, fastnc
from fastnc import _gpu
fastnc.warm(); fast_thread()
f = sys.argv[1] if len(sys.argv) > 1 else 'extra/OR_ABI-L1b-RadF-M6C02_G19_s20263021200207_e20263021209516_c20263021209551.nc'
v = sys.argv[2] if len(sys.argv) > 2 else 'Rad'
d = fastnc.Dataset(path(f)); x = d[v]


def best(fn, k=5):
    r = []
    for _ in range(k):
        cp.cuda.Device().synchronize()
        t = time.perf_counter(); fn(); cp.cuda.Device().synchronize(); r.append(time.perf_counter() - t)
    return min(r) * 1e3


x.set_auto_maskandscale(False)
raw = x[...]
print('cpu raw read      %7.1f ms (%d MB)' % (best(lambda: x[...]), raw.nbytes >> 20))
dev = cp.empty(raw.shape, raw.dtype)
print('upload pageable   %7.1f ms' % best(lambda: dev.set(raw)))
pin = cp.cuda.alloc_pinned_memory(raw.nbytes); h = np.frombuffer(pin, raw.dtype, raw.size).reshape(raw.shape)
h[...] = raw
print('upload pinned     %7.1f ms' % best(lambda: dev.set(h)))
print('gpu raw read      %7.1f ms   bands: %d' % (best(lambda: x.read(Ellipsis, device='gpu')),
                                                  len(_gpu._bands(x._bv, [(0, s, 1) for s in x.shape]))))
x.set_auto_maskandscale(True)
print('cpu masked read   %7.1f ms' % best(lambda: x[...]))
print('gpu masked read   %7.1f ms' % best(lambda: x.read(Ellipsis, device='gpu')))
