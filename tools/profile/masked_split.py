"""A whole-file masked read split in two: the raw read, then mask/scale on
the raw output (what fusing the two could save)."""
import sys, time
import numpy as np
from common import fast_thread, path, real_files
sys.path.insert(0, path('.'))
import fastnc
from fastnc import _maskscale

fast_thread()
fastnc.warm()


def best(fn, n=5):
    r = []
    for _ in range(n):
        t = time.perf_counter(); fn(); r.append(time.perf_counter() - t)
    return min(r) * 1e3


for f in sys.argv[1:] or real_files():
    ds = fastnc.Dataset(path(f))
    vs = [v for v in ds._walk_vars()] if hasattr(ds, '_walk_vars') else None
    if vs is None:
        vs = []
        for g in ds._walk():
            vs += list(g.variables.values())
    def raw():
        ds.set_auto_maskandscale(False)
        return ds.load(recursive=True)
    def masked():
        ds.set_auto_maskandscale(True)
        return ds.load(recursive=True)
    tr = best(raw); tm = best(masked)
    mb = sum(a.nbytes for a in raw().values() if hasattr(a, 'nbytes')) / 1e6
    print('%-40s raw %7.1f ms  masked %7.1f ms  (+%5.1f)  %6.0f MB raw' % (f[:40], tr, tm, tm - tr, mb))
    ds.close()
