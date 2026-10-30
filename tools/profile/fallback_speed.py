"""Whole-file reads with numba never loaded (what a fresh process gets):
raw and masked, best of 3 in one process, with a cProfile of one."""
import sys, time, cProfile, pstats
from common import fast_thread, path
sys.path.insert(0, path('.'))
import fastnc
from fastnc import _chunks
_chunks.LOAD_AFTER = float('inf')
fast_thread()


def best(fn, n=3):
    r = []
    for _ in range(n):
        t = time.perf_counter(); fn(); r.append(time.perf_counter() - t)
    return min(r) * 1e3


for f in sys.argv[1:]:
    ds = fastnc.Dataset(path(f))
    def load(m):
        ds.set_auto_maskandscale(m)
        return ds.load(recursive=True)
    print('%-40s raw %7.1f ms  masked %7.1f ms' % (f[:40], best(lambda: load(False)), best(lambda: load(True))))
    if '-p' in sys.argv[0:1] or len(sys.argv) == 2:
        pr = cProfile.Profile(); pr.enable(); load(True); pr.disable()
        pstats.Stats(pr).sort_stats('tottime').print_stats(12)
    assert fastnc._jit._numba is None if hasattr(fastnc, '_jit') else True
