"""A test of the Python around the kernels that takes seconds, for working
on it: the small whole-file loads, timed part by part, in one process.

    python tools/profile/quick.py [wrf ku rcm avhrr metop] [--masked] [--rounds N]

Parts are timed with the clock around the functions named here (a profiler
adds more than the work), outermost call only. "kernels" is the time inside
the pool, "python" all the rest of the load. Best round of N, by total.
"""
import argparse
import functools
import sys
import threading
import time

from common import fast_thread, path

sys.path.insert(0, path('.'))
fast_thread()
import fastnc                                       # noqa: E402
from fastnc import _chunks, _dataset, _h5read, _hdf5, _nc4, _pool  # noqa: E402

FILES = {'wrf': 'wrfout_d01_2025-10-26_06_20_00.nc', 'ku': '20260904I1_ku_outer_lfm_VV_L2_pass13_combined.nc',
         'rcm': 'rcm2-sclna-owi-cm-20261014t010230-20261014t010349-00003-xxxxx_ll_gd.nc',
         'avhrr': 'NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc',
         'metop': 'W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20260922172210_G_O_20260922163559_20260922163659_C_N_T__.nc'}
T = {}
DEPTH = threading.local()
MAIN = threading.main_thread()


def timed(owner, name, part):
    """owner.name timed as `part`, where owner has it (a part may not exist
    in every version of the code)."""
    real = getattr(owner, name, None)
    if real is None:
        return

    @functools.wraps(real)
    def w(*a, **k):
        if getattr(DEPTH, part, 0) or threading.current_thread() is not MAIN:
            return real(*a, **k)
        setattr(DEPTH, part, 1)
        t = time.perf_counter()
        try:
            return real(*a, **k)
        finally:
            T[part] = T.get(part, 0.0) + time.perf_counter() - t
            setattr(DEPTH, part, 0)
    setattr(owner, name, w)


timed(_dataset.Dataset, '__init__', 'open')
timed(_nc4.NC4File, 'scan', 'scan')
timed(_nc4.NC4File, 'link', 'link')
timed(_hdf5.H5File, 'chunk_arrays', 'index')
timed(_chunks.ChunkIndex, '__init__', 'index')
timed(_chunks, '_plan', 'plan')
timed(_pool, 'share', 'kernels')
timed(_chunks, 'maybe_stage', 'stage')
timed(_h5read.Reader, 'read', 'reader')
timed(_dataset._Variables, 'fill', 'fill')
timed(_dataset.Group, 'set_auto_maskandscale', 'auto')
timed(_dataset.Group, 'load', 'load')
timed(_h5read, 'read_many', 'many')
timed(_chunks, 'read_batch', 'batch')
timed(_dataset.Dataset, 'close', 'close')
PARTS = ('auto', 'load', 'many', 'batch', 'close')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*')
    ap.add_argument('--masked', action='store_true')
    ap.add_argument('--rounds', type=int, default=30)
    a = ap.parse_args()
    fastnc.warm()
    print('%-6s %7s %7s %7s | %6s %6s %6s %6s %6s | %s' % (
        '', 'total', 'kernels', 'python', 'open', 'scan', 'link', 'index', 'plan',
        ' '.join('%6s' % s for s in PARTS)))
    for key in a.files or ['wrf', 'ku', 'rcm', 'avhrr', 'metop']:
        p = path(FILES[key])
        best = None
        for _ in range(a.rounds + 2):
            T.clear()
            t = time.perf_counter()
            with fastnc.Dataset(p) as d:
                d.set_auto_maskandscale(a.masked)
                d.load(recursive=True)
            t = time.perf_counter() - t
            if best is None or t < best[0]:
                best = (t, dict(T))
        t, parts = best
        ms = lambda k: parts.get(k, 0.0) * 1e3      # noqa: E731
        print('%-6s %7.2f %7.2f %7.2f | %6.2f %6.2f %6.2f %6.2f %6.2f | %s' % (
            key, t * 1e3, ms('kernels'), t * 1e3 - ms('kernels'), ms('open'), ms('scan'), ms('link'), ms('index'), ms('plan'),
            ' '.join('%6.2f' % ms(s) for s in PARTS)), flush=True)


if __name__ == '__main__':
    main()
