"""Timing of fastnc's netCDF3 reader on wrfout d02 against netCDF4-python
(raw), runs alternated A/B, with min and median of each. The file is in the
OS cache after the first pass ("warm"); --first times a fresh process's
first open + read instead (numba's import and cache load included).

    python tools/check/nc3_bench.py [--reps N] [--first]
"""
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from fastnc import _nc3, _pool                                  # noqa: E402

DATA = os.path.dirname(os.path.dirname(os.path.dirname(ROOT))) if '.claude' in ROOT else ROOT
D02 = os.path.join(os.environ.get('FASTNC_DATA', DATA), 'wrfout_d02_2024-11-06_01_30_00')


def timeit(fn, reps):
    r = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        r.append(time.perf_counter() - t)
    return r


def fmt(ts):
    return '%8.3f (%8.3f) ms' % (min(ts) * 1e3, np.median(ts) * 1e3)


def main():
    import netCDF4
    _pool.fast_thread()
    reps = int(sys.argv[sys.argv.index('--reps') + 1]) if '--reps' in sys.argv else 7
    if '--first' in sys.argv:
        code = ('import sys,time;sys.path.insert(0,%r);t=time.perf_counter();'
                'from fastnc import _nc3;f=_nc3.NC3File(%r);'
                't1=time.perf_counter();[v.read(()) for v in f.root.variables.values()];'
                't2=time.perf_counter();[v.read(()) for v in f.root.variables.values()];'
                't3=time.perf_counter();'
                'print("import+open %%.1f ms, first whole read %%.1f ms, second %%.1f ms"'
                '%%((t1-t)*1e3,(t2-t1)*1e3,(t3-t2)*1e3))' % (ROOT, D02))
        for _ in range(3):
            subprocess.run([sys.executable, '-c', code])
        return
    nc = netCDF4.Dataset(D02)
    nc.set_auto_maskandscale(False)
    f = _nc3.NC3File(D02)
    V = f.root.variables
    _nc3.KERNELS = True
    for v in V.values():                       # warm the cache and the kernels
        v.read(())
    for v in nc.variables.values():
        v[...]

    def open_fast():
        _nc3.NC3File(D02).close()

    def open_nc():
        netCDF4.Dataset(D02).close()

    def open_fast_attrs():
        g = _nc3.NC3File(D02)
        g.root.attrs
        for v in g.root.variables.values():
            v.attrs
        g.close()

    rows = [('open (header)', open_fast, open_nc),
            ('open + every attribute', open_fast_attrs, None)]
    rows.append(('all variables', lambda: [v.read(()) for v in V.values()],
                 lambda: [v[...] for v in nc.variables.values()]))
    for name, key in (('T', ()), ('U', ()), ('P_TOP', ()), ('Times', ()), ('HGT', ()),
                      ('T', (0, 10)), ('T', (0, slice(None), 100)),
                      ('T', (0, slice(None), slice(None), 200)),
                      ('T', (0, slice(None), slice(None, None, 2), slice(None, None, 2))),
                      ('T', (0, 5, slice(100, 200), slice(50, 150)))):
        nk = tuple(k if isinstance(k, int) else slice(*k.indices(n))
                   for k, n in zip(key, V[name].shape)) if key else ()
        label = '%s[%s]' % (name, ','.join(str(k) if isinstance(k, int) else
                                           '%s:%s:%s' % (k.start, k.stop, k.step) for k in nk))
        rows.append((label, lambda n=name, k=nk: V[n].read(k),
                     lambda n=name, k=key: nc.variables[n][k] if k else nc.variables[n][...]))
    total = sum(v.dtype.itemsize * int(np.prod(v.shape)) for v in V.values())
    print('d02: %d variables, %.0f MB, %d threads; min (median) of %d alternated runs' % (
        len(V), total / 1e6, _pool.threads(), reps))
    cols = (('fastnc', dict(KERNELS=True)), ('fastnc 1 thread', dict(KERNELS=True, threads=1)),
            ('fastnc no numba', dict(KERNELS=False)))
    print('%-30s' % '' + ''.join('%-26s' % c for c, _ in cols) + '%-26s%s' % ('netCDF4-python', 'nc4 / fastnc'))
    for label, fa, fb in rows:
        ts = [[] for _ in cols]
        tb = []
        for _ in range(reps):                  # alternated
            for (c, st), t in zip(cols, ts):
                _nc3.KERNELS = st['KERNELS']
                _pool.set_threads(st.get('threads'))
                t += timeit(fa, 1)
            _pool.set_threads(None)
            if fb is not None:
                tb += timeit(fb, 1)
        ratio = '%6.1fx (%.1fx)' % (min(tb) / min(ts[0]), np.median(tb) / np.median(ts[0]))             if tb else ''
        print('%-30s' % label + ''.join('%-26s' % fmt(t) for t in ts) +
              '%-26s%s' % (fmt(tb) if tb else '', ratio))
    f.close()
    nc.close()


if __name__ == '__main__':
    main()
