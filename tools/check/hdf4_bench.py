"""Time reading every variable of HDF4 files (raw values) with fastnc
(variable by variable, and all at once with `_hdf4.read_many`), pyhdf and
netCDF4-python, alternating rounds; prints the median of each and
fastnc's speed-up with its spread over rounds.

    python tools/check/hdf4_bench.py [--rounds N] file.hdf ...
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'profile'))
import fastnc  # noqa: E402
from fastnc import _chunks, _pool  # noqa: E402

_pool.fast_thread()


def read_fastnc(path):
    with fastnc.Dataset(path) as d:
        d.set_auto_maskandscale(False)
        return sum(np.asarray(v[...]).nbytes for v in d.variables.values())


def read_fastnc_many(path):
    from fastnc import _hdf4
    with fastnc.Dataset(path) as d:
        return sum(a.nbytes for a in _hdf4.read_many([v._bv for v in d.variables.values()]))


def read_pyhdf(path):
    from pyhdf.SD import SD
    h = SD(path)
    n = 0
    for name, info in h.datasets().items():
        try:
            n += h.select(info[3]).get().nbytes
        except Exception:
            pass
    h.end()
    return n


def read_netcdf4(path):
    import netCDF4
    with netCDF4.Dataset(path) as d:
        d.set_auto_maskandscale(False)
        n = 0
        for v in d.variables.values():
            try:
                n += np.asarray(v[...]).nbytes
            except Exception:
                pass
        return n


def main():
    args = sys.argv[1:]
    rounds = 7
    if '--rounds' in args:
        i = args.index('--rounds')
        rounds = int(args[i + 1])
        del args[i:i + 2]
    _chunks.warm()
    readers = [('fastnc', read_fastnc), ('fastnc read_many', read_fastnc_many), ('pyhdf', read_pyhdf),
               ('netCDF4', read_netcdf4)]
    for path in args:
        times = {k: [] for k, _ in readers}
        nbytes = read_fastnc(path)
        for f in readers:                           # warm the page cache and each library
            f[1](path)
        for r in range(rounds):
            order = readers if r % 2 == 0 else readers[::-1]
            for name, f in order:
                t = time.perf_counter()
                f(path)
                times[name].append(time.perf_counter() - t)
        med = {k: float(np.median(v)) for k, v in times.items()}
        line = '%s (%.0f MB): ' % (os.path.basename(path), nbytes / 1e6)
        line += ', '.join('%s %.1f ms' % (k, 1e3 * med[k]) for k, _ in readers)
        for mine in ('fastnc', 'fastnc read_many'):
            f0 = np.array(times[mine])
            for k in ('pyhdf', 'netCDF4'):
                ratios = np.array(times[k]) / f0
                line += '\n    %s x%.1f vs %s (%.1f-%.1f)' % (mine, np.median(ratios), k, ratios.min(), ratios.max())
        print(line)


if __name__ == '__main__':
    main()
