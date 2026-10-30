"""xarray engine='fastnc' against engine='netcdf4': same Dataset (variables,
dims, coords, attrs, encodings that matter, decoded values); and timing."""
import os, sys, time, warnings
import numpy as np, xarray as xr
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from fastnc._xarray import FastncBackendEntrypoint
warnings.simplefilter('ignore')


def compare(a, b):
    bad = []
    if set(a.variables) != set(b.variables):
        bad.append('variables %s' % sorted(set(a.variables) ^ set(b.variables))[:5])
    if dict(a.sizes) != dict(b.sizes):
        bad.append('sizes')
    if set(a.coords) != set(b.coords):
        bad.append('coords %s vs %s' % (sorted(a.coords)[:5], sorted(b.coords)[:5]))
    for k in b.variables:
        if k not in a.variables:
            continue
        x, y = a[k], b[k]
        if x.dtype != y.dtype:
            bad.append('%s dtype %s != %s' % (k, x.dtype, y.dtype))
        if x.attrs.keys() != y.attrs.keys():
            bad.append('%s attrs %s != %s' % (k, list(x.attrs)[:4], list(y.attrs)[:4]))
        try:
            xv, yv = x.values, y.values
        except Exception as e:
            bad.append('%s load: %r' % (k, e)); continue
        if xv.dtype.kind in 'fc':
            same = xv.shape == yv.shape and np.array_equal(xv, yv, equal_nan=True) and xv.tobytes() == yv.tobytes()
        elif xv.dtype.kind == 'O':
            same = xv.shape == yv.shape and all(np.array_equal(p, q) if isinstance(p, np.ndarray) else p == q for p, q in zip(xv.ravel(), yv.ravel()))
        else:
            same = np.array_equal(xv, yv)
        if not same:
            bad.append('%s values differ' % k)
    if a.attrs.keys() != b.attrs.keys():
        bad.append('global attrs')
    return bad


for p in sys.argv[1:]:
    t = time.perf_counter(); a = xr.open_dataset(p, engine=FastncBackendEntrypoint); ta = time.perf_counter() - t
    t = time.perf_counter(); a.load(); la = time.perf_counter() - t
    t = time.perf_counter(); b = xr.open_dataset(p, engine='netcdf4'); tb = time.perf_counter() - t
    t = time.perf_counter(); b.load(); lb = time.perf_counter() - t
    t = time.perf_counter(); c = xr.open_dataset(p, engine=FastncBackendEntrypoint, preload=True); tc = time.perf_counter() - t
    bad = compare(a, b) + compare(c, b)
    print('%-40s %s  fastnc open %6.1f + load %6.1f ms | preload %6.1f ms | netcdf4 open %7.1f + load %6.1f ms' % (
        os.path.basename(p)[:40], 'OK' if not bad else 'BAD %d' % len(bad), ta * 1e3, la * 1e3, tc * 1e3, tb * 1e3, lb * 1e3))
    for x in bad[:6]:
        print('     ', x[:200])
