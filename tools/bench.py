"""The README's benchmark: fastnc against netCDF4-python, h5py and xarray on
the real files, every reader timed on the same scenario in turn.

    python tools/bench.py [--quick] [--gpu] [--out docs/benchmarks.json]

Method. Files are in the OS cache (each read once before timing; a cold
cache needs admin rights to flush on Windows and is not measured). Warm
timings are the best of N runs, readers alternated run by run so they see
the same machine; "first" is a fresh Python process doing import + open +
read, wall time from just before the import. fastnc runs on every logical
CPU unless a row says 1 thread. Raw reads have mask and scale off;
"masked" rows are netCDF4-python's defaults (fastnc gives the same values).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'profile'))
from common import fast_thread  # noqa: E402

D = ROOT
FILES = [
    # (label, file, a typical variable, a window of it)
    ('WRF d01 netCDF4', 'wrfout_d01_2025-10-26_06_20_00.nc', 'T', (0, slice(None), slice(40, 80), slice(50, 100))),
    ('WRF d02 netCDF3', 'wrfout_d02_2024-11-06_01_30_00', 'T', (0, slice(None), slice(40, 80), slice(50, 100))),
    ('MODIS 1 km L1B', 'MOD021KM.A2025301.1445.007.2025301204859.nc',
     'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive', (3, slice(1000, 1256), slice(500, 756))),
    ('MODIS 250 m L1B', 'MOD02QKM.A2025301.1445.007.2025301204859.nc',
     'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_250_RefSB', (0, slice(4000, 4256), slice(2000, 2256))),
    ('VIIRS I-band', 'VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I05', (slice(3000, 3256), slice(3000, 3256))),
    ('Metop-SG VII', 'W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20260922172210_G_O_20260922163559_20260922163659_C_N_T__.nc',
     'data/measurement_data/vii_10690', (slice(100, 356), slice(1000, 1256))),
    ('AVHRR HRPT', 'NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc', 'avhrr_ch4', (slice(100, 356), slice(1000, 1256))),
    ('RCM SAR winds', 'rcm2-sclna-owi-cm-20261014t010230-20261014t010349-00003-xxxxx_ll_gd.nc', 'wind_speed',
     (0, slice(100, 356), slice(100, 356))),
    ('Ku scatterometer', '20260904I1_ku_outer_lfm_VV_L2_pass13_combined.nc', 'Ze', (slice(100, 356), slice(None))),
    ('GOES-19 ABI RadF C02', 'extra/OR_ABI-L1b-RadF-M6C02_G19_s20263021200207_e20263021209516_c20263021209551.nc',
     'Rad', (slice(10000, 10512), slice(10000, 10512))),
    ('GOES-19 ABI RadF C13', 'extra/OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc',
     'Rad', (slice(2000, 2256), slice(2000, 2256))),
]


def best(fns, n):
    """Best time (ms) of each fn over n rounds, run in turn round by round."""
    times = [[] for _ in fns]
    for _ in range(n):
        for k, fn in enumerate(fns):
            t = time.perf_counter()
            fn()
            times[k].append(time.perf_counter() - t)
    return [min(t) * 1e3 for t in times]


# -- the readers
def nc4_all(p, masked):
    import netCDF4
    with netCDF4.Dataset(p) as d:
        def walk(g):
            for v in g.variables.values():
                v.set_auto_maskandscale(masked)
                try:
                    v[...]
                except Exception:
                    pass
            for s in g.groups.values():
                walk(s)
        walk(d)


def h5_all(p):
    import h5py
    with h5py.File(p, 'r') as f:
        def visit(n, o):
            if isinstance(o, h5py.Dataset) and o.shape:
                try:
                    o[()]
                except Exception:
                    pass
        f.visititems(visit)


def fn_all(p, masked):
    import fastnc
    with fastnc.Dataset(p) as d:
        d.set_auto_maskandscale(masked)
        d.load(recursive=True)


def xr_load(p, engine):
    import xarray as xr
    with xr.open_dataset(p, engine=engine) as ds:
        ds.load()


def one(lib, p, v, key, masked):
    if lib == 'fastnc':
        import fastnc
        with fastnc.Dataset(p) as d:
            x = d[v]
            x.set_auto_maskandscale(masked)
            return x[key]
    if lib == 'netCDF4':
        import netCDF4
        with netCDF4.Dataset(p) as d:
            x = d[v]
            x.set_auto_maskandscale(masked)
            return x[key]
    import h5py
    with h5py.File(p, 'r') as f:
        return f[v][key]


FIRST = r'''
import sys, time
t = time.perf_counter()
lib, p, v = sys.argv[1:4]
if lib == 'fastnc':
    import fastnc
    d = fastnc.Dataset(p); x = d[v]; x.set_auto_maskandscale(False); x[...]
elif lib == 'netCDF4':
    import netCDF4
    d = netCDF4.Dataset(p); x = d[v]; x.set_auto_maskandscale(False); x[...]
else:
    import h5py
    f = h5py.File(p, 'r'); f[v][()]
print((time.perf_counter() - t) * 1e3)
'''


def first_call(lib, p, v):
    env = dict(os.environ)
    env.pop('PYTHONDONTWRITEBYTECODE', None)
    out = subprocess.run([sys.executable, '-c', FIRST, lib, p, v], capture_output=True, text=True,
                         cwd=ROOT, env=env)
    try:
        return float(out.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def machine():
    info = {'python': platform.python_version(), 'platform': platform.platform(),
            'cpu': platform.processor(), 'logical_cpus': os.cpu_count(), 'numpy': np.__version__}
    for mod in ('numba', 'h5py', 'netCDF4', 'xarray', 'h5netcdf', 'cupy'):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import cupy
        info['gpu'] = cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()
    except Exception:
        info['gpu'] = None
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--files', default='')
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'benchmarks.json'))
    a = ap.parse_args()
    fast_thread()
    import fastnc
    from fastnc import _chunks
    _chunks.warm()
    n = 2 if a.quick else 5
    rows = []
    files = [f for f in FILES if not a.files or any(s in f[0] for s in a.files.split(','))]
    for label, f, v, win in files:
        p = os.path.join(D, f)
        if not os.path.exists(p):
            continue
        is_hdf5 = open(p, 'rb').read(4) != b'CDF\x01'[:3] + open(p, 'rb').read(4)[3:4] or True
        hdf5 = not open(p, 'rb').read(3) == b'CDF'
        r = {'file': label, 'path': f, 'MB': os.path.getsize(p) / 1e6, 'variable': v}
        # warm the OS cache and every reader
        fn_all(p, False)
        nc4_all(p, False)
        if hdf5:
            h5_all(p)
        # whole file
        fns = [lambda: fn_all(p, False), lambda: nc4_all(p, False)] + ([lambda: h5_all(p)] if hdf5 else [])
        t = best(fns, max(2, n // 2))
        r['all_raw'] = dict(zip(['fastnc', 'netCDF4', 'h5py'], t))
        try:
            t = best([lambda: fn_all(p, True), lambda: nc4_all(p, True)], max(2, n // 2))
            r['all_masked'] = dict(zip(['fastnc', 'netCDF4'], t))
        except Exception as e:
            r['all_masked'] = {'error': repr(e)[:100]}
        fastnc.set_threads(1)
        r['all_raw_1thread'] = best([lambda: fn_all(p, False)], max(2, n // 2))[0]
        fastnc.set_threads(None)
        engines = ['netcdf4'] + (['h5netcdf'] if hdf5 else ['scipy'])
        try:
            from fastnc._xarray import FastncBackendEntrypoint
            t = best([lambda: xr_load(p, FastncBackendEntrypoint)] + [lambda e=e: xr_load(p, e) for e in engines], 2)
            r['xarray_load'] = dict(zip(['fastnc'] + engines, t))
        except Exception as e:
            r['xarray_load'] = {'error': repr(e)[:100]}
        # one variable, raw and masked; a window
        libs = ['fastnc', 'netCDF4'] + (['h5py'] if hdf5 else [])
        r['var_raw'] = dict(zip(libs, best([lambda l=l: one(l, p, v, Ellipsis, False) for l in libs], n)))
        try:
            r['var_masked'] = dict(zip(libs[:2], best([lambda l=l: one(l, p, v, Ellipsis, True) for l in libs[:2]], n)))
        except Exception as e:
            r['var_masked'] = {'error': repr(e)[:100]}
        r['window_raw'] = dict(zip(libs, best([lambda l=l: one(l, p, v, win, False) for l in libs], n)))
        fastnc.set_threads(1)
        r['var_raw_1thread'] = best([lambda: one('fastnc', p, v, Ellipsis, False)], n)[0]
        fastnc.set_threads(None)
        r['first_call'] = {l: first_call(l, p, v) for l in libs}
        if a.gpu:
            import cupy as cp

            def gpu(masked):
                with fastnc.Dataset(p) as d:
                    x = d[v]
                    x.set_auto_maskandscale(masked)
                    out = x.read(Ellipsis, device='gpu')
                cp.cuda.Device().synchronize()
                return out

            def cpu_up(lib, masked):
                vals = one(lib, p, v, Ellipsis, masked)
                cp.asarray(np.ma.getdata(vals))
                cp.cuda.Device().synchronize()
            gpu(False)
            r['gpu_raw'] = dict(zip(['fastnc gpu', 'fastnc cpu+upload', 'netCDF4+upload'],
                                    best([lambda: gpu(False), lambda: cpu_up('fastnc', False),
                                          lambda: cpu_up('netCDF4', False)], n)))
            try:
                gpu(True)
                r['gpu_masked'] = dict(zip(['fastnc gpu', 'fastnc cpu+upload', 'netCDF4+upload'],
                                           best([lambda: gpu(True), lambda: cpu_up('fastnc', True),
                                                 lambda: cpu_up('netCDF4', True)], n)))
            except Exception as e:
                r['gpu_masked'] = {'error': repr(e)[:100]}
        rows.append(r)
        print(json.dumps(r))
    res = {'machine': machine(), 'when': time.strftime('%Y-%m-%d %H:%M'), 'rows': rows}
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
