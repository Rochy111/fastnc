"""Where each operation's time goes, phase by phase: fastnc's functions
wrapped with timers (outermost call per phase only), pool work timed on
every thread to see how busy the threads are, and the rest -- the
operation's wall time minus the phases -- reported as "other" (per-read
Python, xarray, numpy).

    python tools/profile/phases.py [file-substring ...]

Columns (ms, best run of 5 by wall time): wall; open (Dataset.__init__);
plan (chunk plans); pool (wall time inside _pool.share / map_items: the
kernels and threaded numpy) with its thread efficiency (busy thread-time /
(pool wall x threads asked for)) and number of calls; mask (mask / scale
outside the pool); ra (readahead's batch reads, included in pool); alloc
(arena); other.
"""
import functools
import os
import sys
import threading
import time
import warnings

from common import fast_thread, path

sys.path.insert(0, path('.'))
warnings.simplefilter('ignore')
import numpy as np                                  # noqa: E402

import fastnc                                       # noqa: E402
from fastnc import _arena, _chunks, _dataset, _maskscale, _pool, _readahead  # noqa: E402

fastnc.warm()
fast_thread()

T = {}
TLS = threading.local()
LOCK = threading.Lock()
POOL = {'wall': 0.0, 'busy': 0.0, 'cap': 0.0, 'calls': 0}


def timed(mod, name, phase):
    real = getattr(mod, name)

    @functools.wraps(real)
    def w(*a, **k):
        depth = getattr(TLS, phase, 0)
        setattr(TLS, phase, depth + 1)
        t = time.perf_counter()
        try:
            return real(*a, **k)
        finally:
            setattr(TLS, phase, depth)
            if depth == 0 and threading.current_thread() is threading.main_thread():
                T[phase] = T.get(phase, 0.0) + time.perf_counter() - t
    setattr(mod, name, w)


real_share = _pool.share


def share(work, n):
    n = max(1, min(n, _pool.threads()))
    busy = [0.0]

    def wrapped():
        t = time.perf_counter()
        try:
            work()
        finally:
            d = time.perf_counter() - t
            with LOCK:
                busy[0] += d
    outer = getattr(TLS, 'pool', 0) == 0 and threading.current_thread() is threading.main_thread()
    TLS.pool = getattr(TLS, 'pool', 0) + 1
    t = time.perf_counter()
    try:
        real_share(wrapped, n)
    finally:
        TLS.pool -= 1
        d = time.perf_counter() - t
        if outer:
            POOL['wall'] += d
            POOL['busy'] += busy[0]
            POOL['cap'] += d * n
            POOL['calls'] += 1


_pool.share = share
timed(_dataset.Dataset, '__init__', 'open')
timed(_chunks, 'plan', 'plan')
timed(_maskscale, 'apply', 'mask')
timed(_readahead.Readahead, '_fetch', 'ra')
timed(_arena, 'empty', 'alloc')

FILES = {
    'wrf1': ('wrfout_d01_2025-10-26_06_20_00.nc', 'T', (0, slice(10, 20), slice(50, 80), slice(60, 90))),
    'wrf2': ('wrfout_d02_2024-11-06_01_30_00', 'T', (0, slice(10, 20), slice(50, 80), slice(60, 90))),
    'modis': ('MOD021KM.A2025301.1445.007.2025301204859.nc',
              'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive',
              (slice(2, 4), slice(500, 600), slice(400, 500))),
    'viirs': ('VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I05',
              (slice(3000, 3512), slice(3000, 3512))),
    'metop': ('W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20260922172210_G_O_20260922163559_20260922163659_C_N_T__.nc',
              None, None),
    'avhrr': ('NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc', 'avhrr_ch4', (slice(100, 400), slice(100, 400))),
    'rcm': ('rcm2-sclna-owi-cm-20261014t010230-20261014t010349-00003-xxxxx_ll_gd.nc', 'wind_speed',
            (slice(100, 300), slice(100, 300))),
    'ku': ('20260904I1_ku_outer_lfm_VV_L2_pass13_combined.nc', 'Ze', (slice(100, 200),)),
    'c02': ('extra/OR_ABI-L1b-RadF-M6C02_G19_s20263021200207_e20263021209516_c20263021209551.nc', 'Rad',
            (slice(5000, 6000), slice(5000, 6000))),
    'c13': ('extra/OR_ABI-L1b-RadF-M6C13_G19_s20263021200207_e20263021209527_c20263021209579.nc', 'Rad',
            (slice(1000, 1500), slice(1000, 1500))),
}


def ops(p, var, win):
    def loop(masked):
        with fastnc.Dataset(p) as d:
            d.set_auto_maskandscale(masked)
            for g in d._walk():
                for v in g.variables.values():
                    try:
                        v[...]
                    except Exception:
                        pass

    def load(masked):
        with fastnc.Dataset(p) as d:
            d.set_auto_maskandscale(masked)
            d.load(recursive=True)

    def one(masked, key=Ellipsis):
        with fastnc.Dataset(p) as d:
            x = d[var]
            x.set_auto_maskandscale(masked)
            x[key]

    def xr():
        import xarray
        from fastnc._xarray import FastncBackendEntrypoint
        with xarray.open_dataset(p, engine=FastncBackendEntrypoint) as ds:
            ds.load()

    out = [('open', lambda: fastnc.Dataset(p).close()),
           ('loop raw', lambda: loop(False)), ('loop masked', lambda: loop(True)),
           ('load raw', lambda: load(False)), ('xarray load', xr)]
    if var:
        out += [('var raw', lambda: one(False)), ('var masked', lambda: one(True)),
                ('window', lambda: one(False, win))]
    return out


def measure(fn, n=5):
    best = None
    for _ in range(n):
        T.clear()
        for k in POOL:
            POOL[k] = 0
        t = time.perf_counter()
        fn()
        wall = time.perf_counter() - t
        if best is None or wall < best[0]:
            best = (wall, dict(T), dict(POOL))
    return best


def row(name, r):
    wall, t, pool = r
    ms = lambda x: x * 1e3                          # noqa: E731
    eff = pool['busy'] / pool['cap'] if pool['cap'] else 0
    known = t.get('open', 0) + t.get('plan', 0) + pool['wall'] + t.get('alloc', 0)
    mask_only = max(0.0, t.get('mask', 0) - 0)      # (mask's own pool time is inside pool too)
    other = wall - known
    return ('  %-12s wall %8.2f | open %6.2f plan %6.2f pool %8.2f (eff %3.0f%%, %4d calls) '
            'mask %6.2f ra %6.2f alloc %5.2f | other %7.2f' % (
                name, ms(wall), ms(t.get('open', 0)), ms(t.get('plan', 0)), ms(pool['wall']),
                100 * eff, pool['calls'], ms(mask_only), ms(t.get('ra', 0)), ms(t.get('alloc', 0)),
                ms(other)))


def main():
    want = sys.argv[1:] or list(FILES)
    for key in want:
        f, var, win = FILES[key]
        p = path(f)
        print('%s  (%s)' % (key, f[:60]), flush=True)
        for name, fn in ops(p, var, win):
            try:
                fn()
                print(row(name, measure(fn)), flush=True)
            except Exception as e:
                print('  %-12s failed: %s' % (name, e))


if __name__ == '__main__':
    main()
