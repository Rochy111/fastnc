"""fastnc._maskscale.apply on the real files: equal to netCDF4-python's
masked / scaled read (type, dtype, data bits, mask, fill value), and how
long it takes against netCDF4-python's own mask and scale.

    python tools/check/maskscale_real.py [--gpu] [--reps N]

netCDF4-python's cost is measured two ways, alternating with fastnc's
runs (other work may share the machine): its whole masked read minus its
raw read, and its post-processing alone (the same numpy code, run by
fastnc._maskscale._reference on the raw array). fastnc's apply() runs on
1 thread and on all of them; --gpu adds the cupy path (raw already on
the card, result left there; and with the copies both ways).
"""
import glob
import os
import statistics
import sys
import time
import warnings

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'tools', 'profile'))

from fastnc import _maskscale as M, _pool        # noqa: E402

D = r'C:\Users\Roch\Desktop\Python\fastnc'
CASES = [
    (D + r'\VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I05'),
    (D + r'\VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I01'),
    (D + r'\VJ102IMG_NRT.A2026265.2006.021.nc', 'observation_data/I01_uncert_index'),
    (D + r'\MOD021KM.A2025301.1445.007.2025301204859.nc',
     'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_RefSB'),
    (D + r'\MOD021KM.A2025301.1445.007.2025301204859.nc',
     'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive'),
    (D + r'\MOD021KM.A2025301.1445.007.2025301204859.nc',
     'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/SensorZenith'),
    ('METOP', 'data/measurement_data/vii_443'),
    ('METOP', 'data/measurement_data/vii_12020'),
    ('METOP', 'data/measurement_data/longitude'),
    (D + r'\NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc', 'avhrr_ch1'),
    (D + r'\wrfout_d01_2025-10-26_06_20_00.nc', 'T'),
    (D + r'\wrfout_d02_2024-11-06_01_30_00', 'T'),
]


def fast_thread():
    try:
        from common import fast_thread as f
        f()
    except Exception:
        _pool.fast_thread()


def same(a, b):
    """'' if a and b are the same result to the bit, else what differs."""
    ma = numpy.ma
    if type(a) is not type(b):
        return 'type %s != %s' % (type(a).__name__, type(b).__name__)
    da, db = numpy.asarray(ma.getdata(a)), numpy.asarray(ma.getdata(b))
    if da.dtype != db.dtype or da.shape != db.shape:
        return 'dtype/shape %s%s != %s%s' % (da.dtype, da.shape, db.dtype, db.shape)
    if not numpy.array_equal(da.view(numpy.uint8), db.view(numpy.uint8)):
        n = (da.reshape(-1).view('u%d' % da.itemsize) != db.reshape(-1).view('u%d' % db.itemsize)).sum()
        return 'data differs at %d' % n
    if isinstance(a, ma.MaskedArray):
        if (a._mask is ma.nomask) != (b._mask is ma.nomask):
            return 'nomask vs mask'
        if a._mask is not ma.nomask and not numpy.array_equal(a._mask, b._mask):
            return 'mask differs at %d' % (a._mask != b._mask).sum()
        fa, fb = a.__dict__.get('_fill_value'), b.__dict__.get('_fill_value')
        if (fa is None) != (fb is None) or (fa is not None and (
                numpy.asarray(fa).dtype != numpy.asarray(fb).dtype
                or numpy.asarray(fa).tobytes() != numpy.asarray(fb).tobytes())):
            return 'fill_value %r != %r' % (fa, fb)
    return ''


def timeit(fn, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def main():
    gpu = '--gpu' in sys.argv
    reps = int(sys.argv[sys.argv.index('--reps') + 1]) if '--reps' in sys.argv else 7
    import netCDF4
    fast_thread()
    metop = glob.glob(D + r'\W_XX-EUMETSAT*.nc')[0]
    cp = None
    if gpu:
        import cupy as cp
    print('numpy %s, netCDF4 %s, %d threads' % (numpy.__version__, netCDF4.__version__,
                                               _pool.threads()))
    hdr = ('variable', 'dtype', 'MB', 'nc m+s-raw', 'nc post', 'fastnc 1t', 'fastnc all',
           'post/1t', 'post/all')
    print('%-40s %-7s %6s %10s %9s %9s %10s %16s %16s' % hdr + ('  gpu (copies)' if gpu else ''))
    for path, name in CASES:
        path = metop if path == 'METOP' else path
        ds = netCDF4.Dataset(path)
        v = ds[name]
        attrs = {a: v.getncattr(a) for a in v.ncattrs()}
        no_fill = v.get_fill_value() is None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            exp = v[...]
            v.set_auto_maskandscale(False)
            raw = v[...]
            ok = same(M.apply(raw, attrs, v.dtype, no_fill=no_fill, var_shape=v.shape), exp)
            ok1 = same(M.apply(raw, attrs, v.dtype, no_fill=no_fill, var_shape=v.shape, threads=1),
                       exp)
            okg = ''
            if gpu:
                g = M.apply(cp.asarray(raw), attrs, v.dtype, no_fill=no_fill, var_shape=v.shape,
                            xp=cp)
                okg = same(g.get() if hasattr(g, 'get') else g, exp)

            def nc_ms():
                v.set_auto_maskandscale(True)
                v[...]

            def nc_raw():
                v.set_auto_maskandscale(False)
                v[...]

            def post():
                M._reference(raw, attrs, v.dtype, 'primitive', True, True, True, True, no_fill,
                             v.shape, None, None)

            def f1():
                M.apply(raw, attrs, v.dtype, no_fill=no_fill, var_shape=v.shape, threads=1)

            def fall():
                M.apply(raw, attrs, v.dtype, no_fill=no_fill, var_shape=v.shape)

            graw = cp.asarray(raw) if gpu else None

            def fg():
                M.apply(graw, attrs, v.dtype, no_fill=no_fill, var_shape=v.shape, xp=cp)
                cp.cuda.runtime.deviceSynchronize()

            def fgc():
                r = M.apply(cp.asarray(raw), attrs, v.dtype, no_fill=no_fill, var_shape=v.shape,
                            xp=cp)
                r.get() if hasattr(r, 'get') else r

            fns = [nc_ms, nc_raw, post, f1, fall] + ([fg, fgc] if gpu else [])
            for f in fns:                       # warm (numba compiles, caches)
                f()
            t = {f.__name__: [] for f in fns}
            for _ in range(reps):               # A/B alternating
                for f in fns:
                    t[f.__name__] += timeit(f, 1)
        med = {k: statistics.median(x) * 1e3 for k, x in t.items()}
        rat1 = [a / b for a, b in zip(t['post'], t['f1'])]
        ratn = [a / b for a, b in zip(t['post'], t['fall'])]
        line = '%-40s %-7s %6.1f %10.1f %9.1f %9.2f %10.2f %6.1fx (%.1f-%.1f) %6.1fx (%.1f-%.1f)' % (
            name[-40:], raw.dtype, raw.nbytes / 1e6, med['nc_ms'] - med['nc_raw'], med['post'],
            med['f1'], med['fall'], statistics.median(rat1), min(rat1), max(rat1),
            statistics.median(ratn), min(ratn), max(ratn))
        if gpu:
            line += '  %.2f (%.1f)' % (med['fg'], med['fgc'])
        bad = [s for s in (ok, ok1, okg) if s]
        print(line + ('' if not bad else '   MISMATCH: ' + '; '.join(bad)), flush=True)
        ds.close()


if __name__ == '__main__':
    main()
