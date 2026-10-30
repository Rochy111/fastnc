"""Whole variables read one after another (xarray's load, a loop over
ds.variables): with readahead on and off -- times, and the same values."""
import sys, time, warnings
from common import fast_thread, path
sys.path.insert(0, path('.'))
warnings.simplefilter('ignore')
import numpy as np, fastnc, xarray as xr
from fastnc import _readahead
from fastnc._xarray import FastncBackendEntrypoint as E
fastnc.warm(); fast_thread()


def best(fn, k=5):
    r = []
    for _ in range(k):
        t = time.perf_counter(); fn(); r.append(time.perf_counter() - t)
    return min(r) * 1e3


def loop(p, masked):
    with fastnc.Dataset(p) as d:
        d.set_auto_maskandscale(masked)
        return {k: v[...] for k, v in d.variables.items()}


def enc(x):
    if x is np.ma.masked:
        return 'masked'
    if np.ma.isMaskedArray(x):
        return (np.ma.getdata(x).tobytes(), np.ma.getmaskarray(x).tobytes(), repr(x.fill_value), x.dtype)
    x = np.asarray(x)
    return (x.dtype, x.shape, repr(x.tolist()) if x.dtype.hasobject else x.tobytes())


for f in sys.argv[1:] or ['wrfout_d01_2025-10-26_06_20_00.nc', 'NSS.HRPT.ND.D92236.S1231.E1243.B171717.nc',
                          'rcm2-sclna-owi-cm-20261014t010230-20261014t010349-00003-xxxxx_ll_gd.nc',
                          '20260904I1_ku_outer_lfm_VV_L2_pass13_combined.nc']:
    p = path(f)
    row = []
    res = {}
    for on in (False, True):
        _readahead.ENABLED = on
        res[on] = (loop(p, True), xr.open_dataset(p, engine=E).load())
        row.append('%s: loop raw %6.1f masked %6.1f  xarray load %6.1f' % (
            'on ' if on else 'off', best(lambda: loop(p, False)), best(lambda: loop(p, True)),
            best(lambda: xr.open_dataset(p, engine=E).load())))
    a, b = res[False], res[True]
    same = all(enc(a[0][k]) == enc(b[0][k]) for k in a[0]) and a[1].identical(b[1])
    print('%-28s %s | %s | same: %s' % (f[:28], row[0], row[1], same), flush=True)
