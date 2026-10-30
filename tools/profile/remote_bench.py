"""Remote reads timed: fastnc against h5py over a Python file object on HTTP
ranges (what h5py does with an fsspec file; fsspec's HTTP needs aiohttp,
which the dev env lacks, so tools/profile/remote_common.HTTPFile stands in,
with and without fsspec's default 5 MB read-ahead cache) and netCDF-c's
byte-range mode (netCDF4.Dataset(url + '#mode=bytes')).

By default every client goes through a local logging proxy to the real
server, so that requests and bytes are counted the same way for all (a
local hop added to every request). netCDF-c reads an S3 host's URL its own
way (not through plain HTTP as it does the proxy's), so --direct times
every client on the real URL; counts then come from fastnc's and the file
object's own counters (netCDF-c's: none). Each case opens the file afresh.

    python tools/profile/remote_bench.py URL [--cases open,var,window,whole]
        [--clients fastnc,h5py,h5py-cache,netcdf4] [--var Rad] [--repeat 2]
        [--window y0,x0] [--local copy.nc] [--direct]
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import numpy as np                                      # noqa: E402

import fastnc                                           # noqa: E402
from common import fast_thread                          # noqa: E402
from remote_common import HTTPFile, Proxy               # noqa: E402


def case_fastnc(url, case, var, win):
    with fastnc.Dataset(url) as ds:
        ds.set_auto_maskandscale(False)
        meta = {k: {a: v.getncattr(a) for a in v.ncattrs()} for k, v in ds.variables.items()}
        if case == 'open':
            out = len(meta)
        elif case == 'var':
            out = ds[var][...]
        elif case == 'window':
            y, x = win
            out = ds[var][y:y + 512, x:x + 512]
        else:
            out = ds.load()
        return out, (ds._source.requests, ds._source.fetched)


def case_h5py(url, case, var, win, cache):
    import h5py
    fo = HTTPFile(url, cache=cache)
    with h5py.File(fo, 'r') as f:
        meta = {k: dict(f[k].attrs) for k in f.keys()}
        if case == 'open':
            out = len(meta)
        elif case == 'var':
            out = f[var][()]
        elif case == 'window':
            y, x = win
            out = f[var][y:y + 512, x:x + 512]
        else:
            out = {k: f[k][()] for k in f.keys() if isinstance(f[k], h5py.Dataset)}
    fo.close()
    return out, (fo.requests, fo.fetched)


def case_netcdf4(url, case, var, win):
    import netCDF4
    with netCDF4.Dataset(url + '#mode=bytes') as ds:
        ds.set_auto_maskandscale(False)
        meta = {k: {a: v.getncattr(a) for a in v.ncattrs()} for k, v in ds.variables.items()}
        if case == 'open':
            out = len(meta)
        elif case == 'var':
            out = ds[var][...]
        elif case == 'window':
            y, x = win
            out = ds[var][y:y + 512, x:x + 512]
        else:
            out = {k: v[...] for k, v in ds.variables.items()}
        return out, None


CLIENTS = {
    'fastnc': case_fastnc,
    'h5py': lambda u, c, v, w: case_h5py(u, c, v, w, 0),
    'h5py-cache': lambda u, c, v, w: case_h5py(u, c, v, w, 5 << 20),
    'netcdf4': case_netcdf4,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url')
    ap.add_argument('--cases', default='open,var,window,whole')
    ap.add_argument('--clients', default='fastnc,h5py,h5py-cache,netcdf4')
    ap.add_argument('--var', default='Rad')
    ap.add_argument('--window', default='1000,2000')
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--local', help='a local copy: values compared to fastnc reading it')
    ap.add_argument('--direct', action='store_true', help='no proxy')
    a = ap.parse_args()
    fast_thread()
    fastnc.warm()
    win = tuple(int(x) for x in a.window.split(','))
    base, name = a.url.rsplit('/', 1)
    proxy = None if a.direct else Proxy(base)
    url = a.url if a.direct else proxy.url + '/' + name
    local = fastnc.Dataset(a.local) if a.local else None
    if local is not None:
        local.set_auto_maskandscale(False)
    print('%-8s %-11s %9s %9s %11s' % ('case', 'client', 'seconds', 'requests', 'MB fetched'))
    for case in a.cases.split(','):
        for rep in range(a.repeat):
            for client in a.clients.split(','):
                n0 = len(proxy.log) if proxy else 0
                t = time.perf_counter()
                out, counts = CLIENTS[client](url, case, a.var, win)
                dt = time.perf_counter() - t
                if proxy is not None:
                    log = proxy.log[n0:]
                    counts = len(log), sum(n for _, _, n in log)
                ok = ''
                if local is not None and case in ('var', 'window'):
                    want = local[a.var][...] if case == 'var' else \
                        local[a.var][win[0]:win[0] + 512, win[1]:win[1] + 512]
                    ok = 'same' if np.asarray(out).tobytes() == np.asarray(want).tobytes() else 'DIFFERENT'
                if local is not None and case == 'whole' and client == 'fastnc':
                    want = local.load()
                    ok = 'same' if all(np.asarray(out[k]).tobytes() == np.asarray(want[k]).tobytes()
                                       for k in want) and list(out) == list(want) else 'DIFFERENT'
                print('%-8s %-11s %9.3f %9s %11s %s' % (
                    case, client, dt, counts[0] if counts else '?',
                    '%.1f' % (counts[1] / 1e6) if counts else '?', ok), flush=True)
    if proxy is not None:
        proxy.close()


if __name__ == '__main__':
    main()
