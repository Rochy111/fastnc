"""Where a remote fastnc read spends its time: open (metadata), the chunk
index, the data -- with requests and bytes for each, straight to the URL.

    python tools/profile/remote_phases.py URL [--var Rad] [--window y0,x0]
        [--whole] [--repeat 3] [--set NAME=VALUE ...]  (_remote constants)
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import fastnc                                           # noqa: E402
from fastnc import _remote                              # noqa: E402
from common import fast_thread                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url')
    ap.add_argument('--var', default='Rad')
    ap.add_argument('--window', default='1000,2000')
    ap.add_argument('--whole', action='store_true', help='the whole variable, not a window')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split('=')
        setattr(_remote, k, int(v))
    fast_thread()
    fastnc.warm()
    y, x = (int(v) for v in a.window.split(','))
    for rep in range(a.repeat):
        row = []
        t = time.perf_counter()
        ds = fastnc.Dataset(a.url)
        s = ds._source
        {k: {n: v.getncattr(n) for n in v.ncattrs()} for k, v in ds.variables.items()}
        mark = [(time.perf_counter(), s.requests, s.fetched)]
        v = ds[a.var]
        v.set_auto_maskandscale(False)
        v._bv._reader.index
        mark.append((time.perf_counter(), s.requests, s.fetched))
        out = v[...] if a.whole else v[y:y + 512, x:x + 512]
        mark.append((time.perf_counter(), s.requests, s.fetched))
        prev = (t, 0, 0)
        for name, m in zip(('open', 'index', 'data'), mark):
            row.append('%s %.3f s %3d req %6.1f MB' % (name, m[0] - prev[0], m[1] - prev[1],
                                                       (m[2] - prev[2]) / 1e6))
            prev = m
        print(' | '.join(row) + ' | total %.3f s  %s' % (mark[-1][0] - t, out.shape), flush=True)
        ds.close()


if __name__ == '__main__':
    main()
