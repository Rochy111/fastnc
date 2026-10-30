"""What the remote hooks cost local reads: fastnc at this checkout against
another tree (e.g. the commit before remote reading, unpacked with git
archive), alternately in fresh processes, on local files.

Times (median of --n): open = Dataset() plus every variable's attributes
(every need() of the metadata parse), index = the first small read of the
biggest chunked variable (its chunk index parsed: need() / prefetch() per
B-tree node), read = the same small read again (the engine's fetch hook).

    python tools/profile/remote_local_overhead.py --base DIR FILE... [--rounds 5] [--n 15]
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def child(root, path, n):
    sys.path.insert(0, root)
    sys.path.insert(1, HERE)
    from common import fast_thread
    fast_thread()
    import numpy as np
    import fastnc
    fastnc.warm()
    t_open, t_index, t_read = [], [], []
    for i in range(n):
        t = time.perf_counter()
        ds = fastnc.Dataset(path)
        for g in ds._walk():
            for v in g.variables.values():
                {a: v.getncattr(a) for a in v.ncattrs()}
        t_open.append(time.perf_counter() - t)
        best = None
        for g in ds._walk():
            for v in g.variables.values():
                if v.chunking() != 'contiguous' and len(v.shape) >= 2 and v.dtype != str and \
                        (best is None or v.size > best.size):
                    best = v
        if best is not None:
            best.set_auto_maskandscale(False)
            key = tuple(slice(s // 2, s // 2 + 64) for s in best.shape)
            t = time.perf_counter()
            best[key]
            t_index.append(time.perf_counter() - t)
            t = time.perf_counter()
            best[key]
            t_read.append(time.perf_counter() - t)
        ds.close()
    med = lambda x: float(np.median(x)) if x else float('nan')   # noqa: E731
    print(json.dumps([med(t_open), med(t_index), med(t_read)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--base', required=True)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--n', type=int, default=15)
    a = ap.parse_args()
    for path in a.files:
        res = {'base': [], 'this': []}
        for r in range(a.rounds):
            for name, root in (('base', a.base), ('this', ROOT)):
                out = subprocess.run([sys.executable, __file__, '--child', root, path, str(a.n)],
                                     capture_output=True, text=True)
                if out.returncode:
                    print(out.stderr[-2000:])
                    raise SystemExit(1)
                res[name].append(json.loads(out.stdout.strip().splitlines()[-1]))
        import numpy as np
        b, t = np.array(res['base']), np.array(res['this'])
        print(os.path.basename(path)[:50])
        for k, name in enumerate(('open', 'index', 'read')):
            ratio = t[:, k] / b[:, k]
            print('  %-5s base %8.3f ms  this %8.3f ms  ratio %.3f (min %.3f max %.3f)' % (
                name, np.median(b[:, k]) * 1e3, np.median(t[:, k]) * 1e3, np.median(ratio),
                ratio.min(), ratio.max()))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        main()
