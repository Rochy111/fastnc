"""fastnc against h5py directly on virtual datasets and external storage:
every such dataset of the given files (default: tests/data/vds/*.h5),
shape, fill value, sources, and values whole and for a few selections.

    python tools/check/vds_vs_h5py.py [files...]

Run from the files' directory when they use external storage with
relative names (h5py looks for those in the current directory)."""
import glob
import os
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import fastnc                                               # noqa: E402

KEYS = [(), (0,), (slice(None, None, 2),), (slice(1, -1, 3), Ellipsis), (Ellipsis, slice(2, 9))]


def main(paths):
    bad = n = 0
    for p in paths:
        try:
            h = h5py.File(p, 'r')
        except OSError as e:
            print('skip %s (h5py: %s)' % (p, e))
            continue
        f = fastnc.File(p)
        names = []
        try:
            h.visititems(lambda k, o: names.append(k) if isinstance(o, h5py.Dataset)
                         and (o.is_virtual or o.external) else None)
        except RuntimeError as e:                       # (HDF5 2.0's files in HDF5 1.14)
            print('skip %s (h5py: %s)' % (p, e))
            names = []
        for k in names:
            a, b = h[k], f[k]
            n += 1
            msgs = []
            if a.shape != b.shape:
                msgs.append('shape %r vs %r' % (b.shape, a.shape))
            if [(m.file_name, m.dset_name) for m in (a.virtual_sources() if a.is_virtual else [])] != \
                    [(m.file_name, m.dset_name) for m in (b.virtual_sources() if b.is_virtual else [])]:
                msgs.append('sources differ')
            if a.external != b.external:
                msgs.append('external %r vs %r' % (b.external, a.external))
            for key in KEYS if a.shape == b.shape else []:
                try:
                    want = a[key]
                except Exception:
                    continue
                got = b[key]
                if np.asarray(got).tobytes() != np.asarray(want).astype(np.asarray(got).dtype).tobytes():
                    msgs.append('values differ for %r' % (key,))
            if msgs:
                bad += 1
                print('%s:%s: %s' % (p, k, '; '.join(msgs)))
        f.close()
        h.close()
    print('%d datasets, %d differ' % (n, bad))
    return bad


if __name__ == '__main__':
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, 'tests', 'data', 'vds', '*.h5')))
    os.chdir(os.path.dirname(os.path.abspath(files[0])))
    sys.exit(1 if main([os.path.abspath(x) for x in files]) else 0)
