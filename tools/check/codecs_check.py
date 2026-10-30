"""fastnc's codec kernels against the reference filters (hdf5plugin through
h5py), chunk by chunk, byte for byte, on every single-filter dataset of
the HDF5 files given (default: tests/data/filters.h5 and
tests/data/codecs/*.h5).

    python tools/check/codecs_check.py [file.h5 ...]
"""
import glob
import importlib
import os
import sys

import h5py
import numpy as np

from codecs_ref import ROOT, cd_array, chunks, filters

MODULES = {32000: 'lzf', 32001: 'blosc', 32004: 'lz4', 32008: 'bitshuffle',
           32015: 'zstd', 32026: 'blosc2'}


def module(fid):
    name = MODULES.get(fid)
    if name is None:
        return None
    try:
        return importlib.import_module('fastnc._' + name)
    except ImportError:
        return None


def check_dataset(ds, verbose=False):
    """(chunks checked, failures) for one dataset."""
    fl = filters(ds)
    if len(fl) != 1:
        return 0, []
    m = module(fl[0][0])
    if m is None:
        return 0, []
    cd = cd_array(ds)
    nb = int(cd[1])
    ws = np.zeros(max(m.workspace_size(nb), 1), np.uint8)
    bad = []
    n = 0
    for off, raw, mask, want in chunks(ds):
        if mask & 1:
            continue                    # filter skipped for this chunk
        n += 1
        src = np.frombuffer(raw, np.uint8)
        pad = 7                         # output in the middle of a larger array
        dst = np.full(nb + 2 * pad, 0xA5, np.uint8)
        r = m.decode(src, 0, len(src), dst, pad, pad + nb, cd, ws)
        ok = (r == pad + len(want) and dst[pad:pad + len(want)].tobytes() == want
              and (dst[:pad] == 0xA5).all() and (dst[pad + nb:] == 0xA5).all())
        if not ok:
            bad.append((ds.name, off, r, len(want)))
            if verbose:
                got = dst[pad:pad + len(want)].tobytes()
                i = next((k for k in range(min(len(got), len(want))) if got[k] != want[k]), None)
                print('   FAIL %s %s r=%s want %d, first diff at %s' % (ds.name, off, r, len(want), i))
    return n, bad


def check_file(path, verbose=True):
    tot, fails, sets = 0, [], 0
    with h5py.File(path, 'r') as f:
        items = []
        f.visititems(lambda n, o: items.append(o) if isinstance(o, h5py.Dataset) else None)
        for ds in items:
            n, bad = check_dataset(ds, verbose)
            if n:
                sets += 1
            tot += n
            fails += bad
    print('%-50s %4d datasets %6d chunks %4d failures' % (os.path.relpath(path, ROOT), sets, tot, len(fails)))
    return tot, fails


if __name__ == '__main__':
    paths = sys.argv[1:] or ([os.path.join(ROOT, 'tests', 'data', 'filters.h5')]
                             + sorted(glob.glob(os.path.join(ROOT, 'tests', 'data', 'codecs', '*.h5'))))
    allf = []
    for p in paths:
        allf += check_file(p)[1]
    print('FAILURES: %d' % len(allf))
    sys.exit(1 if allf else 0)
