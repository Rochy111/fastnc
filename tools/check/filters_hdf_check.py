"""Every chunk of every dataset using HDF5's own filters (shuffle,
fletcher32, szip, n-bit, scale-offset; deflate and bzip2 along) in the
files given, decoded by fastnc and by HDF5 (h5py), compared byte for byte
over the whole chunk.

    python tools/check/filters_hdf_check.py [files...]

Default: tests/data/filters.h5, tests/data/filters_hdf/*.h5 and the
synthetic corpus's szip / fletcher / bzip2 files.
"""
import glob
import os
import sys
import time

import numpy as np

from filters_hdf_common import ROOT, decode_chunk, h5open, load, ref_chunk

OURS = {1, 2, 3, 4, 5, 6, 307}


def default_files():
    out = [os.path.join(ROOT, 'tests', 'data', 'filters.h5')]
    out += sorted(glob.glob(os.path.join(ROOT, 'tests', 'data', 'filters_hdf', '*.h5')))
    syn = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(ROOT))), 'corpus', 'synthetic')
    if not os.path.isdir(syn):
        syn = r'C:\Users\Roch\Desktop\Python\fastnc\corpus\synthetic'
    for n in ('h5_szip.h5', 'nc4_szip.nc', 'h5_gzip_shuffle_fletcher.h5',
              'nc4_zlib_shuffle_fletcher.nc', 'nc4_bzip2.nc'):
        p = os.path.join(syn, n)
        if os.path.exists(p):
            out.append(p)
    return out


def check(path, verbose=False):
    b, f, dsets = load(path)
    h = h5open(path)
    bad = 0
    nchunks = 0
    for name, ds in dsets.items():
        ids = [x.id for x in ds.filters]
        if not ids or not set(ids) <= OURS:
            continue
        hd = h[name]
        t = time.perf_counter()
        nb = 0
        nskip = 0
        for offs, addr, size, mask in ds.chunk_table():
            got = decode_chunk(b, ds, addr, size, mask)
            try:
                ref = ref_chunk(hd, offs)
            except OSError as e:                # this HDF5 cannot (szip missing, 1.14's bugs)
                nskip += 1
                why = str(e)
                continue
            nchunks += 1
            if isinstance(got, int) or got.size != ref.size or not np.array_equal(got, ref):
                nb += 1
                if nb <= 2:
                    print('  BAD %s chunk %s: %s' % (name, offs, got if isinstance(got, int) else
                                                     'size %d vs %d, first diff at %s' % (
                                                         got.size, ref.size,
                                                         np.flatnonzero(got[:ref.size] != ref[:got.size])[:3])))
        bad += nb
        if nskip:
            print('  %-40s no reference from this HDF5: %s' % (name, why[:70]))
        elif verbose or nb:
            print('  %-40s %-30s %s %d chunks %s' % (name, ids, ds.dtype, len(ds.chunk_table()),
                                                   'ok' if not nb else '%d BAD' % nb))
    print('%s: %d chunks, %s' % (os.path.basename(path), nchunks, 'all equal' if not bad else '%d BAD' % bad))
    return bad


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    v = '-v' in sys.argv
    files = args or default_files()
    total = sum(check(p, v) for p in files)
    sys.exit(1 if total else 0)
