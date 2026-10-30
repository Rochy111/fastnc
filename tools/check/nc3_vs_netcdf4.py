"""fastnc's netCDF3 reader against netCDF4-python (raw: auto mask/scale and
chartostring off), on every netCDF3 file at hand: tests/data/nc3_*.nc, the
synthetic corpus, wrfout d02, and edge cases made by nc3_make_edge.py into
a temporary folder (with --big, also files big enough for the threaded
paths). Metadata (data model, dimensions, variables, attributes with their
types) and values, whole and for random keys, compared byte for byte.

    python tools/check/nc3_vs_netcdf4.py [--big] [--keys N] [--no-real]
"""
import glob
import os
import sys
import tempfile

import numpy as np
import netCDF4

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from fastnc import _nc3                                         # noqa: E402
import nc3_make_edge                                            # noqa: E402

DATA = os.path.dirname(os.path.dirname(os.path.dirname(ROOT))) \
    if '.claude' in ROOT else ROOT                          # (from a worktree: the main checkout)
DATA = os.environ.get('FASTNC_DATA', DATA)


def same_value(a, b):
    """Equal including type: numpy scalars / arrays by dtype and bytes, str /
    bytes exactly."""
    if type(a) is not type(b):
        return False
    if isinstance(a, (str, bytes)):
        return a == b
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, np.generic):
        return a.dtype == b.dtype and a.tobytes() == b.tobytes()
    return a == b


def random_key(rng, shape):
    key = []
    for n in shape:
        r = rng.random()
        if n == 0:
            key.append(slice(0, 0, 1))
        elif r < 0.25:
            key.append(int(rng.integers(0, n)))
        elif r < 0.45:
            key.append(slice(0, n, 1))
        else:
            a = int(rng.integers(0, n))
            b = int(rng.integers(a, n + 1))
            s = int(rng.choice([1, 1, 2, 3, 7, max(1, n // 3)]))
            key.append(slice(a, b, s))
    return tuple(key)


def compare(path, nkeys, ref_path=None, stats=None):
    """Every check on one file; ref_path: read netCDF4's values from this
    file instead (for streaming copies netCDF4 cannot read)."""
    errs = []
    ref = netCDF4.Dataset(ref_path or path)
    ref.set_auto_maskandscale(False)
    ref.set_auto_chartostring(False)
    f = _nc3.NC3File(path)
    try:
        g = f.root
        if f.data_model != ref.data_model:
            errs.append('data_model %s != %s' % (f.data_model, ref.data_model))
        rd = {k: (len(d), d.isunlimited()) for k, d in ref.dimensions.items()}
        if list(g.dimensions.items()) != list(rd.items()):
            errs.append('dimensions %s != %s' % (g.dimensions, rd))
        ra = {k: ref.getncattr(k) for k in ref.ncattrs()}
        if list(g.attrs) != list(ra):
            errs.append('global attribute names %s != %s' % (list(g.attrs), list(ra)))
        for k in ra:
            if k in g.attrs and not same_value(g.attrs[k], ra[k]):
                errs.append('global attribute %r: %r != %r' % (k, g.attrs[k], ra[k]))
        if list(g.variables) != list(ref.variables):
            errs.append('variables %s != %s' % (list(g.variables), list(ref.variables)))
        rng = np.random.default_rng(len(path))
        for name, rv in ref.variables.items():
            v = g.variables.get(name)
            if v is None:
                continue
            for what, a, b in (('dimensions', v.dimensions, rv.dimensions),
                               ('shape', v.shape, rv.shape), ('dtype', v.dtype, rv.dtype),
                               ('filters', v.filters, rv.filters()),
                               ('endian', v.endian, rv.endian())):
                if a != b:
                    errs.append('%s %s: %r != %r' % (name, what, a, b))
            if rv.chunking() is not None or v.chunks is not None:
                errs.append('%s chunking %r / chunks %r' % (name, rv.chunking(), v.chunks))
            va = v.attrs
            rva = {k: rv.getncattr(k) for k in rv.ncattrs()}
            if list(va) != list(rva):
                errs.append('%s attribute names %s != %s' % (name, list(va), list(rva)))
            for k in rva:
                if k in va and not same_value(va[k], rva[k]):
                    errs.append('%s attribute %r: %r != %r' % (name, k, va[k], rva[k]))
            keys = [tuple(slice(0, n, 1) for n in v.shape)]
            keys += [random_key(rng, v.shape) for _ in range(nkeys)]
            for key in keys:
                want = np.asarray(rv[key] if key else rv[...])
                got = v.read(key)
                if got.dtype != want.dtype or got.shape != want.shape or \
                        got.tobytes() != want.tobytes():
                    errs.append('%s%s: values differ (%s %s vs %s %s)' % (
                        name, key, got.dtype, got.shape, want.dtype, want.shape))
                    break
                if stats is not None:
                    stats['reads'] += 1
                    stats['bytes'] += got.nbytes
            out = np.empty(v.shape, v.dtype)
            r = v.read(keys[0], out=out)
            want = np.asarray(rv[...])
            if r is not out or out.tobytes() != want.tobytes():
                errs.append('%s: read(out=) wrong' % name)
    finally:
        f.close()
        ref.close()
    return errs


def main():
    big = '--big' in sys.argv
    nkeys = 20
    if '--keys' in sys.argv:
        nkeys = int(sys.argv[sys.argv.index('--keys') + 1])
    files = sorted(glob.glob(os.path.join(ROOT, 'tests', 'data', 'nc3_*.nc')))
    files += sorted(glob.glob(os.path.join(DATA, 'corpus', 'synthetic', 'nc3_*.nc')))
    if '--no-real' not in sys.argv:
        d02 = os.path.join(DATA, 'wrfout_d02_2024-11-06_01_30_00')
        if os.path.exists(d02):
            files.append(d02)
    tmp = tempfile.mkdtemp(prefix='fastnc_nc3_')
    files += nc3_make_edge.make_all(tmp, big=big)
    pairs = [(p, None if nc3_make_edge.reference_for(p) == p else nc3_make_edge.reference_for(p))
             for p in files] + nc3_make_edge.streaming_files(tmp)
    stats = {'reads': 0, 'bytes': 0}
    bad = 0
    # (settings of _nc3 to run everything under: the defaults, numba's
    # kernels on, every path taken on tiny requests, the plain-handle reader)
    small = dict(PIECE=4096, RUN_MIN=256, PAR_MIN=8192, SWAP_NUMPY=512, GATHER_PAR=1024,
                 GATHER_BLOCK=300)
    settings = [('defaults', {}), ('kernels', dict(KERNELS=True))]
    if '--small-thresholds' in sys.argv or big:
        settings += [('small, numpy', dict(small, KERNELS=False)),
                     ('small, kernels', dict(small, KERNELS=True)),
                     ('small, kernels, handles', dict(small, KERNELS=True, _IO=_nc3._Handles))]
    for label, st in settings:
        saved = {k: getattr(_nc3, k) for k in st}
        for k, v in st.items():
            setattr(_nc3, k, v)
        for p, ref in pairs:
            try:
                errs = compare(p, nkeys, ref, stats)
            except Exception as e:
                errs = ['%s: %s' % (type(e).__name__, e)]
            status = 'ok' if not errs else 'FAIL'
            print('%-4s %s%s [%s]' % (status, os.path.basename(p),
                                      ' (vs %s)' % os.path.basename(ref) if ref else '', label))
            for e in errs[:10]:
                print('      ', e)
            bad += bool(errs)
        for k, v in saved.items():
            setattr(_nc3, k, v)
    print('%d files x %d settings, %d failing; %d reads, %.1f MB compared' % (
        len(pairs), len(settings), bad, stats['reads'], stats['bytes'] / 1e6))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
