"""fastnc reading a remote file against the same file on disk, read by
fastnc and by h5py: every dimension, attribute and variable (raw values;
variables bigger than --max-mb compared on a window and their first and
last rows), through both front ends (Dataset and File).

    python tools/check/remote_vs_local.py URL LOCAL [--max-mb 64]
    python tools/check/remote_vs_local.py fileobj LOCAL   (the local file through a
        Python file object, its buffer poisoned: what the remote path reads)
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import h5py                                             # noqa: E402
import numpy as np                                      # noqa: E402

import fastnc                                           # noqa: E402
from fastnc._h5read import native                       # noqa: E402


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if a.dtype.kind == 'O' or b.dtype.kind == 'O':
            return all(same(x, y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist()))
        if a.dtype != b.dtype:
            if a.dtype.fields is None and b.dtype.fields is None and a.dtype.kind == b.dtype.kind:
                b = b.astype(native(b.dtype))
                a = a.astype(native(a.dtype))
            if a.dtype != b.dtype:
                return False
        return a.tobytes() == b.tobytes()
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, np.floating)) and a != a:
        return b != b
    return a == b


def keys_for(shape, esize, max_bytes):
    if int(np.prod(shape)) * esize <= max_bytes or not shape:
        return [Ellipsis]
    mid = tuple(slice(s // 2, s // 2 + 512) for s in shape)
    first = (slice(0, 1),) + tuple(slice(None) for _ in shape[1:])
    last = (slice(shape[0] - 1, shape[0]),) + tuple(slice(None) for _ in shape[1:])
    return [mid, first, last]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url')
    ap.add_argument('local')
    ap.add_argument('--max-mb', type=float, default=64)
    a = ap.parse_args()
    max_bytes = int(a.max_mb * 2 ** 20)
    bad = []
    n = 0
    t = time.perf_counter()
    if a.url == 'fileobj':
        from fastnc import _remote
        _remote._POISON = 0xA5                         # bytes read before fetched: wrong values
    remote = (lambda: open(a.local, 'rb')) if a.url == 'fileobj' else (lambda: a.url)
    r, l = fastnc.Dataset(remote()), fastnc.Dataset(a.local)
    h = h5py.File(a.local, 'r')
    r.set_auto_maskandscale(False)
    l.set_auto_maskandscale(False)
    for gr, gl in zip(r._walk(), l._walk()):
        n += 1
        if gr.path != gl.path or list(gr.dimensions) != list(gl.dimensions) \
                or [len(d) for d in gr.dimensions.values()] != [len(d) for d in gl.dimensions.values()]:
            bad.append(('dims', gr.path))
        if not same([gr.getncattr(k) for k in gr.ncattrs()], [gl.getncattr(k) for k in gl.ncattrs()]) \
                or gr.ncattrs() != gl.ncattrs():
            bad.append(('group attrs', gr.path))
        if list(gr.variables) != list(gl.variables):
            bad.append(('variables', gr.path))
        for k, vr in gr.variables.items():
            vl = gl.variables[k]
            n += 1
            if vr.ncattrs() != vl.ncattrs() or not same([vr.getncattr(x) for x in vr.ncattrs()],
                                                        [vl.getncattr(x) for x in vl.ncattrs()]):
                bad.append(('attrs', k))
            if vr.dimensions != vl.dimensions or vr.shape != vl.shape or vr.dtype != vl.dtype:
                bad.append(('shape/dtype', k))
                continue
            esize = vr.dtype.itemsize if isinstance(vr.dtype, np.dtype) else 8
            hpath = ('%s/%s' % (gr.path.rstrip('/'), k))
            for key in keys_for(vr.shape, esize, max_bytes):
                got = vr[key]
                if not same(got, vl[key]):
                    bad.append(('values vs fastnc local', k, key))
                if hpath in h and isinstance(h[hpath], h5py.Dataset) and vr.dtype != str \
                        and vr.dtype.fields is None:            # compounds: netCDF's own dtype
                    want = h[hpath][key if key is not Ellipsis else ()]
                    if not same(got, want):
                        bad.append(('values vs h5py', k, key))
    s = r._source
    print('Dataset: %d groups+variables, %d requests, %.1f MB fetched, %.1f s' % (
        n, s.requests, s.fetched / 1e6, time.perf_counter() - t))
    r.close()
    l.close()
    # the h5py front end
    t = time.perf_counter()
    fr = fastnc.File(remote())
    fl = fastnc.File(a.local)
    m = 0

    def visit(name, obj):
        nonlocal m
        m += 1
        ho = h[name]
        lo = fl[name]                                   # attributes: as fastnc reads them locally
        if list(obj.attrs) != list(lo.attrs) or sorted(obj.attrs) != sorted(ho.attrs) or                 not all(same(obj.attrs[x], lo.attrs[x]) for x in lo.attrs):
            bad.append(('File attrs', name))
        if isinstance(ho, h5py.Dataset):
            if obj.shape != ho.shape or obj.dtype != ho.dtype or obj.chunks != ho.chunks:
                bad.append(('File shape/dtype/chunks', name))
                return
            for key in keys_for(ho.shape, ho.dtype.itemsize, max_bytes):
                k2 = () if key is Ellipsis else key
                if not same(obj[k2], ho[k2]):
                    bad.append(('File values', name, key))
    fr.visititems(visit)
    print('File: %d objects, %d requests, %.1f MB fetched, %.1f s' % (
        m, fr._source.requests, fr._source.fetched / 1e6, time.perf_counter() - t))
    fr.close()
    fl.close()
    print('differences:', bad if bad else 'none')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
