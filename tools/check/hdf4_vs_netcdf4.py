"""fastnc against netCDF4-python (netCDF-c's HDF4 dispatch) and pyhdf on
HDF4 files: data model, dimensions, variables, attributes (names, values,
types), chunking, endianness, raw values whole and for a few selections.

    python tools/check/hdf4_vs_netcdf4.py [--warm] file.hdf ...

--warm loads fastnc's kernels first (else the first reads go through
zlib / numpy, as before numba is up). Prints only differences; exit
status 1 if any.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import fastnc  # noqa: E402
from fastnc import _chunks  # noqa: E402

try:
    import netCDF4
except ImportError:
    netCDF4 = None
try:
    from pyhdf.SD import SD
except ImportError:
    SD = None


def same_value(x, y):
    if type(x) is not type(y):
        return False
    if isinstance(x, (str, bytes)):
        return x == y
    x, y = np.asarray(x), np.asarray(y)
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def same_array(x, y):
    """Values equal (byte order aside; NaNs by their bits)."""
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return False
    if x.dtype.kind == 'S' or y.dtype.kind == 'S':
        return x.tobytes() == y.tobytes()
    return x.dtype.newbyteorder('=') == y.dtype.newbyteorder('=') and \
        x.astype(x.dtype.newbyteorder('=')).tobytes() == y.astype(y.dtype.newbyteorder('=')).tobytes()


def dimlen(d):
    try:
        return len(d)
    except RuntimeError:                            # (netCDF-c fails on some HDF4 dimensions)
        return None


def selections(shape):
    out = [Ellipsis]
    if len(shape) >= 1 and shape[0] > 2:
        out += [(slice(1, None, 2),), (shape[0] // 2,), (slice(None, None, -3),), ([0, shape[0] - 1],)]
    if len(shape) >= 2 and min(shape) > 3:
        out += [(slice(1, -1), slice(None, None, 3)), (Ellipsis, slice(2, 3)),
                tuple(slice(s // 3, s // 3 + max(1, s // 4)) for s in shape)]
    return out


def check(path, bad):
    def err(*a):
        bad.append(' '.join(str(x) for x in (os.path.basename(path),) + a))
        print(bad[-1][:300])
    b = fastnc.Dataset(path)
    b.set_auto_maskandscale(False)
    a = None
    if netCDF4 is not None:
        try:
            a = netCDF4.Dataset(path)
            a.set_auto_maskandscale(False)
        except OSError as e:
            print(os.path.basename(path), 'netCDF4 cannot open it:', e)
    nvals = 0
    if a is not None:
        for attr in ('data_model', 'file_format', 'disk_format'):
            if getattr(a, attr) != getattr(b, attr):
                err(attr, getattr(a, attr), getattr(b, attr))
        da = [(k, dimlen(d), d.isunlimited()) for k, d in a.dimensions.items()]
        db = [(k, len(d), d.isunlimited()) for k, d in b.dimensions.items()]
        if [x[0] for x in da] != [x[0] for x in db] or                 any(x != y for x, y in zip(da, db) if x[1] is not None):
            err('dimensions', da, db)
        if list(a.variables) != list(b.variables):
            err('variables', list(a.variables), list(b.variables))
        if a.ncattrs() != b.ncattrs():
            err('global attribute names', a.ncattrs(), b.ncattrs())
        for k in a.ncattrs():
            if k in b.ncattrs() and not same_value(a.getncattr(k), b.getncattr(k)):
                err('global attribute', k, repr(a.getncattr(k))[:100], repr(b.getncattr(k))[:100])
        for k, va in a.variables.items():
            vb = b.variables.get(k)
            if vb is None:
                continue
            try:
                va.shape
            except RuntimeError as e:
                print(os.path.basename(path), k, 'netCDF4 cannot give its shape:', e, vb.shape)
                continue
            for what, x, y in [('dimensions', va.dimensions, vb.dimensions), ('shape', va.shape, vb.shape),
                               ('dtype', va.dtype.newbyteorder('='), vb.dtype),
                               ('chunking', va.chunking(), vb.chunking()), ('endian', va.endian(), vb.endian()),
                               ('filters', va.filters(), vb.filters()), ('attributes', va.ncattrs(), vb.ncattrs())]:
                if x != y:
                    err(k, what, x, y)
            for n in va.ncattrs():
                if n in vb.ncattrs() and not same_value(va.getncattr(n), vb.getncattr(n)):
                    err(k, 'attribute', n, repr(va.getncattr(n))[:100], repr(vb.getncattr(n))[:100])
            for sel in selections(va.shape):
                try:
                    want = va[sel]
                except Exception as e:                  # (netCDF-c fails past a variable's records)
                    if sel is Ellipsis:
                        print(os.path.basename(path), k, 'netCDF4 cannot read it:', e)
                    continue
                try:
                    got = vb[sel]
                except Exception as e:
                    err(k, sel, 'fastnc error', repr(e))
                    continue
                nvals += 1
                if not same_array(want, got):
                    err(k, sel, 'values differ', np.asarray(want).ravel()[:5], np.asarray(got).ravel()[:5])
        a.close()
    if SD is not None:
        h = SD(path)
        for name, info in h.datasets().items():
            vb = b.variables.get(name)
            if vb is None:
                continue
            s = h.select(info[3])
            try:
                want = s.get()
            except Exception:
                continue                                # (pyhdf: no little-endian types)
            want = np.asarray(want)
            got = vb[tuple(slice(0, n) for n in want.shape)]     # (pyhdf: a variable's own records)
            nvals += 1
            if np.asarray(want).dtype.kind == 'S' or vb.dtype.kind == 'S':
                ok = np.asarray(want).tobytes() == np.asarray(got).tobytes() or \
                    [x for x in np.asarray(want).ravel().tolist()] == [x.decode('latin-1') for x in np.asarray(got).ravel().tolist()]
            else:
                ok = same_array(np.asarray(want).astype(vb.dtype), got)
            if not ok:
                err(name, 'differs from pyhdf', np.asarray(want).ravel()[:5], np.asarray(got).ravel()[:5])
        h.end()
    b.close()
    print('%s: %d variables, %d reads compared' % (os.path.basename(path), len(b.variables), nvals))


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--warm' in args:
        args.remove('--warm')
        _chunks.warm()
    bad = []
    for p in args:
        check(p, bad)
    print('%d differences' % len(bad))
    sys.exit(1 if bad else 0)
