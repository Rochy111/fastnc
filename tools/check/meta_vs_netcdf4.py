"""fastnc.Dataset's metadata and raw values against netCDF4-python."""
import os, sys
import numpy as np, netCDF4
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import fastnc


def attrs_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.tobytes() == b.tobytes()
    if isinstance(a, np.generic):
        return a.dtype == b.dtype and a.tobytes() == b.tobytes()
    return a == b


def cmp_group(g, r, bad, values):
    if list(g.dimensions) != list(r.dimensions):
        bad.append('%s dims %s != %s' % (g.path, list(g.dimensions)[:8], list(r.dimensions)[:8]))
    for n, d in r.dimensions.items():
        if n in g.dimensions and (len(g.dimensions[n]) != len(d) or g.dimensions[n].isunlimited() != d.isunlimited()):
            bad.append('%s dim %s: %d/%s != %d/%s' % (g.path, n, len(g.dimensions[n]), g.dimensions[n].isunlimited(), len(d), d.isunlimited()))
    if list(g.variables) != list(r.variables):
        bad.append('%s vars %s != %s' % (g.path, list(g.variables)[:6], list(r.variables)[:6]))
    if g.ncattrs() != r.ncattrs():
        bad.append('%s attrs %s != %s' % (g.path, g.ncattrs()[:6], r.ncattrs()[:6]))
    for k in r.ncattrs():
        if k in g.ncattrs() and not attrs_equal(g.getncattr(k), r.getncattr(k)):
            bad.append('%s attr %s: %r != %r' % (g.path, k, g.getncattr(k), r.getncattr(k)))
    for n, rv in r.variables.items():
        v = g.variables.get(n)
        if v is None:
            continue
        where = '%s/%s' % (g.path.rstrip('/'), n)
        if v.dimensions != rv.dimensions:
            bad.append('%s dims %s != %s' % (where, v.dimensions, rv.dimensions))
        if v.shape != rv.shape:
            bad.append('%s shape %s != %s' % (where, v.shape, rv.shape))
        if v.dtype != rv.dtype:
            bad.append('%s dtype %s != %s' % (where, v.dtype, rv.dtype))
        if v.ncattrs() != rv.ncattrs():
            bad.append('%s attrs %s != %s' % (where, v.ncattrs(), rv.ncattrs()))
        for k in rv.ncattrs():
            if k in v.ncattrs() and not attrs_equal(v.getncattr(k), rv.getncattr(k)):
                bad.append('%s attr %s: %r (%s) != %r (%s)' % (where, k, v.getncattr(k), type(v.getncattr(k)).__name__, rv.getncattr(k), type(rv.getncattr(k)).__name__))
        for m in ('chunking', 'endian', 'filters'):
            try:
                a, b = getattr(v, m)(), getattr(rv, m)()
            except Exception:
                continue
            if a != b:
                bad.append('%s %s() %r != %r' % (where, m, a, b))
        if values:
            rv.set_auto_maskandscale(False); v.set_auto_maskandscale(False)
            try:
                want = rv[...]
            except Exception as e:
                continue
            try:
                got = v[...]
            except Exception as e:
                bad.append('%s read: %r' % (where, e)); continue
            w = np.asarray(want); gg = np.asarray(got)
            if w.dtype.kind == 'O' or gg.dtype.kind == 'O':
                ok = w.shape == gg.shape and all((np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y) for x, y in zip(w.ravel(), gg.ravel()))
            else:
                ok = w.shape == gg.shape and w.tobytes() == gg.astype(w.dtype).tobytes()
            if not ok:
                bad.append('%s values differ' % where)
    for n, rg in r.groups.items():
        if n not in g.groups:
            bad.append('%s group %s missing' % (g.path, n)); continue
        cmp_group(g.groups[n], rg, bad, values)


if __name__ == '__main__':
    values = '--values' in sys.argv
    for p in [a for a in sys.argv[1:] if not a.startswith('--')]:
        try:
            r = netCDF4.Dataset(p)
        except Exception as e:
            print('%-45s netCDF4 cannot open: %s' % (os.path.basename(p)[:45], str(e)[:60])); continue
        try:
            g = fastnc.Dataset(p)
        except Exception as e:
            import traceback; traceback.print_exc()
            print('%-45s fastnc CRASH %r' % (os.path.basename(p)[:45], e)); continue
        bad = []
        if g.data_model != r.data_model:
            bad.append('data_model %s != %s' % (g.data_model, r.data_model))
        try:
            cmp_group(g, r, bad, values)
        except Exception as e:
            import traceback; traceback.print_exc()
            bad.append('CRASH %r' % e)
        print('%-45s %s' % (os.path.basename(p)[:45], 'OK' if not bad else 'BAD %d' % len(bad)))
        for b in bad[:8]:
            print('     ', b[:220])
