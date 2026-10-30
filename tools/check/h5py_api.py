"""fastnc.File against h5py.File: keys and order, kinds, attributes (names,
order, values, types), dataset properties, and values."""
import os, sys
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import fastnc


def eq(a, b):
    if isinstance(b, h5py.Reference) or isinstance(a, fastnc._h5py.Reference):
        return True
    if isinstance(b, h5py.Empty):
        return isinstance(a, fastnc._h5py.Empty)
    if type(a) is not type(b) and not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
        return False
    if isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False
        if b.dtype.kind == 'O':
            return all(eq(x, y) for x, y in zip(a.ravel(), b.ravel()))
        if b.dtype.fields:
            return all(eq(a[n], b[n]) for n in b.dtype.names)
        return a.dtype.kind == b.dtype.kind and a.tobytes() == b.astype(a.dtype).tobytes()
    if isinstance(b, np.generic):
        return a.dtype == b.dtype.newbyteorder('=') and a.tobytes() == np.asarray(b).astype(a.dtype).tobytes() or \
            (b.dtype.fields is not None and all(eq(a[n], b[n]) for n in b.dtype.names))
    return a == b


def check(p):
    bad = []
    r = h5py.File(p, 'r'); g = fastnc.File(p)
    names = []
    r.visititems(lambda n, o: names.append(n))
    mine = []
    g.visititems(lambda n, o: mine.append(n))
    if names != mine:
        bad.append('visit order %s != %s' % (mine[:5], names[:5]))
    for n in ['/'] + names:
        ro, go = r[n], g[n]
        if type(ro).__name__ != type(go).__name__:
            bad.append('%s kind %s != %s' % (n, type(go).__name__, type(ro).__name__)); continue
        if list(go.attrs.keys()) != list(ro.attrs.keys()):
            bad.append('%s attr keys %s != %s' % (n, list(go.attrs)[:5], list(ro.attrs)[:5]))
        for k in ro.attrs.keys():
            try:
                rv = ro.attrs[k]
            except Exception:
                continue
            if k in ('DIMENSION_LIST', 'REFERENCE_LIST'):
                continue
            gv = go.attrs[k]
            if not eq(gv, rv):
                bad.append('%s attr %s: %r (%s) != %r (%s)' % (n, k, gv, type(gv).__name__, rv, type(rv).__name__))
        if isinstance(ro, h5py.Group):
            if list(go.keys()) != list(ro.keys()):
                bad.append('%s keys %s != %s' % (n, list(go)[:5], list(ro)[:5]))
            continue
        if isinstance(ro, h5py.Datatype):
            if go.dtype != ro.dtype:
                bad.append('%s named dtype' % n)
            continue
        for prop in ('shape', 'maxshape', 'dtype', 'chunks', 'compression', 'compression_opts', 'shuffle',
                     'fletcher32', 'scaleoffset', 'size', 'ndim', 'fillvalue'):
            try:
                a, b = getattr(go, prop), getattr(ro, prop)
            except Exception as e:
                continue
            ok = eq(a, b) if prop == 'fillvalue' else (a == b)
            if prop == 'dtype' and a != b:
                ok = a.newbyteorder('=') == b.newbyteorder('=') if a.kind != 'O' else (h5py.check_string_dtype(b) is not None) == ('vlen' in (a.metadata or {}) and a.metadata['vlen'] is str)
            if not ok:
                bad.append('%s %s %r != %r' % (n, prop, a, b))
        try:
            want = ro[()]
        except Exception:
            continue
        try:
            got = go[()]
        except NotImplementedError as e:
            bad.append('%s: %s' % (n, e)); continue
        except Exception as e:
            bad.append('%s read: %r' % (n, e)); continue
        if not eq(got, want):
            bad.append('%s values differ (%s vs %s)' % (n, type(got).__name__, type(want).__name__))
    return bad


if __name__ == '__main__':
    for p in sys.argv[1:]:
        try:
            bad = check(p)
        except Exception as e:
            import traceback; traceback.print_exc()
            bad = ['CRASH %r' % e]
        print('%-45s %s' % (os.path.basename(p)[:45], 'OK' if not bad else 'BAD %d' % len(bad)))
        for b in bad[:6]:
            print('     ', b[:200])
