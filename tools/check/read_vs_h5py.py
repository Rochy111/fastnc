"""fastnc's raw reads against h5py: every dataset of the given files, whole
and with a few selections; timing of whole-file reads."""
import glob, mmap, os, sys, time
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools', 'check'))
from fastnc import _hdf5, _h5read, _pool
from hdf5_vs_h5py import walk

_pool.fast_thread()


def same(a, b):
    if isinstance(a, (str, bytes)) or isinstance(b, (str, bytes)):
        dec = lambda x: x.decode('utf-8', 'surrogateescape') if isinstance(x, bytes) else x
        return dec(a) == dec(b)                  # the reader gives str; h5py's API bytes
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == 'S' and b.dtype.kind == 'S':
        return a.tolist() == b.tolist()          # h5py strips trailing NULs of scalars
    if a.dtype.kind == 'O' or b.dtype.kind == 'O':
        return [x.decode() if isinstance(x, bytes) else x for x in a.ravel().tolist()] == \
               [x.decode() if isinstance(x, bytes) else x for x in b.ravel().tolist()] if a.dtype.kind == b.dtype.kind or True else False
    if a.dtype.fields is not None and b.dtype.fields is not None:
        return list(a.dtype.names) == list(b.dtype.names) and all(same(a[n], b[n]) for n in b.dtype.names)
    if b.dtype.kind in 'bc' or a.dtype.fields is not None:
        return a.tobytes() == np.ascontiguousarray(b).astype(_h5read.native(b.dtype)).tobytes() or \
               a.tobytes() == np.ascontiguousarray(b).tobytes()
    return a.tobytes() == np.ascontiguousarray(b.astype(_h5read.native(b.dtype))).tobytes()


def selections(shape):
    out = [()]
    if len(shape) >= 1 and shape[0] > 0:
        out.append((shape[0] // 2,))
        out.append((slice(None, None, 3),))
    if len(shape) >= 2 and all(shape):
        out.append((Ellipsis, slice(shape[-1] // 3, shape[-1] // 3 + 7), slice(None, None, 2))
                   if len(shape) >= 2 else ())
        out.append(tuple(slice(s // 4, s // 4 + max(1, s // 3)) for s in shape))
        out.append((slice(None, None, -2),))
    return out


for path in sys.argv[1:]:
    with open(path, 'rb') as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    src = np.frombuffer(mm, np.uint8)
    f = _hdf5.H5File(mm)
    objs = {}
    walk(f, f.root, '/', objs)
    h = h5py.File(path, 'r')
    bad = []
    nds = 0
    dsets = [(p, o) for p, o in objs.items() if isinstance(o, _hdf5.Dataset)]
    for p, o in dsets:
        r = h[p]
        rd = _h5read.Reader(f, o, src)
        for sel in selections(o.shape):
            try:
                want = r[sel] if r.shape is not None else None
            except Exception as e:
                continue
            try:
                got = rd.read(sel)
            except NotImplementedError as e:
                bad.append('%s: %s' % (p, e)); break
            except Exception as e:
                import traceback; traceback.print_exc()
                bad.append('%s %s: %r' % (p, sel, e)); break
            if want is None:
                continue
            if not same(got, want):
                bad.append('%s %s differ' % (p, sel))
        nds += 1
    # timing: every dataset whole, fastnc vs h5py
    def ours():
        for p, o in dsets:
            _h5read.Reader(f, o, src).read(())
    def theirs():
        for p, o in dsets:
            try:
                h[p][()]
            except Exception:
                pass
    tf = []; th = []
    for _ in range(3):
        t = time.perf_counter(); 
        try: ours()
        except Exception: pass
        tf.append(time.perf_counter() - t)
        t = time.perf_counter(); theirs(); th.append(time.perf_counter() - t)
    print('%-45s %3d datasets  %s   fastnc %7.1f ms  h5py %7.1f ms  (%.1fx)' % (
        os.path.basename(path)[:45], nds, 'OK' if not bad else 'BAD %d' % len(bad),
        min(tf) * 1e3, min(th) * 1e3, min(th) / min(tf)))
    for b in bad[:6]:
        print('     ', b[:180])
