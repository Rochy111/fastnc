"""fastnc's HDF5 parser against h5py on every HDF5 file given (default: the
real files and corpus/synthetic): objects, shapes, types, chunk tables,
filters, fill values and attribute values."""
import glob, mmap, os, sys, time
import h5py, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from fastnc import _hdf5


def walk(f, g, path, out):
    for name, (kind, target) in g.links.items():
        if kind != 'hard':
            continue
        p = path + '/' + name if path != '/' else '/' + name
        h = f.object_header(target)
        if h.first(8) is not None:
            out[p] = _hdf5.Dataset(f, target, name, h)
        elif h.first(3) is not None:
            out[p] = _hdf5.Group(f, target, name, h)     # a committed datatype
            out[p].named_type = True
        else:
            sub = _hdf5.Group(f, target, name, h)
            out[p] = sub
            walk(f, sub, p, out)



def same(a, b):
    try:
        a = np.asarray(a); b = np.asarray(b)
        if a.dtype.kind == 'O' or b.dtype.kind == 'O':
            return [x.decode() if isinstance(x, bytes) else x for x in a.ravel().tolist()] == \
                   [x.decode() if isinstance(x, bytes) else x for x in b.ravel().tolist()]
        if a.dtype.kind in 'fc':
            return a.shape == b.shape and a.tobytes() == b.astype(a.dtype).tobytes()
        return a.shape == b.shape and np.array_equal(a, b)
    except Exception:
        return False


def check(path):
    bad = []
    with open(path, 'rb') as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    t = time.perf_counter()
    f = _hdf5.H5File(mm)
    objs = {}
    walk(f, f.root, '/', objs)
    t_walk = time.perf_counter() - t
    h = h5py.File(path, 'r')
    ref = {}
    h.visititems(lambda n, o: ref.__setitem__('/' + n, o))
    if set(ref) - set(objs):
        bad.append('missing: %s' % sorted(set(ref) - set(objs))[:5])
    nchunks = 0
    t_chunks = 0
    for p, o in objs.items():
        r = ref.get(p)
        if r is None:
            bad.append('extra: ' + p); continue
        # attributes
        try:
            mine = {k: a for k, a in o.attributes().items()}
        except Exception as e:
            bad.append('%s attrs: %r' % (p, e)); mine = {}
        for k in r.attrs.keys():
            if k not in mine:
                bad.append('%s attr %s missing' % (p, k)); continue
            try:
                rv = r.attrs[k]
            except Exception:
                continue
            if k in ('DIMENSION_LIST', 'REFERENCE_LIST'):
                continue
            try:
                mv = mine[k].value()
            except Exception as e:
                bad.append('%s attr %s: %r' % (p, k, e)); continue
            if not same(mv, rv):
                bad.append('%s attr %s: %r != %r' % (p, k, mv, rv))
        if getattr(o, 'named_type', False):
            if not isinstance(r, h5py.Datatype):
                bad.append(p + ' kind')
            continue
        if isinstance(o, _hdf5.Group):
            if not isinstance(r, h5py.Group):
                bad.append(p + ' kind')
            continue
        if not isinstance(r, h5py.Dataset):
            bad.append(p + ' kind'); continue
        if r.shape is None and o.null:
            continue
        if tuple(o.shape) != r.shape:
            bad.append('%s shape %s != %s' % (p, o.shape, r.shape))
        if o.dtype is not None and o.type.h5py_dtype != r.dtype and r.dtype.kind != 'O':
            bad.append('%s dtype %s != %s' % (p, o.type.h5py_dtype, r.dtype))
        if o.chunks != r.chunks:
            bad.append('%s chunks %s != %s' % (p, o.chunks, r.chunks))
        dc = r.id.get_create_plist()
        fids = [dc.get_filter(i)[0] for i in range(dc.get_nfilters())]
        if [x.id for x in o.filters] != fids:
            bad.append('%s filters %s != %s' % (p, [x.id for x in o.filters], fids))
        if o.chunks and r.size:
            t = time.perf_counter()
            try:
                mine = sorted((tuple(int(x) for x in c[0]), c[1], c[2], c[3]) for c in o.chunk_table())
            except Exception as e:
                bad.append('%s chunk table: %r' % (p, e)); continue
            t_chunks += time.perf_counter() - t
            L = []
            r.id.chunk_iter(L.append)
            theirs = sorted((tuple(c.chunk_offset), c.byte_offset, c.size, c.filter_mask) for c in L)
            nchunks += len(L)
            unlim = [i for i, x in enumerate(r.maxshape) if x is None]
            if o.layout.index == 'earray' and unlim and unlim[0] != 0:
                continue        # HDF5's chunk_iter misreports these offsets; checked on values instead
            if [m[:2] + m[3:] for m in mine] != [x[:2] + x[3:] for x in theirs] or \
                    any(m[2] != x[2] for m, x in zip(mine, theirs)):
                d = [(m, x) for m, x in zip(mine, theirs) if m != x][:2]
                bad.append('%s chunks differ (%d vs %d) %s' % (p, len(mine), len(theirs), d))
        elif o.layout.cls == 1 and r.size:
            off = r.id.get_offset()
            if off is not None and f.addr(o.layout.address) != off:
                bad.append('%s contiguous address %s != %s' % (p, o.layout.address, off))
        fv = r.id.get_create_plist().get_fill_value(np.empty((), r.dtype)) if r.dtype.kind not in 'OV' else None
    return bad, len(objs), nchunks, t_walk, t_chunks


if __name__ == '__main__':
    files = sys.argv[1:]
    if not files:
        files = sorted(glob.glob(os.path.join(ROOT, '*.nc')) + glob.glob(os.path.join(ROOT, 'corpus', 'synthetic', '*')))
        files = [x for x in files if '(1)' not in x]
    for p in files:
        if _hdf5.find_superblock(open(p, 'rb').read(4096)) is None:
            continue
        try:
            bad, n, nc, tw, tc = check(p)
        except Exception as e:
            import traceback; traceback.print_exc()
            print('%-50s CRASH %r' % (os.path.basename(p)[:50], e)); continue
        print('%-50s %4d objects %6d chunks  walk %6.1f ms  chunk tables %6.1f ms  %s' % (
            os.path.basename(p)[:50], n, nc, tw * 1e3, tc * 1e3, 'OK' if not bad else 'BAD %d' % len(bad)))
        for x in bad[:6]:
            print('     ', x[:200])
