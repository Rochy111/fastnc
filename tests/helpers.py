"""What the reference generator (tools/make_references.py) and the tests
share: how a value read from a file is reduced to a digest, how an
attribute value is written as JSON without losing its type, the standard
selections, and where the reference files and data files are.

The generator imports this module, so a reference and the digest a test
computes from fastnc's result come from the same code. It needs numpy
only: no h5py, netCDF4 or fastnc."""
import gzip
import hashlib
import json
import os
import sys

import numpy as np

TESTS = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(TESTS, 'data')
ROOT = os.path.dirname(TESTS)

REFERENCE_SETS = ('corpus', 'synthetic', 'real')   # tests/data, corpus/synthetic, the big files
NATIVE = '<' if sys.byteorder == 'little' else '>'
INLINE_BYTES = 256              # attribute values up to this size are kept whole, beyond it a sha256


# ------------------------------------------------------------ dtypes

def native(dt):
    """The same type in native byte order (fields too)."""
    return dt.newbyteorder('=') if dt.byteorder not in ('=', '|') or dt.names else dt


def dtype_str(dt):
    """A dtype written in native-order form: '<f4' for both '<f4' and
    '>f4' on a little-endian machine, '|O' for objects, the field list for
    compound types (offsets and padding included, as numpy's descr)."""
    if not isinstance(dt, np.dtype):
        return 'str' if dt is str else repr(dt)             # netCDF4-python's NC_STRING dtype is `str`
    dt = native(dt)
    if dt.names is not None or dt.subdtype is not None:
        return str(_descr(dt))
    return dt.str


def _descr(dt):
    if dt.names is None:
        if dt.subdtype is not None:
            return [dtype_str(dt.subdtype[0]), list(dt.subdtype[1])]
        return dtype_str(dt)
    return [[n, _descr(dt.fields[n][0]), int(dt.fields[n][1])] for n in dt.names] + [int(dt.itemsize)]


def byteorder(dt):
    """'<', '>' or '|': the byte order the reader's dtype states."""
    if not isinstance(dt, np.dtype):
        return '|'
    if dt.names is not None:
        orders = {byteorder(dt.fields[n][0]) for n in dt.names} - {'|'}
        return orders.pop() if len(orders) == 1 else '|'
    if dt.subdtype is not None:
        return byteorder(dt.subdtype[0])
    return NATIVE if dt.byteorder == '=' else dt.byteorder


def dtype_meta(dt):
    """h5py's extra type information carried in dtype.metadata (vlen
    str / bytes / base type, enum mapping, reference kind), JSON-safe; or
    None. Compared so that fastnc.File's dtypes say what h5py's say."""
    md = getattr(dt, 'metadata', None) if isinstance(dt, np.dtype) else None
    if not md:
        return None
    out = {}
    for k in sorted(md):
        v = md[k]
        if isinstance(v, type):
            out[k] = v.__name__
        elif isinstance(v, np.dtype):
            out[k] = dtype_str(v)
        elif isinstance(v, dict):
            out[k] = {str(a): int(b) if isinstance(b, (int, np.integer)) else repr(b) for a, b in v.items()}
        else:
            out[k] = repr(v)
    return out


# ------------------------------------------------------------ values

def _sha(a):
    """sha256 of an array's bytes in native byte order, C order. Compound
    values lose their padding first: what lies between fields is whatever
    the reader's buffer held (netCDF4-python builds compound types with
    align=True and leaves the padding uninitialised), so each field is
    hashed on its own, in field order, recursively."""
    h = hashlib.sha256()
    _feed(h, np.asarray(a))
    return h.hexdigest()


def _feed(h, a):
    """Add an array's values to hash `h`: whole if it has no fields, else
    field by field (each field's values for every point, C order)."""
    if a.dtype.names:
        for n in a.dtype.names:
            _feed(h, a[n])
        return
    if a.dtype.subdtype is not None:                         # a subarray type outside a compound
        a = a.view(a.dtype.subdtype[0]).reshape(a.shape + a.dtype.subdtype[1])
    if a.dtype.byteorder not in ('=', '|'):
        a = a.astype(native(a.dtype))
    a = np.ascontiguousarray(a)
    if a.size:
        h.update(memoryview(a.reshape(-1).view(np.uint8)))


def _bytes(x):
    """An array's or scalar's values as bytes: native byte order, compound
    fields one after another (all points of a field, then the next), no
    padding."""
    a = np.asarray(x)
    if a.dtype.names:
        return b''.join(_bytes(a[n]) for n in a.dtype.names)
    if a.dtype.byteorder not in ('=', '|'):
        a = a.astype(native(a.dtype))
    return np.ascontiguousarray(a).tobytes()


def canon(x, loose=False):
    """A JSON-able canonical form of a non-numeric value (object arrays of
    vlen strings / sequences, str, bytes, lists), whose sha256 is its
    digest. `loose` decodes bytes as UTF-8, so h5py's bytes and
    netCDF4-python's str can be compared for agreement; references store
    the strict form."""
    if isinstance(x, np.ndarray):
        if x.dtype.hasobject:                                # vlen, or a compound with vlen / reference fields
            return ['O', list(x.shape), [canon(e, loose) for e in x.reshape(-1).tolist()]]
        if loose and x.dtype.kind == 'S':
            return ['O', list(x.shape), [canon(e, loose) for e in x.reshape(-1).tolist()]]
        if loose and x.dtype.kind == 'U':
            return ['O', list(x.shape), [canon(str(e), loose) for e in x.reshape(-1).tolist()]]
        return ['a', dtype_str(x.dtype), list(x.shape), _sha(x)]
    if isinstance(x, (bytes, np.bytes_)):
        x = bytes(x)
        return ['s', x.decode('utf-8', 'surrogateescape')] if loose else ['b', x.hex()]
    if isinstance(x, str):
        return ['s', str(x)]
    if isinstance(x, np.generic):
        if x.dtype.hasobject:
            return ['n', dtype_str(x.dtype), canon(x.item(), loose)]
        return ['n', dtype_str(x.dtype), _sha(np.asarray(x))]
    if type(x).__name__ in ('Reference', 'RegionReference'):  # h5py's: no address, only whether it is set
        return ['ref', type(x).__name__, bool(x)]
    if isinstance(x, bool) or x is None:
        return ['k', x]
    if isinstance(x, int):
        return ['i', x]
    if isinstance(x, float):
        return ['f', float.hex(x)]
    if isinstance(x, (list, tuple)):
        return ['l', [canon(e, loose) for e in x]]
    return ['r', repr(x)]


def canon_sha(x, loose=False):
    return hashlib.sha256(json.dumps(canon(x, loose), ensure_ascii=True,
                                     separators=(',', ':')).encode()).hexdigest()


def type_name(x):
    """The Python type of a result, the same under numpy 1 and 2: numpy
    scalars as 'np.float32', 'np.bool' (numpy 1 calls it bool_), 'np.bytes'."""
    name = type(x).__name__
    return 'np.' + name.rstrip('_') if isinstance(x, np.generic) else name


def digest(x):
    """What a read returned, as a JSON-able dict: the Python type, shape,
    dtype in native-order form and the byte order the reader gave it, and
    the sha256 of the values in native byte order. A masked array adds the
    mask's digest, the number of masked points, the fill value, and digests
    the data with masked points set to zero (what lies under a mask is not
    part of netCDF4-python's answer). Object arrays (vlen) and strings are
    digested through `canon`."""
    out = {'type': type_name(x)}
    if type(x).__name__ == 'Empty':                          # h5py.Empty: a null dataspace
        out['dtype'] = dtype_str(x.dtype)
        return out
    if isinstance(x, np.ma.MaskedArray):
        m = _any_mask(np.ma.getmaskarray(x))
        data = np.array(np.ma.getdata(x), copy=True)
        if m.any():
            data[m] = np.zeros((), data.dtype)
        out.update(_plain(data))
        out['mask_sha256'] = _sha(m.astype(np.uint8))
        out['nmasked'] = int(m.sum())
        try:
            out['fill_value'] = encode(x.fill_value)
        except AttributeError:                               # np.ma.masked (a fully masked scalar) has none
            out['fill_value'] = None
        return out
    if isinstance(x, (str, bytes, list, tuple)) or x is None:
        out['sha256'] = canon_sha(x)
        return out
    out.update(_plain(np.asarray(x)))
    return out


def _any_mask(m):
    """A compound type's mask (one flag per field) as one flag per point:
    masked if any field is."""
    if not m.dtype.names:
        return m
    r = np.zeros(m.shape, bool)
    for n in m.dtype.names:
        sub = _any_mask(m[n])
        if sub.ndim > r.ndim:
            sub = sub.reshape(m.shape + (-1,)).any(-1)
        r |= sub
    return r


def _plain(a):
    d = {'shape': list(a.shape), 'dtype': dtype_str(a.dtype), 'byteorder': byteorder(a.dtype)}
    if a.dtype.hasobject:
        d['sha256'] = canon_sha(a)
    else:
        d['sha256'] = _sha(a)
    md = dtype_meta(a.dtype)
    if md:
        d['dtype_meta'] = md
    return d


def loose_digest(x):
    """A digest for comparing two readers' values only (never stored as
    a reference): bytes decoded as str, fixed and vlen strings alike."""
    if isinstance(x, np.ma.MaskedArray):
        x = np.ma.getdata(x)
    if type(x).__name__ == 'Empty':
        return 'Empty'
    a = np.asarray(x) if not isinstance(x, (str, bytes, list)) else x
    if isinstance(a, np.ndarray) and a.dtype.kind not in 'OSU' and not a.dtype.hasobject:
        return [list(a.shape), dtype_str(a.dtype), _sha(a)]
    return canon_sha(a, loose=True)


def same_values(ours, ref):
    """Does digest `ours` (from fastnc) equal reference digest `ref`?
    Everything but the byte order flag counts: fastnc returns native byte
    order where h5py may return the file's. Returns (ok, message)."""
    diff = [k for k in sorted(set(ours) | set(ref)) if k != 'byteorder' and ours.get(k) != ref.get(k)]
    if not diff:
        return True, ''
    return False, '; '.join('%s: got %r, want %r' % (k, ours.get(k), ref.get(k)) for k in diff)


# ------------------------------------------------------------ attribute values

def encode(x):
    """An attribute (or fill) value as JSON without losing what it is:
    str, bytes, numpy scalar of a given type, ndarray, list of str
    (netCDF4-python's NC_STRING arrays), h5py references. Numbers keep
    their exact bytes (hex, native byte order); `repr` is for people and
    not compared."""
    if isinstance(x, np.generic):                            # before bytes/str: np.bytes_ is a bytes too
        out = {'t': 'np', 'type': type_name(x), 'dtype': dtype_str(x.dtype)}
        if x.dtype.hasobject:                                # a compound with reference / vlen fields
            out['items'] = encode(x.item())
        else:
            out['hex'] = _bytes(x).hex()
        md = dtype_meta(x.dtype)
        if md:
            out['dtype_meta'] = md
        out['repr'] = _human(x)
        return out
    if isinstance(x, np.ndarray):
        out = {'t': 'nd', 'type': type(x).__name__, 'dtype': dtype_str(x.dtype), 'shape': list(x.shape)}
        md = dtype_meta(x.dtype)
        if md:
            out['dtype_meta'] = md
        if x.dtype.hasobject:
            items = [encode(e) for e in x.reshape(-1).tolist()]
            if len(items) <= 32:
                out['items'] = items
            else:
                out['sha256'] = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
        elif x.nbytes <= INLINE_BYTES:
            out['hex'] = _bytes(x).hex()
            out['repr'] = _human(x)
        else:
            out['sha256'] = _sha(x)
        return out
    if isinstance(x, str):
        return {'t': 'str', 'v': x}
    if isinstance(x, bytes):
        return {'t': 'bytes', 'v': x.hex()}
    if isinstance(x, bool):
        return {'t': 'bool', 'v': x}
    if isinstance(x, int):
        return {'t': 'int', 'v': x}
    if isinstance(x, float):
        return {'t': 'float', 'v': float.hex(x)}
    if isinstance(x, list):
        return {'t': 'list', 'v': [encode(e) for e in x]}
    if isinstance(x, tuple):
        return {'t': 'tuple', 'v': [encode(e) for e in x]}
    if x is None:
        return {'t': 'none'}
    name = type(x).__name__
    if name == 'Empty':
        return {'t': 'Empty', 'dtype': dtype_str(x.dtype)}
    if name in ('Reference', 'RegionReference'):
        return {'t': name, 'v': bool(x)}
    return {'t': 'other', 'type': name, 'repr': repr(x)}


def _human(x):
    try:
        v = np.asarray(x).tolist()
        s = json.dumps(v, default=lambda o: o.hex() if isinstance(o, bytes) else repr(o))
        return s if len(s) <= 200 else s[:197] + '...'
    except Exception:
        return '?'


def strip_repr(e):
    """An encoded value without its human-readable parts, for comparing."""
    if isinstance(e, dict):
        return {k: strip_repr(v) for k, v in e.items() if k != 'repr'}
    if isinstance(e, list):
        return [strip_repr(v) for v in e]
    return e


def jsonable(x):
    """Plain metadata (filters(), chunking(), h5py's compression_opts ...)
    as JSON: numpy scalars to Python, tuples to lists."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


# ------------------------------------------------------------ selections

def selections(shape):
    """The standard selections read from every variable besides the whole:
    a strided slice of every dimension (step 2, from a third of the way
    in, a bounded number of points), an integer index on the first, and
    a window of the last two (leading dimensions cut to their first
    element). Keys are lists of ['s', start, stop, step] / ['i', index],
    so they survive JSON; `key()` turns one back into a numpy index."""
    shape = tuple(int(n) for n in shape)
    if not shape:
        return {}
    strided = []
    for i, n in enumerate(shape):                            # at most 3 x 3 x 32 x 32 points: netCDF4-python
        a = n // 3                                           # reads strided slices point by point (~9 us each)
        cap = 32 if i >= len(shape) - 2 else 3
        strided.append(['s', a, min(n, a + 2 * cap), 2])
    sel = {'strided': strided}
    if shape[0] > 0:
        sel['index0'] = [['i', shape[0] // 2]] + [['s', None, None, 1] for _ in shape[1:]]
    win = [['s', 0, 1, 1] for _ in shape[:-2]]
    for n in shape[-2:]:
        a = n // 4
        win.append(['s', a, a + max(1, n // 2), 1])
    sel['window'] = win
    return sel


def key(sel):
    return tuple(int(k[1]) if k[0] == 'i' else slice(k[1], k[2], k[3]) for k in sel)


# ------------------------------------------------------------ reference files and data

def reference_path(which):
    return os.path.join(DATA, 'references_%s.json.gz' % which)


_REFS = {}


def references(which):
    """A reference set ('corpus', 'synthetic', 'real') as a dict, or None
    if it was never generated. Read once per process."""
    if which not in _REFS:
        p = reference_path(which)
        _REFS[which] = None
        if os.path.exists(p):
            with gzip.open(p, 'rt', encoding='utf-8') as f:
                _REFS[which] = json.load(f)
    return _REFS[which]


def search_dirs():
    """Where the files that are not in git may be: $FASTNC_REAL_DATA, the
    checkout, and the folders above it (a git worktree under
    .claude/worktrees/ sits three levels below the main checkout)."""
    dirs = []
    if os.environ.get('FASTNC_REAL_DATA'):
        dirs.append(os.environ['FASTNC_REAL_DATA'])
    d = ROOT
    for _ in range(4):
        dirs.append(d)
        d = os.path.dirname(d)
    return dirs


def locate(which, name):
    """The path of a reference set's file on this machine, or None."""
    if which == 'corpus':
        p = os.path.join(DATA, name)
        return p if os.path.exists(p) else None
    sub = ('corpus', 'synthetic') if which == 'synthetic' else ()
    for d in search_dirs():
        p = os.path.join(d, *sub, name)
        if os.path.exists(p):
            return p
    return None
