"""A file's CDL header, as `ncdump -h` prints it, made from what fastnc read
(netCDF4-python's Dataset.tocdl runs the ncdump program; this needs none)."""
import math
import os

import numpy as np

_TYPE = {'i1': 'byte', 'u1': 'ubyte', 'i2': 'short', 'u2': 'ushort', 'i4': 'int', 'u4': 'uint',
         'i8': 'int64', 'u8': 'uint64', 'f4': 'float', 'f8': 'double', 'S1': 'char'}
_SUFFIX = {'i1': 'b', 'u1': 'UB', 'i2': 's', 'u2': 'US', 'i4': '', 'u4': 'U', 'i8': 'LL', 'u8': 'ULL'}
_ESCAPED = set(' !"#$&\'()*,:;<=>?[\\]^`{|}~')    # (what ncdump escapes in names)


def name(s):
    """A name as ncdump writes it: characters CDL gives a meaning to escaped,
    and a leading digit."""
    return ''.join('\\' + c if c in _ESCAPED or (i == 0 and c.isdigit()) else c for i, c in enumerate(s))


def _float(x, digits, suffix):
    if math.isnan(x):
        return 'NaN' + suffix
    if math.isinf(x):
        return ('Infinity' if x > 0 else '-Infinity') + suffix
    s = '%.*g' % (digits, x)
    if '.' not in s:
        s = s.replace('e', '.e') if 'e' in s else s + '.'
    return s + suffix


_ESC = {chr(c): '\\%03o' % c for c in list(range(32)) + [127]}     # (other control characters: octal)
_ESC.update({'\\': '\\\\', '"': '\\"', "'": "\\'", '\b': '\\b', '\f': '\\f', '\r': '\\r', '\t': '\\t',
             '\v': '\\v', '\n': '\\n'})


def _text(s, split=False):
    """A char or string value as ncdump writes it in a header: quoted and
    escaped; split (char text of a classic-model file) a line a newline, as
    ncdump breaks it there (not in netCDF-4 / HDF5 / HDF4 files)."""
    esc = _ESC
    if not split:
        return ''.join(esc.get(c, c) for c in s).join('""')
    parts, cur = [], []
    for c in s:
        cur.append(esc.get(c, c))
        if c == '\n':
            parts.append(''.join(cur))
            cur = []
    if cur or not parts:
        parts.append(''.join(cur))
    return '",\n\t\t\t"'.join(parts).join('""')


def _values(v, nc_string=False, split=False):
    """An attribute's values as ncdump writes them, and the type it states
    (None where the values say it themselves). nc_string: the text is
    NC_STRING, not NC_CHAR (stated, and never split at newlines); split:
    char text is (a classic-model file)."""
    if isinstance(v, str):
        return (_text(v), 'string') if nc_string else (_text(v, split), None)
    if isinstance(v, bytes):
        return _text(v.decode('utf-8', 'replace'), split), None
    if isinstance(v, list):
        return ', '.join(_text(x, False) for x in v), 'string'
    a = np.asarray(v)
    k = a.dtype.newbyteorder('=').str[1:]
    if a.dtype.kind == 'U' or a.dtype.kind == 'O':
        return ', '.join(_text(str(x)) for x in a.reshape(-1).tolist()), 'string'
    if a.dtype.kind == 'S':
        return _text(b''.join(a.reshape(-1).tolist()).decode('utf-8', 'replace'), split), None
    flat = a.reshape(-1).tolist()
    if k == 'f4':
        return ', '.join(_float(float(np.float32(x)), 7, 'f') for x in flat), None
    if k == 'f8':
        return ', '.join(_float(x, 15, '') for x in flat), None
    if k in _SUFFIX:
        return ', '.join('%d%s' % (x, _SUFFIX[k]) for x in flat), None
    return ', '.join(str(x) for x in flat), None


def _stored(attrs, name, value):
    """An attribute's value as ncdump prints it: HDF4 text holding NULs as
    stored (netCDF-c gives it without them), else its value."""
    return getattr(attrs, 'stored', {}).get(name, value)


def _type_name(var):
    dt = var.datatype
    if dt is str or (getattr(dt, 'dtype', None) is str and getattr(dt, 'name', None) is None):
        return 'string'
    if isinstance(dt, np.dtype):
        return _TYPE.get(dt.newbyteorder('=').str[1:], str(dt))
    return name(getattr(dt, 'name', None) or str(dt))


def _types(group, pad):
    """The types: section of a group (compound, enum, vlen), as ncdump: in
    the order netCDF-c read them, kinds mixed, where the backend knows it."""
    out = []
    kinds = {}
    for kind in ('cmptypes', 'enumtypes', 'vltypes'):
        for n, t in getattr(group, kind, {}).items():
            kinds.setdefault(n, (kind, t))
    every = getattr(getattr(group, '_bg', None), 'all_types', None)
    order = [n for n, _ in every if n in kinds] if every is not None else list(kinds)
    for n in order:
        kind, t = kinds[n]
        if kind == 'cmptypes':
            out.append('%s  compound %s {' % (pad, name(n)))
            for f in t.dtype.names:
                ft = t.dtype.fields[f][0]
                base = ft.base if ft.shape else ft
                dims = '(%s)' % ', '.join(str(d) for d in ft.shape) if ft.shape else ''
                out.append('%s    %s %s%s ;' % (pad, _TYPE.get(base.newbyteorder('=').str[1:], str(base)), name(f), dims))
            out.append('%s  }; // %s' % (pad, name(n)))
        elif kind == 'enumtypes':
            members = ', '.join('%s = %s' % (name(k), v) for k, v in t.enum_dict.items())   # (as stored)
            base = _TYPE.get(np.dtype(t.dtype).newbyteorder('=').str[1:], str(t.dtype))
            out.append('%s  %s enum %s {%s} ;' % (pad, base, name(n), members))
        else:
            base = 'string' if t.dtype is str else _TYPE.get(np.dtype(t.dtype).newbyteorder('=').str[1:], str(t.dtype))
            out.append('%s  %s(*) %s ;' % (pad, base, name(n)))
    if out:
        out.insert(0, '%stypes:' % pad)
    return out


def _enum_name(var, value):
    dt = var.datatype
    for k, v in getattr(dt, 'enum_dict', {}).items():
        if v == value:
            return name(k)
    return str(value)


def _group(g, depth, split):
    pad = '  ' * depth
    out = _types(g, pad)
    if g.dimensions:
        out.append('%sdimensions:' % pad)
        for n, d in g.dimensions.items():
            if d.isunlimited():
                out.append('%s\t%s = UNLIMITED ; // (%d currently)' % (pad, name(n), len(d)))
            else:
                out.append('%s\t%s = %d ;' % (pad, name(n), len(d)))
    # (the file's variables as stored: those auto_complex reads as complex, by their parts)
    vs = [(n, getattr(v._bv, '_raw', v)) for n, v in g.variables.items()]
    if depth == 0 and getattr(g, 'disk_format', None) == 'HDF4':
        vs = [(v.name, v) for v in g.hdf4_datasets()]   # (every data set, names taken twice too)
    strings = g._bg.__dict__.get('_strings') if hasattr(g._bg, '__dict__') else None
    if vs:
        out.append('%svariables:' % pad)
        for n, v in vs:
            dims = '(%s)' % ', '.join(name(x) for x in v.dimensions) if v.dimensions else ''
            out.append('%s\t%s %s%s ;' % (pad, _type_name(v), name(n), dims))
            for a in v.ncattrs():
                val = _stored(v._bv.attrs, a, v.getncattr(a))
                if hasattr(v.datatype, 'enum_dict') and a == '_FillValue':
                    out.append('%s\t\t%s %s:%s = %s ;' % (pad, _type_name(v), name(n), name(a),
                                                         _enum_name(v, int(np.asarray(val)))))
                    continue
                is_str = getattr(v._bv.attrs, 'is_nc_string', None)
                text, typ = _values(val, isinstance(val, str) and is_str is not None and is_str(a), split)
                out.append('%s\t\t%s%s:%s = %s ;' % (pad, typ + ' ' if typ else '', name(n), name(a), text))
    atts = g.ncattrs()
    if atts:
        g._bg.attrs                                 # (parsed: which are NC_STRING is known then)
        strings = getattr(g._bg, '_strings', None) or set()
        out.append('')
        out.append('%s// %s attributes:' % (pad, 'global' if depth == 0 else 'group'))
        for a in atts:
            val = _stored(g._bg.attrs, a, g.getncattr(a))
            text, typ = _values(val, isinstance(val, str) and a in strings, split)
            out.append('%s\t\t%s:%s = %s ;' % (pad, typ + ' ' if typ else '', name(a), text))
    for n, sub in g.groups.items():
        out.append('')
        out.append('%sgroup: %s {' % (pad, name(n)))
        out.extend(_group(sub, depth + 1, split))
        out.append('%s  } // group %s' % (pad, name(n)))
    return out


def tocdl(ds):
    """The CDL header of an open fastnc.Dataset, as `ncdump -h` prints it."""
    base = os.path.splitext(os.path.basename(ds.filepath()))[0]
    split = ds.data_model != 'NETCDF4' and ds.disk_format != 'HDF4'
    lines = ['netcdf %s {' % name(base)] + _group(ds, 0, split) + ['}']
    return '\n'.join(lines) + '\n'
