"""netCDF-4 as netCDF-c reads it out of an HDF5 file: dimensions from
dimension scales (ids from _Netcdf4Dimid), each variable's dimensions
from its DIMENSION_LIST or _Netcdf4Coordinates (or phony dimensions when
a dataset has none attached), groups, user types, hidden attributes, and
attribute values in the form netCDF4-python gives them. Implements the
backend contract (docs/contracts.md, section 2) for HDF5 files.

Unlimited dimension lengths are worked out once, at open, from every
variable using them (h5netcdf asks every variable again for every
question, which made xarray take 10 s to open a WRF file).
"""
import threading
from collections.abc import Mapping

import numpy as np

from . import _h5read, _hdf5, _select

_ABSENT = object()
HIDDEN = frozenset(('_Netcdf4Dimid', '_Netcdf4Coordinates', '_nc3_strict', 'DIMENSION_LIST',
                    'REFERENCE_LIST', 'CLASS', 'NAME', '_NCProperties', '_IsNetcdf4',
                    '_SuperblockVersion', '_Format', '_NCZarr_Attr'))
NON_COORD = '_nc4_non_coord_'
NOT_A_VAR = b'This is a netCDF dimension but not a netCDF variable'


class Dim:
    __slots__ = ('name', 'size', 'unlimited', 'id', 'scale', 'group')

    def __init__(self, name, size, unlimited, did, scale, group):
        self.name, self.size, self.unlimited, self.id = name, size, unlimited, did
        self.scale, self.group = scale, group


def _attr_value(a, f):
    """An attribute's value as netCDF4-python's getncattr returns it."""
    t = a.dtype
    if a.shape is None:
        return ''                                   # null dataspace (an empty NC_CHAR)
    if t.cls == _hdf5.Datatype.STRING:
        raw = np.frombuffer(a.raw, np.dtype('S%d' % t.size)) if t.size else np.array([b''])
        if raw.size <= 1:
            s = raw[0] if raw.size else b''
            if a.name == '_FillValue' and isinstance(s, bytes):
                return s                            # (a char variable's: netCDF4 keeps it bytes)
            return s.decode('utf-8', 'replace') if isinstance(s, bytes) else str(s)
        return [x.decode('utf-8', 'replace') for x in raw.tolist()]
    if t.cls == _hdf5.Datatype.VLEN and t.vlen_string:
        v = a.value().ravel().tolist()
        return v[0] if len(v) == 1 else v
    v = a.value()
    if v.dtype.byteorder == '>':
        v = v.astype(v.dtype.newbyteorder('='))
    v = np.array(v).ravel()
    if v.size == 1:
        return v[0]
    return v


def nc_dtype(t):
    """The numpy dtype netCDF4-python gives an HDF5 type: compounds rebuilt
    from their members' netCDF types with align=True (fixed strings inside
    become NC_CHAR, one byte), enums as their base type."""
    if t.cls == _hdf5.Datatype.COMPOUND:
        # (a member of the old format may be an array without being of an array type)
        return np.dtype([(name, nc_dtype(mt), t.dtype.fields[name][0].shape if mt.cls != _hdf5.Datatype.ARRAY else ())
                         for name, off, mt in t.members], align=True)
    if t.cls == _hdf5.Datatype.STRING:
        return np.dtype('S1')
    if t.cls == _hdf5.Datatype.ARRAY:
        return np.dtype((nc_dtype(t.base), t.dims))
    if t.cls == _hdf5.Datatype.ENUM:
        return _h5read.native(t.base.dtype)
    if t.cls == _hdf5.Datatype.FLOAT and (t.dtype is None or t.size == 2 or _h5read.reduced(t)):
        return max(_h5read.float_target(t), np.dtype('f4'), key=lambda d: d.itemsize)   # (netCDF has no float16)
    if t.cls == _hdf5.Datatype.FIXED and t.precision < 8 * t.size:
        # (netCDF-C goes by its bits, not the bytes they are kept in)
        return np.dtype('%s%d' % ('i' if t.signed else 'u', next(n for n in (1, 2, 4, 8) if 8 * n >= t.precision)))
    if t.cls == _hdf5.Datatype.FIXED and t.dtype is None and t.size < 8:
        return _h5read.int_target(t)
    return _h5read.native(t.dtype)


def _shown(t):
    """Whether netCDF4 shows a dataset of this type: not references or opaque
    bytes, nor compounds of those or of variable-length values."""
    if t.cls in (_hdf5.Datatype.REFERENCE, _hdf5.Datatype.OPAQUE):
        return False
    if t.cls == _hdf5.Datatype.ARRAY:
        return _shown(t.base)
    if t.cls == _hdf5.Datatype.COMPOUND:
        return all(_shown(m[2]) and not m[2].holds_vlen() for m in t.members)
    return True


class Attrs(Mapping):
    """A dataset's attributes as netCDF shows them, each decoded when first
    asked for: masking wants a handful of names, and decoding all of them
    was the biggest Python cost of a masked read of a WRF file (1,770
    attributes of 200 variables)."""
    __slots__ = ('f', 'h', 'vals', 'found', 'none', 'safe', 'toma')

    def __init__(self, f, h):
        self.f, self.h, self.vals, self.found = f, h, {}, None
        self.none = set()                           # names it does not have: asked on every read
        self.safe = set()                           # those fit to cast to the variable's type
        self.toma = {}                              # (see _maskscale._toma)

    def among(self, names):
        """Which of names (the same tuple every call) it has; a read asks
        about the same few several times."""
        if self.found is None:
            self.found = [n for n in names if n in self]
        return self.found

    def get(self, name, default=None):
        v = self.vals.get(name, _ABSENT)
        if v is not _ABSENT:
            return v
        if name in self.none or name in HIDDEN:
            return default
        f = self.f
        r = f.find_attribute(self.h, name)
        if r is None:
            self.none.add(name)
            return default
        v = self.vals[name] = _attr_value(f._attribute(*r), f)
        return v

    def __getitem__(self, name):
        v = self.get(name, _ABSENT)
        if v is _ABSENT:
            raise KeyError(name)
        return v

    def __contains__(self, name):
        return self.get(name, _ABSENT) is not _ABSENT

    def _names(self):
        return [k for k in self.f.attribute_refs(self.h) if k not in HIDDEN]

    def whole(self):
        """Every attribute, in one pass (a new dict): each looked up by name
        after listing them was most of what xarray's open pays per variable."""
        f, vals, out = self.f, self.vals, {}
        for k, r in f.attribute_refs(self.h).items():
            if k not in HIDDEN:
                v = vals.get(k, _ABSENT)
                if v is _ABSENT:
                    v = vals[k] = _attr_value(f._attribute(*r), f)
                out[k] = v
        return out

    def __iter__(self):
        return iter(self._names())

    def __len__(self):
        return len(self._names())


class Variable:
    """A netCDF variable over an HDF5 dataset (backend contract). Its shape
    and type are parsed when first wanted, and its dimensions (NC4File.link)
    likewise: reading one variable of 200 needs one of each."""

    def __init__(self, group, name, lname, address, header, ds=None):
        self.group, self.name, self._lname, self._address, self._header = group, name, lname, address, header
        self._ds = ds
        self._read_by = None
        self._short = None
        self._dims = []                             # Dim objects
        self._dnames = None
        self._attrs = None

    @property
    def ds(self):
        if self._ds is None:
            self._ds = _hdf5.Dataset(self.group.nc.f, self._address, self._lname, self._header)
        return self._ds

    @property
    def _reader(self):
        if self._read_by is None:
            nc = self.group.nc
            self._read_by = r = _h5read.Reader(nc.f, self.ds, nc.src)
            t = self.ds.type
            # (for _quick: chars as they lie, and records if netCDF has them as stored)
            r.plain = (3,)
            if t.cls == _hdf5.Datatype.COMPOUND and t.dtype == nc_dtype(t):
                r.plain, r.record = (3, 6), nc_dtype(t)
        return self._read_by

    @property
    def dims(self):
        self.group.nc.link()
        return self._dims

    def short(self):
        """Whether an axis ends before its unlimited dimension does (records
        written to other variables): netCDF-C then gives the rest as fill
        values. The dimension's length is the longest variable using it, so
        this takes all of them (link), unless a look shows that none can
        reach past this one (_cannot_be_short)."""
        if self._short is None:
            nc = self.group.nc
            shape = self.ds.shape
            if not nc._linked and self._cannot_be_short():
                self._short = False
            else:
                nc.link()
                self._short = any(d is not None and d.unlimited and s < d.size
                                  for s, d in zip(shape, self._dims))
        return self._short

    def _cannot_be_short(self):
        """True if no dataset's unlimited axis is longer than any axis of this
        variable that an unlimited dimension could be behind: its own
        unlimited ones, and fixed ones only if its DIMENSION_LIST says a
        scale with an unlimited axis is attached there (a file netCDF-C would
        not write, but could read). False when it cannot tell."""
        ds = self.ds
        shape, most = ds.shape, ds.maxshape
        if not shape:
            return True
        nc = self.group.nc
        if None in most:
            top = nc.longest()
            below = [k for k, s in enumerate(shape) if s < top]
            if not below:
                return True
            if any(most[k] is None for k in below):
                return False
        else:
            # (no unlimited axis: what is attached to its own axes says it
            # all, and the look through every dataset is not needed)
            below = range(len(shape))
        f = nc.f
        attrs = _Lazy(f, self._header)
        dl = attrs.get('DIMENSION_LIST')
        if dl is None:
            # (no scales named: ids, which take every group's dimensions;
            # else its axes match a fixed dimension by size)
            return attrs.get('_Netcdf4Coordinates') is None and not nc.has_scale(self)
        refs = f.vlen_refs(dl)
        for k in below:
            addr = refs[k] if k < len(refs) else None
            space = None if addr is None else f.extent(addr)
            if space is None or space[1] is None or None in space[1]:
                return False
        return True

    def _padding(self, shape):
        """An array of this shape holding what a read past the stored values
        gives, as netCDF-C: _FillValue, else the type's default fill value,
        with or without filling on (checked type by type on netCDF4-python)."""
        t = self.ds.type
        dt = self.dtype
        if dt is str or t.cls == _hdf5.Datatype.VLEN:
            res = np.empty(shape, object)
            flat = res.reshape(-1)
            for i in range(flat.size):
                flat[i] = '' if dt is str else np.array([], dt)
            return res
        if t.cls == _hdf5.Datatype.COMPOUND:
            return np.zeros(shape, dt)
        fill = self.attrs.get('_FillValue')
        if fill is None:
            from ._maskscale import DEFAULT_FILLVALS
            fill = DEFAULT_FILLVALS.get(np.dtype(dt).str[1:], 0)
        return np.full(shape, np.array(fill, dt))

    def read_padded(self, box, threads=None):
        """The values of a box (start, stop, step per dimension) that may run
        past what is stored."""
        res = self._padding(_select.box_shape(box))
        sel, part = [], []
        for (s, e, step), n in zip(box, self.ds.shape):
            stop = min(e, n)
            count = len(range(s, stop, step))
            if not count:
                return res
            sel.append(slice(s, stop, step))
            part.append(slice(0, count))
        res[tuple(part)] = self.read(tuple(sel), threads=threads)
        return res

    @property
    def dimensions(self):
        names = self._dnames
        if names is None:
            names = self._dnames = tuple(d.name for d in self.dims)
        return names

    @property
    def shape(self):
        return tuple(self.ds.shape)

    @property
    def dtype(self):
        t = self.ds.type
        if t.cls == _hdf5.Datatype.VLEN and t.vlen_string:
            return str
        if t.cls == _hdf5.Datatype.STRING:
            # one-byte strings are NC_CHAR; longer fixed strings netCDF-c reads as NC_STRING
            return np.dtype('S1') if t.size == 1 else str
        if t.cls == _hdf5.Datatype.VLEN:
            return _h5read.native(t.base.dtype) if t.base.dtype is not None else object
        return nc_dtype(t)

    def user_type(self):
        """A CompoundType / EnumType / VLType for user-defined types (named by
        the committed datatype, as netCDF-c), else None."""
        from ._dataset import CompoundType, EnumType, VLType
        t = self.ds.type
        if t.cls not in (_hdf5.Datatype.COMPOUND, _hdf5.Datatype.ENUM, _hdf5.Datatype.VLEN,
                         _hdf5.Datatype.STRING) or (t.cls == _hdf5.Datatype.STRING and t.size == 1):
            return None
        if (t.cls == _hdf5.Datatype.VLEN and t.vlen_string) or (t.cls == _hdf5.Datatype.STRING and t.size > 1):
            return VLType(None, str)                # NC_STRING (netCDF4-python's VLType of str)
        names = self.group.nc.type_names
        name = names.get(t.committed) if t.committed is not None else None
        if name is None:
            m = self.ds.header.first(3)
            name = names.get(bytes(self.ds.file.b[m[2]:m[2] + m[3]]))
        if t.cls == _hdf5.Datatype.COMPOUND:
            return CompoundType(name, nc_dtype(t))
        if t.cls == _hdf5.Datatype.ENUM:
            return EnumType(name, _h5read.native(t.base.dtype), dict(t.enum))
        return VLType(name, _h5read.native(t.base.dtype) if t.base.dtype is not None else str)

    @property
    def attrs(self):
        if self._attrs is None:
            self._attrs = Attrs(self.group.nc.f, self._header)
        return self._attrs

    @property
    def chunks(self):
        return self.ds.chunks

    @property
    def no_fill(self):
        """What nc_inq_var_fill says: netCDF-c sets no_fill unless HDF5 has a
        user-defined fill value (a stored one; checked on 727 variables)."""
        return self.ds.fill is None

    @property
    def filters(self):
        ids = {f.id: f for f in self.ds.filters}
        out = {'zlib': 1 in ids, 'szip': False, 'zstd': 32015 in ids, 'bzip2': 307 in ids,
               'blosc': False, 'shuffle': 2 in ids, 'complevel': 0, 'fletcher32': 3 in ids}
        if 1 in ids:
            out['complevel'] = ids[1].cd[0] if ids[1].cd else 0
        if 32015 in ids:
            out['complevel'] = ids[32015].cd[0] if ids[32015].cd else 0
        if 307 in ids:
            out['complevel'] = ids[307].cd[0] if ids[307].cd else 0
        if 4 in ids:
            cd = ids[4].cd
            out['szip'] = {'coding': 'nn' if cd and cd[0] & 32 else 'ec',
                           'pixels_per_block': cd[1] if len(cd) > 1 else 0}
        if 32001 in ids:
            cd = ids[32001].cd
            names = {0: 'blosc_lz', 1: 'blosc_lz4', 2: 'blosc_lz4hc', 4: 'blosc_zlib', 5: 'blosc_zstd'}
            out['blosc'] = {'compressor': names.get(cd[6] if len(cd) > 6 else 0, 'blosc_lz'),
                            'shuffle': cd[5] if len(cd) > 5 else 1}
            out['complevel'] = cd[4] if len(cd) > 4 else 0
        return out

    @property
    def widened(self):
        """Whether netCDF has these numbers in another type than they are stored
        in: reads then come here, not straight to the file's (see nc_dtype)."""
        t = self.ds.type
        return t.cls == _hdf5.Datatype.FLOAT and t.size == 2

    @property
    def endian(self):
        t = self.ds.type
        if t.cls not in (0, 1):
            return 'native'                         # chars, strings, user types
        return 'big' if t.order == '>' else 'little'

    def read(self, key=(), out=None, threads=None):
        t = self.ds.type
        if t.cls == _hdf5.Datatype.STRING and t.size > 1:
            # a fixed-length string longer than a byte is NC_STRING to
            # netCDF-c: str values (up to the first NUL), as netCDF4-python
            raw = self._reader.read(key, threads=threads, scalars=False)
            dec = [x.split(b'\0', 1)[0].decode('utf-8', 'replace') for x in raw.reshape(-1).tolist()]
            res = np.empty(raw.shape, object)
            res.reshape(-1)[:] = dec
            return res[()] if res.ndim == 0 else res
        if t.cls == _hdf5.Datatype.COMPOUND:
            raw = self._reader.read(key, threads=threads)
            dt = nc_dtype(t)
            res = np.zeros(raw.shape, dt)
            odd = {m[0]: m[2] for m in t.members if m[2].cls == _hdf5.Datatype.FIXED and m[2].dtype is None and m[2].size < 8}
            for name in dt.names:
                # strings cut to their first byte, as netCDF-c
                res[name] = raw[name] if name not in odd else _h5read.odd_int(raw[name], odd[name])
            if res.ndim == 0 and self.ds.shape:
                res = res[()]                       # (one element; of a scalar variable, an array)
            if out is not None:
                out[...] = res
                return out
            return res
        if ((t.cls == _hdf5.Datatype.FLOAT and t.size < 4)
                or (t.cls == _hdf5.Datatype.FIXED and t.precision < 8 * t.size)):   # (see nc_dtype)
            res = self._reader.read(key, threads=threads)
            if res is None:
                return None
            res = res.astype(nc_dtype(t))
            if out is not None:
                out[...] = res
                return out
            return res
        res = self._reader.read(key, out=out, threads=threads)
        if t.cls == _hdf5.Datatype.VLEN and not self.ds.shape and isinstance(res, np.ndarray) and res.dtype == object:
            v = res[()]                             # (one sequence: its values, as netCDF-c gives them,
            if isinstance(v, np.ndarray) and v.size == 1:   # and netCDF4-python squeezes them)
                return v.reshape(())
            return v
        return res


class Group:
    """A netCDF group over an HDF5 group (backend contract)."""

    def __init__(self, nc, h5group, name, parent):
        self.nc, self.h5, self.name, self.parent = nc, h5group, name, parent
        self._dims = {}
        self._vars = {}
        self._groups = {}
        self._types = {}
        self._peeked = {}                           # name -> Variable found by peek()
        self._subs = {}                             # name -> Group found by peek_group()
        self._attrs = None

    @property
    def dimensions(self):
        if self.holds_no_data():
            return self._dims
        self.nc.link()
        return self._dims

    @property
    def variables(self):
        self.nc.scan()
        return self._vars

    def holds_no_data(self):
        """Whether this group has no dataset of its own, from its own links
        alone: no variables, no dimensions, whatever the rest of the file
        says (xarray opening a root group that only holds groups)."""
        nc = self.nc
        if nc._scanned:
            return not self._vars and not self._dims
        got = self.__dict__.get('_no_data')
        if got is None:
            got = self._no_data = self._look_for_data()
        return got

    def _look_for_data(self):
        f = self.nc.f
        for kind, target in self.h5.links.values():
            if kind != 'hard' or f.object_header(target).first(8) is not None:
                return False
        return True

    def peek_group(self, name):
        """The group called `name` without reading its siblings; None when
        there is none, or only the whole scan can say."""
        nc = self.nc
        if nc._scanned:
            return self._groups.get(name)
        sub = self._subs.get(name)
        if sub is not None:
            return sub
        link = self.h5.link(name)
        if link is None or link[0] != 'hard':
            return None
        f = nc.f
        h = f.object_header(link[1])
        if h.first(8) is not None or h.first(3) is not None:
            return None                             # (a dataset, a type)
        return self._subs.setdefault(name, Group(nc, _hdf5.Group(f, link[1], name, h), name, self))

    @property
    def groups(self):
        self.nc.scan()
        return self._groups

    @property
    def types(self):
        self.nc.scan()
        return self._types

    def peek(self, name):
        """The variable called `name` without reading any other: its link and
        object header alone. KeyError if there is no such variable, None
        when only the whole scan can say: a dimension scale, or anything
        else that takes more than one object to settle."""
        nc = self.nc
        if nc._scanned:
            return self._vars[name]
        v = self._peeked.get(name)
        if v is not None:
            return v
        link = self.h5.link
        plain, other = link(name), link(NON_COORD + name)
        if plain is not None and other is not None:
            return None
        link, lname = (plain, name) if plain is not None else (other, NON_COORD + name)
        if link is None:
            raise KeyError(name)
        kind, target = link
        if kind == 'soft':
            target = nc._resolve(self.h5, target)
        elif kind != 'hard':
            raise KeyError(name)
        if target is None:
            raise KeyError(name)
        f = nc.f
        h = f.object_header(target)
        if h.first(8) is None:
            raise KeyError(name)                    # (a group, a type)
        # (a look through its bytes settles most; attributes in dense storage
        # it cannot see into, so those are asked for by name)
        if f.may_have_attribute(h, b'CLASS\0') and f.find_attribute(h, 'CLASS') is not None:
            return None
        return self._peeked.setdefault(name, Variable(self, name, lname, target, h))

    @property
    def attrs(self):
        if self._attrs is None:
            out = {}
            for k, a in self.h5.attributes().items():
                if k in HIDDEN:
                    continue
                out[k] = _attr_value(a, self.nc.f)
            self._attrs = out
        return self._attrs


class NC4File:
    """A netCDF-4 (or plain HDF5) file read through the netCDF data model."""

    def __init__(self, f, src):
        self.f = f
        self.src = src
        self.next_dimid = 0
        self.by_scale = {}                          # dimension scale object address -> Dim
        self._type_names = {}                       # committed datatype address -> name
        # (one lookup, not every root attribute parsed: WRF's root has 150)
        root = f.object_header(f.root.address)
        self.data_model = 'NETCDF4_CLASSIC' if f.find_attribute(root, '_nc3_strict') is not None             else 'NETCDF4'
        # as netCDF-c: read every group's scales and variables first (scan),
        # then match variables to dimensions over the whole tree, children
        # first, making phony dimensions where a dataset has no scales
        # attached (link). Each runs once, whole, when first needed: one
        # variable of a file needs neither, and they were most of an open
        self.root = Group(self, f.root, '/', None)
        self._scanned = self._linked = False
        self._longest = None
        self._lock = threading.RLock()
        self._ids = {}                              # id of a group -> (its dimensions, {id: Dim})

    @property
    def type_names(self):
        self.scan()
        return self._type_names

    def has_scale(self, v):
        """Whether v may itself be a dimension scale (its first axis then is
        its own dimension)."""
        f = self.f
        return f.may_have_attribute(v._header, b'CLASS\0') and f.find_attribute(v._header, 'CLASS') is not None

    def longest(self):
        """No unlimited axis of any dataset in the file reaches past this (an
        infinity when that takes more than a look: soft links, a remote
        file). Their dataspaces, not all of their headers."""
        if self._longest is None:
            self._longest = float('inf') if self.f.source is not None else self._reach(self.root.h5, set())
        return self._longest

    def _reach(self, h5g, seen):
        top = 0
        f = self.f
        for kind, target in h5g.links.values():
            if kind != 'hard':
                return float('inf')
            if target in seen:
                continue
            seen.add(target)
            space = f.extent(target)
            if space is None:
                h = f.object_header(target)
                if h.first(3) is None:              # (else a type: nothing to read)
                    top = max(top, self._reach(_hdf5.Group(f, target, '', h), seen))
            elif space[0] is not None:              # (not a null dataspace)
                for s, m in zip(*space):
                    if m is None and s > top:
                        top = s
        return top

    def scan(self):
        """Every group's dimension scales, variables, groups and types."""
        if not self._scanned:
            with self._lock:
                if not self._scanned:
                    self._fill(self.root)
                    self._scanned = True

    def link(self):
        """Variables matched to dimensions (done once, whole, so phony
        dimensions get the numbers netCDF-c gives them whoever asks first)."""
        if not self._linked:
            with self._lock:
                if not self._linked:
                    self.scan()
                    self._match(self.root)
                    self._fix_unlimited(self.root)
                    self._linked = True

    def close(self):
        self.src = None
        self.f.close()

    # -- reading the structure
    def _fill(self, g):
        h5g = g.h5
        f = self.f
        pending = []
        children = []
        for lname, (kind, target) in h5g.links.items():
            if kind == 'soft':
                target = self._resolve(h5g, target)   # netCDF-c follows soft links
                if target is None:
                    continue
            elif kind != 'hard':
                continue
            h = f.object_header(target)
            if h.first(8) is None:
                if h.first(3) is not None:
                    g._types[lname] = f.message_datatype(h.first(3))
                    self._type_names[target] = lname
                    m = h.first(3)                    # and by encoding: datasets carry a copy
                    self._type_names.setdefault(bytes(f.b[m[2]:m[2] + m[3]]), lname)
                else:
                    children.append((lname, target, h))
                continue
            # (most datasets have no attribute of that name: a look through
            # their bytes, nothing decoded)
            cls = None
            if f.may_have_attribute(h, b'CLASS\0'):
                cls = _Lazy(f, h).get('CLASS')
            if cls is not None and _bytes(cls) == b'DIMENSION_SCALE':
                ds = _hdf5.Dataset(f, target, lname, h)
                attrs = _Lazy(f, h)
                nm = attrs.get('NAME')
                dim_only = nm is not None and _bytes(nm).startswith(NOT_A_VAR)
                did = attrs.get('_Netcdf4Dimid')
                did = int(np.asarray(did.value()).ravel()[0]) if did is not None else None
                if did is None:
                    did = self.next_dimid
                self.next_dimid = max(self.next_dimid, did + 1)
                dname = lname[len(NON_COORD):] if lname.startswith(NON_COORD) else lname
                size = ds.shape[0] if ds.shape else 0
                unlim = bool(ds.maxshape and ds.maxshape[0] is None)
                d = Dim(dname, size, unlim, did, target, g)
                g._dims[dname] = d
                self.by_scale[target] = d
                if not dim_only:
                    pending.append((lname, target, h, ds))
            else:
                pending.append((lname, target, h, None))
        for lname, target, h, ds in pending:
            m = h.first(3)
            if m is None or m[1] & 2 or f.b[m[2]] & 15 in (5, 6, 7, 10):    # (only those can be hidden)
                if m is not None and not _shown(f.message_datatype(m)):
                    continue
            vname = lname[len(NON_COORD):] if lname.startswith(NON_COORD) else lname
            v = g._peeked.get(vname)
            g._vars[vname] = Variable(g, vname, lname, target, h, ds) if v is None else v
        for lname, target, h in children:
            up = g
            while up is not None and up.h5.address != target:
                up = up.parent
            if up is not None:                      # (a group inside itself: once is enough)
                continue
            sub = g._subs.get(lname)
            if sub is None:
                sub = Group(self, _hdf5.Group(f, target, lname, h), lname, g)
            g._groups[lname] = sub
            self._fill(sub)

    def _match(self, g):
        for sub in g._groups.values():
            self._match(sub)
        for v in g._vars.values():
            self._attach(g, v, v.ds, _Lazy(self.f, v._header))
        g._dims = dict(sorted(g._dims.items(), key=lambda kv: kv[1].id))

    def _resolve(self, h5g, path, depth=0):
        """The object header address a soft link's path names, or None. (A
        link may lead to another, as the library's limit of 16 allows.)"""
        f = self.f
        g = f.root if path.startswith('/') else h5g
        parts = [p for p in path.split('/') if p]
        for k, p in enumerate(parts):
            link = g.link(p)
            if link is None:
                return None
            target = link[1]
            if link[0] == 'soft' and depth < 16:
                target = self._resolve(g, target, depth + 1)
            elif link[0] != 'hard':
                return None
            if target is None:
                return None
            if k == len(parts) - 1:
                return target
            g = _hdf5.Group(f, target, p)
        return None

    def _attach(self, g, v, ds, attrs):
        """v.dims from _Netcdf4Coordinates, DIMENSION_LIST, a scale of its own,
        or phony dimensions."""
        rank = len(ds.shape)
        if rank == 0:
            return
        f = self.f
        dims = [None] * rank
        coords = attrs.get('_Netcdf4Coordinates')
        if coords is not None:
            ids = [int(x) for x in np.asarray(coords.value()).ravel()]
            for k, did in enumerate(ids[:rank]):
                dims[k] = self._dim_by_id(g, did)
        if any(d is None for d in dims):
            dl = attrs.get('DIMENSION_LIST')
            if dl is not None:
                for k, addr in enumerate(f.vlen_refs(dl)[:rank]):
                    if addr is not None and dims[k] is None:
                        dims[k] = self.by_scale.get(addr)
        if dims[0] is None and ds.address in self.by_scale:
            dims[0] = self.by_scale[ds.address]
        if any(d is None for d in dims):
            for k in range(rank):
                if dims[k] is not None:
                    continue
                n = ds.shape[k]
                unlim = ds.maxshape[k] is None or n == 0
                match = None
                for d in g._dims.values():
                    if d.size == n and d.unlimited == unlim and d not in dims:
                        match = d
                        break
                if match is None:
                    did = self.next_dimid
                    self.next_dimid += 1
                    match = Dim('phony_dim_%d' % did, n, unlim, did, None, g)
                    g._dims[match.name] = match
                dims[k] = match
        v._dims = dims

    def _dim_by_id(self, g, did):
        ids = self._ids
        while g is not None:
            known = ids.get(id(g))
            if known is None or known[0] != len(g._dims):   # (phony ones are added as it goes)
                known = ids[id(g)] = (len(g._dims), {d.id: d for d in g._dims.values()})
            d = known[1].get(did)
            if d is not None:
                return d
            g = g.parent
        return None

    def _fix_unlimited(self, g):
        """An unlimited dimension is as long as the longest variable along it."""
        for v in g._vars.values():
            for k, d in enumerate(v._dims):
                if d is not None and d.unlimited and v.ds.shape[k] > d.size:
                    d.size = v.ds.shape[k]
        for sub in g._groups.values():
            self._fix_unlimited(sub)


class _Lazy:
    """A dataset's attributes found and decoded one by one, as asked for
    (by name: dense storage is looked up by hash, not read through)."""
    __slots__ = ('f', 'h')

    def __init__(self, f, h):
        self.f, self.h = f, h

    def get(self, name):
        r = self.f.find_attribute(self.h, name)
        return None if r is None else self.f._attribute(r[0], r[1], r[2])

    def __contains__(self, name):
        return self.f.find_attribute(self.h, name) is not None


def _bytes(a):
    v = a.value()
    v = np.asarray(v).ravel()
    x = v[0] if v.size else b''
    if isinstance(x, str):
        return x.encode()
    return bytes(x).rstrip(b'\0')
