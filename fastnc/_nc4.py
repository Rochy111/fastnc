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
import numpy as np

from . import _h5read, _hdf5

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
        return np.dtype([(name, nc_dtype(mt)) for name, off, mt in t.members], align=True)
    if t.cls == _hdf5.Datatype.STRING:
        return np.dtype('S1')
    if t.cls == _hdf5.Datatype.ARRAY:
        return np.dtype((nc_dtype(t.base), t.dims))
    if t.cls == _hdf5.Datatype.ENUM:
        return _h5read.native(t.base.dtype)
    return _h5read.native(t.dtype)


class Variable:
    """A netCDF variable over an HDF5 dataset (backend contract)."""

    def __init__(self, group, name, ds, reader):
        self.group, self.name, self.ds, self._reader = group, name, ds, reader
        self.dims = []                              # Dim objects
        self._attrs = None

    @property
    def dimensions(self):
        return tuple(d.name for d in self.dims)

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
        if t.cls == _hdf5.Datatype.COMPOUND:
            return nc_dtype(t)
        return _h5read.native(t.dtype)

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
            out = {}
            for k, a in self.ds.attributes().items():
                if k in HIDDEN:
                    continue
                out[k] = _attr_value(a, self.ds.file)
            self._attrs = out
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
            for name in dt.names:
                res[name] = raw[name]               # strings cut to their first byte, as netCDF-c
            if res.ndim == 0:
                res = res[()]
            if out is not None:
                out[...] = res
                return out
            return res
        return self._reader.read(key, out=out, threads=threads)


class Group:
    """A netCDF group over an HDF5 group (backend contract)."""

    def __init__(self, nc, h5group, name, parent):
        self.nc, self.h5, self.name, self.parent = nc, h5group, name, parent
        self.dimensions = {}
        self.variables = {}
        self.groups = {}
        self.types = {}
        self._attrs = None

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
        self.type_names = {}                        # committed datatype address -> name
        attrs = f.root.attributes()
        self.data_model = 'NETCDF4_CLASSIC' if '_nc3_strict' in attrs else 'NETCDF4'
        # as netCDF-c: read every group's scales and variables first, then
        # match variables to dimensions over the whole tree (children first),
        # making phony dimensions where a dataset has no scales attached
        self.root = self._group(f.root, '/', None)
        self._match(self.root)
        self._fix_unlimited(self.root)

    def close(self):
        pass

    # -- reading the structure
    def _group(self, h5g, name, parent):
        g = Group(self, h5g, name, parent)
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
                    g.types[lname] = f.message_datatype(h.first(3))
                    self.type_names[target] = lname
                    m = h.first(3)                    # and by encoding: datasets carry a copy
                    self.type_names.setdefault(bytes(f.b[m[2]:m[2] + m[3]]), lname)
                else:
                    children.append((lname, target, h))
                continue
            ds = _hdf5.Dataset(f, target, lname, h)
            attrs = _Lazy(f, h)
            cls = attrs.get('CLASS')
            is_scale = cls is not None and _bytes(cls) == b'DIMENSION_SCALE'
            if is_scale:
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
                g.dimensions[dname] = d
                self.by_scale[target] = d
                if not dim_only:
                    pending.append((lname, ds, attrs))
            else:
                pending.append((lname, ds, attrs))
        for lname, ds, attrs in pending:
            vname = lname[len(NON_COORD):] if lname.startswith(NON_COORD) else lname
            v = Variable(g, vname, ds, _h5read.Reader(f, ds, self.src))
            v._h5attrs = attrs
            g.variables[vname] = v
        for lname, target, h in children:
            g.groups[lname] = self._group(_hdf5.Group(f, target, lname, h), lname, g)
        return g

    def _match(self, g):
        for sub in g.groups.values():
            self._match(sub)
        for v in g.variables.values():
            self._attach(g, v, v.ds, v._h5attrs)
            del v._h5attrs
        g.dimensions = dict(sorted(g.dimensions.items(), key=lambda kv: kv[1].id))

    def _resolve(self, h5g, path):
        """The object header address a soft link's path names, or None."""
        f = self.f
        g = f.root if path.startswith('/') else h5g
        parts = [p for p in path.split('/') if p]
        for k, p in enumerate(parts):
            link = g.links.get(p)
            if link is None or link[0] != 'hard':
                return None
            if k == len(parts) - 1:
                return link[1]
            g = _hdf5.Group(f, link[1], p)
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
                unlim = ds.maxshape[k] is None
                match = None
                for d in g.dimensions.values():
                    if d.size == n and d.unlimited == unlim and d not in dims:
                        match = d
                        break
                if match is None:
                    did = self.next_dimid
                    self.next_dimid += 1
                    match = Dim('phony_dim_%d' % did, n, unlim, did, None, g)
                    g.dimensions[match.name] = match
                dims[k] = match
        v.dims = dims

    def _dim_by_id(self, g, did):
        while g is not None:
            for d in g.dimensions.values():
                if d.id == did:
                    return d
            g = g.parent
        return None

    def _fix_unlimited(self, g):
        """An unlimited dimension is as long as the longest variable along it."""
        for v in g.variables.values():
            for k, d in enumerate(v.dims):
                if d is not None and d.unlimited and v.ds.shape[k] > d.size:
                    d.size = v.ds.shape[k]
        for sub in g.groups.values():
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
