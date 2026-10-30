"""The h5py-style front end: File, Group, Dataset, AttributeManager, read-only.

Same names, properties and values as h5py 3 (vlen strings as bytes,
enum-of-FALSE/TRUE as bool, {r, i} compounds as complex, attributes and
links in name order unless the file tracks creation order), so code
written for h5py.File runs unchanged for reading.
"""
import mmap
import os
import posixpath

import numpy as np

from . import _h5read, _hdf5, _select


class Reference:
    """An object reference read from a dataset or attribute (h5py.Reference)."""
    __slots__ = ('address',)

    def __init__(self, address):
        self.address = address

    def __bool__(self):
        return self.address != 0

    def __repr__(self):
        return '<HDF5 object reference %d>' % self.address


def h5py_dtype(t):
    """The dtype h5py reports for datatype t."""
    if t.cls == _hdf5.Datatype.VLEN:
        return np.dtype('O', metadata={'vlen': str if t.vlen_string else h5py_dtype(t.base)})
    if t.cls == _hdf5.Datatype.REFERENCE:
        return np.dtype('O', metadata={'ref': Reference})
    if t.cls == _hdf5.Datatype.ENUM and t.h5py_dtype.kind != 'b':
        return np.dtype(t.base.dtype, metadata={'enum': dict(t.enum)})
    if t.cls == _hdf5.Datatype.STRING:
        return np.dtype(t.dtype.str, metadata={'h5py_encoding': 'utf-8' if t.charset == 1 else 'ascii'})
    return t.h5py_dtype


def _objects(shape, dt, items):
    out = np.empty(shape, dt)
    flat = out.reshape(-1)
    for i, x in enumerate(items):
        flat[i] = x
    return out


def _to_h5py(values, t, strings=bytes, top=True):
    """Values decoded by the parser in h5py's form: variable-length strings
    as bytes from datasets and str from attributes, object arrays carrying
    h5py's dtype metadata (vlen, h5py_encoding), references as Reference
    objects (also inside compounds), bool and complex as h5py maps them."""
    arr = isinstance(values, np.ndarray)
    if t.cls == _hdf5.Datatype.VLEN and t.vlen_string:
        conv = (lambda x: x) if strings is str else             (lambda x: x.encode('utf-8') if isinstance(x, str) else x)
        if not arr:
            return conv(values)
        return _objects(values.shape, np.dtype('O', metadata={'vlen': str}),
                        (conv(x) for x in values.reshape(-1)))
    if t.cls == _hdf5.Datatype.VLEN:
        if not arr:
            return values
        inner = (_to_h5py(x, t.base, strings, False) for x in values.reshape(-1))
        return _objects(values.shape, h5py_dtype(t) if top else np.dtype('O'), inner)
    if t.cls == _hdf5.Datatype.REFERENCE and t.ref_kind == 0:
        if not arr:
            return values
        refs = (Reference(int.from_bytes(bytes(x)[:8], 'little')) for x in values.reshape(-1))
        return _objects(values.shape, h5py_dtype(t) if top else np.dtype('O'), refs)
    if t.cls == _hdf5.Datatype.COMPOUND and arr and any(
            m[2].cls in (_hdf5.Datatype.REFERENCE, _hdf5.Datatype.VLEN) for m in t.members):
        names = [m[0] for m in t.members]
        fields = [_to_h5py(values[name], mt, strings, False) for name, off, mt in t.members]
        dt = np.dtype({'names': names,
                       'formats': [f.dtype if f.dtype.kind != 'O' else np.dtype('O') for f in fields],
                       'offsets': [m[1] for m in t.members], 'itemsize': max(16, t.size)})
        out = np.zeros(values.shape, dt)
        for n, f in zip(names, fields):
            out[n] = f
        return out
    if t.cls == _hdf5.Datatype.ENUM and arr and top and t.h5py_dtype.kind != 'b':
        return values.view(h5py_dtype(t))
    if t.cls == _hdf5.Datatype.STRING and arr and top:
        enc = 'utf-8' if t.charset == 1 else 'ascii'
        return values.view(np.dtype(values.dtype.str, metadata={'h5py_encoding': enc}))
    hd = t.h5py_dtype
    if hd is not None and hd.subdtype is not None:
        return values
    if arr and hd is not None and hd != values.dtype and values.dtype.fields is None:
        return values.view(_h5read.native(hd))
    if arr and hd is not None and hd.kind == 'c':
        return values.view(_h5read.native(hd))
    return values


class AttributeManager:
    """obj.attrs: read-only mapping of attribute names to values."""

    def __init__(self, obj):
        self._obj = obj
        self._attrs = None

    def _load(self):
        if self._attrs is None:
            h = self._obj._header
            a = self._obj.file._f.attributes(h)
            tracked = h.version == 2 and h.flags & 4
            if not tracked:
                a = dict(sorted(a.items()))
            self._attrs = a
        return self._attrs

    def __getitem__(self, name):
        a = self._load()[name]
        t = a.dtype
        if a.shape is None:
            return Empty(t.h5py_dtype if t.dtype is not None else None)
        v = a.value()
        v = _to_h5py(v, t, str)
        if isinstance(v, np.ndarray) and v.dtype.byteorder == '>':
            v = v.astype(_h5read.native(v.dtype))
        if isinstance(v, np.ndarray) and v.shape == ():
            return v[()]
        return v

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def __contains__(self, name):
        return name in self._load()

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())

    def keys(self):
        return list(self._load())

    def values(self):
        return [self[k] for k in self._load()]

    def items(self):
        return [(k, self[k]) for k in self._load()]

    def __repr__(self):
        return '<Attributes of HDF5 object (%d)>' % len(self)


class Empty:
    """A null dataspace's value (h5py.Empty)."""

    def __init__(self, dtype):
        self.dtype = dtype

    def __eq__(self, other):
        return isinstance(other, Empty) and self.dtype == other.dtype

    def __repr__(self):
        return 'Empty(dtype=%r)' % self.dtype


class HLObject:
    def __init__(self, file, address, name, header):
        self.file = file
        self._address = address
        self.name = name
        self._header = header
        self.attrs = AttributeManager(self)

    @property
    def parent(self):
        return self.file[posixpath.dirname(self.name) or '/']

    def __eq__(self, other):
        return isinstance(other, HLObject) and other.file is self.file and other._address == self._address

    def __hash__(self):
        return hash((id(self.file), self._address))


class Group(HLObject):
    """An HDF5 group (h5py.Group)."""

    def __init__(self, file, address, name, header=None):
        hdr = header or file._f.object_header(address)
        HLObject.__init__(self, file, address, name, hdr)
        self._g = _hdf5.Group(file._f, address, name, hdr)
        self._links = None

    def _linkmap(self):
        if self._links is None:
            links = self._g.links
            h = self._header
            # h5py iterates by name unless creation order is tracked
            li = h.first(0x02)
            tracked = li is not None and self.file._f.b[li[2] + 1] & 1
            self._links = links if tracked else dict(sorted(links.items()))
        return self._links

    def __getitem__(self, path):
        if isinstance(path, Reference):
            return self.file._object(path.address, None)
        if path == '':
            return self
        if path == '/':
            return self.file._root_group()
        g = self.file if path.startswith('/') else self
        parts = [p for p in path.split('/') if p and p != '.']
        for k, p in enumerate(parts):
            if not isinstance(g, Group):
                raise KeyError(path)
            if p == '..':
                g = g.parent
                continue
            link = g._linkmap().get(p)
            if link is None:
                raise KeyError("Unable to open object (object '%s' doesn't exist)" % p)
            kind, target = link
            name = posixpath.join(g.name, p)
            if kind == 'soft':
                g = g[target] if target.startswith('/') else g[target]
                continue
            if kind != 'hard':
                raise KeyError('external links are not supported: %s' % path)
            g = self.file._object(target, name)
        return g

    def get(self, name, default=None, getclass=False, getlink=False):
        try:
            obj = self[name]
        except KeyError:
            return default
        return type(obj) if getclass else obj

    def __contains__(self, name):
        try:
            self[name]
            return True
        except KeyError:
            return False

    def __iter__(self):
        return iter(self._linkmap())

    def __len__(self):
        return len(self._linkmap())

    def keys(self):
        return list(self._linkmap())

    def values(self):
        return [self[k] for k in self._linkmap()]

    def items(self):
        return [(k, self[k]) for k in self._linkmap()]

    def visit(self, func):
        return self.visititems(lambda name, obj: func(name))

    def visititems(self, func):
        """func(name, obj) for every object below, each once (by name, depth
        first, as H5Ovisit), names relative to this group; stops when func
        returns something other than None."""
        seen = {self._address}
        base = self.name.rstrip('/')

        def walk(g):
            for k in sorted(g._linkmap()):
                kind, target = g._linkmap()[k]
                if kind != 'hard' or target in seen:
                    continue
                seen.add(target)
                obj = g[k]
                rel = obj.name[len(base) + 1:] if base else obj.name.lstrip('/')
                r = func(rel, obj)
                if r is not None:
                    return r
                if isinstance(obj, Group):
                    r = walk(obj)
                    if r is not None:
                        return r
            return None
        return walk(self)

    def __repr__(self):
        return '<HDF5 group "%s" (%d members)>' % (self.name, len(self))


class Datatype(HLObject):
    """A committed (named) datatype (h5py.Datatype)."""

    @property
    def dtype(self):
        return h5py_dtype(self.file._f.message_datatype(self._header.first(3)))


class Dataset(HLObject):
    """An HDF5 dataset (h5py.Dataset), read-only."""

    def __init__(self, file, address, name, header=None):
        hdr = header or file._f.object_header(address)
        HLObject.__init__(self, file, address, name, hdr)
        self._ds = _hdf5.Dataset(file._f, address, name.rsplit('/', 1)[-1], hdr)
        self._reader = _h5read.Reader(file._f, self._ds, file._src)

    # -- properties
    @property
    def shape(self):
        return None if self._ds.null else tuple(self._ds.shape)

    @property
    def maxshape(self):
        return None if self._ds.null else tuple(self._ds.maxshape)

    @property
    def dtype(self):
        return h5py_dtype(self._ds.type)

    @property
    def size(self):
        return int(np.prod(self._ds.shape)) if self.shape is not None else None

    @property
    def nbytes(self):
        return self.size * self._ds.type.size

    @property
    def ndim(self):
        return len(self._ds.shape)

    def __len__(self):
        if not self._ds.shape:
            raise TypeError("Attempt to take len() of scalar dataset")
        return self._ds.shape[0]

    @property
    def chunks(self):
        return self._ds.chunks

    def _filter(self, fid):
        for f in self._ds.filters:
            if f.id == fid:
                return f
        return None

    def _unknown(self):
        """The first filter h5py has no name for (it calls it 'unknown')."""
        for f in self._ds.filters:
            if f.id not in (1, 2, 3, 4, 6, 32000):
                return f
        return None

    @property
    def compression(self):
        for fid, name in ((1, 'gzip'), (32000, 'lzf'), (4, 'szip')):
            if self._filter(fid) is not None:
                return name
        return 'unknown' if self._unknown() is not None else None

    @property
    def compression_opts(self):
        if self.compression == 'unknown':
            return None
        f = self._filter(1)
        if f is not None:
            return f.cd[0] if f.cd else None
        f = self._filter(4)
        if f is not None:
            return ('nn' if f.cd[0] & 32 else 'ec', f.cd[1])
        return None

    @property
    def shuffle(self):
        return self._filter(2) is not None

    @property
    def fletcher32(self):
        return self._filter(3) is not None

    @property
    def scaleoffset(self):
        f = self._filter(6)
        return f.cd[1] if f is not None else None

    @property
    def is_virtual(self):
        return self._ds.layout.cls == 3

    @property
    def external(self):
        return None

    @property
    def fillvalue(self):
        t = self._ds.type
        if t.cls == _hdf5.Datatype.VLEN:
            if t.vlen_string:
                return b''
            return None if self._ds.fill is None else np.array([], _h5read.native(t.base.dtype))
        dt = t.h5py_dtype
        if self._ds.fill is None or dt is None or t.cls == _hdf5.Datatype.REFERENCE:
            return np.zeros((), dt)[()] if dt is not None and t.cls not in (9, 7) else None
        v = np.frombuffer(self._ds.fill, t.dtype, 1)
        v = _to_h5py(v, t)
        return v.astype(_h5read.native(v.dtype))[0]

    # -- values
    def __getitem__(self, key):
        if self._ds.null:
            if key == () or key is Ellipsis:
                return Empty(self.dtype)
            raise ValueError('Empty datasets have no numpy representation')
        if isinstance(key, str) or (isinstance(key, tuple) and any(isinstance(k, str) for k in key)):
            names = [k for k in (key if isinstance(key, tuple) else (key,)) if isinstance(k, str)]
            rest = tuple(k for k in (key if isinstance(key, tuple) else (key,)) if not isinstance(k, str))
            v = self[rest if rest else ()]
            return v[names[0]] if len(names) == 1 else v[names]
        k = key if isinstance(key, tuple) else (key,)
        v = self._reader.read(k, scalars=False)
        v = _to_h5py(v, self._ds.type)
        if isinstance(v, np.ndarray) and v.ndim == 0:
            if not self._ds.shape and key is Ellipsis:
                return v                            # dset[...] of a scalar dataset: 0-d array
            return v[()]
        return v

    def read_direct(self, dest, source_sel=None, dest_sel=None):
        """Read into an existing array (dest[dest_sel] = self[source_sel])."""
        v = self[source_sel if source_sel is not None else ()]
        if dest_sel is None:
            dest[...] = v
        else:
            dest[dest_sel] = v

    def __array__(self, dtype=None, copy=None):
        a = np.asarray(self[()])
        return a.astype(dtype) if dtype is not None else a

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return '<HDF5 dataset "%s": shape %s, type "%s">' % (
            posixpath.basename(self.name), self.shape, self.dtype.str if self.dtype is not None else '?')


class File(Group):
    """An open HDF5 file (h5py.File), read-only."""

    def __init__(self, name, mode='r', **kwargs):
        if mode not in ('r', None):
            raise ValueError('fastnc reads only (mode must be "r")')
        self.filename = os.fspath(name)
        self._fh = open(self.filename, 'rb')
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._src = np.frombuffer(self._mm, np.uint8)
        self._f = _hdf5.H5File(self._mm)
        self._objects = {}
        self.mode = 'r'
        Group.__init__(self, self, self._f.root_address, '/')
        self.file = self

    def _object(self, address, name):
        obj = self._objects.get(address)
        if obj is not None and (name is None or obj.name == name):
            return obj
        h = self._f.object_header(address)
        name = name or '/?'
        if h.first(8) is not None:
            obj = Dataset(self, address, name, h)
        elif h.first(3) is not None:
            obj = Datatype(self, address, name, h)
        else:
            obj = Group(self, address, name, h)
        self._objects.setdefault(address, obj)
        return obj

    @property
    def parent(self):
        return self._root_group()

    def _root_group(self):
        g = self._objects.get(('root',))
        if g is None:
            g = self._objects[('root',)] = Group(self, self._f.root_address, '/')
        return g

    @property
    def libver(self):
        return ('earliest', 'latest') if self._f.sb_version < 2 else ('v108', 'latest')

    def close(self):
        self._objects = {}
        self._src = None
        mm, self._mm = self._mm, None
        if mm is not None:
            try:
                mm.close()
            except BufferError:
                pass
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __bool__(self):
        return self._mm is not None

    def __repr__(self):
        return '<HDF5 file "%s" (mode r)>' % os.path.basename(self.filename)
