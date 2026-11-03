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

from . import _h5read, _hdf5, _pool, _quick, _readahead, _select


class _LazyRemote:
    """fastnc._remote, imported when a URL or file object is opened: its
    HTTP machinery (http.client, ssl, urllib) costs ~40 ms to import, a
    fifth of a fresh process's first local read."""

    def __getattr__(self, name):
        from . import _remote
        return getattr(_remote, name)

    @staticmethod
    def open_source(name, storage_options=None):
        import os
        if isinstance(name, os.PathLike) or (isinstance(name, str) and '://' not in name):
            return None
        from . import _remote
        return _remote.open_source(name, storage_options)


_remote = _LazyRemote()


class Reference:
    """An object reference read from a dataset or attribute (h5py.Reference)."""
    __slots__ = ('address',)

    def __init__(self, address):
        self.address = address

    def __bool__(self):
        return self.address != 0

    def __repr__(self):
        return '<HDF5 object reference %d>' % self.address


class RegionReference(Reference):
    """h5py.RegionReference: a dataset and a selection of it, kept in the
    file's global heap. `dset[ref]` gives the values, `file[ref]` the dataset."""
    __slots__ = ('index',)

    def __init__(self, collection, index):
        Reference.__init__(self, collection)
        self.index = index

    def __repr__(self):
        return '<HDF5 region reference%s>' % ('' if self else ' (null)')


def h5py_dtype(t):
    """The dtype h5py reports for datatype t."""
    if t.cls == _hdf5.Datatype.VLEN:
        if t.vlen_string:
            return np.dtype('O', metadata={'vlen': str if t.charset == 1 else bytes})
        return np.dtype('O', metadata={'vlen': h5py_dtype(t.base)})
    if t.cls == _hdf5.Datatype.REFERENCE:
        return np.dtype('O', metadata={'ref': RegionReference if t.ref_kind == 1 else Reference})
    if t.cls == _hdf5.Datatype.ENUM and t.h5py_dtype.kind != 'b':
        return np.dtype(t.base.dtype, metadata={'enum': dict(t.enum)})
    if t.cls == _hdf5.Datatype.STRING:
        return np.dtype(t.dtype.str, metadata={'h5py_encoding': 'utf-8' if t.charset == 1 else 'ascii'})
    if t.cls == _hdf5.Datatype.COMPOUND and any(
            m[2].cls == _hdf5.Datatype.REFERENCE or m[2].holds_vlen() for m in t.members):
        return _in_memory(t)
    return t.h5py_dtype


def _in_memory(t):
    """A compound's dtype laid out as h5py's are: a string is a pointer there,
    a sequence a pointer and a length, not the 16 bytes of the file's."""
    names, formats, at = [], [], []
    moved = 0
    for name, off, mt in t.members:
        hd, held = _held(mt)
        names.append(name)
        formats.append(hd)
        at.append(off + moved)
        moved += held - mt.size
    return np.dtype({'names': names, 'formats': formats, 'offsets': at, 'itemsize': t.size + moved})


def _held(t):
    """A member's dtype, and the bytes it takes, in memory."""
    if t.cls == _hdf5.Datatype.VLEN:
        return h5py_dtype(t), 8 if t.vlen_string else 16
    if t.cls == _hdf5.Datatype.ARRAY and t.base.holds_vlen():
        hd, held = _held(t.base)
        return np.dtype((hd, t.dims)), held * int(np.prod(t.dims))
    return h5py_dtype(t), t.size


def _objects(shape, dt, items):
    out = np.empty(shape, dt)
    out.reshape(-1)[:] = _hdf5.object_array(list(items), out.size)
    return out


def _retyped(values, dt):
    """An object array as one of h5py's dtype, with its metadata."""
    out = np.empty(values.shape, dt)
    out[...] = values
    return out


_PADDED_IN = (_hdf5.Datatype.STRING, _hdf5.Datatype.COMPOUND, _hdf5.Datatype.ARRAY)


def _padded(t):
    """Whether t holds fixed strings the library would change on reading:
    those ended by a NUL (bytes after it are not the string) or by spaces."""
    if t.cls == _hdf5.Datatype.STRING:
        return t.strpad != 1
    if t.cls == _hdf5.Datatype.ARRAY:
        return _padded(t.base)
    return t.cls == _hdf5.Datatype.COMPOUND and any(_padded(m[2]) for m in t.members)


def _trim(v, t):
    """Fixed strings in v, in place, as HDF5 converts them for h5py's type
    (padded with NULs): cut at a first NUL, or trailing spaces gone."""
    if t.cls == _hdf5.Datatype.ARRAY:
        _trim(v, t.base)
    elif t.cls == _hdf5.Datatype.COMPOUND:
        for name, _, mt in t.members:
            if _padded(mt):
                _trim(v[name], mt)
    elif t.strpad != 1 and t.size and v.size:
        b = np.frombuffer(v.tobytes(), np.uint8).reshape(v.shape + (t.size,))
        if t.strpad == 0:
            keep = np.logical_and.accumulate(b != 0, axis=-1)
        else:
            keep = np.logical_or.accumulate((b != 32)[..., ::-1], axis=-1)[..., ::-1]
        v[...] = np.frombuffer((b * keep).tobytes(), v.dtype).reshape(v.shape)


def _to_h5py(values, t, strings=bytes, top=True):
    """Values decoded by the parser in h5py's form: variable-length strings
    as bytes from datasets and str from attributes, object arrays carrying
    h5py's dtype metadata (vlen, h5py_encoding), references as Reference
    objects (also inside compounds), bool and complex as h5py maps them."""
    arr = isinstance(values, np.ndarray)
    if arr and t.cls in _PADDED_IN and _padded(t):
        values = values.copy()
        _trim(values, t)
    if t.cls == _hdf5.Datatype.ARRAY and t.base.holds_vlen():
        return _to_h5py(values, t.base, strings, False)
    if t.cls == _hdf5.Datatype.VLEN and t.vlen_string:
        conv = (lambda x: x) if strings is str else             (lambda x: x.encode('utf-8') if isinstance(x, str) else x)
        if not arr:
            return conv(values)
        if values.size and isinstance(values.flat[0], strings):
            return _retyped(values, h5py_dtype(t))  # (as they are: all alike, from one decoding)
        return _objects(values.shape, h5py_dtype(t), (conv(x) for x in values.reshape(-1)))
    if t.cls == _hdf5.Datatype.VLEN:
        if not arr:
            return values
        dt = h5py_dtype(t) if top else np.dtype('O')
        if t.base.cls in (0, 1) and not t.base.odd():
            return _retyped(values, dt)
        inner = (_to_h5py(x, t.base, strings, False) for x in values.reshape(-1))
        return _objects(values.shape, dt, inner)
    if t.cls == _hdf5.Datatype.REFERENCE and t.ref_kind == 0:
        if not arr:
            return values
        if values.dtype.itemsize == 8:
            refs = map(Reference, np.frombuffer(values.tobytes(), '<u8').tolist())
        else:
            refs = (Reference(int.from_bytes(bytes(x)[:8], 'little')) for x in values.reshape(-1))
        return _objects(values.shape, h5py_dtype(t) if top else np.dtype('O'), refs)
    if t.cls == _hdf5.Datatype.REFERENCE and t.ref_kind == 1 and arr:
        refs = (RegionReference(int.from_bytes(bytes(x)[:8], 'little'), int.from_bytes(bytes(x)[8:12], 'little'))
                for x in values.reshape(-1))
        return _objects(values.shape, h5py_dtype(t) if top else np.dtype('O'), refs)
    if t.cls == _hdf5.Datatype.COMPOUND and arr and t.h5py_dtype.kind != 'c' and any(
            m[2].cls == _hdf5.Datatype.REFERENCE or m[2].holds_vlen() or m[2].retyped() for m in t.members):
        out = np.zeros(values.shape, _in_memory(t))
        for name, off, mt in t.members:
            out[name] = _to_h5py(values[name], mt, bytes, False)
        return out
    if t.cls == _hdf5.Datatype.ENUM and arr and (top and t.h5py_dtype.kind != 'b' or t.retyped() and t.size > 1):
        return values.astype(h5py_dtype(t))         # (not a view: the values are native by now)
    if t.cls == _hdf5.Datatype.STRING and arr and top:
        enc = 'utf-8' if t.charset == 1 else 'ascii'
        return values.view(np.dtype(values.dtype.str, metadata={'h5py_encoding': enc}))
    hd = t.h5py_dtype
    if hd is not None and hd.subdtype is not None:
        return values
    if arr and hd is not None and hd.kind == 'c':
        # (a read's values are native by now, an attribute's or a fill's as stored)
        first = values.dtype if values.dtype.fields is None else values.dtype[0]
        return values.view(hd).astype(_h5read.native(hd)) if first.byteorder == '>' else values.view(_h5read.native(hd))
    if arr and hd is not None and hd != values.dtype and values.dtype.fields is None:
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


def _member_of(file, name):
    """A pickled object of a file, where it is unpickled."""
    return file[name]


def _reopened(name, kwargs):
    """A pickled File: opened again where it is unpickled."""
    return File(name, **kwargs)


class HLObject:
    def __init__(self, file, address, name, header):
        self.file = file
        self._address = address
        self._name = name
        self._header = header
        self.attrs = AttributeManager(self)

    def __reduce__(self):
        return _member_of, (self.file, self.name)

    @property
    def name(self):
        if self._name == '/?':                      # (reached by a reference: looked for when asked)
            self._name = self.file._path_of(self._address)
        return self._name

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
        if isinstance(path, RegionReference):
            f = self.file._f
            return self.file._object(_hdf5._uint(f.global_heap_object(path.address, path.index), 0, f.O), None)
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
            link = g._g.link(p)
            if link is None:
                raise KeyError("Unable to open object (object '%s' doesn't exist)" % p)
            kind, target = link
            name = posixpath.join(g.name, p)
            if kind == 'soft':
                g = g[target]
                if g.file is self.file:             # (by the name it was asked for, as h5py)
                    g = g.file._object(g._address, name)
                continue
            if kind == 'external':
                g = self.file._follow(target, p)
                continue
            if kind != 'hard':
                raise KeyError('link of an unknown kind: %s' % path)
            g = g.file._object(target, name)
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
            self.file._by_name.add(g.name)          # (what a loop's readahead follows)
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
        self._reader.strings = bytes
        self._reader.plain = (6,)
        self._quick = None                          # see _quick (False: no read of it can go so)

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
        dt = self.dtype                             # (h5py's: an object's size in memory, not in the file)
        return 0 if self.size is None else self.size * dt.itemsize

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

    def virtual_sources(self):
        """[VDSmap(vspace, file_name, dset_name, src_space)] of a virtual
        dataset, as h5py (the spaces: fastnc._vds.SelectionSpace, with the
        h5py SpaceID methods for reading a selection)."""
        if not self.is_virtual:
            raise RuntimeError('Not a virtual dataset')
        from . import _vds
        return _vds.virtual_sources(self._ds)

    @property
    def external(self):
        """[(file name, offset, size)] of external storage, or None."""
        ext = self._ds.external
        return list(ext) if ext else None

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
        q = self._quick
        if q is None:
            q = self._quick = _quick.reader(self._reader)
        if q:
            v = q(key)
            if v is not None:
                return v
        if isinstance(key, RegionReference):
            return self._region(key)
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
        t = self._ds.type
        v = self._read(k)
        if t.cls == _hdf5.Datatype.COMPOUND and t.holds_vlen():
            v = self.file._f.decode_members(t, v)
        v = _to_h5py(v, t)
        if not self._ds.shape and (key is Ellipsis or (isinstance(key, tuple) and key == (Ellipsis,))):
            return np.asarray(v)                    # dset[...] of a scalar dataset: 0-d array
        if isinstance(v, np.ndarray) and v.ndim == 0:
            return v[()]
        return v

    def _region(self, ref):
        """dset[region reference], in the shape h5py guesses for it: a box that
        fills its bounds, else one dimension (values in the order of the dataset,
        points as listed)."""
        f = self.file._f
        data = bytes(f.global_heap_object(ref.address, ref.index))
        if _hdf5._uint(data, 0, f.O) != self._address:
            raise ValueError('Region reference must point to this dataset')
        rank = len(self.shape)
        sel = _hdf5.region_selection(data[f.O:], rank)
        if sel is None:
            return np.zeros((0,) * rank, self.dtype)
        if isinstance(sel, str):
            return self[()]
        if isinstance(sel, np.ndarray):                 # points
            lo, hi = sel.min(axis=0), sel.max(axis=0)
            box = self[tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))]
            return box[tuple((sel - lo).T)]
        first, last = sel
        lo, hi = first.min(axis=0), last.max(axis=0)
        box = self[tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))]
        if len(first) == 1:
            return box
        mask = np.zeros(box.shape, bool)
        for a, b in zip((first - lo).tolist(), (last - lo + 1).tolist()):
            mask[tuple(slice(x, y) for x, y in zip(a, b))] = True
        out = box[mask]
        shape = tuple(int(mask.any(axis=tuple(k for k in range(rank) if k != d)).sum()) for d in range(rank))
        return out.reshape(shape) if np.prod(shape) == out.size else out

    def _read(self, k):
        """The values of index k, through the group's readahead if this reads
        all of a dataset: a loop over a group's datasets decoded together."""
        shape = self._ds.shape
        if not _readahead.ENABLED or not shape or self._name == '/?' or not _readahead.whole(k, shape):
            return self._reader.read(k, scalars=False)
        ahead, me = self.file._ahead(self)
        got = ahead.take(me)
        if got is not None:
            return got[0]
        v = self._reader.read(k, scalars=False)
        ahead.note(me)
        return v

    def read_direct(self, dest, source_sel=None, dest_sel=None):
        """Read into an existing array (dest[dest_sel] = self[source_sel])."""
        if self._ds.null:
            raise TypeError('Empty datasets have no numpy representation')
        v = self[source_sel if source_sel is not None else ()]
        if dest_sel is None:
            dest[...] = v
        else:
            dest[dest_sel] = v

    def __array__(self, dtype=None, copy=None):
        if self._ds.null:
            raise TypeError('Empty datasets have no numpy representation')
        a = np.asarray(self[()])
        return a.astype(dtype) if dtype is not None else a

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return '<HDF5 dataset "%s": shape %s, type "%s">' % (
            posixpath.basename(self.name), self.shape, self.dtype.str if self.dtype is not None else '?')


class _Member:
    """A dataset as _readahead sees a variable: raw values are what it gets."""
    __slots__ = ('_bv', 'shape', 'dtype', '_reader', 'name')

    def __init__(self, ds):
        self._bv, self._reader = self, ds._reader
        self.shape = tuple(ds._ds.shape)
        t = ds._ds.type.dtype
        self.dtype = np.dtype(object) if t is None else _h5read.native(t)

    def _short(self):
        return False

    def _probe(self):
        return None


class _Siblings:
    """A group's datasets, in the order h5py lists them, as _readahead sees a
    group's variables. Looked at only once two have been read in a row."""

    def __init__(self, file, path):
        self.file, self.path = file, path
        self.members = {}
        self.seen = None                            # the first two datasets read, as link names

    def member(self, ds):
        m = self.members.get(ds._address)
        if m is None:
            m = self.members[ds._address] = _Member(ds)
            m.name = posixpath.basename(ds.name)
        return m

    def arrange(self, first, second):
        """Two datasets just read in a row: which order a loop goes in."""
        self.seen = first.name, second.name

    def _order(self, links):
        """The links in the order of the program's loop. A group can be walked
        in two: by iterating it (creation order, if the file tracks it) and by
        visititems (names): whichever puts the second read soonest after the
        first."""
        names = sorted(links)
        order = list(links)
        if self.path in self.file._by_name:
            return names
        if self.seen is None or names == order or not all(s in links for s in self.seen):
            return order

        def gap(x):
            return (x.index(self.seen[1]) - x.index(self.seen[0])) % len(x)
        return names if gap(names) < gap(order) else order

    @property
    def variables(self):
        g = self.file[self.path]
        out = {}
        links = g._linkmap()
        for name in self._order(links):
            kind, target = links[name]
            if kind == 'hard' and target not in out:
                obj = self.file._object(target, posixpath.join(self.path, name))
                if isinstance(obj, Dataset) and not obj._ds.null:
                    out[target] = self.member(obj)
        return out


class File(Group):
    """An open HDF5 file (h5py.File), read-only."""

    def __init__(self, name, mode='r', **kwargs):
        if mode not in ('r', None):
            raise ValueError('fastnc reads only (mode must be "r")')
        self._opened = name, kwargs
        _pool.fast_thread()
        self._others = {}                           # files external links lead to
        self._source = _remote.open_source(name, kwargs.get('storage_options'))
        if self._source is None:
            self.filename = os.fspath(name)
            self._fh = open(self.filename, 'rb')
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        else:                                       # a URL or a file-like object (_remote)
            self.filename, self._fh, self._mm = self._source.name, None, _remote.readonly(self._source)
        self._src = np.frombuffer(self._mm, np.uint8)
        self._f = _hdf5.H5File(self._mm, source=self._source)
        if self._source is None:
            self._f.path = self.filename                # positional reads for big batches
        self._objects = {}
        self._aheads = {}                           # a group's path -> its Readahead
        self._by_name = set()                       # groups visititems has gone through
        self.mode = 'r'
        Group.__init__(self, self, self._f.root_address, '/')
        self.file = self

    def __reduce__(self):
        return _reopened, self._opened

    def _path_of(self, address):
        """A path to the object at an address, or None: h5py asks the library,
        which looks for it. Found once, in one walk of the file's groups."""
        paths = self.__dict__.get('_paths')
        if paths is None:
            paths = self._paths = {}
            todo = [(self, '')]
            seen = {self._address}
            while todo:
                g, base = todo.pop()
                for name, (kind, target) in g._linkmap().items():
                    if kind != 'hard':
                        continue
                    path = base + '/' + name
                    paths.setdefault(target, path)
                    if target not in seen:
                        seen.add(target)
                        obj = self._object(target, path)
                        if isinstance(obj, Group):
                            todo.append((obj, path))
        return '/' if address == self._address else paths.get(address)

    def _ahead(self, ds):
        """The readahead of ds's group, and ds as it knows it."""
        path = posixpath.dirname(ds.name) or '/'
        ahead = self._aheads.get(path)
        if ahead is None:
            ahead = self._aheads.setdefault(path, _readahead.Readahead(_Siblings(self, path)))
        return ahead, ahead.group.member(ds)

    def _object(self, address, name):
        obj = self._objects.get(address)
        if obj is not None and (name is None or obj._name == name):
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

    def _follow(self, target, name):
        """What an external link names: the object in that file, which is
        looked for beside this one (as the library does without a prefix) and
        kept open until this one closes."""
        if target is None:
            raise KeyError('external link %s: of a kind not understood' % name)
        there, inside = target
        path = there if os.path.isabs(there) else os.path.join(os.path.dirname(os.path.abspath(self.filename)), there)
        if not os.path.exists(path):
            raise KeyError('Unable to open external link %s (%s not found)' % (name, there))
        path = os.path.normcase(os.path.normpath(path))
        f = self._others.get(path)
        if f is None:
            f = self._others[path] = File(path)
        return f[inside]

    def close(self):
        for f in self._others.values():
            f.close()
        self._others.clear()
        if getattr(self, '_f', None) is not None:
            self._f.close()
        self._objects = {}
        self._aheads = {}
        self._src = None
        mm, self._mm = self._mm, None
        if mm is not None and hasattr(mm, 'close'):
            try:
                mm.close()
            except BufferError:
                pass
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self._source is not None:
            self._source.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __bool__(self):
        return self._mm is not None

    def __repr__(self):
        return '<HDF5 file "%s" (mode r)>' % os.path.basename(self.filename)
