"""The netCDF4-python-style front end: Dataset, Group, Dimension, Variable.

Switching from netCDF4 should be an import change: the same attributes
and methods, the same values (raw and masked/scaled), read-only. A
Dataset opens netCDF-3 (classic, 64-bit offset, CDF-5) through `_nc3`,
netCDF-4 / plain HDF5 through `_nc4` and HDF4 (SD) through `_hdf4`;
each gives backend objects (docs/contracts.md, section 2) wrapped here.
"""
import mmap
import os

import numpy as np

from . import _hdf5, _select


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


def _open_bytes(path):
    """The file's bytes, mapped read-only (and the handle keeping it open)."""
    fh = open(path, 'rb')
    try:
        size = os.fstat(fh.fileno()).st_size
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) if size else b''
    except Exception:
        fh.close()
        raise
    return fh, mm


def _format(head):
    if head[:3] == b'CDF':
        return {1: 'NETCDF3_CLASSIC', 2: 'NETCDF3_64BIT_OFFSET', 5: 'NETCDF3_64BIT_DATA'}.get(head[3])
    if _hdf5.find_superblock(head) is not None:
        return 'HDF5'
    if head[:4] == b'\x0e\x03\x13\x01':
        return 'HDF4'
    return None


class Dimension:
    """A netCDF dimension (netCDF4.Dimension)."""

    def __init__(self, group, name, size, unlimited):
        self._group, self.name, self._size, self._unlimited = group, name, size, unlimited

    @property
    def size(self):
        return self._size

    def __len__(self):
        return self._size

    def isunlimited(self):
        return self._unlimited

    def group(self):
        return self._group

    def __repr__(self):
        if self._unlimited:
            return "<class 'fastnc.Dimension'> (unlimited): name = '%s', size = %d" % (self.name, self._size)
        return "<class 'fastnc.Dimension'>: name = '%s', size = %d" % (self.name, self._size)


class CompoundType:
    """A netCDF compound type (netCDF4.CompoundType)."""

    def __init__(self, name, dtype):
        self.name, self.dtype = name, dtype
        self.dtype_view = dtype

    def __repr__(self):
        return "<class 'fastnc.CompoundType'>: name = '%s', numpy dtype = %s" % (self.name, self.dtype)


class EnumType:
    """A netCDF enum type (netCDF4.EnumType)."""

    def __init__(self, name, dtype, enum_dict):
        self.name, self.dtype, self.enum_dict = name, dtype, enum_dict

    def __repr__(self):
        return "<class 'fastnc.EnumType'>: name = '%s', numpy dtype = %s, fields/values =%s" % (
            self.name, self.dtype, self.enum_dict)


class VLType:
    """A netCDF variable-length type (netCDF4.VLType)."""

    def __init__(self, name, dtype):
        self.name, self.dtype = name, dtype

    def __repr__(self):
        return "<class 'fastnc.VLType'>: name = '%s', numpy dtype = %s" % (self.name, self.dtype)


class Variable:
    """A netCDF variable (netCDF4.Variable), read-only. Indexing returns
    values with netCDF4-python's automatic masking and scaling unless
    turned off, exactly as netCDF4-python would."""

    def __init__(self, group, bv):
        ds = group._dataset
        self.__dict__.update(_group=group, _bv=bv, name=bv.name, _mask=ds._auto_mask,
                             _scale=ds._auto_scale, _always_mask=ds._always_mask,
                             _chartostring=ds._chartostring, _device=ds._device)

    # -- metadata
    @property
    def dimensions(self):
        return self._bv.dimensions

    @property
    def shape(self):
        bv = self._bv
        if hasattr(bv, 'dims'):                     # unlimited dims share the longest length
            return tuple(max(s, d.size) if d is not None and d.unlimited else s
                         for s, d in zip(bv.shape, bv.dims))
        return tuple(bv.shape)

    @property
    def dtype(self):
        return self._bv.dtype

    @property
    def datatype(self):
        """The dtype, or for user-defined types a CompoundType / EnumType /
        VLType carrying its name (as netCDF4-python)."""
        ut = getattr(self._bv, 'user_type', None)
        t = ut() if ut is not None else None
        return t if t is not None else self._bv.dtype

    @property
    def ndim(self):
        return len(self._bv.shape)

    @property
    def size(self):
        return int(np.prod(self.shape)) if self.shape else 1

    def __len__(self):
        if not self.shape:
            raise TypeError('len() of unsized object')
        return self.shape[0]

    def ncattrs(self):
        return list(self._bv.attrs)

    def getncattr(self, name):
        try:
            return self._bv.attrs[name]
        except KeyError:
            raise AttributeError("NetCDF: Attribute not found: %s" % name) from None

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        attrs = self.__dict__.get('_bv').attrs
        if name in attrs:
            return attrs[name]
        if name == '__dict__':
            raise AttributeError(name)
        raise AttributeError("NetCDF: Attribute not found: %s" % name)

    def __setattr__(self, name, value):
        if name.startswith('_') or name == 'name':
            object.__setattr__(self, name, value)
        else:
            raise AttributeError('fastnc reads only: cannot set attribute %r' % name)

    @property
    def __dict__attrs(self):
        return dict(self._bv.attrs)

    def group(self):
        return self._group

    def filters(self):
        return self._bv.filters

    def chunking(self):
        c = self._bv.chunks
        if c is None:
            # netCDF-4 / HDF4 contiguous storage says so; netCDF-3 has no choice (None)
            return None if self._group._dataset.disk_format in ('NETCDF3', 'CDF5') else 'contiguous'
        return list(c)

    def endian(self):
        return self._bv.endian

    def get_var_chunk_cache(self):
        return (0, 0, 0.0)

    def set_auto_maskandscale(self, flag):
        self._mask = self._scale = bool(flag)

    def set_auto_mask(self, flag):
        self._mask = bool(flag)

    def set_auto_scale(self, flag):
        self._scale = bool(flag)

    def set_always_mask(self, flag):
        self._always_mask = bool(flag)

    def set_auto_chartostring(self, flag):
        self._chartostring = bool(flag)

    def isunlimited(self):
        bv = self._bv
        return any(d is not None and d.unlimited for d in getattr(bv, 'dims', ()))

    # -- values
    def __getitem__(self, key):
        return self.read(key)

    def read(self, key=Ellipsis, out=None, device=None, threads=None):
        """Values for key (any netCDF4-python index), masked and scaled as
        set. device 'gpu' returns a cupy array (identical values)."""
        device = device or self._device
        if device == 'gpu':
            from . import _gpu
            return _gpu.read(self, key, threads=threads)
        k = key if isinstance(key, tuple) else (key,)
        bv = self._bv
        box, post, drop = _select.normalize(k, bv.shape)
        if hasattr(bv, '_reader'):                  # the HDF5 reader takes any index
            raw = bv.read(k) if threads is None else bv.read(k, threads=threads)
        else:
            # other backends read boxes (contract section 2): reduce the index
            # here, and apply the rest (dropped, reversed, picked) afterwards
            raw = bv.read(tuple(slice(s, e, st) for s, e, st in box))
            raw = _select.finish(raw, post, drop)
        return self._post(raw, box, drop, threads)

    def _post(self, raw, box, drop, threads=None):
        """netCDF4-python's masking / scaling / char conversion of raw values."""
        bv = self._bv
        chars = self._chartostring and bv.dtype == np.dtype('S1') and '_Encoding' in bv.attrs
        if not (self._mask or self._scale or chars):
            return raw
        from . import _maskscale
        dt = self.datatype
        return _maskscale.apply(raw, bv.attrs, bv.dtype, mask=self._mask, scale=self._scale,
                                always_mask=self._always_mask, chartostring=self._chartostring,
                                var_dtype=None if isinstance(dt, np.dtype) or dt is str else dt,
                                no_fill=getattr(bv, 'no_fill', False), var_shape=tuple(bv.shape),
                                key=_norm_key(box, drop), threads=threads)

    def __array__(self, dtype=None, copy=None):
        a = np.asarray(self[...])
        return a.astype(dtype) if dtype is not None else a

    def __repr__(self):
        return "<class 'fastnc.Variable'>\n%s %s(%s)\nshape = %s" % (
            self.dtype, self.name, ', '.join(self.dimensions), self.shape)


class Group:
    """A netCDF group (netCDF4.Group)."""

    def __init__(self, dataset, bg, parent):
        self._dataset = dataset
        self._bg = bg
        self.parent = parent
        self.name = bg.name
        self.dimensions = {name: Dimension(self, name, *_dim_info(d)) for name, d in bg.dimensions.items()}
        self.variables = {name: Variable(self, bv) for name, bv in bg.variables.items()}
        self.groups = {name: Group(dataset, sub, self) for name, sub in bg.groups.items()}

    @property
    def path(self):
        if self.parent is None:
            return '/'
        p = self.parent.path
        return (p if p.endswith('/') else p + '/') + self.name

    def ncattrs(self):
        return list(self._bg.attrs)

    def getncattr(self, name):
        try:
            return self._bg.attrs[name]
        except KeyError:
            raise AttributeError("NetCDF: Attribute not found: %s" % name) from None

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        attrs = self.__dict__.get('_bg').attrs
        if name in attrs:
            return attrs[name]
        raise AttributeError("NetCDF: Attribute not found: %s" % name)

    def __getitem__(self, path):
        g = self
        parts = [p for p in path.split('/') if p]
        if path.startswith('/'):
            while g.parent is not None:
                g = g.parent
        for k, p in enumerate(parts):
            last = k == len(parts) - 1
            if last and p in g.variables:
                return g.variables[p]
            if p in g.groups:
                g = g.groups[p]
            elif p == '..' and g.parent is not None:
                g = g.parent
            else:
                raise IndexError('%s not found in %s' % (p, g.path))
        return g

    def _walk(self):
        yield self
        for sub in self.groups.values():
            yield from sub._walk()

    def load(self, names=None, recursive=False, threads=None):
        """{name: values} for the named variables (default: all of this group,
        and of its subgroups too if recursive, keyed by path), each as
        var[...] would give it. Decoded together: every chunk of every
        variable in one pass of the cores, biggest first -- a file of many
        small variables no longer waits on each variable in turn."""
        if names is None:
            groups = list(self._walk()) if recursive else [self]
            vs = [(('%s/%s' % (g.path.rstrip('/'), k)).lstrip('/') if recursive else k, v)
                  for g in groups for k, v in g.variables.items()]
        else:
            vs = [(n, self[n]) for n in names]
        raws = {}
        batch = [(k, v) for k, v in vs if getattr(v._bv, '_reader', None) is not None
                 and v._bv.ds.type.cls in (0, 1, 4, 8) and v._bv.ds.shape]
        if batch:
            from ._h5read import read_many
            for (k, v), a in zip(batch, read_many([v._bv._reader for _, v in batch], threads=threads)):
                raws[k] = a
        h4 = [(k, v) for k, v in vs if type(v._bv).__module__.endswith('_hdf4')]
        if h4:
            from ._hdf4 import read_many as read_many4
            for (k, v), a in zip(h4, read_many4([v._bv for _, v in h4], threads=threads)):
                if a is not None:
                    raws[k] = a
        out = {}
        for k, v in vs:
            if k not in raws:
                out[k] = v.read(Ellipsis, threads=threads)
                continue
            out[k] = v._post(raws[k], [(0, n, 1) for n in v._bv.shape], (), threads)
        return out

    def set_auto_maskandscale(self, flag):
        for g in self._walk():
            for v in g.variables.values():
                v.set_auto_maskandscale(flag)

    def set_auto_mask(self, flag):
        for g in self._walk():
            for v in g.variables.values():
                v.set_auto_mask(flag)

    def set_auto_scale(self, flag):
        for g in self._walk():
            for v in g.variables.values():
                v.set_auto_scale(flag)

    def set_always_mask(self, flag):
        for g in self._walk():
            for v in g.variables.values():
                v.set_always_mask(flag)

    def set_auto_chartostring(self, flag):
        for g in self._walk():
            for v in g.variables.values():
                v.set_auto_chartostring(flag)

    @property
    def cmptypes(self):
        return {k: t for k, t in getattr(self._bg, 'types', {}).items() if t.cls == 6}

    @property
    def enumtypes(self):
        return {k: t for k, t in getattr(self._bg, 'types', {}).items() if t.cls == 8}

    @property
    def vltypes(self):
        return {k: t for k, t in getattr(self._bg, 'types', {}).items() if t.cls == 9}

    def __repr__(self):
        return "<class 'fastnc.Group'>\ngroup %s:\n    dimensions: %s\n    variables: %s\n    groups: %s" % (
            self.path, ', '.join(self.dimensions), ', '.join(self.variables), ', '.join(self.groups))


def _norm_key(box, drop):
    """The index read, one int (dropped dimension) or slice per dimension."""
    return tuple(s if i in drop else slice(s, e, st) for i, (s, e, st) in enumerate(box))


def _dim_info(d):
    if isinstance(d, tuple):
        return d[0], d[1]
    return d.size, d.unlimited


class Dataset(Group):
    """An open netCDF / HDF5 file (netCDF4.Dataset), read-only.

    device: 'cpu' (default) or 'gpu' (values come back as cupy arrays).
    filename may also be an http(s):// or s3:// URL (anonymous; options
    such as region in storage_options) or a file-like object: its bytes
    are then fetched as needed (`_remote`).
    """

    def __init__(self, filename, mode='r', device=None, **kwargs):
        if mode != 'r':
            raise ValueError('fastnc reads only (mode must be "r")')
        self._auto_mask = self._auto_scale = True
        self._always_mask = True
        self._chartostring = True
        self._device = device
        self._fh = self._mm = None
        self._source = src = _remote.open_source(filename, kwargs.get('storage_options'))
        if src is None:
            self._filepath = os.fspath(filename)
            with open(self._filepath, 'rb') as fh:
                head = fh.read(4096)
        else:                                       # a URL or a file-like object
            self._filepath = src.name
            head = src.buf[:4096].tobytes()
        fmt = _format(head)
        if fmt is None:
            raise OSError('[Errno -51] NetCDF: Unknown file format: %r' % self._filepath)
        if fmt == 'HDF5':
            from ._nc4 import NC4File
            self._fh, self._mm = _open_bytes(self._filepath) if src is None else (None, _remote.readonly(src))
            self._backend = NC4File(_hdf5.H5File(self._mm, source=src), np.frombuffer(self._mm, np.uint8))
            self.file_format = self._backend.data_model
            self.disk_format = 'HDF5'
        elif fmt == 'HDF4':
            from ._hdf4 import HDF4File
            self._fh, self._mm = _open_bytes(self._filepath) if src is None else (None, _remote.whole(src))
            self._backend = HDF4File(self._mm, np.frombuffer(self._mm, np.uint8), self._filepath)
            self.file_format = self._backend.data_model
            self.disk_format = 'HDF4'
        else:
            from ._nc3 import NC3File                 # positional reads of its own
            self._backend = NC3File(self._filepath) if src is None else NC3File(src.name, source=src)
            self.file_format = fmt
            self.disk_format = {'NETCDF3_CLASSIC': 'NETCDF3', 'NETCDF3_64BIT_OFFSET': 'NETCDF3',
                                'NETCDF3_64BIT_DATA': 'CDF5'}[fmt]
        self.data_model = self._backend.data_model
        Group.__init__(self, self, self._backend.root, None)
        self.name = '/'

    def filepath(self):
        return self._filepath

    @property
    def isopen(self):
        return getattr(self, '_backend', None) is not None

    def close(self):
        b = getattr(self, '_backend', None)
        if b is not None:
            b.close()
        self._backend = None
        mm, self._mm = getattr(self, '_mm', None), None
        if mm is not None and hasattr(mm, 'close'):
            try:
                mm.close()
            except BufferError:
                pass                                # arrays still view it: closed when they go
        fh = getattr(self, '_fh', None)
        if fh is not None:
            fh.close()
        self._fh = None
        if getattr(self, '_source', None) is not None:
            self._source.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return "<class 'fastnc.Dataset'>\nroot group (%s data model, file format %s):\n    dimensions: %s\n    variables: %s\n    groups: %s" % (
            self.data_model, self.disk_format, ', '.join(self.dimensions), ', '.join(self.variables),
            ', '.join(self.groups))
