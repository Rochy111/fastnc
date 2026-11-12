"""The netCDF4-python-style front end: Dataset, Group, Dimension, Variable.

Switching from netCDF4 should be an import change: the same attributes
and methods, the same values (raw and masked/scaled), read-only. A
Dataset opens netCDF-3 (classic, 64-bit offset, CDF-5) through `_nc3`,
netCDF-4 / plain HDF5 through `_nc4` and HDF4 (SD) through `_hdf4`;
each gives backend objects (docs/contracts.md, section 2) wrapped here.
"""
import itertools
import math
import mmap
import os

import numpy as np

from . import _hdf5, _hw, _pool, _quick, _readahead, _select

FUSED = True                 # masked / scaled HDF5 reads through _fused when it can
FUSED_LOAD_MIN = None           # ... in load(): variables of at least this many raw bytes
#                                 (None: this CPU's last cache: an array that fits is still
#                                 there for the mask / scale pass, and fusing saves no memory
#                                 traffic but costs its own pass of the pool)
_lazy = {}                  # modules a raw read never needs (numpy.ma: 8 ms of a fresh one)


def _masking():
    """fastnc._maskscale, imported at first need (`from . import` in a
    function runs the import machinery every time: 950 times a WRF load)."""
    m = _lazy.get('_maskscale')
    if m is None:
        from . import _maskscale
        m = _lazy['_maskscale'] = _maskscale
    return m


def _fusing():
    m = _lazy.get('_fused')
    if m is None:
        from . import _fused
        m = _lazy['_fused'] = _fused
    return m


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


def _past_user_block(fh):
    """'HDF5' if its signature lies at a power of two past the first 4096 bytes."""
    size = os.fstat(fh.fileno()).st_size
    at = 4096
    while at + 8 <= size:
        fh.seek(at)
        if fh.read(8) == _hdf5.SIGNATURE:
            return 'HDF5'
        at *= 2
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
        # (netCDF4-python's quotes the class: it prints it with %r)
        if self._unlimited:
            return "\"<class 'fastnc.Dimension'>\" (unlimited): name = '%s', size = %d" % (self.name, self._size)
        return "\"<class 'fastnc.Dimension'>\": name = '%s', size = %d" % (self.name, self._size)


class CompoundType:
    """A netCDF compound type (netCDF4.CompoundType)."""

    def __init__(self, name, dtype):
        self.name, self.dtype = name, dtype
        self._view = None

    @property
    def dtype_view(self):
        if self._view is None:
            self._view = _masking()._set_viewdtype(self.dtype)
        return self._view

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


def _attribute(b, name):
    """The netCDF attribute `name` of b (a backend variable or group), as
    netCDF4-python's v.<name> gives it: '_' names too (v._FillValue,
    v._Unsigned, hasattr(v, '_FillValue')), any that ncattrs() lists.
    Python's special names (__array__, __dict__ ...) are never attributes,
    nor is anything before b is set (an object half made or unpickled)."""
    if b is None or (name[:2] == '__' and name[-2:] == '__'):
        raise AttributeError(name)
    attrs = b.attrs
    if name in attrs:
        return attrs[name]
    raise AttributeError("NetCDF: Attribute not found: %s" % name)


class Variable:
    """A netCDF variable (netCDF4.Variable), read-only. Indexing returns
    values with netCDF4-python's automatic masking and scaling unless
    turned off, exactly as netCDF4-python would."""

    def __init__(self, group, bv):
        ds = group._dataset
        self.__dict__.update(_group=group, _bv=bv, name=bv.name, _mask=ds._auto_mask,
                             _scale=ds._auto_scale, _always_mask=ds._always_mask,
                             _chartostring=ds._chartostring, _device=ds._device, _quick=None,
                             _nan=ds._auto_nan)

    def __reduce__(self):
        flags = (self._mask, self._scale, self._always_mask, self._chartostring, self._nan)
        if '_sds_index' in self.__dict__:           # (a data set whose name another holds)
            return _sds_in, (self._group, self.__dict__['_sds_index']) + flags
        return _variable_in, (self._group, self.name) + flags

    # -- metadata
    @property
    def dimensions(self):
        return self._bv.dimensions

    @property
    def shape(self):
        bv = self._bv
        if self._short() or (not hasattr(bv, 'short') and hasattr(bv, 'dims')):
            # unlimited dims share the longest length
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
        t = self.__dict__.get('_datatype')
        if t is None:
            ut = getattr(self._bv, 'user_type', None)
            t = ut() if ut is not None else None
            t = self._datatype = t if t is not None else self._bv.dtype
        return t

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
        return _attribute(self.__dict__.get('_bv'), name)

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
            return None if self._group._dataset.disk_format == 'NETCDF3' else 'contiguous'
        return list(c)

    def endian(self):
        return self._bv.endian

    def get_var_chunk_cache(self):
        return (0, 0, 0.0)

    def set_var_chunk_cache(self, size=None, nelems=None, preemption=None):
        """Accepted and ignored: fastnc's chunk cache sizes itself to the
        machine (netCDF4-python's sets netCDF-C's per-variable cache)."""

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

    # (netCDF4-python's view of the switches, read-only here as there: set_auto_*)
    mask = property(lambda self: self._mask)
    scale = property(lambda self: self._scale)
    always_mask = property(lambda self: self._always_mask)
    chartostring = property(lambda self: self._chartostring)
    auto_complex = property(lambda self: self._group._dataset._auto_complex)

    def get_dims(self):
        """The Dimension of each of this variable's dimensions, looked up from
        its group outwards, as netCDF4-python's."""
        out = []
        for name in self.dimensions:
            g = self._group
            while g is not None and name not in g.dimensions:
                g = g.parent
            out.append(g.dimensions[name] if g is not None else None)
        return tuple(out)

    def get_fill_value(self):
        """The fill value: `_FillValue`, else netCDF's default for the type
        (None where filling is off, or for a type without one), as
        netCDF4-python's."""
        bv = self._bv
        if getattr(bv, 'no_fill', False):
            return None
        if '_FillValue' in bv.attrs:
            return bv.attrs['_FillValue']
        dt = self.datatype
        if not isinstance(dt, np.dtype) or dt.str[1:] not in _masking().DEFAULT_FILLVALS:
            return None
        # (a 0-d array of the type, as netCDF4-python gives it; native order, so
        # the value is right where netCDF4-python byte-swaps big-endian HDF4's)
        return np.array(_masking().DEFAULT_FILLVALS[dt.str[1:]], dt.newbyteorder('='))

    def quantization(self):
        """(significant digits or bits, algorithm) of a quantized variable,
        from netCDF-C's attributes, else None (netCDF4-python's)."""
        for att, alg in (('_QuantizeBitGroomNumberOfSignificantDigits', 'BitGroom'),
                         ('_QuantizeGranularBitRoundNumberOfSignificantDigits', 'GranularBitRound'),
                         ('_QuantizeBitRoundNumberOfSignificantBits', 'BitRound')):
            if att in self._bv.attrs:
                return int(self._bv.attrs[att]), alg
        return None

    def getValue(self):
        """A scalar variable's value (netCDF4-python's)."""
        if self.dimensions:
            raise IndexError('to retrieve values from a non-scalar variable, use slicing')
        return self[slice(None)]

    def set_auto_nan(self, flag):
        """Masked float values as NaN in a plain array, xarray's way, instead
        of a numpy masked array (off by default, as netCDF4-python has it).
        Applies where masking is on; integer results (not scaled to floats)
        stay masked arrays, having no NaN."""
        self._nan = bool(flag)

    def isunlimited(self):
        """Whether one of the variable's dimensions is unlimited (fastnc's:
        netCDF4-python's Variable has no such method)."""
        bv = self._bv
        dims = getattr(bv, 'dims', None)
        if dims is not None:                        # (netCDF-4: the backend knows them)
            return any(d is not None and d.unlimited for d in dims)
        return any(d is not None and d.isunlimited() for d in self.get_dims())

    # -- values
    def __getitem__(self, key):
        return self.read(key)

    def read(self, key=Ellipsis, out=None, device=None, threads=None, mask=None, scale=None,
             nan=None):
        """Values for key (any netCDF4-python index), masked and scaled as
        set. device 'gpu' returns a cupy array (identical values).

        mask, scale, nan: for this read alone, in place of the variable's
        set_auto_mask / set_auto_scale / set_auto_nan (`v.read(k, mask=False,
        scale=False)` is a raw read, and changes nothing another thread sees).
        out: an array of the read's shape and dtype exactly, which gets the
        values: a raw read of a box (ints and slices) is decoded straight
        into it; any other is copied in. Returns out (a masked result: a
        masked array whose data is out)."""
        if mask is not None or scale is not None or nan is not None:
            v = object.__new__(Variable)
            v.__dict__.update(self.__dict__)
            for name, flag in (('_mask', mask), ('_scale', scale), ('_nan', nan)):
                if flag is not None:
                    v.__dict__[name] = bool(flag)
            return v.read(key, out, device, threads)
        try:
            if out is not None:
                return self._read_out(key, out, device, threads)
            r = self._values(key, device, threads)
        except Exception:
            # closed while this read was under way: whatever that made of it
            # (a parser's TypeError, numba typing the bytes that were let go)
            if not self._group._dataset.isopen():
                raise ValueError('the file of variable %s is closed' % self.name) from None
            raise
        if self._nan and self._mask:
            return _nan_filled(self, r, threads)
        return r

    def _read_out(self, key, out, device, threads):
        """read(key, out=...)."""
        if (device or self._device) == 'gpu':
            raise ValueError("out= takes a numpy array: it reads to the CPU (device='cpu')")
        if not isinstance(out, np.ndarray) or not out.flags.writeable:
            raise TypeError('out must be a writable numpy array')
        bv = self._bv
        chars = self._chartostring and bv.dtype == np.dtype('S1') and '_Encoding' in bv.attrs
        scales = self._scale and any(a in bv.attrs for a in ('scale_factor', 'add_offset', '_Unsigned'))
        if (not (self._mask or scales or chars) and bv.shape and isinstance(bv.dtype, np.dtype)
                and out.flags.c_contiguous and not self._short() and not self._null):
            k = key if isinstance(key, tuple) else (key,)
            box, post, drop = _select.normalize(k, bv.shape)
            if post is None:                        # (a box: the backend writes into out)
                full = _select.box_shape(box)
                _check_out(out, [n for i, n in enumerate(full) if i not in drop], bv.dtype)
                if hasattr(bv, '_reader'):
                    r = bv.read(k, out=out) if threads is None else bv.read(k, out=out, threads=threads)
                else:
                    r = bv.read(tuple(s if i in drop else slice(s, e, st)
                                      for i, (s, e, st) in enumerate(box)), out=out)
                if r is not out:
                    np.copyto(out, r)
                return out
        return _into(self.read(key, device='cpu', threads=threads), out)

    def read_async(self, key=Ellipsis, **kwargs):
        """read(key, **kwargs) on a worker of fastnc's: a
        concurrent.futures.Future of its result (fastnc's own), so the next
        read can be under way while this one's values are used."""
        from . import _async
        return _async.submit(self.read, key, **kwargs)

    def memmap(self):
        """The stored values in place, with no copy: a read-only numpy array
        over the file's mapping, for a variable stored uncompressed in one
        piece (netCDF-3 variables, record ones too; contiguous netCDF-4 /
        HDF5 and HDF4 ones). Raw: no mask or scale, the file's byte order,
        and slicing it reads only what is touched. ValueError where there is
        no such piece (compressed or chunked data, a remote file). It stays
        valid while anything holds it, the file closed or not."""
        if not self._group._dataset.isopen():
            raise ValueError('the file of variable %s is closed' % self.name)
        bv = self._bv
        r = getattr(bv, '_reader', None)
        v = r.view() if r is not None else (bv.view() if hasattr(bv, 'view') else None)
        if v is None:
            raise ValueError('%s is not stored in one uncompressed piece of a local file: '
                             'there is nothing to view in place (read it instead)' % self.name)
        return v

    def _result_dtype(self):
        """The dtype a read of this variable gives (masked or not)."""
        bv = self._bv
        r = _masking().apply(np.zeros(1, bv.dtype), bv.attrs, bv.dtype, mask=False, scale=self._scale)
        return np.asarray(r).dtype

    def _values(self, key, device, threads):
        """read() without out or nan."""
        if not self._bv.shape and not _select.whole_scalar(key) and not self._null:
            # a scalar: v[:], v[0], v[[0]] give its value, as in netCDF4-python
            # (a null dataset, no value at all, refuses them as too many indices)
            asked = key
            key, pick = _select.scalar_nc(key)
            if pick:
                if (device or self._device) == 'gpu':
                    from . import _gpu
                    return _gpu._host_then_upload(self, asked, threads)
                return self._pick(threads)
        device = device or self._device
        if self._short():
            # past what is stored: fill values, as netCDF-C gives them
            if device == 'gpu':
                from . import _gpu
                return _gpu._host_then_upload(self, key, threads)
            if not self._group._dataset.isopen:
                raise ValueError('the file of variable %s is closed' % self.name)
            box, post, drop = _select.normalize(key if isinstance(key, tuple) else (key,), self.shape)
            raw = _select.finish(self._bv.read_padded(box, threads), post, drop)
            return self._post(raw, box, drop, threads)
        if device == 'gpu' and not self._null:      # (a null dataset: nothing to put on the card)
            from . import _gpu
            return _gpu.read(self, key, threads=threads)
        k = key if isinstance(key, tuple) else (key,)
        bv = self._bv
        if _readahead.ENABLED and hasattr(bv, '_reader') and _readahead.whole(k, bv.shape):
            # whole variables read in the file's order: decoded several at a time
            ra = self._group._readahead()
            got = ra.take(self)
            if got is None:
                res = self._read(k, threads)
                ra.note(self)
                return res
            box, post, drop = _select.normalize(k, bv.shape)
            return self._post(got[0], box, drop, threads, got[1])
        return self._read(k, threads)

    def _read(self, k, threads):
        bv = self._bv
        q = self._quick
        if q is None:
            q = self._quick = False if getattr(bv, 'widened', False) else _quick.reader(getattr(bv, '_reader', None))
        if q:
            raw = q(k)
            if raw is not None:
                box, _, drop = _select.normalize(k, bv.shape)
                return self._post(raw, box, drop, threads)
        if FUSED and (self._mask or self._scale) and hasattr(bv, '_reader') and not getattr(bv, 'widened', False):
            res = _fusing().read(self, k, threads)  # mask / scale while decoding
            if res is not None:
                return res
        box, post, drop = _select.normalize(k, bv.shape)
        probe = None
        if hasattr(bv, '_reader'):                  # the HDF5 reader takes any index
            raw = bv.read(k) if threads is None else bv.read(k, threads=threads)
        else:
            # other backends read boxes (contract section 2): reduce the index
            # here, and apply the rest (dropped, reversed, picked) afterwards
            sel = tuple(slice(s, e, st) for s, e, st in box)
            if getattr(bv, 'probes', False):
                probe = self._probe(_select.box_shape(box))
                raw = bv.read(sel, probe=probe)
            else:
                raw = bv.read(sel)
            raw = _select.finish(raw, post, drop)
        return self._post(raw, box, drop, threads, probe is not None and probe.proven)

    def _pick(self, threads):
        """v[0] (any integer from -1 up) of a scalar: netCDF4-python reads it
        as the one value of a (1,) array, numpy's scalar, and masks and
        scales that (a 0-d array where always_mask is off, np.void for a
        compound ...), as for an integer index into an array variable."""
        bv = self._bv
        if hasattr(bv, '_reader'):
            raw = bv.read((Ellipsis,)) if threads is None else bv.read((Ellipsis,), threads=threads)
        else:
            raw = bv.read(())
        if type(raw) is np.ndarray and raw.ndim == 0:
            raw = raw[()]
        return self._post(raw, [(0, 1, 1)], (0,), threads, shape=(1,))

    def _probe(self, shape=None):
        """What a read of this shape (default: all of it) can test as it goes
        to spare the pass over its values, or None."""
        if not self._mask:                          # (no numpy.ma for a raw read)
            return None
        ms = _masking()
        if not ms._kernels_ready():
            return None
        bv = self._bv
        if math.prod(bv.shape if shape is None else shape) < ms.FAST_MIN_SIZE:
            return None
        dt = self.datatype
        if not isinstance(dt, np.dtype) or not isinstance(bv.dtype, np.dtype):
            return None
        return ms.probe(bv.attrs, bv.dtype, self._mask, self._scale, getattr(bv, 'no_fill', False))

    @property
    def _null(self):
        """An HDF5 null dataspace: no values at all. netCDF-C will not open
        such a file; here reading one gives None, masked or not."""
        ds = getattr(getattr(self._bv, '_reader', None), 'ds', None)
        return ds is not None and ds.null

    def _short(self):
        """Whether this variable has fewer records than its unlimited
        dimension (netCDF-4 only: netCDF-3 records are the file's)."""
        short = getattr(self._bv, 'short', None)
        return short is not None and short()

    def _fast_plan(self, raw):
        """What _post would hand the kernels raw with, or None if it would
        do something else with it."""
        if not (self._mask or self._scale):
            return None
        bv = self._bv
        dt = self.datatype
        ms = _masking()
        return ms.fast_plan(raw, bv.attrs, bv.dtype,
                            ms._kind(bv.dtype, None if isinstance(dt, np.dtype) or dt is str else dt),
                            self._mask, self._scale, getattr(bv, 'no_fill', False))

    def _post(self, raw, box, drop, threads=None, clean=False, shape=None):
        """netCDF4-python's masking / scaling / char conversion of raw values;
        clean: the read proved no value here is masked; shape: the variable's
        as the index saw it (default: its own)."""
        if raw is None:
            return None                             # (see _null)
        bv = self._bv
        chars = self._chartostring and bv.dtype == np.dtype('S1') and '_Encoding' in bv.attrs
        if not (self._mask or self._scale or chars) or getattr(bv, 'kind', None) == 'compound':
            return raw                              # (complex compounds: as any compound, as they are)
        dt = self.datatype
        return _masking().apply(raw, bv.attrs, bv.dtype, mask=self._mask, scale=self._scale,
                                always_mask=self._always_mask, chartostring=self._chartostring,
                                var_dtype=None if isinstance(dt, np.dtype) or dt is str else dt,
                                no_fill=getattr(bv, 'no_fill', False),
                                var_shape=self.shape if shape is None else shape,
                                key=_norm_key(box, drop), threads=threads, clean=clean)

    def __array__(self, dtype=None, copy=None):
        a = np.asarray(self[...])
        return a.astype(dtype) if dtype is not None else a

    def __repr__(self):
        """As netCDF4-python prints a variable: type, dimensions, attributes,
        unlimited dimensions, shape and whether it is filled."""
        out = ["<class 'fastnc.Variable'>"]
        dt = self.datatype
        kind = ('compound' if isinstance(dt, CompoundType) else 'vlen' if isinstance(dt, VLType) or dt is str
                else 'enum' if isinstance(dt, EnumType) else None)
        out.append('%s %s(%s)' % (kind or self.dtype, self.name, ', '.join(self.dimensions)))
        for name in self.ncattrs():
            out.append('    %s: %s' % (name, self.getncattr(name)))
        if kind:
            out.append('%s data type: %s' % (kind, self.dtype))
        unlim = [n for n, d in zip(self.dimensions, self.get_dims()) if d is not None and d.isunlimited()]
        if self._group.path != '/':
            out.append('path = %s' % self._group.path)
        out.append('unlimited dimensions: %s' % ', '.join(unlim))
        out.append('current shape = %r' % (self.shape,))
        if not kind:
            if getattr(self._bv, 'no_fill', False):
                out.append('filling off')
            elif '_FillValue' in self._bv.attrs:
                out.append('filling on')
            else:
                key = np.dtype(self.dtype).str[1:]
                fv = _masking().DEFAULT_FILLVALS.get(key)
                out.append('filling on, default _FillValue of %s %s' % (fv, 'ignored' if key in ('u1', 'i1') else 'used'))
        return '\n'.join(out)


def _variable(group, bv):
    """The Variable of backend variable bv: of complex values where the file
    opened with auto_complex=True stores them so (_complex)."""
    if group._dataset._auto_complex:
        from . import _complex
        k = _complex.kind(bv)
        if k is not None:
            return Variable(group, _complex.ComplexBackend(bv, k, group))
    return Variable(group, bv)


class _Variables(dict):
    """Group.variables, a dict as in netCDF4-python, that looks a name up by
    reading that variable alone (`d['T']` in a loop over many files: most of
    what an open cost was for the other variables). Anything that needs them
    all reads them all first, and is then the plain dict in the file's order."""

    def __init__(self, group):
        self.group, self.whole = group, False

    def fill(self):
        if self.whole:
            return
        g = self.group
        if g._bg.holds_no_data():
            self.whole = True
            return
        found = dict(dict.items(self))              # (those handed out stay the ones)
        dict.clear(self)
        for name, bv in g._bg.variables.items():
            v = found.get(name)
            dict.__setitem__(self, name, _variable(g, bv) if v is None else v)
        self.whole = True

    def __missing__(self, name):
        if self.whole:
            raise KeyError(name)
        bv = self.group._bg.peek(name)
        if bv is None:
            self.fill()
            return dict.__getitem__(self, name)
        return dict.setdefault(self, name, _variable(self.group, bv))

    def __contains__(self, name):
        try:
            self[name]
        except KeyError:
            return False
        return True

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def __iter__(self):
        self.fill()
        return dict.__iter__(self)

    def __reversed__(self):
        self.fill()
        return dict.__reversed__(self)

    def __len__(self):
        self.fill()
        return dict.__len__(self)

    def __eq__(self, other):
        self.fill()
        return dict.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

    __hash__ = None

    def __repr__(self):
        self.fill()
        return dict.__repr__(self)

    def keys(self):
        self.fill()
        return dict.keys(self)

    def values(self):
        self.fill()
        return dict.values(self)

    def items(self):
        self.fill()
        return dict.items(self)

    def copy(self):
        self.fill()
        return dict(self)


class Group:
    """A netCDF group (netCDF4.Group)."""

    def __init__(self, dataset, bg, parent):
        self._dataset = dataset
        self._bg = bg
        self.parent = parent
        self.name = bg.name
        if hasattr(bg, 'peek'):
            self.variables = _Variables(self)
        else:
            self.variables = {name: _variable(self, bv) for name, bv in bg.variables.items()}

    def __reduce__(self):
        return _group_at, (self._dataset, self.path)

    @property
    def groups(self):
        d = self.__dict__.get('_groups')
        if d is None:
            kids = self.__dict__.get('_kids', {})
            d = self.__dict__['_groups'] = {name: kids.get(name) or Group(self._dataset, sub, self)
                                            for name, sub in self._bg.groups.items()}
        return d

    def _child(self, name):
        """The subgroup called `name`, or None: that one alone where it can
        be (a path walks one group of each level, not the whole file)."""
        d = self.__dict__.get('_groups')
        if d is not None:
            return d.get(name)
        kids = self.__dict__.setdefault('_kids', {})
        g = kids.get(name)
        if g is None:
            peek = getattr(self._bg, 'peek_group', None)
            bg = peek(name) if peek is not None else None
            if bg is None:
                return self.groups.get(name)
            g = kids.setdefault(name, Group(self._dataset, bg, self))
        return g

    @property
    def dimensions(self):
        d = self.__dict__.get('_dimensions')
        if d is None:
            d = self.__dict__['_dimensions'] = {name: Dimension(self, name, *_dim_info(x))
                                                for name, x in self._bg.dimensions.items()}
        return d

    def _readahead(self):
        ra = self.__dict__.get('_ra')
        if ra is None:
            ra = self.__dict__.setdefault('_ra', _readahead.Readahead(self))
            self._dataset._readaheads.append(ra)
        return ra

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
        return _attribute(self.__dict__.get('_bg'), name)

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
            sub = g._child(p)
            if sub is not None:
                g = sub
            elif p == '..' and g.parent is not None:
                g = g.parent
            else:
                raise IndexError('%s not found in %s' % (p, g.path))
        return g

    def _walk(self):
        yield self
        for sub in self.groups.values():
            yield from sub._walk()

    keepweakref = False                             # (netCDF4-python's, read-only)
    auto_complex = property(lambda self: self._dataset._auto_complex)

    # (netCDF4-python's: whether the library can decode these; fastnc's own decoders can)
    def has_blosc_filter(self):
        return True

    def has_zstd_filter(self):
        return True

    def has_bzip2_filter(self):
        return True

    def has_szip_filter(self):
        return True

    def get_variables_by_attributes(self, **kwargs):
        """This group's variables whose attributes match every keyword: equal
        to the value given, or the value a callable taking the attribute (None
        where there is none) and saying whether it matches. netCDF4-python's
        code, so its quirks too (any attribute getattr finds)."""
        vs = []
        has_value_flag = False
        for vname, var in self.variables.items():
            for k, v in kwargs.items():
                if callable(v):
                    has_value_flag = v(getattr(var, k, None))
                    if has_value_flag is False:
                        break
                elif hasattr(var, k) and getattr(var, k) == v:
                    has_value_flag = True
                else:
                    has_value_flag = False
                    break
            if has_value_flag is True:
                vs.append(var)
        return vs

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
        fused = {}
        if FUSED:
            # big scaled variables of many chunks: masked / scaled while
            # decoding, each on every core (faster than raw in the batch,
            # then mask/scale: GOES-19 C02 269 -> ~176 ms)
            least = _hw.last_cache_bytes() if FUSED_LOAD_MIN is None else FUSED_LOAD_MIN
            for k, v in vs:
                if ((v._mask or v._scale) and getattr(v._bv, '_reader', None) is not None
                        and _readahead._nbytes(v) >= least and not v._short()):
                    r = _fusing().read(v, (Ellipsis,), threads)
                    if r is not None:
                        fused[k] = r
        batch = [(k, v) for k, v in vs if getattr(v._bv, '_reader', None) is not None
                 and k not in fused and v._bv.ds.type.cls in (0, 1, 4, 8) and v._bv.ds.shape
                 and not v._short()]
        clean = set()
        if batch:
            from ._h5read import read_many
            probes = [v._probe() for _, v in batch]
            arrays = read_many([v._bv._reader for _, v in batch], threads=threads, probes=probes)
            for (k, v), a, pr in zip(batch, arrays, probes):
                raws[k] = a
                if pr is not None and pr.proven:
                    clean.add(k)
        h4 = [(k, v) for k, v in vs if type(v._bv).__module__.endswith('_hdf4')]
        if h4:
            from ._hdf4 import read_many as read_many4
            for (k, v), a in zip(h4, read_many4([v._bv for _, v in h4], threads=threads)):
                if a is not None:
                    raws[k] = a
        rest = [(k, v) for k, v in vs if k not in raws and k not in fused]
        if len(rest) > 1:
            # the others (netCDF-3, special types) read concurrently on the pool:
            # each read stays parallel inside, and many small variables no
            # longer go one after another
            got = {}

            def one(kv):
                got[kv[0]] = kv[1].read(Ellipsis, threads=threads)
            _pool.map_items(one, rest)
        else:
            got = {k: v.read(Ellipsis, threads=threads) for k, v in rest}

        todo = [kv for kv in vs if kv[0] in raws]
        if len(todo) > 1 and any(v._mask or v._scale for _, v in todo) and _masking()._kernels_ready():
            # as for decoding: all the arrays in one pass of the pool, not one
            # after another with the threads woken for each
            items = [(k, v, v._fast_plan(raws[k])) for k, v in todo]
            items = [it for it in items if it[2] is not None and not it[2].nan_const]
            if len(items) > 1:
                done = _masking().run_plans([(p, raws[k], v._always_mask, k in clean)
                                             for k, v, p in items], threads)
                got.update((it[0], r) for it, r in zip(items, done))
        for k, v in todo:
            if k not in got:
                got[k] = v._post(raws[k], [(0, n, 1) for n in v._bv.shape], (), threads, k in clean)
        res = {}
        for k, v in vs:
            r = fused[k] if k in fused else got[k]
            res[k] = _nan_filled(v, r, threads) if v._nan and v._mask else r
        return res

    def load_async(self, names=None, recursive=False, threads=None):
        """load() on a worker of fastnc's: a concurrent.futures.Future of its
        {name: values} (fastnc's own)."""
        from . import _async
        return _async.submit(self.load, names, recursive, threads)

    def _set_all(self, method, flag, *defaults):
        """A setter over every variable of this group and those below; on the
        Dataset also its HDF4 data sets and the default of variables made later."""
        for g in self._walk():
            for v in g.variables.values():
                getattr(v, method)(flag)
        if self is self._dataset:
            for v in self.__dict__.get('_sds') or ():
                getattr(v, method)(flag)
            for d in defaults:
                self.__dict__[d] = bool(flag)

    def set_auto_maskandscale(self, flag):
        self._set_all('set_auto_maskandscale', flag, '_auto_mask', '_auto_scale')

    def set_auto_mask(self, flag):
        self._set_all('set_auto_mask', flag, '_auto_mask')

    def set_auto_scale(self, flag):
        self._set_all('set_auto_scale', flag, '_auto_scale')

    def set_always_mask(self, flag):
        self._set_all('set_always_mask', flag, '_always_mask')

    def set_auto_chartostring(self, flag):
        self._set_all('set_auto_chartostring', flag, '_chartostring')

    def set_auto_nan(self, flag):
        """Masked float values as NaN in plain arrays, for every variable
        (Variable.set_auto_nan)."""
        self._set_all('set_auto_nan', flag, '_auto_nan')

    def _user_types(self, cls):
        """{name: CompoundType / EnumType / VLType} of this group's types of
        HDF5 class cls, as netCDF4-python gives them (they were the backend's
        own datatype objects, without name, dtype or enum_dict)."""
        got = self.__dict__.setdefault('_types', {})
        if cls not in got:
            from ._h5read import native
            from ._nc4 import nc_dtype
            out = {}
            every = getattr(self._bg, 'all_types', None)    # (with those netCDF-c names: _Anonymous...)
            for k, t in every if every is not None else getattr(self._bg, 'types', {}).items():
                if t.cls != cls:
                    continue
                if cls == 6:
                    out[k] = CompoundType(k, nc_dtype(t))
                elif cls == 8:
                    out[k] = EnumType(k, native(t.base.dtype), dict(t.enum))
                else:
                    out[k] = VLType(k, native(t.base.dtype) if t.base.dtype is not None else str)
            got[cls] = out
        return dict(got[cls])

    @property
    def cmptypes(self):
        return self._user_types(6)

    @property
    def enumtypes(self):
        return self._user_types(8)

    @property
    def vltypes(self):
        return self._user_types(9)

    def __repr__(self):
        return _group_repr(self, "<class 'fastnc.Group'>")


def _group_repr(g, typ):
    """print(group) as netCDF4-python prints one: attributes, dimensions with
    their sizes, variables with their types and dimensions, groups."""
    out = [typ]
    if g.path == '/':
        out.append('root group (%s data model, file format %s):' % (g.data_model, g.disk_format))
    else:
        out.append('group %s:' % g.path)
    for name in g.ncattrs():
        out.append('    %s: %s' % (name, g.getncattr(name)))
    out.append('    dimensions(sizes): %s' % ', '.join('%s(%s)' % (n, len(d)) for n, d in g.dimensions.items()))
    out.append('    variables(dimensions): %s' % ', '.join(
        '%s %s%s' % (v.dtype, n, str(tuple(v.dimensions)).replace(',)', ')').replace("'", ''))
        for n, v in g.variables.items()))
    out.append('    groups: %s' % ', '.join(g.groups))
    return '\n'.join(out)


def _norm_key(box, drop):
    """The index read, one int (dropped dimension) or slice per dimension."""
    return tuple(s if i in drop else slice(s, e, st) for i, (s, e, st) in enumerate(box))


def _dim_info(d):
    if isinstance(d, tuple):
        return d[0], d[1]
    return d.size, d.unlimited


def _reopened(filename, device, options, state):
    """A pickled Dataset, in another process: the file opened there."""
    d = Dataset(filename, device=device, **options)
    d.__dict__.update(state)
    return d


def _group_at(ds, path):
    g = ds
    for name in path.split('/'):
        if name:
            g = g._child(name)
    return g


def _variable_in(group, name, mask, scale, always_mask, chartostring, nan=False):
    v = group.variables[name]
    v.__dict__.update(_mask=mask, _scale=scale, _always_mask=always_mask, _chartostring=chartostring,
                      _nan=nan)
    return v


def _sds_in(dataset, index, mask, scale, always_mask, chartostring, nan=False):
    v = dataset.hdf4_datasets()[index]
    v.__dict__.update(_mask=mask, _scale=scale, _always_mask=always_mask, _chartostring=chartostring,
                      _nan=nan)
    return v


class _Flag(int):
    """A bool that can be called for itself (Dataset.isopen)."""

    def __call__(self):
        return bool(self)

    def __bool__(self):
        return int(self) != 0

    def __repr__(self):
        return repr(bool(self))


NAN_BLOCK = 1 << 18             # values a pool thread fills with NaN at a time


def _fill_nan(data, mask, threads=None):
    """NaN into data (float, writable) where mask is True, on the pool for big arrays."""
    if data.size < 4 * NAN_BLOCK or not (data.flags.c_contiguous and mask.flags.c_contiguous):
        np.copyto(data, np.nan, where=mask)
        return
    d, m = data.reshape(-1), mask.reshape(-1)
    n = d.size
    blocks = itertools.count()
    nblk = (n + NAN_BLOCK - 1) // NAN_BLOCK

    def work():
        while True:
            b = next(blocks)
            if b >= nblk:
                return
            s = slice(b * NAN_BLOCK, min(n, (b + 1) * NAN_BLOCK))
            np.copyto(d[s], np.nan, where=m[s])     # (numpy lets go of the GIL)
    _pool.share(work, _pool.memory_threads(data.nbytes, min(nblk, threads or _pool.threads())))


def _nan_filled(var, r, threads=None):
    """A masked float result as a plain array with NaN where it is masked
    (set_auto_nan); anything else as it is: integers have no NaN."""
    if isinstance(r, np.ma.MaskedArray):
        if r is np.ma.masked:                       # (one value, masked: of the read's type)
            dt = var._result_dtype()
            return dt.type(np.nan) if dt.kind == 'f' else r
        if r.dtype.kind != 'f':
            return r
        data, m = r.data, r._mask
        if m is np.ma.nomask or not m.any():
            return data
        if not data.flags.writeable:                # (a view of the file: not ours to change)
            data = data.copy()
        _fill_nan(data, m, threads)
        return data
    if type(r).__name__ == 'MaskedArray' and type(r).__module__.endswith('_maskscale_gpu'):
        if r.data.dtype.kind != 'f':
            return r
        if r.mask is not None:
            r.data[r.mask] = np.nan
        return r.data
    return r


def _into(r, out):
    """A read's result r in `out` (its shape and dtype exactly): out itself,
    or for a masked result a masked array whose data is out."""
    if r is None or r is np.ma.masked:
        return r
    data = r.data if isinstance(r, np.ma.MaskedArray) else np.asarray(r)
    _check_out(out, data.shape, data.dtype)
    np.copyto(out, data)
    if isinstance(r, np.ma.MaskedArray):
        res = out.view(type(r))
        res.__dict__.update(r.__dict__)
        res._mask = r._mask
        return res
    return out


def _check_out(out, shape, dtype):
    if out.shape != tuple(shape) or out.dtype != dtype:
        raise ValueError('out has shape %s and dtype %s; this read gives shape %s, dtype %s'
                         % (out.shape, out.dtype, tuple(shape), dtype))


class Dataset(Group):
    """An open netCDF / HDF5 file (netCDF4.Dataset), read-only.

    device: 'cpu' (default) or 'gpu' (values come back as cupy arrays).
    filename may also be an http(s):// or s3:// URL (anonymous; options
    such as region in storage_options) or a file-like object: its bytes
    are then fetched as needed (`_remote`).

    It pickles, as do its groups and variables: by what they name, the file
    being opened again where they are unpickled (dask's workers, multiprocessing).
    """

    def __init__(self, filename, mode='r', device=None, **kwargs):
        if mode != 'r':
            raise ValueError('fastnc reads only (mode must be "r")')
        self._opened = filename, device, kwargs
        _pool.fast_thread()
        self._auto_mask = self._auto_scale = True
        self._always_mask = True
        self._chartostring = True
        self._auto_nan = False
        self._auto_complex = bool(kwargs.get('auto_complex', False))
        self._device = device
        self._fh = self._mm = None
        self._readaheads = []                       # (each group's: see Group._readahead)
        self._source = src = _remote.open_source(filename, kwargs.get('storage_options'))
        if src is None:
            self._filepath = os.fspath(filename)
            with open(self._filepath, 'rb') as fh:
                head = fh.read(4096)
        else:                                       # a URL or a file-like object
            self._filepath = src.name
            head = src.buf[:4096].tobytes()
        fmt = _format(head)
        if fmt is None and src is None:
            with open(self._filepath, 'rb') as fh:  # (past a user block of any size)
                fmt = _past_user_block(fh)
        if fmt is None:
            raise OSError('[Errno -51] NetCDF: Unknown file format: %r' % self._filepath)
        if fmt == 'HDF5':
            from ._nc4 import NC4File
            self._fh, self._mm = _open_bytes(self._filepath) if src is None else (None, _remote.readonly(src))
            h5 = _hdf5.H5File(self._mm, source=src)
            if src is None:
                h5.path = self._filepath                # positional reads for big batches
            self._backend = NC4File(h5, np.frombuffer(self._mm, np.uint8))
            self.file_format = self._backend.data_model
            self.disk_format = 'HDF5'
        elif fmt == 'HDF4':
            from ._hdf4 import HDF4File
            self._fh, self._mm = _open_bytes(self._filepath) if src is None else (None, _remote.whole(src))
            self._backend = HDF4File(self._mm, np.frombuffer(self._mm, np.uint8), self._filepath)
            self._backend.f.source = src                # (external elements beside a remote file)
            self.file_format = self._backend.data_model
            self.disk_format = 'HDF4'
        else:
            from ._nc3 import NC3File                 # positional reads of its own
            self._backend = NC3File(self._filepath) if src is None else NC3File(src.name, source=src)
            self.file_format = fmt
            self.disk_format = 'NETCDF3'
        self.data_model = self._backend.data_model
        Group.__init__(self, self, self._backend.root, None)
        self.name = '/'

    def __reduce__(self):
        return _reopened, self._opened + ({k: self.__dict__[k] for k in (
            '_auto_mask', '_auto_scale', '_always_mask', '_chartostring', '_auto_nan')},)

    def filepath(self):
        return self._filepath

    def tocdl(self, coordvars=False, data=False, outfile=None):
        """The file's CDL header, as `ncdump -h` prints it (netCDF4-python's
        tocdl runs the ncdump program; fastnc writes it itself). outfile: a
        path to write it to, else it is returned. Data (data=True,
        coordvars=True) is not written."""
        if data or coordvars:
            raise NotImplementedError('tocdl writes the header only (data=False, coordvars=False)')
        from ._cdl import tocdl
        cdl = tocdl(self)
        if outfile is None:
            return cdl
        with open(outfile, 'w', encoding='utf-8', newline='\n') as f:
            f.write(cdl)

    def hdf4_datasets(self):
        """Every SD data set of an HDF4 file, in the file's order: the i-th is
        the one pyhdf's `SD(path).select(i)` reads. `variables` holds one data
        set of a name, the last, as netCDF-C does; HDF-EOS files can hold
        several (ASTER L1B: a `Latitude` and a `Longitude` per swath), told
        apart by their dimensions ('GeoTrack:VNIR_Swath' ...). Those in
        `variables` come back as the same Variable objects. ValueError for a
        file that is not HDF4."""
        sds = getattr(self._bg, 'sds', None)
        if sds is None:
            raise ValueError('not an HDF4 file: %s' % (self._filepath,))
        got = self.__dict__.get('_sds')
        if got is None:
            got = []
            for i, bv in enumerate(sds):
                v = self.variables.get(bv.name)
                if v is None or v._bv is not bv:
                    v = Variable(self, bv)
                    v.__dict__['_sds_index'] = i
                got.append(v)
            self.__dict__['_sds'] = got
        return list(got)

    @property
    def isopen(self):
        """Whether the file is open: `ds.isopen()` as in netCDF4-python, and
        `ds.isopen` as fastnc had it, both."""
        return _Flag(getattr(self, '_backend', None) is not None)

    def close(self):
        for ra in self.__dict__.get('_readaheads', ()):
            ra.cache.clear()                        # (what was decoded ahead is the file's too)
        b = getattr(self, '_backend', None)
        self._backend = None                        # (closed first: a read failing meanwhile says so)
        if b is not None:
            b.close()
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
        if not self.isopen():
            return "<class 'fastnc.Dataset'>\nroot group (closed): %s" % self._filepath
        return _group_repr(self, "<class 'fastnc.Dataset'>")
