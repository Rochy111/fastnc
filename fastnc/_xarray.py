"""xarray backend: xarray.open_dataset(path, engine='fastnc').

Built as xarray's netCDF4 backend is -- a data store whose variables are
lazily indexed arrays of raw values (orthogonal indexing, mask and scale
off), with xarray's CF decoding on top -- so the Dataset is the one
engine='netcdf4' gives, value for value. What differs is the cost: no
filters() probe per variable (36-48 ms each where netCDF4's plugins do
not load: 7 s for a WRF file) and no rescans for unlimited dimensions
(h5netcdf's quadratic open, 10 s); opening wrfout d01 takes milliseconds.

preload=True reads every variable at open in one pass of the decoder
(all chunks of all variables at once, biggest first), which is what
.load() would do variable by variable.
"""
import os
import threading

import numpy as np

from xarray.backends import BackendArray, BackendEntrypoint
from xarray.backends.common import AbstractDataStore
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import FrozenDict
from xarray.core.variable import Variable

from . import _dataset
from ._intrin import take
from ._jit import njit

_LOCK = threading.Lock()


class FastncArray(BackendArray):
    """A variable's raw values, read when indexed."""

    def __init__(self, var, preloaded=None):
        self.var = var
        self.shape = var.shape
        dt = var.dtype
        self.dtype = np.dtype(object) if dt is str else np.dtype(dt)
        self._values = preloaded

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.OUTER,
                                                  self._getitem)

    def _getitem(self, key):
        if self._values is not None:
            return self._values[key]
        v = self.var
        v.set_auto_maskandscale(False)
        v.set_auto_chartostring(False)
        try:
            return v[key]
        except IndexError as err:
            raise IndexError('The indexing operation you are attempting to perform is not valid on '
                             'a fastnc.Variable. Try loading your data into memory first by calling '
                             '.load().') from err


class DecodedArray(BackendArray):
    """A variable's values with xarray's own mask and scale functions
    applied block by block on fastnc's threads (each block a few hundred
    kB, in cache): xarray applies them to the whole array on one thread,
    in several passes -- 2.5 s of a GOES-19 full-disk band's 2.7 s load.
    The functions are the very ones xarray's coders make (taken from
    decoding a one-element stand-in with the same attributes), and they
    are elementwise, so the values are xarray's by construction."""

    BLOCK = 1 << 18
    LUT_MIN = 1 << 16               # values from which an 8 / 16-bit read goes through a table

    def __init__(self, raw, funcs, dtype):
        self.raw, self.funcs = raw, funcs
        self.shape = raw.shape
        self.dtype = np.dtype(dtype)
        self._luts = {}

    def _lut(self, rdt):
        """For raw values of 8 or 16 bits: xarray's functions applied once
        to every possible value -- a table (64K entries at most) the decoded
        array is then looked up in, value by value. The same functions on
        the same values: the same results, bit for bit. None where it does
        not apply (wider types; a NaN scale_factor / add_offset, where NaN
        times NaN would depend on numpy's loop order)."""
        if rdt.kind not in 'iu' or rdt.itemsize > 2:
            return None
        if rdt.str in self._luts:
            return self._luts[rdt.str]
        lut = None
        try:
            for f in self.funcs:
                for k in ('scale_factor', 'add_offset'):
                    c = getattr(f, 'keywords', {}).get(k)
                    if c is not None and np.isnan(np.asarray(c, np.float64)).any():
                        raise ValueError('NaN constant')
            x = np.arange(1 << (8 * rdt.itemsize), dtype='u%d' % rdt.itemsize).view(rdt)
            for f in self.funcs:
                x = f(x)
            x = np.asarray(x)
            if x.shape == (1 << (8 * rdt.itemsize),) and x.dtype == self.dtype:
                lut = np.ascontiguousarray(x)
        except Exception:
            lut = None
        self._luts[rdt.str] = lut
        return lut

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.OUTER,
                                                  self._getitem)

    FUSED_THREADS = None            # (tests: 1, so that small arrays take the fused path)

    def _getitem(self, key):
        var = getattr(self.raw, 'var', None)
        if var is not None and getattr(self.raw, '_values', None) is None \
                and isinstance(var.dtype, np.dtype) and var.dtype.isnative:
            lut = self._lut(np.dtype(var.dtype))
            if lut is not None:
                # looked up as each chunk is decoded: the raw array never
                # goes to memory and back
                from . import _fused
                var.set_auto_maskandscale(False)
                var.set_auto_chartostring(False)
                res = _fused.read_lut(var, key, lut, self.FUSED_THREADS, self.LUT_MIN)
                if res is not None:
                    return res
        raw = np.asarray(self.raw._getitem(key))   # (ascontiguousarray makes 0-d 1-d)
        if not raw.flags.c_contiguous:
            raw = raw.copy()
        funcs = self.funcs
        if raw.size >= self.LUT_MIN and raw.dtype.isnative:
            lut = self._lut(raw.dtype)
            if lut is not None:
                return _lookup(raw, lut)
        if raw.size <= self.BLOCK:
            x = raw
            for f in funcs:
                x = f(x)
            return np.asarray(x, self.dtype)
        from . import _arena, _pool
        out = _arena.empty(raw.shape, self.dtype)
        src, dst = raw.reshape(-1), out.reshape(-1)
        n = src.size
        blk = self.BLOCK

        def one(i):
            x = src[i * blk:(i + 1) * blk]
            for f in funcs:
                x = f(x)
            dst[i * blk:(i + 1) * blk] = x
        _pool.map_items(one, range(-(-n // blk)))
        return out


def _lookup(raw, lut):
    """lut[raw] (raw 8 / 16-bit, viewed unsigned as the index), copied as
    bits; on every thread."""
    from . import _arena, _chunks, _pool
    u = np.dtype('u%d' % lut.dtype.itemsize) if lut.dtype.itemsize in (1, 2, 4, 8) else None
    out = _arena.empty(raw.shape, lut.dtype)
    if u is None:
        out[...] = lut[raw.view('u%d' % raw.dtype.itemsize)]
        return out
    idx = raw.reshape(-1).view('u%d' % raw.dtype.itemsize)
    tab = lut.view(u)
    dst = out.reshape(-1).view(u)
    n = idx.size
    blk = 1 << 18
    nb = -(-n // blk)
    if _chunks._READY.is_set():
        ctr = np.zeros(1, np.int64)
        _pool.share(lambda: _gather(idx, tab, dst, ctr, blk), nb)
    else:
        def one(i):
            np.take(tab, idx[i * blk:(i + 1) * blk], out=dst[i * blk:(i + 1) * blk])
        _pool.map_items(one, range(nb))
    return out


@njit(cache=True, nogil=True)
def _gather(idx, tab, dst, ctr, blk):
    """dst[i] = tab[idx[i]], blocks of blk taken off ctr[0]."""
    n = idx.size
    nb = (n + blk - 1) // blk
    while True:
        b = take(ctr, 0)
        if b >= nb:
            return
        for i in range(b * blk, min(n, b * blk + blk)):
            dst[i] = tab[idx[i]]


_MASK_SCALE = ('_FillValue', 'missing_value', 'scale_factor', 'add_offset', '_Unsigned')


def _decoded(name, dims, data, attrs, encoding):
    """The variable with xarray's mask / scale decoding applied by fastnc
    (DecodedArray), attributes and encoding as xarray's coders leave them;
    None to leave it to xarray (non-numeric, time-like, or anything the
    stand-in decoding does not explain)."""
    dt = data.dtype
    if dt.kind not in 'iuf' or not any(k in attrs for k in _MASK_SCALE):
        return None
    units = attrs.get('units')
    if isinstance(units, str) and ' since ' in units:
        return None                                 # time-like: xarray's time coders handle it
    try:
        from xarray.coding import common as xcommon
        from xarray.coding import variables as xvars
        probe = Variable(dims, np.zeros((1,) * len(dims), dt), dict(attrs), dict(encoding))
        dec = probe
        for coder in (xvars.CFMaskCoder(decode_times=False, decode_timedelta=False),
                      xvars.CFScaleOffsetCoder(decode_times=False, decode_timedelta=False)):
            dec = coder.decode(dec, name=name)
        funcs = []
        a = dec._data
        while isinstance(a, indexing.LazilyIndexedArray):
            a = a.array
        while isinstance(a, xcommon._ElementwiseFunctionArray):
            funcs.append(a.func)
            a = a.array
            while isinstance(a, indexing.LazilyIndexedArray):
                a = a.array
        if not isinstance(a, (np.ndarray, indexing.NumpyIndexingAdapter)):
            return None
        funcs.reverse()
    except Exception:
        return None
    if not funcs:
        return None
    raw = data.array if isinstance(data, indexing.LazilyIndexedArray) else data
    return Variable(dims, indexing.LazilyIndexedArray(DecodedArray(raw, funcs, dec.dtype)),
                    dec.attrs, dec.encoding)


class FastncDataStore(AbstractDataStore):
    """An open fastnc.Dataset (or one of its groups) as an xarray data store."""

    def __init__(self, ds, group=None, preload=False, decode_mask_scale=False):
        self.ds = ds
        self.decode_mask_scale = decode_mask_scale
        g = ds
        if group not in (None, '', '/'):
            g = ds[group]
            if not isinstance(g, _dataset.Group):
                raise OSError('group not found: %s' % group)
        self.group = g
        self._preloaded = _preload(g) if preload else {}

    @classmethod
    def open(cls, filename, group=None, preload=False, storage_options=None,
             decode_mask_scale=False):
        return cls(_dataset.Dataset(filename, storage_options=storage_options), group=group,
                   preload=preload, decode_mask_scale=decode_mask_scale)

    def get_variables(self):
        return FrozenDict((k, self.open_store_variable(k, v)) for k, v in self.group.variables.items())

    def get_attrs(self):
        g = self.group
        return FrozenDict((k, g.getncattr(k)) for k in g.ncattrs())

    def get_dimensions(self):
        return FrozenDict((k, len(v)) for k, v in self.group.dimensions.items())

    def get_encoding(self):
        return {'unlimited_dims': {k for k, v in self.group.dimensions.items() if v.isunlimited()}}

    def open_store_variable(self, name, var):
        attributes = {k: var.getncattr(k) for k in var.ncattrs()}
        data = indexing.LazilyIndexedArray(FastncArray(var, self._preloaded.get(name)))
        encoding = {'dtype': var.dtype}
        if isinstance(var.dtype, np.dtype) and var.dtype.kind in 'iuf' and '_FillValue' in attributes:
            pass
        _ensure_fill_value_valid(data, attributes)
        filters = var.filters()
        if filters is not None:
            encoding.update(filters)
        chunking = var.chunking()
        if chunking is not None:
            if chunking == 'contiguous':
                encoding['contiguous'] = True
                encoding['chunksizes'] = None
            else:
                encoding['contiguous'] = False
                encoding['chunksizes'] = tuple(chunking)
                encoding['preferred_chunks'] = dict(zip(var.dimensions, chunking))
        if 'least_significant_digit' in attributes:
            encoding['least_significant_digit'] = attributes.pop('least_significant_digit')
        encoding['source'] = self.ds.filepath()
        encoding['original_shape'] = data.shape
        if self.decode_mask_scale:
            v = _decoded(name, var.dimensions, data, attributes, encoding)
            if v is not None:
                return v
        return Variable(var.dimensions, data, attributes, encoding)

    def close(self, **kwargs):
        self.ds.close()


def _ensure_fill_value_valid(data, attributes):
    # netCDF4 attributes of string variables cannot be _FillValue (xarray's rule)
    if data.dtype.kind == 'S' and '_FillValue' in attributes:
        attributes['_FillValue'] = np.bytes_(attributes['_FillValue'])


def _preload(group):
    """Every variable of the group, raw, read in one batched pass (Group.load)."""
    for v in group.variables.values():
        v.set_auto_maskandscale(False)
        v.set_auto_chartostring(False)
    return group.load()


class FastncBackendEntrypoint(BackendEntrypoint):
    """xarray backend for netCDF-3, netCDF-4 and HDF5 files through fastnc."""

    description = 'Open netCDF (.nc, .nc4, .cdf) and HDF5 files with fastnc'
    url = 'https://github.com/Rochy111/fastnc'
    open_dataset_parameters = ('filename_or_obj', 'mask_and_scale', 'decode_times', 'concat_characters',
                               'decode_coords', 'drop_variables', 'use_cftime', 'decode_timedelta',
                               'group', 'preload', 'storage_options')
    supports_groups = True

    def guess_can_open(self, filename_or_obj):
        if isinstance(filename_or_obj, str) and filename_or_obj.startswith(('http://', 'https://', 's3://')):
            return filename_or_obj.split('#')[0].endswith(('.nc', '.nc4', '.h5', '.hdf5', '.he5', '.hdf'))
        if isinstance(filename_or_obj, (str, os.PathLike)):
            try:
                with open(filename_or_obj, 'rb') as f:
                    head = f.read(8)
            except OSError:
                return False
            return head.startswith((b'CDF', b'\x89HDF\r\n\x1a\n', b'\x0e\x03\x13\x01'))
        return False

    def open_dataset(self, filename_or_obj, *, mask_and_scale=True, decode_times=True,
                     concat_characters=True, decode_coords=True, drop_variables=None,
                     use_cftime=None, decode_timedelta=None, group=None, preload=False,
                     storage_options=None, **kwargs):
        store = FastncDataStore.open(_name(filename_or_obj), group=group, preload=preload,
                                     storage_options=storage_options,
                                     decode_mask_scale=bool(mask_and_scale))
        store_entrypoint = StoreBackendEntrypoint()
        with _close_on_error(store):
            return store_entrypoint.open_dataset(
                store, mask_and_scale=mask_and_scale, decode_times=decode_times,
                concat_characters=concat_characters, decode_coords=decode_coords,
                drop_variables=drop_variables, use_cftime=use_cftime,
                decode_timedelta=decode_timedelta)

    def open_groups_as_dict(self, filename_or_obj, **kwargs):
        from xarray.core.treenode import NodePath
        ds = _dataset.Dataset(_name(filename_or_obj), storage_options=kwargs.get('storage_options'))
        out = {}
        for g in ds._walk():
            path = g.path
            out[str(NodePath(path))] = self.open_dataset(filename_or_obj, group=path, **kwargs) \
                if path != '/' else self.open_dataset(filename_or_obj, **kwargs)
        ds.close()
        return out

    def open_datatree(self, filename_or_obj, **kwargs):
        from xarray.core.datatree import DataTree
        return DataTree.from_dict(self.open_groups_as_dict(filename_or_obj, **kwargs))


def _name(f):
    """A path as a string; URLs and file objects as they are."""
    return os.fspath(f) if isinstance(f, os.PathLike) else f


class _close_on_error:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, typ, val, tb):
        if typ is not None:
            self.store.close()
