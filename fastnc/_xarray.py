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


class FastncDataStore(AbstractDataStore):
    """An open fastnc.Dataset (or one of its groups) as an xarray data store."""

    def __init__(self, ds, group=None, preload=False):
        self.ds = ds
        g = ds
        if group not in (None, '', '/'):
            g = ds[group]
            if not isinstance(g, _dataset.Group):
                raise OSError('group not found: %s' % group)
        self.group = g
        self._preloaded = _preload(g) if preload else {}

    @classmethod
    def open(cls, filename, group=None, preload=False, device=None):
        return cls(_dataset.Dataset(filename), group=group, preload=preload)

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
        return Variable(var.dimensions, data, attributes, encoding)

    def close(self, **kwargs):
        self.ds.close()


def _ensure_fill_value_valid(data, attributes):
    # netCDF4 attributes of string variables cannot be _FillValue (xarray's rule)
    if data.dtype.kind == 'S' and '_FillValue' in attributes:
        attributes['_FillValue'] = np.bytes_(attributes['_FillValue'])


def _preload(group):
    """Every variable of the group, raw, read in one pass where the backend can."""
    out = {}
    vs = list(group.variables.items())
    hdf = [(k, v) for k, v in vs if hasattr(v._bv, '_reader')]
    if hdf:
        from ._h5read import read_many
        vals = read_many([v._bv._reader for _, v in hdf])
        for (k, v), a in zip(hdf, vals):
            if a is not None and v._bv.ds.type.cls != 6 and v.dtype is not str:
                out[k] = a
    for k, v in vs:
        if k not in out:
            v.set_auto_maskandscale(False)
            v.set_auto_chartostring(False)
            out[k] = v[...]
    return out


class FastncBackendEntrypoint(BackendEntrypoint):
    """xarray backend for netCDF-3, netCDF-4 and HDF5 files through fastnc."""

    description = 'Open netCDF (.nc, .nc4, .cdf) and HDF5 files with fastnc'
    url = 'https://github.com/Rochy111/fastnc'
    open_dataset_parameters = ('filename_or_obj', 'mask_and_scale', 'decode_times', 'concat_characters',
                               'decode_coords', 'drop_variables', 'use_cftime', 'decode_timedelta',
                               'group', 'preload')
    supports_groups = True

    def guess_can_open(self, filename_or_obj):
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
                     use_cftime=None, decode_timedelta=None, group=None, preload=False, **kwargs):
        store = FastncDataStore.open(os.fspath(filename_or_obj), group=group, preload=preload)
        store_entrypoint = StoreBackendEntrypoint()
        with _close_on_error(store):
            return store_entrypoint.open_dataset(
                store, mask_and_scale=mask_and_scale, decode_times=decode_times,
                concat_characters=concat_characters, decode_coords=decode_coords,
                drop_variables=drop_variables, use_cftime=use_cftime,
                decode_timedelta=decode_timedelta)

    def open_groups_as_dict(self, filename_or_obj, **kwargs):
        from xarray.core.treenode import NodePath
        ds = _dataset.Dataset(os.fspath(filename_or_obj))
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


class _close_on_error:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, typ, val, tb):
        if typ is not None:
            self.store.close()
