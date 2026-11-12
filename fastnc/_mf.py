"""MFDataset: files read as one, variables joined along an aggregation
dimension (netCDF4-python's MFDataset, read-only). netCDF4-python's takes
netCDF-3 and NETCDF4_CLASSIC files only; this one any file fastnc opens.

The first file (or `master_file`) gives the metadata: dimensions, global
attributes, variables, their attributes. A variable whose first dimension is
the aggregation dimension (by default the master's unlimited one) spans every
file; any other is the master's own. A read of a spanning variable reads the
part of each file its index selects, in order, and joins them."""
import glob as _glob
import os

import numpy as np

from ._dataset import Dataset


class MFDataset:
    """Several files as one Dataset (netCDF4.MFDataset): `files` a list of
    paths or a glob pattern (sorted), `aggdim` the dimension to join along
    (default: the master file's unlimited dimension), `exclude` variable
    names to leave out, `check` whether to make sure every file has the
    aggregated variables, `master_file` the file whose metadata is used."""

    def __init__(self, files, check=False, aggdim=None, exclude=(), master_file=None):
        if isinstance(files, (str, os.PathLike)):
            pattern = os.fspath(files)
            files = sorted(_glob.glob(pattern))
            if not files:
                raise OSError('no files match %s' % pattern)
        files = [os.fspath(f) for f in files]
        if not files:
            raise ValueError('MFDataset needs at least one file')
        if master_file is not None:
            master_file = os.fspath(master_file)
            if master_file not in files:
                raise ValueError('master_file not in files list')
        else:
            master_file = files[0]
        self._files = files
        self._datasets = [Dataset(f) for f in files]
        m = self._master = self._datasets[files.index(master_file)]
        if aggdim is None:
            unlim = [n for n, d in m.dimensions.items() if d.isunlimited()]
            if not unlim:
                self.close()
                raise OSError('master dataset %s does not have an aggregation dimension' % master_file)
            aggdim = unlim[0]
        elif aggdim not in m.dimensions:
            self.close()
            raise OSError('master dataset %s does not have a dimension %r' % (master_file, aggdim))
        self._aggdim = aggdim
        exclude = set(exclude)
        lengths = [len(d.dimensions[aggdim]) if aggdim in d.dimensions else 0 for d in self._datasets]
        if check:
            for f, d in zip(files, self._datasets):
                if aggdim not in d.dimensions:
                    self.close()
                    raise OSError('file %s has no dimension %r' % (f, aggdim))
        self.dimensions = {}
        for n, dim in m.dimensions.items():
            size = sum(lengths) if n == aggdim else len(dim)
            self.dimensions[n] = _MFDimension(n, size, dim.isunlimited() or n == aggdim)
        self.variables = {}
        for n, v in m.variables.items():
            if n in exclude:
                continue
            if v.dimensions and v.dimensions[0] == aggdim:
                parts = []
                for f, d in zip(files, self._datasets):
                    if n not in d.variables:
                        if check:
                            self.close()
                            raise OSError('variable %s is not in file %s' % (n, f))
                        continue
                    parts.append(d.variables[n])
                self.variables[n] = MFVariable(self, n, v, parts)
            else:
                self.variables[n] = v
        self.groups = {}
        self.file_format = m.file_format
        self.data_model = m.data_model
        self.disk_format = m.disk_format
        self.path = '/'
        self.parent = None
        self.name = '/'

    # -- attributes, as the master file's
    def ncattrs(self):
        return self._master.ncattrs()

    def getncattr(self, name):
        return self._master.getncattr(name)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self.__dict__['_master'], name)

    def __getitem__(self, name):
        return self.variables[name]

    def filepath(self):
        return self._master.filepath()

    def isopen(self):
        return all(d.isopen() for d in self._datasets)

    def _setall(self, method, flag):
        for v in self.variables.values():
            getattr(v, method)(flag)

    def set_auto_maskandscale(self, flag):
        self._setall('set_auto_maskandscale', flag)

    def set_auto_mask(self, flag):
        self._setall('set_auto_mask', flag)

    def set_auto_scale(self, flag):
        self._setall('set_auto_scale', flag)

    def set_always_mask(self, flag):
        self._setall('set_always_mask', flag)

    def set_auto_chartostring(self, flag):
        self._setall('set_auto_chartostring', flag)

    def set_auto_nan(self, flag):
        self._setall('set_auto_nan', flag)

    def close(self):
        for d in self.__dict__.get('_datasets', ()):
            d.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return "<class 'fastnc.MFDataset'>\nroot group (%s data model, file format %s), %d files:\n" \
               "    dimensions: %s\n    variables: %s" % (
                   self.data_model, self.disk_format, len(self._files), ', '.join(self.dimensions),
                   ', '.join(self.variables))


class _MFDimension:
    """A dimension of an MFDataset: the aggregation one sums the files'."""

    def __init__(self, name, size, unlimited):
        self.name, self.size, self._unlimited = name, size, unlimited

    def __len__(self):
        return self.size

    def isunlimited(self):
        return self._unlimited

    def __repr__(self):
        return "<class 'fastnc._mf._MFDimension'>%s: name = '%s', size = %d" % (
            ' (unlimited)' if self._unlimited else '', self.name, self.size)


class MFVariable:
    """A variable spanning the files of an MFDataset along its first
    dimension; metadata from the master file's variable."""

    def __init__(self, mf, name, master, parts):
        self._mf, self.name, self._master, self._parts = mf, name, master, parts
        self._lengths = [p.shape[0] for p in parts]
        self._starts = np.concatenate([[0], np.cumsum(self._lengths)]).astype(np.int64)

    # -- metadata
    @property
    def dimensions(self):
        return self._master.dimensions

    @property
    def shape(self):
        return (int(self._starts[-1]),) + tuple(self._master.shape[1:])

    @property
    def dtype(self):
        return self._master.dtype

    @property
    def datatype(self):
        return self._master.datatype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def __len__(self):
        return self.shape[0]

    def ncattrs(self):
        return self._master.ncattrs()

    def getncattr(self, name):
        return self._master.getncattr(name)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self.__dict__['_master'], name)

    def _each(self, method, flag):
        for p in self._parts:
            getattr(p, method)(flag)
        if self._master not in self._parts:
            getattr(self._master, method)(flag)

    def set_auto_maskandscale(self, flag):
        self._each('set_auto_maskandscale', flag)

    def set_auto_mask(self, flag):
        self._each('set_auto_mask', flag)

    def set_auto_scale(self, flag):
        self._each('set_auto_scale', flag)

    def set_always_mask(self, flag):
        self._each('set_always_mask', flag)

    def set_auto_chartostring(self, flag):
        self._each('set_auto_chartostring', flag)

    def set_auto_nan(self, flag):
        self._each('set_auto_nan', flag)

    # -- values
    def __getitem__(self, key):
        k = key if isinstance(key, tuple) else (key,)
        if any(x is Ellipsis for x in k):           # (the dimensions it stands for, spelled out)
            i = next(j for j, x in enumerate(k) if x is Ellipsis)
            k = k[:i] + (slice(None),) * (self.ndim - len(k) + 1) + k[i + 1:]
        first, rest = (k[0], k[1:]) if k else (slice(None), ())
        n = self.shape[0]
        drop = False
        if isinstance(first, slice):
            idx = np.arange(n)[first]
        elif isinstance(first, (int, np.integer)):
            i = int(first)
            if not -n <= i < n:
                raise IndexError('index %d is out of bounds for the aggregation dimension of size %d' % (i, n))
            idx = np.array([i % n])
            drop = True
        else:
            a = np.asarray(first)
            idx = np.arange(n)[a] if a.dtype == bool else a.astype(np.int64) % n
        f = np.searchsorted(self._starts, idx, side='right') - 1
        pieces = []
        s = 0
        while s < len(idx):                         # (runs of one file, in the order asked)
            e = s + 1
            while e < len(idx) and f[e] == f[s]:
                e += 1
            local = idx[s:e] - self._starts[f[s]]
            sel = _as_slice(local)
            pieces.append(self._parts[f[s]][(sel,) + tuple(rest)])
            s = e
        if not pieces:
            out = self._parts[0][(slice(0, 0),) + tuple(rest)] if self._parts else np.zeros((0,), self.dtype)
        elif any(isinstance(p, np.ma.MaskedArray) for p in pieces):
            # (as netCDF4-python's: joined by numpy.ma, whose fill value is
            # numpy's default, and a plain array if nothing is masked)
            out = np.ma.concatenate(pieces)
            if not out.mask.any():
                out = out.data
        else:
            out = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        if drop:
            out = out[0]
        return out

    def __array__(self, dtype=None, copy=None):
        a = np.asarray(self[...])
        return a.astype(dtype) if dtype is not None else a

    def __repr__(self):
        return "<class 'fastnc.MFVariable'>\n%s %s(%s)\nshape = %s" % (
            self.dtype, self.name, ', '.join(self.dimensions), self.shape)


def _as_slice(local):
    """A slice for evenly spaced increasing indices (one read), else the list."""
    if len(local) == 1:
        return slice(int(local[0]), int(local[0]) + 1)
    step = local[1] - local[0]
    if step > 0 and np.all(np.diff(local) == step):
        return slice(int(local[0]), int(local[-1]) + 1, int(step))
    return local.tolist()
