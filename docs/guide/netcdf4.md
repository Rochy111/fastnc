# netCDF4-python's API

`fastnc.Dataset` reads any file fastnc supports (netCDF-3, netCDF-4, HDF5,
HDF4) through netCDF4-python's `Dataset`, read-only. Code written for
netCDF4-python runs unchanged:

```python
import fastnc as netCDF4          # or keep both and pick per file

with netCDF4.Dataset('wrfout_d01_2025-10-26_06_20_00.nc') as ds:
    print(ds.data_model, ds.file_format, ds.disk_format)
    print(ds.dimensions['Time'].isunlimited(), len(ds.dimensions['bottom_top']))
    t = ds.variables['T']
    print(t.dimensions, t.shape, t.dtype, t.ncattrs())
    level0 = t[0, 0]              # any netCDF4-python index
```

## What is there

- **The file:** `Dataset(path_or_url, mode='r', device=None, storage_options=None)`,
  `data_model`, `file_format`, `disk_format`, `filepath()`, `isopen`,
  `close()` and `with` blocks.
- **Groups:** `groups`, `path`, `parent`, `ds['group/sub/var']` paths,
  `dimensions`, `variables`, attributes as properties or through `ncattrs()`
  and `getncattr()`, and user types (`cmptypes`, `enumtypes`, `vltypes`).
- **Dimensions:** `name`, `size`, `len()`, `isunlimited()`, and netCDF-c's
  phony dimensions for HDF5 datasets without dimension scales.
- **Variables:**
  - `dimensions`, `shape`, `dtype`, `datatype`, `ndim`, `size`, `len()`
  - attributes as properties
  - `filters()`, `chunking()`, `endian()`, `isunlimited()`, `group()`
- **Indexing:** integers, slices with steps, negative indices, `Ellipsis`,
  integer and boolean arrays, and a list per dimension, all as
  netCDF4-python interprets them. A scalar variable takes them as one of
  shape (1,) does: `v[:]`, `v[0]` and `v[0:1]` give its value.
- **Reads into existing memory:** `var.read(key, out=array)`.

Writing (`mode='w'`, `createVariable` and so on) raises an error: fastnc
reads only.

## Masking and scaling

By default, values come back exactly as netCDF4-python returns them:

- **Masked:** values equal to `_FillValue` or the default fill value, to
  `missing_value`, or outside `valid_min`, `valid_max` or `valid_range`,
  come back as a masked array.
- **Scaled:** `scale_factor` and `add_offset` are applied, with
  netCDF4-python's dtype promotion, which differs between numpy 1 and 2.
- **Unsigned:** `_Unsigned` is honoured.
- **Masked or not:** the choice between a masked and a plain array, and the
  masked array's fill value, follow netCDF4-python too.

The switches work as in netCDF4-python, on a variable or on a whole file:

```python
ds.set_auto_maskandscale(False)       # raw values, every variable
var.set_auto_mask(False)              # scaled, never masked
var.set_auto_scale(False)             # masked, never scaled
var.set_always_mask(False)            # a plain array when nothing is masked
var.set_auto_chartostring(False)      # char arrays stay char arrays
```

Big scaled variables are masked and scaled while they are decoded, chunk by
chunk on every core, so the raw integers never go to memory and back.

## Many variables

`ds.load()` returns every variable of a group decoded in one pass, and
`ds.load(['T', 'QVAPOR'])` the ones named. See
[Many variables at once](many-variables.md).

## On the GPU

`var.read(key, device='gpu')` returns a cupy array, or opening with
`Dataset(path, device='gpu')` makes that every variable's default. See
[GPU, threads and hardware](hardware.md).

## With dask and multiprocessing

Datasets, groups and variables pickle by what they name. The file is opened
again where they are unpickled, so dask workers and `multiprocessing` pools
can use them.
