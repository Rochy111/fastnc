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
  `data_model`, `file_format`, `disk_format`, `filepath()`, `isopen()`,
  `close()` and `with` blocks; `get_variables_by_attributes(**kwargs)` on
  any group; `has_blosc_filter()`, `has_zstd_filter()`,
  `has_bzip2_filter()`, `has_szip_filter()` (all True: fastnc decodes
  them itself); `print()` of a dataset, group, variable or dimension shows
  what netCDF4-python's shows; `tocdl(outfile=None)`, the header `ncdump -h` prints, made by
  fastnc itself (netCDF4-python's runs the `ncdump` program), the same to
  the byte on every one of 325 satellite and model files `ncdump` reads,
  and on every test file, HDF5 types without a name (h5py's) included.
- **Groups:** `groups`, `path`, `parent`, `ds['group/sub/var']` paths,
  `dimensions`, `variables`, attributes as properties or through `ncattrs()`
  and `getncattr()`, and user types (`cmptypes`, `enumtypes`, `vltypes`).
- **Dimensions:** `name`, `size`, `len()`, `isunlimited()`, and netCDF-c's
  phony dimensions for HDF5 datasets without dimension scales.
- **Variables:**
  - `dimensions`, `shape`, `dtype`, `datatype`, `ndim`, `size`, `len()`
  - attributes as properties
  - `filters()`, `chunking()`, `endian()`, `isunlimited()`, `group()`,
    `get_var_chunk_cache()` and `set_var_chunk_cache()` (accepted, and
    ignored: fastnc's cache sizes itself)
  - `get_dims()`, `get_fill_value()`, `quantization()`, `getValue()`, and
    the switches as attributes: `mask`, `scale`, `always_mask`,
    `chartostring`
- **Indexing:** integers, slices with steps, negative indices, `Ellipsis`,
  integer and boolean arrays, and a list per dimension, all as
  netCDF4-python interprets them. A scalar variable takes them as one of
  shape (1,) does: `v[:]`, `v[0]` and `v[0:1]` give its value.
- **Reads into existing memory:** `var.read(key, out=array)`, with `array`
  of the read's shape and dtype exactly. A raw read of a box (integers and
  slices) is decoded straight into it; any other is copied in. It returns
  `array`, or for a masked result a masked array whose data is `array`.
- **Reads in the background:** `var.read_async(key, **kwargs)` and
  `group.load_async(names)` (fastnc's own) return a
  `concurrent.futures.Future` of what `read` / `load` would return, so the
  next read is under way while the last result is used. (`fastnc.File`
  datasets: `d.read_async(key)`.)
- **The stored values in place, no copy:** `var.memmap()` (fastnc's own) is
  a read-only numpy array over the file's mapping, for a variable stored
  uncompressed in one piece: every netCDF-3 variable (record variables
  strided as they lie), contiguous netCDF-4 / HDF5 and HDF4 ones. Raw (no
  mask or scale) and in the file's byte order; slicing it touches only what
  it selects, and it stays valid while held, the file closed or not.
  Compressed or chunked variables raise `ValueError`.
- **Every data set of an HDF4 file:** `ds.hdf4_datasets()`, a list in the
  file's order, the i-th being what pyhdf's `SD(path).select(i)` reads.
  `variables` holds one data set per name, the last, as netCDF-c does;
  HDF-EOS files can hold several of one name (ASTER L1B has a `Latitude` and
  a `Longitude` per swath), told apart by their dimensions:

    ```python
    lat = {v.dimensions[0]: v for v in ds.hdf4_datasets() if v.name == 'Latitude'}
    vnir = lat['GeoTrack:VNIR_Swath'][:]
    ```

- **Several files as one:** `fastnc.MFDataset(files_or_glob, aggdim=None,
  exclude=(), check=False, master_file=None)`, netCDF4-python's
  `MFDataset`: variables whose first dimension is the aggregation dimension
  (by default the unlimited one) span every file, read as one array, and
  the rest come from the first (or master) file. netCDF4-python's takes
  only netCDF-3 and NETCDF4_CLASSIC files; this one any that fastnc opens.
- **Complex numbers:** `Dataset(path, auto_complex=True)`, as
  netCDF4-python 1.7: a compound of two floats named `r...` and `i...`
  (`{r, i}`, `{real, imag}`, h5py's complex datasets) reads as complex64 /
  complex128, unmasked as any compound; a float variable whose last
  dimension has length 2 and is named `ri`, `complex` or `_pfnc_complex`,
  or the start of one (`r`, `c`, `comp`; case counts), loses that dimension
  and reads as complex, masked and scaled as netCDF4-python does it.
  netCDF4-python's nc-complex also takes such a dimension in any position,
  and integer variables, then reads them wrong or not at all; fastnc leaves
  those as stored.
- **Module helpers:** `fastnc.default_fillvals`, `fastnc.chartostring()`,
  and `num2date`, `date2num`, `date2index` from cftime (installed
  separately), as netCDF4-python re-exports them.

Writing (`mode='w'`, `createVariable` and so on) raises an error: fastnc
reads only.

## Masking and scaling

By default, values come back exactly as netCDF4-python returns them:

- **Masked:** values equal to `_FillValue` or the default fill value, to
  `missing_value`, or outside `valid_min`, `valid_max` or `valid_range`,
  come back as a masked array. One difference, on purpose: a range typed
  like `scale_factor` / `add_offset` is in unpacked units, as the netCDF
  User Guide and CF say, and is applied to the unpacked values
  ([Accuracy](../reference/accuracy.md#where-fastnc-differs-on-purpose)).
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

Two more of fastnc's own:

```python
ds.set_auto_nan(True)                 # masked floats as NaN in plain arrays, as xarray gives them
var.read(key, mask=False, scale=False)    # this read alone raw; the variable's settings stay
var.read(key, nan=True)                   # ... or NaN-filled, or any of mask / scale / nan
```

`set_auto_nan` applies where masking is on and the values are floats
(scaled, or stored as floats): the masked points come back as NaN, in an
ordinary numpy array. Integers stay masked arrays, having no NaN. It costs
a pass over the masked points on every core (a GOES full disk: 32 ms, the
masked array 25, `np.ma.filled(v[:], np.nan)` 53). The per-read
arguments change nothing another thread reading the same variable sees,
unlike the `set_auto_*` switches.

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
